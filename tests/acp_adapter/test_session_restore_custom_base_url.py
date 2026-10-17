"""ACP session restore keeps a named custom provider's persisted endpoint (#135688).

The session row persists the runtime name ``custom`` plus the endpoint the conversation ran on.
``_make_agent`` resolved ``requested="custom"`` without that URL, the bare-custom resolve raised
``AuthError`` (caught at debug level), and the agent silently fell back to the config default
provider — whose URL the next persist wrote back, permanently rerouting the session. The restore
now hands the persisted http(s) URL back as ``explicit_base_url`` so the resolver rebuilds the
runtime on the stored endpoint; ``process://`` rows and URL-less sessions resolve as before.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import run_agent  # noqa: F401 — settles _early_recovery's gitdir probe at collection time, before

# home_io_guard activates; a worktree's gitdir lives under the real home tree (#135554 precedent).
import yaml

from acp_adapter.session import SessionManager


class _CapturingAgent:
    model = "fake-model"

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def acp_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("run_agent.AIAgent", _CapturingAgent)
    resolve_calls: list[dict] = []

    def fake_resolve(requested=None, explicit_base_url=None, **_kwargs):
        resolve_calls.append({
            "requested": requested,
            "explicit_base_url": explicit_base_url,
        })
        if requested == "custom" and explicit_base_url:
            # The direct-alias rung a real resolve takes for bare custom + explicit URL; the
            # credential itself is out of scope here (None — a no-auth local endpoint).
            return {
                "provider": "custom",
                "api_mode": "chat_completions",
                "base_url": explicit_base_url,
                "api_key": None,
            }
        return {
            "provider": "default-provider",
            "api_mode": "chat_completions",
            "base_url": "https://default.example/v1",
            "api_key": None,
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider", fake_resolve
    )
    monkeypatch.setattr(
        "acp_adapter.session._register_task_cwd", lambda task_id, cwd: None
    )
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build",
        lambda **_kwargs: None,
    )

    def _write_config(cfg: dict) -> None:
        (Path(os.environ["HERMES_HOME"]) / "config.yaml").write_text(
            yaml.safe_dump(cfg), encoding="utf-8"
        )

    return _write_config, resolve_calls


def test_custom_restore_forwards_persisted_base_url_to_resolver(monkeypatch, tmp_path):
    """A persisted http(s) base_url under runtime name ``custom`` reaches the resolver as
    ``explicit_base_url`` instead of resolving credential-less and falling back."""
    _write_config, resolve_calls = acp_env(monkeypatch, tmp_path)
    _write_config({"model": {"default": "sub-model", "provider": "process://sub"}})
    sm = SessionManager(db=None)
    sm._get_db = lambda: None

    agent = sm._make_agent(
        session_id="s1",
        cwd=".",
        model="local-model",
        requested_provider="custom",
        base_url="http://127.0.0.1:8080/v1",
    )

    assert resolve_calls[-1] == {
        "requested": "custom",
        "explicit_base_url": "http://127.0.0.1:8080/v1",
    }
    assert agent.kwargs["base_url"] == "http://127.0.0.1:8080/v1"
    assert agent.kwargs["provider"] == "custom"


def test_non_http_base_url_is_not_forwarded(monkeypatch, tmp_path):
    """A ``process://`` (or missing) base_url resolves exactly as before — no explicit URL."""
    _write_config, resolve_calls = acp_env(monkeypatch, tmp_path)
    _write_config({"model": {"default": "sub-model", "provider": "process://sub"}})
    sm = SessionManager(db=None)
    sm._get_db = lambda: None

    sm._make_agent(
        session_id="s1",
        cwd=".",
        requested_provider="process://sub",
        base_url="process://sub-provider",
    )
    assert resolve_calls[-1]["explicit_base_url"] is None

    sm._make_agent(session_id="s2", cwd=".", requested_provider="process://sub")
    assert resolve_calls[-1]["explicit_base_url"] is None


def test_restore_rebuilds_agent_on_the_stored_endpoint(monkeypatch, tmp_path):
    """End to end: an acp session row naming runtime ``custom`` with a local endpoint restores an
    agent pointed at that endpoint, not the config's default (process://) provider."""
    _write_config, _resolve_calls = acp_env(monkeypatch, tmp_path)
    _write_config({"model": {"default": "sub-model", "provider": "process://sub"}})

    row = {
        "source": "acp",
        "model": "local-model",
        "model_config": json.dumps({
            "provider": "custom",
            "base_url": "http://127.0.0.1:8080/v1",
            "cwd": str(tmp_path),
        }),
    }
    fake_db = SimpleNamespace(
        get_session=lambda sid: dict(row),
        reopen_session=lambda sid: None,
        get_messages_as_conversation=lambda sid, **_kw: [],
    )
    sm = SessionManager(db=None)
    sm._get_db = lambda: fake_db

    state = sm._restore("s1")

    assert state is not None
    assert state.agent.kwargs["provider"] == "custom"
    assert state.agent.kwargs["base_url"] == "http://127.0.0.1:8080/v1"
