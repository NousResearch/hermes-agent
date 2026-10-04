"""Plugin LLM calls re-enter the owning profile's scope outside any turn (#132887).

On a multiplexed gateway, plugins fire from contexts that never run inside a turn's
``_profile_runtime_scope`` (bridge host handlers, boot-time memory flushes). The
auxiliary runtime resolver reads ``OPENROUTER_BASE_URL`` through ``get_secret_str``
BEFORE the trusted ``model.base_url`` rung is consulted, so an unscoped host-owned
``ctx.llm`` call fails closed with ``UnscopedSecretError`` even when the profile's
config would answer. ``PluginLlm`` binds the PluginManager's immutable home and
re-enters its secret scope + HERMES_HOME override for the duration of the call; a
turn's own scope is never replaced, and non-multiplex deployments keep os.environ
semantics.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from agent.plugin_llm import PluginLlm, _TrustPolicy
from agent.secret_scope import (
    current_secret_scope,
    reset_secret_scope,
    set_secret_scope,
)

ENV_VALUE = "https://served-profile.example/v1"
TURN_VALUE = "https://turn-scope.example/v1"


def _fake_response(text: str = "ok") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, role="assistant"),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def _make_llm(home: Path, **kwargs) -> PluginLlm:
    return PluginLlm(
        plugin_id="demo",
        policy_loader=lambda _pid: _TrustPolicy(plugin_id="demo"),
        profile_home=str(home),
        **kwargs,
    )


def _install_probe(monkeypatch, record: dict) -> None:
    """Replace ``agent.auxiliary_client.call_llm`` with a probe that reads a profile secret
    exactly the way the runtime resolver's first line does — under multiplexing with no
    scope installed this is the line that failed closed."""
    from agent.secret_scope import get_secret_str

    def probe(**call_kwargs):
        from hermes_constants import get_hermes_home

        record["secret"] = get_secret_str("OPENROUTER_BASE_URL")
        record["scope_installed"] = current_secret_scope() is not None
        record["hermes_home"] = str(get_hermes_home())
        return _fake_response()

    monkeypatch.setattr("agent.auxiliary_client.call_llm", probe)


def test_unscoped_multiplex_call_reenters_owning_profile_scope(tmp_path, monkeypatch):
    home = tmp_path / "served-profile"
    home.mkdir()
    (home / ".env").write_text(f"OPENROUTER_BASE_URL={ENV_VALUE}\n")
    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", True)
    record: dict = {}
    _install_probe(monkeypatch, record)
    llm = _make_llm(home)

    result = llm.complete([{"role": "user", "content": "hi"}])

    assert result.text == "ok"
    assert record["secret"] == ENV_VALUE
    assert record["scope_installed"] is True
    assert Path(record["hermes_home"]).resolve() == home.resolve()
    # The re-entered scope is transient: nothing leaks past the call.
    assert current_secret_scope() is None


def test_turn_scoped_call_keeps_its_own_scope(tmp_path, monkeypatch):
    home = tmp_path / "served-profile"
    home.mkdir()
    (home / ".env").write_text(f"OPENROUTER_BASE_URL={ENV_VALUE}\n")
    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", True)
    record: dict = {}
    _install_probe(monkeypatch, record)
    llm = _make_llm(home)
    token = set_secret_scope(
        {"OPENROUTER_BASE_URL": TURN_VALUE}, profile_home=str(tmp_path / "turn")
    )
    try:
        llm.complete([{"role": "user", "content": "hi"}])
    finally:
        reset_secret_scope(token)

    assert record["secret"] == TURN_VALUE
    assert current_secret_scope() is None


def test_single_profile_deployment_stays_noop(tmp_path, monkeypatch):
    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", False)
    record: dict = {}
    _install_probe(monkeypatch, record)
    llm = _make_llm(tmp_path)

    llm.complete([{"role": "user", "content": "hi"}])

    assert record["scope_installed"] is False


def test_plugin_context_binds_manager_home(tmp_path):
    from hermes_cli.plugins import PluginContext, PluginManager

    manager = PluginManager(scope_key=str(tmp_path))
    manifest = SimpleNamespace(key="", name="demo")
    ctx = PluginContext(manifest=manifest, manager=manager)

    assert ctx.llm._profile_home == str(manager.home_path)
