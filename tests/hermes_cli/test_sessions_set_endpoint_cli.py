"""CLI tests for ``hermes sessions set-endpoint`` (issue #77831).

Follows the FakeDB pattern of tests/hermes_cli/test_sessions_delete.py:
monkeypatch ``hermes_state.SessionDB`` with a fake, drive ``main()`` via
``sys.argv``, and assert on captured writes + stdout.
"""

import json
import sys

import pytest


SESSION_ID = "20260315_092437_c9a6ff"
OLD_URL = "http://model-host.local:8355/v1"
NEW_URL = "http://203.0.113.7:8355/v1"


def _run(monkeypatch, capsys, argv_tail, row, **rt_patches):
    """Run `hermes sessions set-endpoint <argv_tail>` against a FakeDB."""
    import hermes_cli.main as main_mod
    import hermes_cli.runtime_provider as rt_mod
    import hermes_state

    captured = {"meta": None, "billing": None, "closed": False}

    class FakeDB:
        def resolve_session_id(self, session_id):
            captured["resolved_from"] = session_id
            return SESSION_ID if session_id.startswith("20260315") else None

        def get_session(self, session_id):
            return dict(row)

        def update_session_meta(self, session_id, model_config_json, model=None):
            captured["meta"] = (session_id, model_config_json)

        def update_session_billing_route(
            self, session_id, *, provider, base_url, billing_mode=None
        ):
            captured["billing"] = (session_id, provider, base_url, billing_mode)

        def close(self):
            captured["closed"] = True

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: FakeDB())
    # Config-backed lookups are imported lazily inside the helper — patch the
    # module of origin.
    monkeypatch.setattr(
        rt_mod,
        "find_custom_provider_identity",
        rt_patches.get("find", lambda url: None),
    )
    monkeypatch.setattr(
        rt_mod, "_get_named_custom_provider", rt_patches.get("entry", lambda pid: None)
    )
    monkeypatch.setattr(sys, "argv", ["hermes", "sessions", "set-endpoint", *argv_tail])

    main_mod.main()
    return captured, capsys.readouterr().out


def _stuck_row():
    """The issue's repro row: session pinned to the dead endpoint."""
    return {
        "id": SESSION_ID,
        "model_config": json.dumps({
            "provider": "custom:model-host.local:8355",
            "base_url": OLD_URL,
            "model": "qwen3",
        }),
        "billing_provider": "custom",
        "billing_base_url": OLD_URL,
        "billing_mode": "chat_completions",
    }


def test_set_endpoint_rewrites_row_and_billing(monkeypatch, capsys):
    captured, output = _run(monkeypatch, capsys, [SESSION_ID, NEW_URL], _stuck_row())

    assert f"Session '{SESSION_ID}' re-pointed to {NEW_URL}." in output
    assert "stored bare 'custom'" in output

    # model_config: provider/base_url rewritten, other knobs preserved.
    sid, meta_json = captured["meta"]
    assert sid == SESSION_ID
    meta = json.loads(meta_json)
    assert meta["provider"] == "custom"
    assert meta["base_url"] == NEW_URL
    assert meta["model"] == "qwen3"

    # billing route rewritten; billing_mode preserved (None -> COALESCE).
    bsid, provider, base_url, billing_mode = captured["billing"]
    assert bsid == SESSION_ID
    assert provider == "custom"
    assert base_url == NEW_URL
    assert billing_mode is None
    assert captured["closed"] is True


def test_set_endpoint_uses_configured_entry_slug(monkeypatch, capsys):
    captured, output = _run(
        monkeypatch,
        capsys,
        [SESSION_ID, NEW_URL],
        _stuck_row(),
        find=lambda url: "custom:local",
    )

    assert "derived from the configured entry" in output
    assert "  provider: custom:local (was custom:model-host.local:8355)" in output
    meta = json.loads(captured["meta"][1])
    assert meta["provider"] == "custom:local"
    assert meta["base_url"] == NEW_URL
    # custom:* slugs bill under the bare class.
    assert captured["billing"][1] == "custom"
    assert captured["billing"][2] == NEW_URL


def test_set_endpoint_keeps_builtin_provider(monkeypatch, capsys):
    row = dict(_stuck_row())
    row["model_config"] = json.dumps({
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com",
    })
    captured, output = _run(
        monkeypatch,
        capsys,
        [SESSION_ID, NEW_URL],
        row,
    )

    assert "  provider: anthropic" in output
    meta = json.loads(captured["meta"][1])
    assert meta["provider"] == "anthropic"
    assert meta["base_url"] == NEW_URL
    assert captured["billing"][1] == "anthropic"


def test_set_endpoint_explicit_provider(monkeypatch, capsys):
    captured, output = _run(
        monkeypatch,
        capsys,
        [SESSION_ID, NEW_URL, "--provider", "custom:local"],
        _stuck_row(),
        entry=lambda pid: {"name": "local", "base_url": NEW_URL},
    )

    assert "re-pointed to" in output
    meta = json.loads(captured["meta"][1])
    assert meta["provider"] == "custom:local"
    assert meta["base_url"] == NEW_URL
    assert captured["billing"][1] == "custom"


def test_set_endpoint_unknown_provider_rejected(monkeypatch, capsys):
    captured, output = _run(
        monkeypatch,
        capsys,
        [SESSION_ID, NEW_URL, "--provider", "custom:nope"],
        _stuck_row(),
    )

    assert "Error: Unknown provider 'custom:nope'" in output
    assert captured["meta"] is None
    assert captured["billing"] is None


def test_set_endpoint_session_not_found(monkeypatch, capsys):
    captured, output = _run(monkeypatch, capsys, ["nope", NEW_URL], _stuck_row())

    assert "Session 'nope' not found." in output
    assert captured["meta"] is None
    assert captured["billing"] is None


def test_set_endpoint_invalid_url(monkeypatch, capsys):
    captured, output = _run(
        monkeypatch, capsys, [SESSION_ID, "ftp://host/v1"], _stuck_row()
    )

    assert "Error: invalid endpoint URL 'ftp://host/v1'" in output
    assert captured["meta"] is None
    assert captured["billing"] is None


def test_set_endpoint_unique_prefix_resolution(monkeypatch, capsys):
    captured, output = _run(monkeypatch, capsys, ["20260315", NEW_URL], _stuck_row())

    assert captured["resolved_from"] == "20260315"
    assert f"Session '{SESSION_ID}' re-pointed to {NEW_URL}." in output


@pytest.fixture(autouse=True)
def _isolate_hermes_home(monkeypatch, tmp_path):
    """Keep real config out of the lookups (conftest-level safety net)."""
    import os

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    os.environ.pop("HERMES_INFERENCE_PROVIDER", None)
