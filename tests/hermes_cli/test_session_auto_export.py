"""Auto-export writes a finished session to Markdown only when asked to."""

import json
from pathlib import Path
from types import SimpleNamespace

from agent import relay_runtime
from hermes_cli import lifecycle, observability, plugins, session_auto_export


def _session(session_id="sess-1", *, title="Nightly run", messages=None):
    if messages is None:
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
    return {
        "id": session_id,
        "title": title,
        "messages": messages,
    }


def _db(data):
    """Minimal SessionDB stand-in: export_session + close."""
    closed = []
    return (
        SimpleNamespace(
            export_session=lambda sid: data if data and data.get("id") == sid else None,
            close=lambda: closed.append(True),
        ),
        closed,
    )


def _config(**session_overrides):
    return {"session": {**session_overrides}}


def _exports(directory):
    """Export artifacts in a directory (the shared conftest drops its own
    scratch dirs into tmp_path, so a bare iterdir() is never empty)."""
    return sorted(
        p.name
        for p in directory.iterdir()
        if p.suffix in {".md", ".qmd"} or p.name == "manifest.jsonl"
    )


# ── resolve_settings ────────────────────────────────────────────────────────


def test_disabled_by_default():
    enabled, _, _ = session_auto_export.resolve_settings({"session": {}})
    assert enabled is False


def test_missing_session_section_is_not_an_error():
    enabled, _, fmt = session_auto_export.resolve_settings({})
    assert enabled is False
    assert fmt == "md"


def test_explicit_dir_is_expanded(monkeypatch):
    monkeypatch.setenv("HOME", "/home/tester")
    _, output_dir, _ = session_auto_export.resolve_settings(
        _config(auto_export=True, auto_export_dir="~/vault/hermes")
    )
    assert output_dir == Path("/home/tester/vault/hermes")


def test_empty_dir_falls_back_to_hermes_home(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "hermes_constants.get_hermes_home", lambda: tmp_path, raising=False
    )
    _, output_dir, _ = session_auto_export.resolve_settings(
        _config(auto_export=True, auto_export_dir="")
    )
    # Shares the directory `hermes sessions export` uses, so both paths
    # contribute to one manifest.
    assert output_dir == tmp_path / "session-exports"


def test_unknown_format_falls_back_to_md():
    _, _, fmt = session_auto_export.resolve_settings(
        _config(auto_export=True, auto_export_format="pdf")
    )
    assert fmt == "md"


def test_qmd_format_is_honored():
    _, _, fmt = session_auto_export.resolve_settings(
        _config(auto_export=True, auto_export_format="QMD")
    )
    assert fmt == "qmd"


# ── export_finalized_session ────────────────────────────────────────────────


def test_writes_nothing_when_disabled(tmp_path):
    db, _ = _db(_session())
    path = session_auto_export.export_finalized_session(
        "sess-1", config=_config(auto_export=False, auto_export_dir=str(tmp_path)), db=db
    )
    assert path is None
    assert _exports(tmp_path) == []


def test_writes_markdown_and_manifest_when_enabled(tmp_path):
    db, _ = _db(_session())
    path = session_auto_export.export_finalized_session(
        "sess-1", config=_config(auto_export=True, auto_export_dir=str(tmp_path)), db=db
    )

    assert path is not None and path.exists()
    assert path.suffix == ".md"
    text = path.read_text(encoding="utf-8")
    assert "hello" in text and "hi" in text

    manifest = tmp_path / "manifest.jsonl"
    entry = json.loads(manifest.read_text(encoding="utf-8").strip())
    assert entry["session_id"] == "sess-1"
    assert entry["format"] == "md"
    assert entry["path"] == str(path)


def test_empty_transcript_is_skipped(tmp_path):
    db, _ = _db(_session(messages=[]))
    path = session_auto_export.export_finalized_session(
        "sess-1", config=_config(auto_export=True, auto_export_dir=str(tmp_path)), db=db
    )
    # A `hermes` run that produced no turn must not litter the vault.
    assert path is None
    assert _exports(tmp_path) == []


def test_unknown_session_is_skipped(tmp_path):
    db, _ = _db(_session())
    path = session_auto_export.export_finalized_session(
        "nope", config=_config(auto_export=True, auto_export_dir=str(tmp_path)), db=db
    )
    assert path is None


def test_blank_session_id_is_skipped(tmp_path):
    db, _ = _db(_session())
    path = session_auto_export.export_finalized_session(
        "  ", config=_config(auto_export=True, auto_export_dir=str(tmp_path)), db=db
    )
    assert path is None


def test_second_finalize_overwrites_with_longer_transcript(tmp_path):
    """A resumed session finalizes again; the newer transcript is a superset."""
    cfg = _config(auto_export=True, auto_export_dir=str(tmp_path))

    db, _ = _db(_session())
    first = session_auto_export.export_finalized_session("sess-1", config=cfg, db=db)

    longer = _session(
        messages=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "and one more thing"},
        ]
    )
    db2, _ = _db(longer)
    second = session_auto_export.export_finalized_session("sess-1", config=cfg, db=db2)

    assert second == first
    assert "and one more thing" in second.read_text(encoding="utf-8")


def test_secrets_are_redacted(tmp_path):
    secret = "sk-ant-api03-" + "A" * 40
    db, _ = _db(
        _session(messages=[{"role": "user", "content": f"key is {secret}"}])
    )
    path = session_auto_export.export_finalized_session(
        "sess-1", config=_config(auto_export=True, auto_export_dir=str(tmp_path)), db=db
    )
    # Unattended writes into a synced vault get redaction unconditionally —
    # there is no operator present to pass --redact.
    assert secret not in path.read_text(encoding="utf-8")


def test_owned_db_is_closed(tmp_path, monkeypatch):
    data = _session()
    db, closed = _db(data)
    monkeypatch.setattr(
        "hermes_state.SessionDB", lambda *a, **k: db, raising=False
    )
    session_auto_export.export_finalized_session(
        "sess-1", config=_config(auto_export=True, auto_export_dir=str(tmp_path))
    )
    assert closed == [True]


def test_injected_db_is_not_closed(tmp_path):
    db, closed = _db(_session())
    session_auto_export.export_finalized_session(
        "sess-1", config=_config(auto_export=True, auto_export_dir=str(tmp_path)), db=db
    )
    assert closed == []


# ── finalize_session wiring ─────────────────────────────────────────────────


def _stub_finalize_deps(monkeypatch, calls):
    monkeypatch.setattr(
        observability, "observe_lifecycle", lambda name, **kw: calls.append("builtin")
    )
    monkeypatch.setattr(plugins, "invoke_hook", lambda name, **kw: calls.append("plugin") or [])
    monkeypatch.setattr(
        relay_runtime,
        "SESSION_COORDINATOR",
        SimpleNamespace(finalize_conversation=lambda **kw: calls.append("core")),
    )
    monkeypatch.setattr(relay_runtime, "current_profile_key", lambda: "profile-1")


def test_finalize_session_triggers_export(monkeypatch):
    calls = []
    _stub_finalize_deps(monkeypatch, calls)
    monkeypatch.setattr(
        session_auto_export,
        "export_finalized_session",
        lambda sid: calls.append(("export", sid)),
    )

    lifecycle.finalize_session(session_id="sess-1", platform="cli")

    assert ("export", "sess-1") in calls
    # After the Relay close, before plugin dispatch.
    assert calls.index("core") < calls.index(("export", "sess-1")) < calls.index("plugin")


def test_export_failure_does_not_break_finalize(monkeypatch):
    calls = []
    _stub_finalize_deps(monkeypatch, calls)

    def _boom(session_id):
        raise RuntimeError("disk full")

    monkeypatch.setattr(session_auto_export, "export_finalized_session", _boom)

    # Shutdown must survive a broken export.
    lifecycle.finalize_session(session_id="sess-1", platform="cli")

    assert "plugin" in calls


def test_finalize_without_session_id_skips_export(monkeypatch):
    calls = []
    _stub_finalize_deps(monkeypatch, calls)
    monkeypatch.setattr(
        session_auto_export,
        "export_finalized_session",
        lambda sid: calls.append("export"),
    )

    lifecycle.finalize_session(session_id="", platform="cli")

    assert "export" not in calls
