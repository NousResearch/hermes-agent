"""Title normalization and background fallbacks must not persist active secrets."""
from types import SimpleNamespace

import pytest

from agent import title_generator
from agent.secret_scope import reset_secret_scope, set_secret_scope


@pytest.mark.parametrize("path", ["instant", "fallback", "generated"])
def test_title_projection_masks_after_whitespace_normalization(monkeypatch, path):
    secret = "normalized credential phrase"
    rendered = "normalized   credential\tphrase"
    stored = []
    db = SimpleNamespace(
        get_session_title_source=lambda _sid: "derived",
        get_conversation_root=lambda sid: sid,
        set_auto_title=lambda _sid, title, **kw: stored.append(title) or True,
    )
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)
    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", True)
    monkeypatch.setattr(title_generator, "_auto_title_enabled", lambda: True)
    monkeypatch.setattr(title_generator, "_title_language", lambda: "")
    monkeypatch.setattr(title_generator, "call_llm", lambda **kw: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=rendered))]
    ))
    token = set_secret_scope({"TITLE_TOKEN": secret})
    try:
        if path == "instant":
            title_generator.apply_instant_title(db, "fixture", rendered)
        elif path == "fallback":
            monkeypatch.setattr(title_generator, "generate_title", lambda *a, **kw: None)
            title_generator.auto_title_session(db, "fixture", rendered)
        else:
            title_generator.auto_title_session(db, "fixture", "ordinary request")
    finally:
        reset_secret_scope(token)
    assert stored == ["***"]
