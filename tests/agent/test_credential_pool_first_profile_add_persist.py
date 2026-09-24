"""Regression test for #120177: the FIRST Codex OAuth credential added in a
named profile (neither the profile nor the global root has a Codex pool row)
must persist to the profile's own auth.json and read back.

Real imports, real temp HERMES_HOME root + named profile, real auth.json I/O.
"""
from __future__ import annotations

import json

import pytest


@pytest.fixture
def homes(tmp_path, monkeypatch):
    """Root HERMES_HOME whose auth.json has NO openai-codex pool rows."""
    root = tmp_path / "hermes-root"
    root.mkdir()
    (tmp_path / "fakehome").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "fakehome"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "fakehome"))
    for var in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CODEX_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(root))
    import hermes_constants
    hermes_constants._default_hermes_root_memo = None  # type: ignore[attr-defined]

    store = {"version": 1, "providers": {}, "credential_pool": {}}
    (root / "auth.json").write_text(json.dumps(store))

    import hermes_cli.auth as _auth_mod
    _auth_mod._oauth_heal_notices.clear()
    _auth_mod._oauth_heal_clean_marks.clear()

    def use(home):
        monkeypatch.setenv("HERMES_HOME", str(home))
        hermes_constants._default_hermes_root_memo = None  # type: ignore[attr-defined]
        import hermes_cli.auth as auth_mod
        auth_mod._global_auth_store_cache = None
        auth_mod._oauth_heal_clean_marks.clear()

    def pool_rows(home, provider="openai-codex"):
        p = home / "auth.json"
        if not p.exists():
            return None
        return (json.loads(p.read_text()).get("credential_pool") or {}).get(provider)

    return {"root": root, "use": use, "rows": pool_rows}


def test_first_codex_oauth_add_persists_to_named_profile(homes):
    from agent.credential_pool import AUTH_TYPE_OAUTH, PooledCredential, load_pool
    from hermes_cli.profiles import create_profile

    homes["use"](homes["root"])
    profile = create_profile("codexfirst")
    homes["use"](profile)

    pool = load_pool("openai-codex")
    assert pool.entries() == []
    assert pool._borrowed_root_ids == set()

    pool.add_entry(PooledCredential(
        provider="openai-codex", id="own001", label="mine",
        auth_type=AUTH_TYPE_OAUTH, priority=0, source="device_code",
        access_token="AT", refresh_token="RT",
    ))

    assert [r["id"] for r in (homes["rows"](profile) or [])] == ["own001"], \
        "first Codex OAuth add was not written to the named profile"
    assert not homes["rows"](homes["root"]), \
        "first profile-owned Codex credential leaked into the global root"

    reread = load_pool("openai-codex")
    assert [e.id for e in reread.entries()] == ["own001"], \
        "added Codex credential did not read back from the named profile"
