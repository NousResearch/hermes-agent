"""Repro for #123747: the documented read-only pool resolver must not mutate the
credential store.

``_resolve_anthropic_pool_token`` documents itself as read-only ("never mutate
auth.json"), but it obtained the pool through ``load_pool``, whose seeding
PERSISTS the hermes_pkce singleton into auth.json (and takes ``auth.lock`` /
writes ``auth.json.corrupt``). The resolver now reads through ``peek_pool`` —
the same seeding in memory, never a write.

The HERMES_HOME skeleton / SOUL.md / ``backups/config`` copy are
``_load_config_impl``'s own cold-boot behavior (shared by every consumer,
idempotent, first-boot only), not this resolver's mutation — out of scope here.
"""
import json


def _snap(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


_CREDENTIAL_STORE_FILES = ("auth.json", "auth.lock", "auth.json.corrupt")


def test_resolve_anthropic_pool_token_does_not_mutate_the_credential_store(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HOME", str(tmp_path))
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    (home / "config.yaml").write_text("model:\n  provider: anthropic\n")
    (home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}, "credential_pool": {}}))
    (home / ".anthropic_oauth.json").write_text(json.dumps(
        {"accessToken": "sk-ant-oat01-example", "refreshToken": "r", "expiresAt": 4102444800000}))
    import hermes_cli.config  # noqa: F401
    import agent.anthropic_credentials as creds
    monkeypatch.setattr(creds, "read_claude_code_credentials", lambda: None)
    before = _snap(tmp_path)
    assert creds._resolve_anthropic_pool_token(skip_borrowed=True) == "sk-ant-oat01-example"
    after = _snap(tmp_path)
    changed = sorted(
        k for k in set(before) | set(after)
        if before.get(k) != after.get(k) and k.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] in _CREDENTIAL_STORE_FILES)
    assert changed == [], changed
    # The singleton seed happens in memory only: auth.json stays empty.
    assert json.loads((home / "auth.json").read_text())["credential_pool"] == {}


def test_peek_pool_never_persists(tmp_path, monkeypatch):
    """``peek_pool`` seeds the same rows load_pool would, in memory only."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HOME", str(tmp_path))
    (home / "config.yaml").write_text("model:\n  provider: anthropic\n")
    (home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}, "credential_pool": {}}))
    (home / ".anthropic_oauth.json").write_text(json.dumps(
        {"accessToken": "sk-ant-oat01-example", "refreshToken": "r", "expiresAt": 4102444800000}))
    from agent.credential_pool import load_pool, peek_pool

    peeked = peek_pool("anthropic")
    assert any((getattr(e, "access_token", None) or "") == "sk-ant-oat01-example"
               for e in peeked._entries)
    assert json.loads((home / "auth.json").read_text())["credential_pool"] == {}

    # The owning store's load_pool persists the same seed (the heal is its job).
    loaded = load_pool("anthropic")
    assert any((getattr(e, "access_token", None) or "") == "sk-ant-oat01-example"
               for e in loaded._entries)
    assert json.loads((home / "auth.json").read_text())["credential_pool"] != {}
