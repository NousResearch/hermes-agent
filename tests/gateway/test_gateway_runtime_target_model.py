"""The gateway resolves its default runtime against the configured model.

``_resolve_runtime_agent_kwargs`` is the credential resolution for every gateway-created
agent (turns, api_server, context-window probe). Every other caller of
``resolve_runtime_with_fallback`` passes the model it is about to send; the gateway did not.
Without a model ``pool.select(model=None)`` benches every entry carrying ANY model cooldown,
the ladder drops to the env/borrowed-token rung, and ``resolve_anthropic_token(model=None)``
hands out a token without a cooldown check — so a turn for a model the pool already knows is
spent on every seat starts on that spent quota and 429s, while the ``fallback_model`` chain
that resolution-time fallback exists for (#81209) is never walked.

Real credential pool against a real temp auth store and a real config.yaml.
"""
import json
import time

import pytest

KEY_A = "sk-ant-oat01-seat-a"
KEY_B = "sk-ant-oat01-seat-b"
KEY_CC = "sk-ant-oat01-claude-code"
MODEL_X = "claude-opus-4-5"
MODEL_Y = "claude-sonnet-4-5"
FIXTURE_KEY = "fixture-key"


def _entry(entry_id: str, key: str, priority: int, cooldowns: dict | None = None) -> dict:
    # OAuth rows: the env/borrowed rung hands those out model-blind (``_resolve_anthropic_pool_token``).
    row = {"id": entry_id, "label": entry_id, "auth_type": "oauth", "priority": priority,
           "source": "manual:hermes_pkce", "access_token": key, "refresh_token": f"rt-{entry_id}",
           "expires_at_ms": int((time.time() + 86400) * 1000)}
    if cooldowns:
        row["model_cooldowns"] = cooldowns
    return row


@pytest.fixture
def home(tmp_path, monkeypatch):
    root = tmp_path / "hermes-root"
    root.mkdir()
    (tmp_path / "fakehome").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "fakehome"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "fakehome"))
    for var in ("ANTHROPIC_TOKEN", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(root))
    import hermes_constants
    hermes_constants._default_hermes_root_memo = None  # type: ignore[attr-defined]
    (root / "config.yaml").write_text(json.dumps({
        "model": {"default": MODEL_X, "provider": "anthropic"},
        "providers": {"local-fixture": {"api": "http://127.0.0.1:1/v1", "api_key": FIXTURE_KEY}},
        "fallback_model": [{"provider": "local-fixture", "model": "fixture"}],
    }))
    return root


def _write_pool(home, entries):
    (home / "auth.json").write_text(json.dumps({"credential_pool": {"anthropic": entries}}))


def _write_claude_code_login(tmp_home, token):
    """A borrowed Claude Code login: the env/borrowed rung's last resort, read without a model check."""
    (tmp_home / ".credentials.json").write_text(json.dumps({"claudeAiOauth": {
        "accessToken": token, "refreshToken": f"rt-{token}", "expiresAt": int((time.time() + 86400) * 1000),
        "scopes": ["user:inference", "user:profile"]}}))


def test_default_model_benched_on_every_seat_walks_the_fallback_chain(home, tmp_path):
    benched = {MODEL_X: time.time() + 3600}
    _write_claude_code_login(tmp_path / "fakehome", KEY_CC)
    cc_row = dict(_entry("seat-cc", KEY_CC, 2, benched), source="claude_code")
    _write_pool(home, [_entry("seat-a", KEY_A, 0, benched), _entry("seat-b", KEY_B, 1, benched), cc_row])
    from gateway.run import _resolve_runtime_agent_kwargs

    runtime = _resolve_runtime_agent_kwargs()

    # Not the borrowed login's token on the spent quota — the configured chain.
    assert runtime["api_key"] == FIXTURE_KEY
    assert runtime["model"] == "fixture"
    assert runtime["provider"] == "local-fixture"
    pool = runtime.get("credential_pool")  # a fallback entry may carry its own pool, never the benched one
    assert pool is None or getattr(pool, "provider", "") != "anthropic"
    assert runtime.get("_fallback_notice")


def test_cooldown_on_another_model_keeps_the_pool_for_the_default_model(home):
    other = {MODEL_Y: time.time() + 3600}
    _write_pool(home, [_entry("seat-a", KEY_A, 0, other), _entry("seat-b", KEY_B, 1, other)])
    from gateway.run import _resolve_runtime_agent_kwargs

    runtime = _resolve_runtime_agent_kwargs()

    assert runtime["provider"] == "anthropic"
    assert runtime["api_key"] == KEY_A
    # Resolved THROUGH the pool (model-scoped select), not via the unscoped env/borrowed rung.
    assert runtime.get("credential_pool") is not None
    assert "model" not in runtime
