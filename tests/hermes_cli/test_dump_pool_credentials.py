"""Persisted credential visibility in the public dump path (regression for #20675)."""
import json
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("auth_type,expected", [("oauth", "oauth"), ("api_key", "auth pool"), ("", "auth pool")])
def test_dump_reports_stored_credentials_without_disclosing_or_changing_them(
    tmp_path, monkeypatch, capsys, auth_type, expected
):
    from hermes_cli import dump
    from hermes_cli import auth

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(auth, "_load_global_auth_store", lambda: {})
    for env_var, _ in dump._API_KEYS:
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-openrouter")
    payload = {"version": 1, "providers": {}, "credential_pool": {
        "nous": [{"id": "fixture", "auth_type": auth_type,
                  "access_token": "fixture-token-never-display"}],
        "openai-codex": [{"id": "codex", "auth_type": "oauth",
                          "access_token": "fixture-codex-never-display"}],
    }}
    path = home / "auth.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    before = path.read_bytes()
    monkeypatch.setattr(dump, "load_hermes_dotenv", lambda **kw: None)
    monkeypatch.setattr(dump, "load_config", lambda: {})
    monkeypatch.setattr(dump, "_gateway_status", lambda: "fixture")
    monkeypatch.setattr(dump, "_version_line", lambda root: "fixture")
    monkeypatch.setattr(dump, "_openai_version", lambda: "fixture")
    # The dump must observe stored credentials, not lease/refresh/seed the pool.
    import agent.credential_pool as pools
    monkeypatch.setattr(pools, "load_pool", lambda *a, **kw: pytest.fail("mutating pool load"))
    dump.run_dump(SimpleNamespace(show_keys=True))
    output = capsys.readouterr().out
    rows = {line.strip().split()[0]: line for line in output.splitlines() if line.strip()}
    assert f"set ({expected})" in rows["nous"]
    assert "set (oauth)" in rows["pool:openai-codex"]
    assert "fixture-token-never-display" not in output
    assert "fixture-codex-never-display" not in output
    assert path.read_bytes() == before

    # Qwen's runtime source is an external CLI file, not a persisted pool.
    qwen = tmp_path / "qwen.json"
    qwen.write_text(json.dumps({"access_token": "fixture-qwen-never-display",
                                "refresh_token": "fixture-refresh", "expiry_date": 1}), encoding="utf-8")
    monkeypatch.setattr(auth, "_qwen_cli_auth_path", lambda: qwen)
    monkeypatch.setattr(auth, "_refresh_qwen_cli_tokens", lambda *a, **kw: pytest.fail("display refreshed token"))
    qwen_before = qwen.read_bytes()
    dump.run_dump(SimpleNamespace(show_keys=True))
    output = capsys.readouterr().out
    assert "qwen-cli" in output
    assert "fixture-qwen-never-display" not in output
    assert qwen.read_bytes() == qwen_before
    assert path.read_bytes() == before
    for bad_tokens in ["{broken", "[]", '{"access_token": ""}']:
        qwen.write_text(bad_tokens, encoding="utf-8")
        dump.run_dump(SimpleNamespace(show_keys=True))
        assert "qwen-cli" not in capsys.readouterr().out
        assert qwen.read_text(encoding="utf-8") == bad_tokens


@pytest.mark.parametrize("pool,env,expected", [
    ({}, None, "not set"),
    ({"nous": [None, {"access_token": ""}, {"access_token": 7}]}, None, "not set"),
    ({"nous": [{"access_token": "stored", "auth_type": "oauth"}]}, "env-secret", "set"),
    ({"nous": [{"access_token": "api", "auth_type": "api_key"},
                {"access_token": "oauth", "auth_type": "oauth"}]}, None, "set (auth pool, oauth)"),
    (OSError("fixture read failure"), None, "not set"),
])
def test_dump_pool_fallback_preserves_environment_and_handles_bad_rows(monkeypatch, pool, env, expected):
    from hermes_cli import dump, auth
    monkeypatch.delenv("NOUS_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-openrouter")
    if env:
        monkeypatch.setenv("NOUS_API_KEY", env)
    def read():
        if isinstance(pool, Exception):
            raise pool
        return pool
    monkeypatch.setattr(auth, "read_credential_pool", read)
    line = next(row for row in dump._api_key_lines(False) if row.strip().startswith("nous "))
    assert line.strip().removeprefix("nous").strip().split(" (shell only")[0] == expected
    assert "env-secret" not in line
