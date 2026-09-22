"""Write-guard tests — managed keys can't be set/removed by the user."""
import pytest


@pytest.fixture
def homes(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    import hermes_cli.config as cfg
    from hermes_cli import managed_scope

    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    managed_scope.invalidate_managed_cache()
    (managed / "config.yaml").write_text(
        "model:\n  default: managed/model\n", encoding="utf-8"
    )
    managed_scope.invalidate_managed_cache()
    return home, managed


def test_config_set_managed_key_rejected(homes, capsys):
    from hermes_cli.config import set_config_value

    with pytest.raises(SystemExit) as exc:
        set_config_value("model.default", "user/override")
    assert exc.value.code != 0
    captured = capsys.readouterr()
    assert "managed" in (captured.out + captured.err).lower()




# ── env write guards ─────────────────────────────────────────────────────────


@pytest.fixture
def env_homes(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    (managed / ".env").write_text(
        "OPENAI_API_BASE=https://org.example/v1\n", encoding="utf-8"
    )
    from hermes_cli import managed_scope

    managed_scope.invalidate_managed_cache()
    return home, managed


def test_save_env_value_managed_key_rejected(env_homes, capsys):
    from hermes_cli.config import save_env_value, get_env_path

    save_env_value("OPENAI_API_BASE", "https://user.example/v1")
    assert "managed" in capsys.readouterr().err.lower()
    env_path = get_env_path()
    body = env_path.read_text() if env_path.exists() else ""
    assert "user.example" not in body


# ── refused .env writes must fail, not half-apply ────────────────────────────

# Runtime-constructed fake credentials (never literal key-shaped strings).
ADMIN_KEY = "sk-ds-" + "a" * 24
OLD_KEY = "sk-ds-" + "b" * 24
NEW_KEY = "sk-ds-" + "c" * 24


@pytest.fixture
def pinned_env(tmp_path, monkeypatch):
    """The administrator pins DEEPSEEK_API_KEY and TELEGRAM_HOME_CHANNEL; the user's own stores
    still hold an older key in .env, a config.yaml mirror of it, its env-seeded pool entry, and a
    stale top-level config.yaml copy of the channel."""
    import json

    import hermes_cli.config as cfg
    from hermes_cli import managed_scope

    home, managed = tmp_path / "home", tmp_path / "managed"
    home.mkdir()
    managed.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    (managed / ".env").write_text(
        f"DEEPSEEK_API_KEY={ADMIN_KEY}\nTELEGRAM_HOME_CHANNEL=111\n", encoding="utf-8")
    (home / ".env").write_text(
        f"DEEPSEEK_API_KEY={OLD_KEY}\nTELEGRAM_HOME_CHANNEL=222\n", encoding="utf-8")
    (home / "config.yaml").write_text(
        f"model:\n  provider: deepseek\n  default: deepseek-chat\n  api_key: {OLD_KEY}\n"
        "TELEGRAM_HOME_CHANNEL: '333'\n", encoding="utf-8")
    (home / "auth.json").write_text(json.dumps({"credential_pool": {"deepseek": [{
        "id": "e1", "label": "env", "auth_type": "api_key", "priority": 0,
        "source": "env:DEEPSEEK_API_KEY", "access_token": OLD_KEY}]}}), encoding="utf-8")
    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    cfg.invalidate_env_cache()
    managed_scope.invalidate_managed_cache()
    return home, _user_stores(home)


def _user_stores(home):
    return {name: (home / name).read_text(encoding="utf-8") for name in (".env", "config.yaml", "auth.json")}


def _assert_user_stores_untouched(home, before):
    assert _user_stores(home) == before


@pytest.mark.parametrize("command, args", [
    pytest.param("set", ("DEEPSEEK_API_KEY", NEW_KEY), id="set-credential"),
    pytest.param("set", ("TELEGRAM_HOME_CHANNEL", "999"), id="set-env-setting"),
    pytest.param("unset", ("DEEPSEEK_API_KEY",), id="unset-credential"),
    pytest.param("unset", ("TELEGRAM_HOME_CHANNEL",), id="unset-env-setting"),
])
def test_config_set_unset_of_pinned_env_key_exits_nonzero_and_writes_nothing(pinned_env, capsys, command, args):
    from hermes_cli.config import set_config_value, unset_config_value

    home, before = pinned_env
    with pytest.raises(SystemExit) as exc:
        (set_config_value if command == "set" else unset_config_value)(*args)
    assert exc.value.code == 1
    out = capsys.readouterr()
    assert "managed by your administrator" in out.err
    assert "✓" not in out.out
    _assert_user_stores_untouched(home, before)


@pytest.mark.parametrize("method", ["PUT", "DELETE"])
@pytest.mark.parametrize("lock, refusal", [
    pytest.param("scope", "managed by your administrator", id="managed-scope"),
    pytest.param("install", "managed by nixos", id="package-managed"),
])
def test_api_env_write_of_locked_key_is_refused_and_writes_nothing(pinned_env, monkeypatch, method, lock, refusal):
    from fastapi.testclient import TestClient

    from hermes_cli.web_server import _SESSION_TOKEN, app

    if lock == "install":
        monkeypatch.setenv("HERMES_MANAGED", "nixos")
    home, before = pinned_env
    body = {"key": "DEEPSEEK_API_KEY", **({"value": NEW_KEY} if method == "PUT" else {})}
    resp = TestClient(app).request(method, "/api/env", json=body, headers={"X-Hermes-Session-Token": _SESSION_TOKEN})
    assert resp.status_code == 400
    assert refusal in resp.json()["detail"]
    _assert_user_stores_untouched(home, before)


_ANTHROPIC_ENV_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN")


@pytest.fixture
def anthropic_homes(tmp_path, monkeypatch):
    """A user home whose auth.json holds a manually added anthropic key, and an empty managed dir."""
    import json

    import hermes_cli.config as cfg
    from hermes_cli import managed_scope

    home, managed = tmp_path / "home", tmp_path / "managed"
    home.mkdir()
    managed.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    for name in _ANTHROPIC_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    (home / "auth.json").write_text(json.dumps({"credential_pool": {"anthropic": [{
        "id": "m1", "label": "manual", "auth_type": "api_key", "priority": 0,
        "source": "manual", "access_token": OLD_KEY}]}}), encoding="utf-8")

    def reload():
        cfg.invalidate_env_cache()
        managed_scope.invalidate_managed_cache()

    reload()
    return home, managed, reload


def _disconnect(slug):
    from tui_gateway import server

    return server._methods["model.disconnect"](1, {"slug": slug})


def test_disconnect_on_a_package_managed_install_still_clears_the_auth_store(anthropic_homes, monkeypatch):
    """The .env lock refuses every key on such an install, but a key that holds nothing is nothing to refuse."""
    import json

    home, _managed, _reload = anthropic_homes
    monkeypatch.setenv("HERMES_MANAGED", "nixos")
    resp = _disconnect("anthropic")
    assert resp.get("result", {}).get("disconnected") is True, resp
    assert "anthropic" not in json.loads((home / "auth.json").read_text(encoding="utf-8"))["credential_pool"]


def test_disconnect_refuses_a_pinned_key_before_removing_any_other(anthropic_homes):
    home, managed, reload = anthropic_homes
    (managed / ".env").write_text(f"ANTHROPIC_TOKEN={ADMIN_KEY}\n", encoding="utf-8")
    (home / ".env").write_text(f"ANTHROPIC_API_KEY={OLD_KEY}\n", encoding="utf-8")
    reload()
    before = {name: (home / name).read_text(encoding="utf-8") for name in (".env", "auth.json")}
    resp = _disconnect("anthropic")
    assert "managed by your administrator" in resp.get("error", {}).get("message", ""), resp
    assert {name: (home / name).read_text(encoding="utf-8") for name in (".env", "auth.json")} == before


# ── bulk save strips managed leaves ──────────────────────────────────────────


