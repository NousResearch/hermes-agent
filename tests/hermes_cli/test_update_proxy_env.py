import os

from hermes_cli.update_proxy_env import restore_user_proxy_env


def test_marked_update_proxy_env_is_undone_and_the_users_own_env_kept():
    user = {"PATH": "/usr/bin", "no_proxy": "corp.internal", "ALL_PROXY": "socks5h://user:1080"}
    env = {**user, "HTTPS_PROXY": "http://10.0.0.7:7890", "https_proxy": "http://10.0.0.7:7890",
           "NO_PROXY": "corp.internal,localhost,10.0.0.0/8", "no_proxy": "corp.internal,localhost,10.0.0.0/8",
           "HERMES_UPDATE_SYSTEM_PROXY": "HTTPS_PROXY,https_proxy,NO_PROXY,PATH",
           "HERMES_UPDATE_USER_no_proxy": "corp.internal"}

    restore_user_proxy_env(env)

    assert env == user  # a key outside the proxy set (PATH) is never dropped
    unmarked = {"HTTPS_PROXY": "http://mine:3128", "NO_PROXY": "lan"}
    restore_user_proxy_env(unmarked)
    assert unmarked == {"HTTPS_PROXY": "http://mine:3128", "NO_PROXY": "lan"}
    assert "HERMES_UPDATE_SYSTEM_PROXY" not in os.environ



def test_a_proxy_the_users_dotenv_sets_survives_the_restore(tmp_path, monkeypatch):
    """`hermes update` restarts a gateway/backend with the Desktop's marked env; that process loads the
    user's .env (override) at import, then restores. The .env value is the user's own. The .env spells the
    proxy lower-case: on Windows that is the same variable as the inherited upper-case one."""
    from hermes_cli import env_loader

    monkeypatch.setattr(env_loader, "_DOTENV_PUBLISHED", {})
    monkeypatch.setattr(env_loader, "_LOADED_DOTENV_KEYS", set())
    monkeypatch.setenv("HTTPS_PROXY", "http://10.0.0.7:7890")
    monkeypatch.setenv("HTTP_PROXY", "http://10.0.0.7:7890")
    monkeypatch.setenv("NO_PROXY", "corp.internal,localhost,10.0.0.0/8")
    monkeypatch.setenv("HERMES_UPDATE_SYSTEM_PROXY", "HTTPS_PROXY,https_proxy,HTTP_PROXY,http_proxy")
    monkeypatch.setenv("HERMES_UPDATE_USER_NO_PROXY", "corp.internal")
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / ".env").write_text("https_proxy=http://user:pw@proxy.corp:3128\nNO_PROXY=lan.example\n", encoding="utf-8")

    env_loader.load_hermes_dotenv(hermes_home=home, load_external_secrets=False)
    restore_user_proxy_env()

    proxy_env = {key: os.environ.get(key) for key in ("https_proxy", "HTTP_PROXY", "NO_PROXY", "HERMES_UPDATE_USER_NO_PROXY")}
    assert proxy_env == {"https_proxy": "http://user:pw@proxy.corp:3128", "HTTP_PROXY": None, "NO_PROXY": "lan.example",
                         "HERMES_UPDATE_USER_NO_PROXY": None}



def _desktop_injected_env(monkeypatch, tmp_path, dotenv: str, own_no_proxy: str | None = None):
    """The env the Desktop hands `hermes update` (system proxy in both spellings), then the .env load."""
    from hermes_cli import env_loader

    monkeypatch.setattr(env_loader, "_DOTENV_PUBLISHED", {})
    monkeypatch.setattr(env_loader, "_LOADED_DOTENV_KEYS", set())
    for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        monkeypatch.setenv(key, "http://10.0.0.7:7890")
    marked = "HTTPS_PROXY,https_proxy,HTTP_PROXY,http_proxy"
    for key in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(key, ",".join(filter(None, [own_no_proxy, "localhost", "pypi.org"])))
        if own_no_proxy is None:
            marked += "," + key
        else:
            monkeypatch.setenv("HERMES_UPDATE_USER_" + key, own_no_proxy)
    monkeypatch.setenv("HERMES_UPDATE_SYSTEM_PROXY", marked)
    home = tmp_path / "hermes-home"
    home.mkdir(exist_ok=True)
    (home / ".env").write_text(dotenv, encoding="utf-8")
    env_loader.load_hermes_dotenv(hermes_home=home, load_external_secrets=False)


def test_hermes_update_sends_the_update_to_the_users_dotenv_proxy(tmp_path, monkeypatch):
    """S1: the .env load replaces only the spelling the file uses, and urllib (like curl) prefers the
    lower-case one. Once `hermes update` applied its .env, the .env proxy and NO_PROXY must win."""
    import urllib.request

    from hermes_cli.update_proxy_env import prefer_dotenv_proxy_env

    _desktop_injected_env(monkeypatch, tmp_path, "HTTPS_PROXY=http://proxy.corp:3128\nNO_PROXY=mirror.corp\n")
    prefer_dotenv_proxy_env()

    assert urllib.request.getproxies_environment().get("https") == "http://proxy.corp:3128"
    assert urllib.request.getproxies_environment().get("http") is None
    assert urllib.request.proxy_bypass_environment("mirror.corp")
    assert "HERMES_UPDATE_SYSTEM_PROXY" not in os.environ

    # A .env NO_PROXY alone replaces the user's old entries and keeps the bypass the Desktop added.
    _desktop_injected_env(monkeypatch, tmp_path, "NO_PROXY=mirror.corp\n", own_no_proxy="old.lan")
    prefer_dotenv_proxy_env()

    assert os.environ["NO_PROXY"] == os.environ["no_proxy"] == "mirror.corp,localhost,pypi.org"
