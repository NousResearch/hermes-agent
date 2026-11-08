"""Platform env UI registration must not prevent the platform's own native OAuth sign-in."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import ssl
import subprocess
import sys
from urllib.request import urlopen

import pytest

from tests.providers.fake_pkce_idp import FakeIdP
from tests.waiting import wait_until

pytestmark = pytest.mark.platforms("posix")  # native terminal stdin for auth add
ROOT = Path(__file__).resolve().parents[2]


def _plugin(home, name, entry=None, *, field="optional_env", names=(), kind="platform"):
    directory = home / "plugins" / name
    directory.mkdir(parents=True)
    manifest = {"name": directory.name, "version": "1.0.0", "description": "OAuth test platform",
                field: list(names)}
    if kind is not None:
        manifest["kind"] = kind
    if entry:
        manifest["oauth"] = [entry]
    (directory / "plugin.yaml").write_text(json.dumps(manifest), encoding="utf-8")
    (directory / "__init__.py").write_text("def register(ctx): pass\n", encoding="utf-8")
    return directory


def _entry(base, **overrides):
    return {"id": "rooms", "authorize_url": base + "/authorize", "token_url": base + "/token",
            "scopes": ["rooms"], "hosts": ["api.example.com"], "client_id_env": "NATIVE_CLIENT_ID",
            "client_secret_env": "NATIVE_CLIENT_SECRET", **overrides}


def _env(tmp_path, home):
    # No inherited agent/gateway flags, credentials, default profile or network proxies.
    env = {"HOME": str(tmp_path), "HERMES_HOME": str(home), "PATH": os.environ["PATH"],
           "PYTHONPATH": str(ROOT), "PYTHONNOUSERSITE": "1", "TERM": "dumb",
           "NATIVE_CLIENT_ID": "native-client", "NATIVE_CLIENT_SECRET": "native-secret"}
    for name in ("TMPDIR", "SYSTEMROOT"):
        if name in os.environ:
            env[name] = os.environ[name]
    return env


def _start(env, output, timeout="20"):
    import pty

    master, slave = pty.openpty()
    try:
        process = subprocess.Popen([sys.executable, "-m", "hermes_cli.main", "auth", "add", "native:rooms",
                                    "--no-browser", "--timeout", timeout],
                                   env=env, cwd=ROOT, stdin=slave, stdout=output, stderr=output)
    finally:
        os.close(slave)
    return process, master


@pytest.mark.parametrize("field", ["requires_env", "optional_env"])
def test_native_platform_auth_add_stores_grant(tmp_path, field):
    home = tmp_path / "profile"
    idp = FakeIdP().start(tls_dir=tmp_path / "tls")
    _plugin(home, "native", _entry(idp.base), field=field, names=("NATIVE_CLIENT_ID", "NATIVE_CLIENT_SECRET"))
    env = {**_env(tmp_path, home), "SSL_CERT_FILE": str(idp.ca_pem)}
    log = tmp_path / "auth.log"
    process = None
    master = None
    try:
        with log.open("w") as output:
            process, master = _start(env, output)
            def authorized_url():
                match = re.search(r"https://127\.0\.0\.1:[0-9]+/authorize\?[^\s\x1b]+", log.read_text())
                assert process.poll() is None or match, log.read_text()
                return match.group() if match else None
            url = wait_until(authorized_url, timeout=30, what="native auth authorize URL")
            with urlopen(url, timeout=10, context=ssl.create_default_context(cafile=str(idp.ca_pem))) as response:
                assert response.status == 200
            process.wait(timeout=30)
        assert process.returncode == 0, log.read_text()
        store = json.loads((home / "auth.json").read_text())
        assert store["plugin_oauth"]["native:rooms"]["access_token"]
        assert idp.token_requests[0]["client_id"] == "native-client"
        assert idp.token_requests[0]["client_secret"] == "native-secret"
        assert "native-secret" not in log.read_text()
    finally:
        if process and process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        if master is not None:
            os.close(master)
        idp.stop()


@pytest.mark.parametrize("name,other_kind", [
    ("OPENAI_API_KEY", "platform"), ("TELEGRAM_BOT_TOKEN", "platform"),
    ("DISCORD_BOT_TOKEN", "platform"), ("TEAMS_CLIENT_SECRET", "platform"),
    ("OTHER_CLIENT_ID", "platform"), ("OTHER_CLIENT_ID", "standalone"),
])
@pytest.mark.parametrize("field", ["client_id_env", "client_secret_env"])
def test_native_platform_auth_add_refuses_foreign_variables(tmp_path, name, other_kind, field):
    home = tmp_path / "profile"
    _plugin(home, "other", names=("OTHER_CLIENT_ID",), kind=other_kind)
    # Claiming the variable too must not steal its ownership from core or another plugin.
    _plugin(home, "native", _entry("https://login.example.com", **{field: name}),
            names=("NATIVE_CLIENT_ID", "NATIVE_CLIENT_SECRET", name))
    log = tmp_path / "denied.log"
    with log.open("w") as output:
        process, master = _start(_env(tmp_path, home), output)
        try:
            process.wait(timeout=30)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
            os.close(master)
    assert process.returncode != 0
    assert name in log.read_text()
    assert "declare the plugin's own" in log.read_text()
    assert not (home / "auth.json").exists()


@pytest.mark.parametrize("declaration", ["requires_env", "optional_env"])
@pytest.mark.parametrize("field", ["client_id_env", "client_secret_env"])
def test_native_standalone_auth_add_refuses_kindless_platform_variable(tmp_path, declaration, field):
    home = tmp_path / "profile"
    name = "LEGACY_BOT_TOKEN"
    # Config recognizes platforms/* by location, even when discovery defaults its kind to standalone.
    _plugin(home, "platforms/legacy", field=declaration, names=(name,), kind=None)
    _plugin(home, "native", _entry("https://login.example.com", **{field: name}), kind="standalone",
            names=("NATIVE_CLIENT_ID", "NATIVE_CLIENT_SECRET", name))
    log = tmp_path / "denied.log"
    with log.open("w") as output:
        process, master = _start({**_env(tmp_path, home), name: "legacy-secret"}, output, timeout="1")
        try:
            process.wait(timeout=15)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
            os.close(master)
    assert process.returncode != 0
    text = log.read_text()
    assert f"{name} is declared by another plugin (platforms/legacy); declare the plugin's own" in text, text
    assert "legacy-secret" not in text
    assert not (home / "auth.json").exists()
