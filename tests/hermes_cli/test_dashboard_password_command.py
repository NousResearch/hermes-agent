"""``hermes dashboard password``: set the dashboard sign-in without the TTY prompt ``hermes dashboard`` offers.

The command writes what that prompt writes (``dashboard.basic_auth``: username, an scrypt hash, a signing
secret when ``config.yaml`` has none; see ``test_dashboard_auth_prompt.py``), reads the password from
standard input with ``--password-stdin``, never from argv, and never prints or stores it. The same
credentials then sign in on an old ``hermes serve`` bound to a wildcard address and on the gateway's own
Webapp listener, each started after the change; a listener restarted after a change ends the sign-ins
made with the old password.
"""
import contextlib
import json
import os
import stat
import subprocess
import sys
import threading

import httpx
import pytest
import hermes_yaml as yaml

from plugins.dashboard_auth.basic import _verify_password, hash_password
from tests.hermes_cli.live_gateway import REPO_ROOT, free_port, home_env, live_gateway, run_cli, write_config

pytestmark = pytest.mark.platforms("linux")

USER, PASSWORD = "spike-admin", "throwaway-Pw-7f3a9c"  # invented for this test


def _home(tmp_path):
    home, user = tmp_path / "state", tmp_path / "user"
    home.mkdir(mode=0o700)
    user.mkdir()
    write_config(home)
    return home, home_env(home, user)


def _set_password(env, password=PASSWORD, *extra):
    return run_cli(env, "dashboard", "password", "--username", USER, "--password-stdin", *extra,
                   stdin=password + "\n")


_COMMAND = [sys.executable, "-m", "hermes_cli.main", "dashboard", "password", "--username", USER, "--password-stdin"]


def _set_password_bytes(env, stdin: bytes) -> subprocess.CompletedProcess:
    return subprocess.run(_COMMAND, cwd=REPO_ROOT, env=env, input=stdin, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=120, check=False)


def _basic_auth(home) -> dict:
    return (yaml.safe_load((home / "config.yaml").read_text()).get("dashboard") or {}).get("basic_auth") or {}


def _files_holding(home, needle: str) -> list[str]:
    return [str(p) for p in home.rglob("*") if p.is_file() and needle.encode() in p.read_bytes()]


def _sign_in(base: str, password: str) -> tuple[int, int | None]:
    """(password-login status, /api/auth/me status after it, None when the login failed)."""
    with httpx.Client(base_url=base, timeout=20) as client:
        login = client.post("/auth/password-login",
                            json={"provider": "basic", "username": USER, "password": password, "next": "/"})
        if login.status_code != 200:
            return login.status_code, None
        me = client.get("/api/auth/me")
        return login.status_code, me.status_code if me.json().get("user_id") == USER else -1


def _signed_in(client: httpx.Client) -> bool:
    me = client.get("/api/auth/me")
    return me.status_code == 200 and me.json().get("user_id") == USER


def test_the_password_is_stored_as_a_hash_with_a_signing_secret_and_never_shown(tmp_path):
    home, env = _home(tmp_path)
    result = _set_password(env)
    assert result.returncode == 0, result.stdout
    section = _basic_auth(home)
    # The keys the interactive prompt stores (test_dashboard_auth_prompt.py); the emptied plaintext is not written.
    assert sorted(section) == ["password_hash", "secret", "username"]
    assert section["username"] == USER
    assert _verify_password(PASSWORD, section["password_hash"])
    assert PASSWORD not in result.stdout
    assert _files_holding(home, PASSWORD) == []
    assert stat.S_IMODE((home / "config.yaml").stat().st_mode) == 0o600
    assert "The gateway is running" not in result.stdout  # no gateway in this home
    assert "A gateway or hermes dashboard / hermes serve started from now on uses it." in result.stdout


def test_with_the_secret_in_the_environment_a_secret_is_still_written_like_the_prompt_does(tmp_path):
    home, env = _home(tmp_path)
    env["HERMES_DASHBOARD_BASIC_AUTH_SECRET"] = "an-invented-environment-signing-secret-0123456789"
    assert _set_password(env).returncode == 0
    assert _basic_auth(home)["secret"]


def test_without_a_terminal_or_password_stdin_nothing_is_written(tmp_path):
    home, env = _home(tmp_path)
    result = run_cli(env, "dashboard", "password", "--username", USER, stdin=PASSWORD + "\n")
    assert result.returncode == 2, result.stdout
    assert "--password-stdin" in result.stdout
    assert _basic_auth(home) == {}


def test_an_empty_password_is_refused(tmp_path):
    home, env = _home(tmp_path)
    for stdin in (b"\n", b""):
        result = _set_password_bytes(env, stdin)
        assert result.returncode == 2
        assert b"Empty password; nothing was changed" in result.stdout
    assert _basic_auth(home) == {}


def test_only_the_first_line_counts_and_its_line_ending_is_dropped(tmp_path):
    home, env = _home(tmp_path)
    result = _set_password_bytes(env, f"{PASSWORD}\r\nsecond line\n".encode())
    assert result.returncode == 0, result.stdout
    assert _verify_password(PASSWORD, _basic_auth(home)["password_hash"])


def test_a_lone_carriage_return_is_part_of_the_password(tmp_path):
    home, env = _home(tmp_path)
    assert _set_password_bytes(env, b"throwaway-cr\r").returncode == 0
    assert _verify_password("throwaway-cr\r", _basic_auth(home)["password_hash"])


def test_password_stdin_on_a_terminal_is_refused_before_anything_is_typed(tmp_path):
    home, env = _home(tmp_path)
    master, slave = os.openpty()
    try:
        result = subprocess.run(_COMMAND, cwd=REPO_ROOT, env=env, stdin=slave, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=120, check=False, start_new_session=True)
    finally:
        os.close(slave)
        os.close(master)
    assert result.returncode == 2, result.stdout
    assert "run without --password-stdin to type it at a hidden prompt" in result.stdout.decode()
    assert _basic_auth(home) == {}


def test_a_non_ascii_password_is_read_as_utf8(tmp_path):
    home, env = _home(tmp_path)
    password = "thröwaway-Pässwort-✓"
    assert _set_password_bytes(env, password.encode("utf-8")).returncode == 0  # no trailing newline either
    assert _verify_password(password, _basic_auth(home)["password_hash"])


@pytest.mark.parametrize("stdin, reason", [(b"x" * 1025 + b"\n", "longer than 1024 bytes"),
                                           (b"x" * 1024 + b"\r", "longer than 1024 bytes"),
                                           (b"\xff\xfe-not-utf8\n", "not UTF-8")],
                         ids=["1025-bytes", "1024-bytes-and-a-lone-cr", "not-utf8"])
def test_an_overlong_or_undecodable_password_is_refused(tmp_path, stdin, reason):
    home, env = _home(tmp_path)
    result = _set_password_bytes(env, stdin)
    assert result.returncode == 2, result.stdout
    assert reason in result.stdout.decode()
    assert _basic_auth(home) == {}


def test_a_password_in_dotenv_that_would_override_it_is_removed_and_named(tmp_path):
    home, env = _home(tmp_path)
    dotenv = home / ".env"
    dotenv.write_text("HERMES_DASHBOARD_BASIC_AUTH_PASSWORD=stale-env-value\nOTHER_KEY=kept\n")
    dotenv.chmod(0o600)
    result = _set_password(env)
    assert result.returncode == 0, result.stdout
    assert "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD" in result.stdout
    assert "stale-env-value" not in result.stdout
    text = dotenv.read_text()
    assert "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD" not in text and "OTHER_KEY=kept" in text
    assert stat.S_IMODE(dotenv.stat().st_mode) == 0o600
    assert "⚠" not in result.stdout  # the .env copy is not the calling environment


def test_a_key_the_administrator_pins_is_refused_before_config_yaml_changes(tmp_path):
    home, env = _home(tmp_path)
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / ".env").write_text("HERMES_DASHBOARD_BASIC_AUTH_PASSWORD=pinned-by-admin\n")
    before = (home / "config.yaml").read_bytes()
    result = _set_password({**env, "HERMES_MANAGED_DIR": str(managed)})
    assert result.returncode == 1, result.stdout
    assert "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD" in result.stdout and "nothing was changed" in result.stdout
    assert "pinned-by-admin" not in result.stdout
    assert (home / "config.yaml").read_bytes() == before


def test_changing_it_keeps_the_signing_secret(tmp_path):
    home, env = _home(tmp_path)
    assert _set_password(env).returncode == 0
    first = _basic_auth(home)
    assert _set_password(env, "another-throwaway-Pw").returncode == 0
    second = _basic_auth(home)
    assert second["secret"] == first["secret"]
    assert second["password_hash"] != first["password_hash"]


def test_password_rotation_preserves_the_username_from_this_homes_dotenv(tmp_path):
    home, env = _home(tmp_path)
    assert _set_password(env).returncode == 0
    (home / ".env").write_text("HERMES_DASHBOARD_BASIC_AUTH_USERNAME=local-operator\n")
    result = run_cli(env, "dashboard", "password", "--password-stdin", stdin="new-test-input\n")
    assert result.returncode == 0, result.stdout
    assert _basic_auth(home)["username"] == "local-operator"
    assert _verify_password("new-test-input", _basic_auth(home)["password_hash"])
    assert "HERMES_DASHBOARD_BASIC_AUTH_USERNAME" not in (home / ".env").read_text()


@pytest.mark.parametrize("template", ["${HERMES_DASHBOARD_BASIC_AUTH_USERNAME}",
                                      "${env:HERMES_DASHBOARD_BASIC_AUTH_USERNAME}"])
def test_a_username_template_on_an_override_it_removes_is_refused_before_any_write(tmp_path, template):
    home, env = _home(tmp_path)
    write_config(home, {"dashboard": {"basic_auth": {"username": template}}})
    (home / ".env").write_text("HERMES_DASHBOARD_BASIC_AUTH_USERNAME=local-operator\n")
    config_before, dotenv_before = (home / "config.yaml").read_bytes(), (home / ".env").read_bytes()
    result = run_cli(env, "dashboard", "password", "--password-stdin", stdin="new-test-input\n")
    assert result.returncode == 1, result.stdout
    assert "Set dashboard.basic_auth.username to a literal first" in result.stdout
    assert (home / "config.yaml").read_bytes() == config_before
    assert (home / ".env").read_bytes() == dotenv_before


def test_a_named_profile_gets_its_own_sign_in(tmp_path):
    _, env = _home(tmp_path)
    default_root = tmp_path / "user" / ".hermes"
    profile = default_root / "profiles" / "work"
    profile.mkdir(parents=True)
    write_config(default_root)
    write_config(profile)
    env.pop("HERMES_HOME")
    result = run_cli(env, "-p", "work", "dashboard", "password", "--username", USER, "--password-stdin",
                     stdin=PASSWORD + "\n")
    assert result.returncode == 0, result.stdout
    assert _verify_password(PASSWORD, _basic_auth(profile)["password_hash"])
    assert _basic_auth(default_root) == {}
    assert "This is profile 'work''s own sign-in" in result.stdout
    assert "set those with hermes -p default dashboard password." in result.stdout  # the shared listeners'


def _wait_ready(proc) -> list[str]:
    lines, hit = [], threading.Event()

    def pump():
        for line in proc.stdout:
            lines.append(line)
            if "HERMES_BACKEND_READY" in line:
                hit.set()

    threading.Thread(target=pump, daemon=True).start()
    assert hit.wait(120), "".join(lines)
    return lines


@contextlib.contextmanager
def _old_backend(env, port):
    backend = subprocess.Popen([sys.executable, "-m", "hermes_cli.main", "serve", "--host", "0.0.0.0",
                                "--port", str(port)], cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    try:
        _wait_ready(backend)
        yield f"http://127.0.0.1:{port}"
    finally:
        backend.terminate()
        try:
            backend.wait(timeout=30)
        except subprocess.TimeoutExpired:
            backend.kill()
            backend.wait(timeout=5)


def test_the_new_password_signs_in_on_an_old_backend_on_a_wildcard_address(tmp_path):
    _, env = _home(tmp_path)
    assert _set_password(env).returncode == 0
    with _old_backend(env, free_port()) as base:
        assert _sign_in(base, PASSWORD) == (200, 200)
        assert _sign_in(base, "wrong-password")[0] == 401


def test_a_restart_keeps_sign_ins_and_one_after_a_change_ends_them(tmp_path):
    _, env = _home(tmp_path)
    assert _set_password(env).returncode == 0
    port = free_port()
    with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=20) as client:
        with _old_backend(env, port):
            login = client.post("/auth/password-login",
                                json={"provider": "basic", "username": USER, "password": PASSWORD, "next": "/"})
            assert login.status_code == 200 and _signed_in(client)
        with _old_backend(env, port):
            assert _signed_in(client)  # the stored signing secret carries the sign-in across a restart
        assert _set_password(env, "another-throwaway-Pw").returncode == 0
        with _old_backend(env, port) as base:
            assert not _signed_in(client)
            assert _sign_in(base, PASSWORD)[0] == 401


def test_an_override_in_the_checkouts_dotenv_is_named_and_refused(tmp_path):
    """The CLI and ``gateway.run`` both load ``<checkout>/.env`` after the home's, so a sign-in key there
    comes back on every restart. Checked in-process against a real file: the shared checkout's own
    ``.env`` would leak into every other test the runner starts."""
    from hermes_cli.dashboard_password import _checkout_refusal
    project_env = tmp_path / ".env"
    assert _checkout_refusal(project_env) is None  # absent
    project_env.write_text("OTHER_KEY=x\nHERMES_DASHBOARD_BASIC_AUTH_USERNAME=\n")
    assert _checkout_refusal(project_env) is None  # empty values override nothing
    project_env.write_text("export HERMES_DASHBOARD_BASIC_AUTH_PASSWORD='throwaway-checkout-pw'\n")
    refusal = _checkout_refusal(project_env)
    assert refusal is not None
    assert "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD" in refusal and str(project_env) in refusal
    assert "throwaway-checkout-pw" not in refusal
    assert project_env.read_text().count("throwaway-checkout-pw") == 1  # never edited


def _managed_dir(tmp_path, basic_auth: dict):
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text(yaml.safe_dump({"dashboard": {"basic_auth": basic_auth}}))
    return managed


def test_a_password_hash_the_administrator_pins_is_refused_before_anything_is_written(tmp_path):
    home, env = _home(tmp_path)
    managed = _managed_dir(tmp_path, {"username": USER, "password_hash": hash_password("throwaway-admin-pinned")})
    before = (home / "config.yaml").read_bytes()
    result = _set_password({**env, "HERMES_MANAGED_DIR": str(managed)})
    assert result.returncode == 1, result.stdout
    assert "dashboard.basic_auth.password_hash is managed by your administrator" in result.stdout
    assert "nothing was changed" in result.stdout and "scrypt$" not in result.stdout
    assert (home / "config.yaml").read_bytes() == before


def test_a_pinned_username_refuses_another_name_and_keeps_its_own(tmp_path):
    home, env = _home(tmp_path)
    env = {**env, "HERMES_MANAGED_DIR": str(_managed_dir(tmp_path, {"username": "pinned-operator"}))}
    renamed = _set_password(env)  # --username spike-admin
    assert renamed.returncode == 1, renamed.stdout
    assert "dashboard.basic_auth.username is managed by your administrator" in renamed.stdout
    assert _basic_auth(home) == {}
    kept = run_cli(env, "dashboard", "password", "--password-stdin", stdin=PASSWORD + "\n")
    assert kept.returncode == 0, kept.stdout
    assert "user 'pinned-operator'" in kept.stdout
    assert _verify_password(PASSWORD, _basic_auth(home)["password_hash"])


def test_a_managed_username_template_is_compared_by_its_effective_value(tmp_path):
    home, env = _home(tmp_path)
    (home / ".env").write_text("PR4_TEST_LOGIN=throwaway-login\n")
    env = {**env, "HERMES_MANAGED_DIR": str(_managed_dir(tmp_path, {"username": "${PR4_TEST_LOGIN}"}))}
    result = run_cli(env, "dashboard", "password", "--password-stdin", stdin=PASSWORD + "\n")
    assert result.returncode == 0, result.stdout
    assert "user 'throwaway-login'" in result.stdout
    assert _verify_password(PASSWORD, _basic_auth(home)["password_hash"])


def test_a_managed_username_template_on_an_override_it_removes_is_refused(tmp_path):
    home, env = _home(tmp_path)
    (home / ".env").write_text("HERMES_DASHBOARD_BASIC_AUTH_USERNAME=throwaway-login\n")
    managed = _managed_dir(tmp_path, {"username": "${HERMES_DASHBOARD_BASIC_AUTH_USERNAME}"})
    before = (home / ".env").read_bytes()
    result = run_cli({**env, "HERMES_MANAGED_DIR": str(managed)}, "dashboard", "password", "--password-stdin",
                     stdin=PASSWORD + "\n")
    assert result.returncode == 1, result.stdout
    assert "references an override this command removes" in result.stdout
    assert (home / ".env").read_bytes() == before and _basic_auth(home) == {}


def test_a_password_the_calling_environment_also_sets_is_named_and_still_wins_where_it_is_set(tmp_path):
    home, env = _home(tmp_path)
    (home / ".env").write_text("HERMES_DASHBOARD_BASIC_AUTH_PASSWORD=throwaway-dotenv-pw\n")
    inherited = {**env, "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD": "throwaway-inherited-pw"}
    result = _set_password(inherited)
    assert result.returncode == 0, result.stdout
    assert "⚠ HERMES_DASHBOARD_BASIC_AUTH_PASSWORD is set in the environment this command runs in" in result.stdout
    assert "throwaway-inherited-pw" not in result.stdout and "throwaway-dotenv-pw" not in result.stdout
    assert "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD" not in (home / ".env").read_text()
    port = free_port()
    with _old_backend(inherited, port) as base:  # started from that environment: the inherited value wins
        assert _sign_in(base, "throwaway-inherited-pw") == (200, 200)
        assert _sign_in(base, PASSWORD)[0] == 401
    with _old_backend(env, port) as base:  # without it, the new password
        assert _sign_in(base, PASSWORD) == (200, 200)


def test_a_profile_password_never_claims_the_shared_gateway_that_serves_the_profile(tmp_path):
    """Dashboard sign-in belongs to the listener's launch home: a gateway serving several profiles signs
    in with its own home's credentials, so a profile's password must not send the user to restart it."""
    def prepare(home, _env):
        config = json.loads((home / "config.yaml").read_text())
        config["gateway"] = {"multiplex_profiles": True}
        (home / "config.yaml").write_text(json.dumps(config))
        (home / "profiles" / "work").mkdir(parents=True)
        write_config(home / "profiles" / "work")

    with live_gateway(tmp_path, prepare=prepare) as gateway:
        result = gateway.cli("-p", "work", "dashboard", "password", "--username", USER, "--password-stdin",
                             stdin=PASSWORD + "\n")
        assert result.returncode == 0, result.stdout
        assert "restart it to use the new sign-in" not in result.stdout
        assert f"serves this profile from {gateway.home}" in result.stdout
        assert _verify_password(PASSWORD, _basic_auth(gateway.home / "profiles" / "work")["password_hash"])
        assert _basic_auth(gateway.home) == {}


def test_with_a_sticky_profile_every_printed_command_names_its_home(tmp_path):
    """After ``hermes profile use work`` a bare ``hermes ...`` runs as ``work``: the commands printed for
    the default home's shared gateway must say ``-p default``, or following them rotates ``work`` again."""
    def prepare(home, _env):
        config = json.loads((home / "config.yaml").read_text())
        config["gateway"] = {"multiplex_profiles": True}
        (home / "config.yaml").write_text(json.dumps(config))
        (home / "profiles" / "work").mkdir(parents=True)
        write_config(home / "profiles" / "work")

    with live_gateway(tmp_path, prepare=prepare) as gateway:
        assert gateway.cli("profile", "use", "work").returncode == 0
        result = gateway.cli("dashboard", "password", "--username", USER, "--password-stdin", stdin=PASSWORD + "\n")
        assert result.returncode == 0, result.stdout
        assert _verify_password(PASSWORD, _basic_auth(gateway.home / "profiles" / "work")["password_hash"])
        assert "change them with hermes -p default dashboard password." in result.stdout
        assert "set those with hermes -p default dashboard password." in result.stdout
        bare = [line for line in result.stdout.splitlines() if " hermes dashboard password" in line]
        assert bare == [], bare


def test_the_new_password_signs_in_on_the_gateways_webapp(tmp_path):
    def prepare(_home, env):
        assert _set_password(env).returncode == 0

    with live_gateway(tmp_path, webapp_host="0.0.0.0", prepare=prepare) as gateway:
        base = f"http://127.0.0.1:{gateway.port}"
        assert _sign_in(base, PASSWORD) == (200, 200)
        assert _sign_in(base, "wrong-password")[0] == 401
        again = gateway.cli("dashboard", "password", "--username", USER, "--password-stdin", stdin=PASSWORD + "\n")
        assert again.returncode == 0, again.stdout
        assert "The gateway is running: restart it to use the new sign-in (hermes -p default gateway restart)." in again.stdout
