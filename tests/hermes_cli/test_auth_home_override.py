from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_constants import (
    get_hermes_auth_home,
    get_hermes_auth_home_override,
    get_hermes_home,
)


_AUTH_WORKER = r"""
import json
import sys
import time
from pathlib import Path

from hermes_constants import get_hermes_home
from hermes_cli.auth import (
    _auth_store_lock,
    _load_auth_store,
    _read_shared_nous_state,
    _save_auth_store,
    _store_provider_state,
    _write_shared_nous_state,
    clear_provider_auth,
    get_provider_auth_state,
)

home = get_hermes_home()
home.mkdir(parents=True, exist_ok=True)
(home / "process.marker").write_text(home.name, encoding="utf-8")

for line in sys.stdin:
    command = json.loads(line)
    operation = command["operation"]
    if operation == "put":
        with _auth_store_lock():
            store = _load_auth_store()
            if command.get("signal_path"):
                Path(command["signal_path"]).touch()
            time.sleep(command.get("hold_lock_seconds", 0))
            _store_provider_state(
                store,
                command["provider"],
                command["state"],
                set_active=False,
            )
            _save_auth_store(store)
        result = True
    elif operation == "get":
        result = get_provider_auth_state(command["provider"])
    elif operation == "remove":
        result = clear_provider_auth(command["provider"])
    elif operation == "shared_read":
        result = _read_shared_nous_state()
    elif operation == "shared_write":
        _write_shared_nous_state(command["state"])
        result = True
    else:
        raise ValueError(operation)
    print(json.dumps(result), flush=True)
"""


def test_auth_home_defaults_to_runtime_home_and_validates_override(
    monkeypatch, tmp_path
):
    runtime_home = tmp_path / "runtime"
    residence = tmp_path / "auth-residence"
    monkeypatch.setenv("HERMES_HOME", str(runtime_home))

    assert get_hermes_auth_home_override() is None
    assert get_hermes_auth_home() == get_hermes_home() == runtime_home

    monkeypatch.setenv("HERMES_AUTH_HOME", str(residence))
    assert get_hermes_auth_home_override() == residence.resolve()
    assert get_hermes_auth_home() == residence.resolve()


def test_auth_home_rejects_os_home_and_its_ancestor(monkeypatch, tmp_path):
    import hermes_constants

    operator_home = tmp_path / "operator" / "home"
    runtime_home = tmp_path / "runtime"
    monkeypatch.setenv("HOME", str(operator_home))
    monkeypatch.setenv("HERMES_HOME", str(runtime_home))

    for residence in (operator_home, operator_home.parent):
        monkeypatch.setenv("HERMES_AUTH_HOME", str(residence))
        with pytest.raises(
            hermes_constants.HermesAuthHomeError,
            match="OS user home",
        ):
            hermes_constants.validate_hermes_auth_home()


def test_auth_home_uses_real_home_when_home_is_profile_scoped(
    monkeypatch, tmp_path
):
    import hermes_constants

    runtime_home = tmp_path / "runtime"
    profile_home = runtime_home / "home"
    profile_home.mkdir(parents=True)
    operator_home = tmp_path / "operator"
    monkeypatch.setenv("HOME", str(profile_home))
    monkeypatch.setenv("HERMES_REAL_HOME", str(operator_home))
    monkeypatch.setenv("HERMES_HOME", str(runtime_home))
    monkeypatch.setenv("HERMES_AUTH_HOME", str(operator_home))

    with pytest.raises(
        hermes_constants.HermesAuthHomeError,
        match="OS user home",
    ):
        hermes_constants.validate_hermes_auth_home()


def test_auth_home_rejects_runtime_ancestor(monkeypatch, tmp_path):
    import hermes_constants

    operator_home = tmp_path / "operator"
    runtime_root = tmp_path / "runtime"
    runtime_home = runtime_root / "sessions" / "one"
    monkeypatch.setenv("HOME", str(operator_home))
    monkeypatch.setenv("HERMES_HOME", str(runtime_home))
    monkeypatch.setenv("HERMES_AUTH_HOME", str(runtime_root))

    with pytest.raises(
        hermes_constants.HermesAuthHomeError,
        match="must not contain HERMES_HOME",
    ):
        hermes_constants.validate_hermes_auth_home()


def test_auth_home_rejects_active_runtime_ancestor(monkeypatch, tmp_path):
    import hermes_constants

    process_home = tmp_path / "runtime"
    residence = tmp_path / "auth"
    active_home = residence / "tenant"
    monkeypatch.setenv("HOME", str(tmp_path / "operator"))
    monkeypatch.setenv("HERMES_HOME", str(process_home))
    monkeypatch.setenv("HERMES_AUTH_HOME", str(residence))

    token = hermes_constants.set_hermes_home_override(active_home)
    try:
        with pytest.raises(
            hermes_constants.HermesAuthHomeError,
            match="must not contain HERMES_HOME",
        ):
            hermes_constants.validate_hermes_auth_home()
    finally:
        hermes_constants.reset_hermes_home_override(token)


def test_auth_home_boundary_validation_preserves_supported_layouts(
    monkeypatch, tmp_path
):
    import hermes_constants

    operator_home = tmp_path / "operator"
    runtime_home = tmp_path / "runtime"
    monkeypatch.setenv("HOME", str(operator_home))

    monkeypatch.setenv("HERMES_HOME", str(operator_home))
    monkeypatch.setenv("HERMES_AUTH_HOME", str(operator_home))
    assert hermes_constants.get_hermes_auth_home_strict() == operator_home.resolve()

    monkeypatch.setenv("HERMES_HOME", str(runtime_home))
    monkeypatch.setenv("HERMES_AUTH_HOME", str(runtime_home))
    profile_home = runtime_home / "profiles" / "work"
    token = hermes_constants.set_hermes_home_override(profile_home)
    try:
        assert hermes_constants.get_hermes_auth_home_strict() == profile_home
    finally:
        hermes_constants.reset_hermes_home_override(token)

    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setenv("HERMES_AUTH_HOME", str(runtime_home))
    assert hermes_constants.get_hermes_auth_home_strict() == profile_home

    monkeypatch.setenv("HERMES_HOME", str(runtime_home))
    for residence in (operator_home / ".hermes-auth", tmp_path / "auth-residence"):
        monkeypatch.setenv("HERMES_AUTH_HOME", str(residence))
        assert (
            hermes_constants.get_hermes_auth_home_strict()
            == residence.resolve()
        )

    profile_home = runtime_home / "profiles" / "work"
    residence = tmp_path / "profile-auth-residence"
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setenv("HERMES_AUTH_HOME", str(residence))
    assert hermes_constants.get_hermes_auth_home_strict() == (
        residence.resolve() / "profiles" / "work"
    )


@pytest.mark.parametrize(
    ("value", "message"),
    [("", "set but empty"), ("   ", "set but empty"),
     ("relative/auth", "absolute path")],
)
def test_invalid_override_is_rejected_by_validation_not_by_the_resolver(
    monkeypatch, tmp_path, value, message
):
    """An unusable value must fail loudly at startup, never mid-run.

    The resolver itself has to stay total: it runs from import-time module
    constants and from the file-safety guards, and a raise there either aborts
    startup or gets swallowed into a fail-open credential read.
    """
    import hermes_constants

    runtime_home = tmp_path / "runtime"
    monkeypatch.setenv("HERMES_HOME", str(runtime_home))
    monkeypatch.setenv("HERMES_AUTH_HOME", value)
    monkeypatch.setattr(hermes_constants, "_auth_home_invalid_warned", False)

    with pytest.raises(hermes_constants.HermesAuthHomeError, match=message):
        hermes_constants.validate_hermes_auth_home()

    assert get_hermes_auth_home_override() is None
    assert get_hermes_auth_home() == runtime_home


def test_invalid_override_keeps_the_credential_read_guard_closed(
    monkeypatch, tmp_path
):
    """Regression: an empty HERMES_AUTH_HOME must not disable the read deny.

    get_read_block_error() resolves several Hermes homes; when that raised,
    raise_if_read_blocked's best-effort ``except Exception: return`` turned
    every image-gen/vision reference-file guard into a no-op at once.
    """
    from agent.file_safety import raise_if_read_blocked

    runtime_home = tmp_path / "runtime"
    runtime_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(runtime_home))

    for name in ("auth.json", ".env", ".anthropic_oauth.json"):
        target = runtime_home / name
        target.write_text("secret", encoding="utf-8")
        for value in ("", "relative/auth"):
            monkeypatch.setenv("HERMES_AUTH_HOME", value)
            with pytest.raises(ValueError, match="Access denied"):
                raise_if_read_blocked(str(target))


def test_auth_store_temp_files_are_denied_for_read_and_write(
    monkeypatch, tmp_path
):
    """Atomic-write temps hold the same plaintext tokens as auth.json."""
    from agent.file_safety import get_read_block_error, is_write_denied

    residence = tmp_path / "auth-residence"
    residence.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "runtime"))
    monkeypatch.setenv("HERMES_AUTH_HOME", str(residence))

    for name in ("auth.json.tmp.4242.deadbeef", "auth.json.corrupt"):
        target = residence / name
        target.write_text("{}", encoding="utf-8")
        assert get_read_block_error(str(target)), name
        assert is_write_denied(str(target)), name


def test_legacy_rebootstrap_temp_is_exactly_classified_and_uninstalled(
    tmp_path,
):
    from hermes_cli.auth_artifacts import is_primary_auth_transient
    from hermes_cli.uninstall import _clean_auth_residence

    legacy = tmp_path / "auth.json.rebootstrap.tmp"
    unknown = tmp_path / "auth.json.rebootstrap.tmp.extra"
    legacy.write_text("credential copy", encoding="utf-8")
    unknown.write_text("operator data", encoding="utf-8")

    assert is_primary_auth_transient(legacy.name)
    assert not is_primary_auth_transient(unknown.name)

    _clean_auth_residence(tmp_path)

    assert not legacy.exists()
    assert unknown.read_text(encoding="utf-8") == "operator data"


def test_override_routes_current_credential_paths_and_guards(
    monkeypatch, tmp_path
):
    runtime_home = tmp_path / "runtime"
    residence = tmp_path / "auth-residence"
    global_home = tmp_path / "operator" / ".hermes"
    global_home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path / "operator"))
    monkeypatch.setenv("HERMES_HOME", str(runtime_home))
    monkeypatch.setenv("HERMES_AUTH_HOME", str(residence))

    from agent.anthropic_adapter import (
        _get_hermes_oauth_file,
        read_hermes_oauth_credentials,
    )
    from agent.auxiliary_client import _auth_json_path as auxiliary_auth_json_path
    from agent.credential_sources import _remove_hermes_pkce
    from agent.file_safety import get_read_block_error, is_write_denied
    from hermes_cli.auth import (
        _auth_file_path,
        _auth_lock_path,
        _auth_store_lock,
        _global_auth_file_path,
        _load_auth_store,
        _read_shared_nous_state,
        _save_auth_store,
        _store_provider_state,
        _write_shared_nous_state,
        get_provider_auth_state,
    )
    from hermes_cli.models import _credential_fingerprint
    from hermes_cli.web_server import _save_anthropic_oauth_creds
    from plugins.platforms.photon.auth import (
        load_photon_token,
        store_photon_token,
    )
    from tools.managed_tool_gateway import auth_json_path as managed_auth_json_path
    from tools.xai_http import has_xai_credentials

    assert _auth_file_path() == residence / "auth.json"
    assert _auth_lock_path() == residence / "auth.lock"
    assert _get_hermes_oauth_file() == residence / ".anthropic_oauth.json"
    assert auxiliary_auth_json_path() == residence / "auth.json"
    assert get_hermes_auth_home() / "auth.json" == residence / "auth.json"
    assert managed_auth_json_path() == residence / "auth.json"
    assert _global_auth_file_path() is None

    for path in (
        residence / "auth.json",
        residence / "auth.lock",
        residence / ".anthropic_oauth.json",
    ):
        assert get_read_block_error(str(path)) is not None
        assert is_write_denied(str(path))

    global_home.joinpath("auth.json").write_text(
        json.dumps({"providers": {"global-only": {"value": "must-not-leak"}}}),
        encoding="utf-8",
    )
    shared_path = global_home / "shared" / "nous_auth.json"
    shared_path.parent.mkdir()
    shared_payload = json.dumps(
        {"access_token": "shared-access", "refresh_token": "shared-refresh"}
    )
    shared_path.write_text(shared_payload, encoding="utf-8")

    with _auth_store_lock():
        auth = _load_auth_store()
        _store_provider_state(
            auth,
            "xai-oauth",
            {
                "tokens": {
                    "access_token": "residence-access",
                    "refresh_token": "residence-refresh",
                }
            },
            set_active=False,
        )
        _save_auth_store(auth)
    assert get_provider_auth_state("global-only") is None
    assert has_xai_credentials()

    # The shared Nous store relocates into the residence rather than being
    # switched off. Disabling it would strand every profile with its own
    # refresh-token chain, which is how single-use tokens get replayed
    # (#48415); relocating keeps the sharing inside the residence boundary.
    _write_shared_nous_state(
        {"access_token": "new-access", "refresh_token": "new-refresh"}
    )
    residence_shared = residence / "shared" / "nous_auth.json"
    assert residence_shared.is_file()
    assert (_read_shared_nous_state() or {}).get("access_token") == "new-access"
    assert shared_path.read_text(encoding="utf-8") == shared_payload

    store_photon_token("photon-token")
    assert load_photon_token() == "photon-token"
    assert not (runtime_home / "auth.json").exists()

    from hermes_cli.config import invalidate_env_cache
    from hermes_cli.credential_lifecycle import remove_provider_env_credential

    runtime_home.mkdir(parents=True, exist_ok=True)
    runtime_home.joinpath(".env").write_text(
        "ZAI_API_KEY=runtime-key\n",
        encoding="utf-8",
    )
    invalidate_env_cache()
    with _auth_store_lock():
        auth = _load_auth_store()
        auth.setdefault("credential_pool", {})["zai"] = [
            {
                "id": "env-entry",
                "source": "env:ZAI_API_KEY",
                "access_token": "runtime-key",
            }
        ]
        _save_auth_store(auth)
    result = remove_provider_env_credential("ZAI_API_KEY")
    assert result["found"]
    assert "zai" not in _load_auth_store().get("credential_pool", {})
    assert "ZAI_API_KEY" not in runtime_home.joinpath(".env").read_text(
        encoding="utf-8"
    )

    fingerprint = _credential_fingerprint("xai-oauth")
    (runtime_home / "auth.json").write_text("{}", encoding="utf-8")
    assert _credential_fingerprint("xai-oauth") == fingerprint
    auth_stat = (residence / "auth.json").stat()
    os.utime(
        residence / "auth.json",
        ns=(auth_stat.st_atime_ns, auth_stat.st_mtime_ns + 1_000_000),
    )
    assert _credential_fingerprint("xai-oauth") != fingerprint

    residence_oauth = residence / ".anthropic_oauth.json"
    _save_anthropic_oauth_creds(
        "residence-anthropic",
        "residence-refresh",
        1,
    )
    runtime_oauth = runtime_home / ".anthropic_oauth.json"
    runtime_oauth.write_text(
        json.dumps({"accessToken": "runtime-must-not-be-read"}),
        encoding="utf-8",
    )
    assert read_hermes_oauth_credentials()["accessToken"] == "residence-anthropic"
    result = _remove_hermes_pkce("anthropic", None)
    assert result.cleaned
    assert not residence_oauth.exists()
    assert runtime_oauth.exists()


def test_two_runtime_processes_share_only_the_auth_residence(tmp_path):
    operator_home = tmp_path / "operator"
    global_home = operator_home / ".hermes"
    global_home.mkdir(parents=True)
    global_auth = global_home / "auth.json"
    global_auth.write_text(
        json.dumps({"providers": {"global-only": {"value": "must-not-leak"}}}),
        encoding="utf-8",
    )
    shared_path = global_home / "shared" / "nous_auth.json"
    shared_path.parent.mkdir()
    shared_payload = json.dumps(
        {"access_token": "shared-access", "refresh_token": "shared-refresh"}
    )
    shared_path.write_text(shared_payload, encoding="utf-8")

    residence = tmp_path / "auth-residence"
    homes = (tmp_path / "runtime-a", tmp_path / "runtime-b")
    processes: list[subprocess.Popen[str]] = []

    def start(home: Path) -> subprocess.Popen[str]:
        env = os.environ.copy()
        env.update(
            {
                "HOME": str(operator_home),
                "HERMES_HOME": str(home),
                "HERMES_AUTH_HOME": str(residence),
                "PYTHONUNBUFFERED": "1",
            }
        )
        process = subprocess.Popen(
            [sys.executable, "-c", _AUTH_WORKER],
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append(process)
        return process

    def send(process: subprocess.Popen[str], payload: dict) -> None:
        assert process.stdin is not None
        process.stdin.write(json.dumps(payload) + "\n")
        process.stdin.flush()

    def receive(process: subprocess.Popen[str]):
        assert process.stdout is not None
        line = process.stdout.readline()
        assert line, process.stderr.read() if process.stderr is not None else ""
        return json.loads(line)

    def request(process: subprocess.Popen[str], payload: dict):
        send(process, payload)
        return receive(process)

    first = start(homes[0])
    second = start(homes[1])
    try:
        assert request(first, {"operation": "get", "provider": "global-only"}) is None
        assert request(second, {"operation": "shared_read"}) is None
        assert request(
            first,
            {
                "operation": "shared_write",
                "state": {
                    "access_token": "must-not-write",
                    "refresh_token": "must-not-write",
                },
            },
        )

        first_has_lock = tmp_path / "first-has-lock"
        send(
            first,
            {
                "operation": "put",
                "provider": "first",
                "state": {"value": "one"},
                "hold_lock_seconds": 0.2,
                "signal_path": str(first_has_lock),
            },
        )
        deadline = time.monotonic() + 2
        while not first_has_lock.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert first_has_lock.exists()
        send(
            second,
            {
                "operation": "put",
                "provider": "second",
                "state": {"value": "two"},
            },
        )
        assert receive(first)
        assert receive(second)
        assert request(second, {"operation": "get", "provider": "first"}) == {
            "value": "one"
        }
        assert request(first, {"operation": "get", "provider": "second"}) == {
            "value": "two"
        }

        assert request(second, {"operation": "remove", "provider": "first"})
        assert request(first, {"operation": "get", "provider": "first"}) is None
    finally:
        for process in processes:
            if process.stdin is not None:
                process.stdin.close()
            process.wait(timeout=10)
            assert process.returncode == 0, (
                process.stderr.read() if process.stderr is not None else ""
            )

    persisted = json.loads((residence / "auth.json").read_text(encoding="utf-8"))
    assert persisted["providers"]["second"] == {"value": "two"}
    assert "first" not in persisted["providers"]
    assert not any((home / "auth.json").exists() for home in homes)
    assert all((home / "process.marker").is_file() for home in homes)
    assert global_auth.read_text(encoding="utf-8") == json.dumps(
        {"providers": {"global-only": {"value": "must-not-leak"}}}
    )
    assert shared_path.read_text(encoding="utf-8") == shared_payload
    if os.name != "nt":
        assert (residence / "auth.json").stat().st_mode & 0o777 == 0o600


def test_profiles_keep_separate_stores_inside_the_residence(monkeypatch, tmp_path):
    """Two multiplexed profiles must not share one auth.json or active_provider.

    The gateway scopes each profile turn with a context-local HERMES_HOME
    (``_profile_runtime_scope``). A residence resolved purely from the process
    environment would collapse every profile onto one store.
    """
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from hermes_cli.auth import (
        _auth_file_path,
        _auth_store_lock,
        _load_auth_store,
        _save_auth_store,
        _store_provider_state,
        get_provider_auth_state,
    )

    root = tmp_path / "root"
    residence = tmp_path / "auth-residence"
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_AUTH_HOME", str(residence))

    def write_profile_state(profile: str, value: str) -> Path:
        token = set_hermes_home_override(str(root / "profiles" / profile))
        try:
            with _auth_store_lock():
                store = _load_auth_store()
                _store_provider_state(store, "nous", {"value": value}, set_active=True)
                _save_auth_store(store)
            return _auth_file_path()
        finally:
            reset_hermes_home_override(token)

    def read_profile_state(profile: str):
        token = set_hermes_home_override(str(root / "profiles" / profile))
        try:
            return get_provider_auth_state("nous")
        finally:
            reset_hermes_home_override(token)

    work_path = write_profile_state("work", "work-token")
    play_path = write_profile_state("play", "play-token")

    assert work_path == residence / "profiles" / "work" / "auth.json"
    assert play_path == residence / "profiles" / "play" / "auth.json"
    assert work_path != play_path
    assert read_profile_state("work") == {"value": "work-token"}
    assert read_profile_state("play") == {"value": "play-token"}

    # active_provider is per-profile too, not one field they fight over.
    work_store = json.loads(work_path.read_text(encoding="utf-8"))
    play_store = json.loads(play_path.read_text(encoding="utf-8"))
    assert work_store["providers"].keys() == {"nous"}
    assert work_store["providers"]["nous"] == {"value": "work-token"}
    assert play_store["providers"]["nous"] == {"value": "play-token"}


def test_residence_equal_to_home_is_a_no_op_not_a_fallback_teardown(
    monkeypatch, tmp_path
):
    """Pointing the residence at the directory already in use changes nothing.

    _global_auth_file_path() used to return None whenever the env var was set
    at all, so this silently disabled the global-root read fallback *and* the
    write-through that keeps rotated single-use refresh tokens in sync.
    """
    from hermes_cli.auth import _global_auth_file_path

    root = tmp_path / "operator" / ".hermes"
    profile_home = root / "profiles" / "work"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path / "operator"))
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    monkeypatch.delenv("HERMES_AUTH_HOME", raising=False)
    assert _global_auth_file_path() == root / "auth.json"

    # Same directory, spelled explicitly — must not change the resolution.
    monkeypatch.setenv("HERMES_AUTH_HOME", str(profile_home))
    assert _global_auth_file_path() == root / "auth.json"

    # A residence that genuinely relocates the store does move the fallback,
    # into the residence root, so write-through still has somewhere to go.
    residence = tmp_path / "auth-residence"
    monkeypatch.setenv("HERMES_AUTH_HOME", str(residence))
    assert _global_auth_file_path() == residence / "auth.json"


def test_generated_service_values_use_strict_resolution(monkeypatch, tmp_path):
    from hermes_cli.gateway import _service_auth_home_directive
    from hermes_cli.gateway_windows import _auth_home_for_service

    residence = tmp_path / "auth-residence"
    monkeypatch.delenv("HERMES_AUTH_HOME", raising=False)
    assert _service_auth_home_directive() == ""
    assert _auth_home_for_service() == ""

    monkeypatch.setenv("HERMES_AUTH_HOME", str(residence))
    assert f'Environment="HERMES_AUTH_HOME={residence}"' in _service_auth_home_directive()
    assert _auth_home_for_service() == str(residence)


@pytest.mark.parametrize(
    "value",
    (
        "",
        "relative/auth",
        "~/auth",
        " /absolute/with-leading-space",
        "/absolute/with-trailing-space ",
        "bad\npath",
    ),
)
def test_service_generation_rejects_invalid_auth_home(monkeypatch, value):
    from hermes_cli.gateway import _service_auth_home_directive
    from hermes_cli.gateway_windows import _auth_home_for_service

    monkeypatch.setenv("HERMES_AUTH_HOME", value)
    with pytest.raises(ValueError):
        _service_auth_home_directive()
    with pytest.raises(ValueError):
        _auth_home_for_service()


def test_systemd_auth_home_value_is_escaped_and_parseable(monkeypatch, tmp_path):
    from hermes_cli.gateway import _service_auth_home_directive, generate_systemd_unit

    residence = tmp_path / 'auth\\store"100%ready'
    runtime_home = tmp_path / "runtime"
    runtime_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(runtime_home))
    monkeypatch.setenv("HERMES_AUTH_HOME", str(residence))

    expected = (
        str(residence.resolve())
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("%", "%%")
    )
    assert _service_auth_home_directive() == (
        f'Environment="HERMES_AUTH_HOME={expected}"\n'
    )

    systemd_analyze = shutil.which("systemd-analyze")
    if systemd_analyze is None:
        pytest.skip("systemd-analyze is unavailable")
    unit_path = tmp_path / "hermes-auth-home.service"
    unit_path.write_text(generate_systemd_unit(), encoding="utf-8")
    result = subprocess.run(
        [systemd_analyze, "verify", str(unit_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_windows_cmd_wrapper_preserves_literal_percent(monkeypatch, tmp_path):
    import hermes_cli.gateway_windows as gateway_windows

    residence = tmp_path / "%LOCALAPPDATA%" / "auth"
    monkeypatch.setenv("HERMES_AUTH_HOME", str(residence))
    monkeypatch.setattr(
        gateway_windows,
        "_resolve_detached_python",
        lambda python_path: (python_path, tmp_path / "venv", []),
    )
    monkeypatch.setattr(
        gateway_windows,
        "_preserve_hermes_home_path",
        lambda path: str(path),
    )

    script = gateway_windows._build_gateway_cmd_script(
        sys.executable,
        str(tmp_path),
        str(tmp_path / "runtime"),
        "",
    )

    escaped = str(residence.resolve()).replace("%", "%%")
    assert f'set "HERMES_AUTH_HOME={escaped}"' in script.splitlines()


@pytest.mark.parametrize(
    "value",
    (
        "",
        "   ",
        "relative/auth",
        "~/auth",
        " /absolute/with-leading-space",
        "/absolute/with-trailing-space ",
        "bad\npath",
    ),
)
@pytest.mark.parametrize("termux_fast", (False, True))
def test_invalid_auth_home_version_exits_two_without_fallback_write(
    tmp_path, value, termux_fast
):
    runtime_home = tmp_path / ("termux" if termux_fast else "normal")
    env = os.environ.copy()
    env.update(
        {
            "HERMES_HOME": str(runtime_home),
            "HERMES_AUTH_HOME": value,
            "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
        }
    )
    if termux_fast:
        env["TERMUX_VERSION"] = "test"
        env.pop("HERMES_TERMUX_DISABLE_FAST_CLI", None)
    else:
        env["HERMES_TERMUX_DISABLE_FAST_CLI"] = "1"
        env.pop("TERMUX_VERSION", None)

    result = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "--version"],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "HERMES_AUTH_HOME" in result.stderr
    assert not (runtime_home / "auth.json").exists()


def test_version_reports_valid_effective_auth_home(tmp_path):
    runtime_home = tmp_path / "runtime"
    residence = tmp_path / "residence"
    env = os.environ.copy()
    env.update(
        {
            "HERMES_HOME": str(runtime_home),
            "HERMES_AUTH_HOME": str(residence),
            "HERMES_TERMUX_DISABLE_FAST_CLI": "1",
            "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
        }
    )
    env.pop("TERMUX_VERSION", None)

    result = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "--version"],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert f"Auth home: {residence.resolve()} (HERMES_AUTH_HOME)" in result.stdout


@pytest.mark.parametrize(
    "command",
    (
        ("-m", "acp_adapter.entry", "--version"),
        ("-m", "hermes_agent_entry"),
        (str(Path(__file__).resolve().parents[2] / "run_agent.py"), "--list-tools"),
    ),
)
@pytest.mark.parametrize(
    "value",
    (
        "",
        "   ",
        "relative/auth",
        "~/auth",
        " /absolute/with-leading-space",
        "/absolute/with-trailing-space ",
        "bad\npath",
        "/",
    ),
)
def test_other_operational_entrypoints_reject_invalid_auth_home(
    tmp_path, command, value
):
    runtime_home = tmp_path / "runtime"
    env = os.environ.copy()
    env.update(
        {
            "HERMES_HOME": str(runtime_home),
            "HERMES_AUTH_HOME": value,
            "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
        }
    )

    result = subprocess.run(
        [sys.executable, *command],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    error_lines = [line for line in result.stderr.splitlines() if line.strip()]
    assert len(error_lines) == 1
    assert "HERMES_AUTH_HOME" in error_lines[0]
    assert "ignored" not in error_lines[0]
    assert not runtime_home.exists()


def test_installed_agent_command_uses_the_early_validation_wrapper():
    project = Path(__file__).resolve().parents[2] / "pyproject.toml"
    text = project.read_text(encoding="utf-8")

    assert 'hermes-agent = "hermes_agent_entry:main"' in text
    assert '"hermes_agent_entry"' in text


def test_xai_login_success_names_the_actual_auth_residence(
    monkeypatch, tmp_path, capsys
):
    """The login summary must point at the store that was actually written."""
    import base64
    from types import SimpleNamespace

    from hermes_cli.auth import DEFAULT_XAI_OAUTH_BASE_URL, _login_xai_oauth

    runtime_home = tmp_path / "runtime"
    runtime_home.mkdir()
    residence = tmp_path / "auth-residence"
    monkeypatch.setenv("HERMES_HOME", str(runtime_home))
    monkeypatch.setenv("HERMES_AUTH_HOME", str(residence))

    payload = base64.urlsafe_b64encode(
        json.dumps({"exp": int(time.time()) + 7200}).encode("utf-8")
    ).rstrip(b"=").decode("utf-8")
    access_token = f"h.{payload}.s"
    monkeypatch.setattr(
        "hermes_cli.auth._xai_oauth_device_code_login",
        lambda **kwargs: {
            "tokens": {
                "access_token": access_token,
                "refresh_token": "rt-new",
                "id_token": "",
                "token_type": "Bearer",
            },
            "discovery": {"token_endpoint": "https://auth.x.ai/token"},
            "redirect_uri": "",
            "base_url": DEFAULT_XAI_OAUTH_BASE_URL,
            "last_refresh": "2026-07-31T10:00:00Z",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.auth._update_config_for_provider",
        lambda *args, **kwargs: "config.yaml",
    )

    _login_xai_oauth(
        SimpleNamespace(no_browser=True, timeout=3),
        None,
        force_new_login=True,
    )

    output = capsys.readouterr().out
    assert f"Auth state: {residence.resolve()}/auth.json" in output
    assert f"{runtime_home}/auth.json" not in output
    assert (residence / "auth.json").is_file()
    assert not (runtime_home / "auth.json").exists()


def test_quick_snapshot_captures_credentials_from_the_residence(
    monkeypatch, tmp_path
):
    """Backups resolved auth.json under HERMES_HOME, so they captured nothing."""
    from hermes_cli.backup import _resolve_state_path

    runtime_home = tmp_path / "runtime"
    residence = tmp_path / "auth-residence"
    monkeypatch.setenv("HERMES_HOME", str(runtime_home))

    monkeypatch.delenv("HERMES_AUTH_HOME", raising=False)
    assert _resolve_state_path(runtime_home, "auth.json") == runtime_home / "auth.json"
    assert _resolve_state_path(runtime_home, "config.yaml") == runtime_home / "config.yaml"

    monkeypatch.setenv("HERMES_AUTH_HOME", str(residence))
    assert _resolve_state_path(runtime_home, "auth.json") == residence / "auth.json"
    # Non-credential state still follows HERMES_HOME.
    assert _resolve_state_path(runtime_home, "config.yaml") == runtime_home / "config.yaml"
