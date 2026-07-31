"""Docker credential-residence bootstrap and ownership behavior."""
from __future__ import annotations

import json
import subprocess
import time

import pytest

from tests.docker.conftest import docker_exec, docker_exec_sh


def _volume(name: str, suffix: str) -> str:
    volume = f"{name}-{suffix}"
    subprocess.run(
        ["docker", "volume", "create", volume],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return volume


def _wait_until(container: str, command: str, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = docker_exec_sh(container, command, user="root", timeout=5)
        if result.returncode == 0:
            return
        time.sleep(0.2)
    raise AssertionError(f"timed out waiting for: {command}")


def _wait_for_log(container: str, text: str, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["docker", "logs", container],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if text in result.stdout + result.stderr:
            return
        time.sleep(0.2)
    raise AssertionError(f"timed out waiting for log text: {text}")


def _remove_container_and_volumes(container: str, *volumes: str) -> None:
    subprocess.run(
        ["docker", "rm", "-f", container],
        capture_output=True,
        text=True,
        timeout=10,
    )
    for volume in volumes:
        subprocess.run(
            ["docker", "volume", "rm", "-f", volume],
            capture_output=True,
            text=True,
            timeout=10,
        )


def test_installed_agent_entry_rejects_before_home_mutation(
    built_image: str,
) -> None:
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "sh",
            "-e",
            "HERMES_HOME=/opt/entry-home",
            "-e",
            "HERMES_AUTH_HOME=relative/auth",
            built_image,
            "-c",
            (
                "set +e; "
                "/opt/hermes/.venv/bin/hermes-agent >/tmp/entry.out "
                "2>/tmp/entry.err; "
                "rc=$?; "
                "test \"$rc\" = 2 && "
                "test ! -e /opt/entry-home && "
                "test \"$(wc -l </tmp/entry.err)\" = 1 && "
                "grep -q HERMES_AUTH_HOME /tmp/entry.err && "
                "! grep -q ignored /tmp/entry.err"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


def test_named_profile_bootstrap_targets_residence_store(
    built_image: str, container_name: str,
) -> None:
    runtime_volume = _volume(container_name, "runtime")
    residence_volume = _volume(container_name, "residence")
    seed = json.dumps(
        {"version": 1, "providers": {"nous": {"access_token": "seed"}}}
    )
    try:
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                container_name,
                "-v",
                f"{runtime_volume}:/opt/data",
                "-v",
                f"{residence_volume}:/opt/auth",
                "-e",
                "HERMES_HOME=/opt/data/profiles/work",
                "-e",
                "HERMES_AUTH_HOME=/opt/auth",
                "-e",
                f"HERMES_AUTH_JSON_BOOTSTRAP={seed}",
                built_image,
                "sleep",
                "infinity",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        _wait_until(container_name, "test -f /opt/auth/profiles/work/auth.json")

        result = docker_exec_sh(
            container_name,
            (
                "test -r /opt/auth/profiles/work/auth.json && "
                "test ! -e /opt/data/profiles/work/auth.json && "
                "test \"$(stat -c %U /opt/auth/profiles/work/auth.json)\" = hermes && "
                "test \"$(stat -c %a /opt/auth/profiles/work/auth.json)\" = 600"
            ),
            timeout=10,
        )
        assert result.returncode == 0, result.stderr
    finally:
        _remove_container_and_volumes(
            container_name, runtime_volume, residence_volume
        )


@pytest.mark.parametrize(
    "invalid_override",
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
    ids=(
        "empty",
        "whitespace",
        "relative",
        "tilde",
        "leading",
        "trailing",
        "control",
        "filesystem-root",
    ),
)
def test_invalid_override_seeds_nowhere(
    built_image: str, container_name: str, invalid_override: str,
) -> None:
    runtime_volume = _volume(container_name, "runtime")
    seed = json.dumps(
        {"version": 1, "providers": {"nous": {"access_token": "seed"}}}
    )
    try:
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                container_name,
                "-v",
                f"{runtime_volume}:/opt/data",
                "-e",
                f"HERMES_AUTH_HOME={invalid_override}",
                "-e",
                f"HERMES_AUTH_JSON_BOOTSTRAP={seed}",
                built_image,
                "sleep",
                "infinity",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        _wait_until(
            container_name,
            "grep -q 'profile=default' /opt/data/logs/container-boot.log 2>/dev/null",
        )
        result = docker_exec_sh(
            container_name,
            (
                "test ! -e /opt/data/auth.json && "
                "test ! -e /auth.json && "
                "test \"$(stat -c %U:%G:%a /)\" = root:root:755"
            ),
            user="root",
            timeout=10,
        )
        assert result.returncode == 0, result.stderr
    finally:
        _remove_container_and_volumes(container_name, runtime_volume)


def test_profile_component_symlink_is_not_followed(
    built_image: str, container_name: str,
) -> None:
    runtime_volume = _volume(container_name, "runtime")
    residence_volume = _volume(container_name, "residence")
    seed = json.dumps(
        {"version": 1, "providers": {"nous": {"access_token": "seed"}}}
    )
    try:
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "sh",
                "-v",
                f"{residence_volume}:/opt/auth",
                built_image,
                "-c",
                "ln -s /opt/data/escape /opt/auth/profiles",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                container_name,
                "-v",
                f"{runtime_volume}:/opt/data",
                "-v",
                f"{residence_volume}:/opt/auth",
                "-e",
                "HERMES_HOME=/opt/data/profiles/work",
                "-e",
                "HERMES_AUTH_HOME=/opt/auth",
                "-e",
                f"HERMES_AUTH_JSON_BOOTSTRAP={seed}",
                built_image,
                "sleep",
                "infinity",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        _wait_for_log(container_name, "credential residence creation")
        result = docker_exec_sh(
            container_name,
            (
                "test -L /opt/auth/profiles && "
                "test ! -e /opt/data/escape/work/auth.json"
            ),
            user="root",
            timeout=10,
        )
        assert result.returncode == 0, result.stderr
    finally:
        _remove_container_and_volumes(
            container_name, runtime_volume, residence_volume
        )


@pytest.mark.parametrize("recovery_kind", ("terminal", "newer"))
def test_stage2_rebootstrap_recovers_with_stale_legacy_temp(
    built_image: str, container_name: str, recovery_kind: str,
) -> None:
    runtime_volume = _volume(container_name, "runtime")
    residence_volume = _volume(container_name, "residence")
    auth_dir = "/opt/auth/profiles/work"
    if recovery_kind == "terminal":
        local_nous = {
            "client_id": "hermes-cli-vps",
            "last_auth_error": {"relogin_required": True},
        }
        expected_log = "Nous bootstrap session was terminal"
    else:
        local_nous = {
            "client_id": "hermes-cli-vps",
            "access_token": "OLD-at",
            "refresh_token": "OLD-rt",
            "obtained_at": "2026-07-31T00:00:00Z",
        }
        expected_log = "Applied newer orchestrator-issued Nous bootstrap session"
    local_store = json.dumps(
        {"version": 1, "providers": {"nous": local_nous}},
        separators=(",", ":"),
    )
    seed = json.dumps(
        {
            "version": 1,
            "providers": {
                "nous": {
                    "client_id": "hermes-cli-vps",
                    "access_token": "FRESH-at",
                    "refresh_token": "FRESH-rt",
                    "obtained_at": "2026-07-31T01:00:00Z",
                }
            },
        },
        separators=(",", ":"),
    )
    try:
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "sh",
                "-v",
                f"{residence_volume}:/opt/auth",
                built_image,
                "-c",
                (
                    f"mkdir -p {auth_dir}; "
                    f"printf '%s' '{local_store}' >{auth_dir}/auth.json; "
                    f"printf legacy >{auth_dir}/auth.json.rebootstrap.tmp; "
                    f"printf current >{auth_dir}/auth.json.tmp.1.stale; "
                    f"chown -R 0:0 {auth_dir}; "
                    f"chmod 700 {auth_dir}; "
                    f"chmod 600 {auth_dir}/auth.json "
                    f"{auth_dir}/auth.json.rebootstrap.tmp "
                    f"{auth_dir}/auth.json.tmp.1.stale"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                container_name,
                "-v",
                f"{runtime_volume}:/opt/data",
                "-v",
                f"{residence_volume}:/opt/auth",
                "-e",
                "HERMES_HOME=/opt/data/profiles/work",
                "-e",
                "HERMES_AUTH_HOME=/opt/auth",
                "-e",
                f"HERMES_AUTH_JSON_REBOOTSTRAP={seed}",
                built_image,
                "sleep",
                "infinity",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        _wait_for_log(container_name, expected_log)

        result = docker_exec_sh(
            container_name,
            (
                f"grep -q FRESH-rt {auth_dir}/auth.json && "
                f"! grep -q OLD-rt {auth_dir}/auth.json && "
                f"test \"$(stat -c %U:%a {auth_dir}/auth.json)\" = hermes:600 && "
                f"test \"$(cat {auth_dir}/auth.json.rebootstrap.tmp)\" = legacy && "
                f"test \"$(stat -c %U:%a {auth_dir}/auth.json.rebootstrap.tmp)\" = root:600 && "
                f"test \"$(stat -c %U:%a {auth_dir}/auth.json.tmp.1.stale)\" = hermes:600"
            ),
            user="root",
            timeout=10,
        )
        assert result.returncode == 0, result.stderr
    finally:
        _remove_container_and_volumes(
            container_name, runtime_volume, residence_volume
        )


@pytest.mark.parametrize("named_profile", (False, True))
def test_warm_residence_repairs_only_known_credentials(
    built_image: str, container_name: str, named_profile: bool,
) -> None:
    runtime_volume = _volume(container_name, "runtime")
    residence_volume = _volume(container_name, "residence")
    auth_dir = "/opt/auth/profiles/work" if named_profile else "/opt/auth"
    hermes_home = "/opt/data/profiles/work" if named_profile else "/opt/data"
    global_auth_setup = (
        "printf global-primary >/opt/auth/auth.json; " if named_profile else ""
    )
    global_auth_assert = (
        'test "$(stat -c %U /opt/auth/auth.json)" = hermes && '
        if named_profile
        else ""
    )
    try:
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "sh",
                "-v",
                f"{residence_volume}:/opt/auth",
                built_image,
                "-c",
                (
                    f"mkdir -p {auth_dir} /opt/auth/profiles "
                    "/opt/auth/shared /opt/auth/unrelated; "
                    f"{global_auth_setup}"
                    f"printf primary >{auth_dir}/auth.json; "
                    f"printf anthropic >{auth_dir}/.anthropic_oauth.json; "
                    f"printf corrupt >{auth_dir}/auth.json.corrupt; "
                    f"printf temp >{auth_dir}/auth.json.tmp.1.stale; "
                    f"printf temp >{auth_dir}/.anthropic_oauth.tmp.1.stale; "
                    f"printf temp >{auth_dir}/..anthropic_oauth_stale.tmp; "
                    "printf shared >/opt/auth/shared/nous_auth.json; "
                    "printf temp >/opt/auth/shared/nous_auth.json.tmp.1.stale; "
                    "printf host >/opt/auth/unrelated/host-file; "
                    "printf target >/opt/auth/unrelated/lock-target; "
                    f"ln -s /opt/auth/unrelated/lock-target {auth_dir}/auth.lock; "
                    "chown -R 0:0 /opt/auth; "
                    f"chmod 700 /opt/auth /opt/auth/profiles {auth_dir} "
                    "/opt/auth/shared /opt/auth/unrelated; "
                    f"chmod 600 {auth_dir}/auth.json "
                    f"{auth_dir}/.anthropic_oauth.json "
                    f"{auth_dir}/auth.json.corrupt "
                    f"{auth_dir}/auth.json.tmp.1.stale "
                    f"{auth_dir}/.anthropic_oauth.tmp.1.stale "
                    f"{auth_dir}/..anthropic_oauth_stale.tmp "
                    "/opt/auth/shared/nous_auth.json "
                    "/opt/auth/shared/nous_auth.json.tmp.1.stale "
                    "/opt/auth/unrelated/host-file "
                    "/opt/auth/unrelated/lock-target"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                container_name,
                "-v",
                f"{runtime_volume}:/opt/data",
                "-v",
                f"{residence_volume}:/opt/auth",
                "-e",
                "HERMES_AUTH_HOME=/opt/auth",
                "-e",
                f"HERMES_HOME={hermes_home}",
                built_image,
                "sleep",
                "infinity",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        _wait_until(
            container_name,
            (
                "su hermes -s /bin/sh -c "
                f"'test -r {auth_dir}/auth.json && "
                f"test -r {auth_dir}/.anthropic_oauth.json && "
                "test -r /opt/auth/shared/nous_auth.json'"
            ),
        )

        result = docker_exec_sh(
            container_name,
            (
                f"{global_auth_assert}"
                f"test \"$(stat -c %U {auth_dir}/auth.json)\" = hermes && "
                f"test \"$(stat -c %U {auth_dir}/.anthropic_oauth.json)\" = hermes && "
                f"test \"$(stat -c %U {auth_dir}/auth.json.corrupt)\" = hermes && "
                f"test \"$(stat -c %U {auth_dir}/auth.json.tmp.1.stale)\" = hermes && "
                f"test \"$(stat -c %U {auth_dir}/.anthropic_oauth.tmp.1.stale)\" = hermes && "
                f"test \"$(stat -c %U {auth_dir}/..anthropic_oauth_stale.tmp)\" = hermes && "
                "test \"$(stat -c %U /opt/auth/shared/nous_auth.json)\" = hermes && "
                "test \"$(stat -c %U /opt/auth/shared/nous_auth.json.tmp.1.stale)\" = hermes && "
                "test \"$(stat -c %U /opt/auth/unrelated/host-file)\" = root && "
                "test \"$(stat -c %U /opt/auth/unrelated/lock-target)\" = root && "
                f"test -L {auth_dir}/auth.lock"
            ),
            user="root",
            timeout=10,
        )
        assert result.returncode == 0, result.stderr
    finally:
        _remove_container_and_volumes(
            container_name, runtime_volume, residence_volume
        )
