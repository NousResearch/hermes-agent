"""Pure regression tests for live-system-guard command classifiers.

These tests only classify synthetic argv.  They never invoke a native service
manager or process killer, so they remain safe even while a new classifier is
being developed.
"""

from pathlib import Path
import sys
import base64


def _guard_module():
    expected = Path(__file__).with_name("conftest.py").resolve()
    for module in tuple(sys.modules.values()):
        module_file = getattr(module, "__file__", None)
        if module_file and Path(module_file).resolve() == expected:
            return module
    raise AssertionError("pytest did not load tests/conftest.py")


def test_launchctl_kickstart_is_a_blocked_service_mutation():
    conftest = _guard_module()

    assert conftest._is_service_manager_mutation_command(
        ["launchctl", "kickstart", "gui/501/ai.hermes.gateway"]
    )


def test_launchctl_bootout_is_a_blocked_service_mutation():
    conftest = _guard_module()

    assert conftest._is_service_manager_mutation_command(
        "launchctl bootout gui/501/ai.hermes.gateway"
    )


def test_taskkill_foreign_synthetic_pid_is_blocked():
    conftest = _guard_module()

    assert conftest._is_process_killer_command(
        ["taskkill", "/PID", "987654321", "/T", "/F"],
        is_own_subtree=lambda _pid: False,
    )


def test_detached_gateway_popen_is_blocked():
    conftest = _guard_module()

    assert conftest._is_detached_gateway_spawn(
        "Popen",
        ["/definitely-not-a-real-hermes-gateway", "-m", "hermes_cli.main", "gateway", "run"],
    )


def test_protected_service_verbs_fail_closed_except_read_only_allowlist():
    conftest = _guard_module()

    assert not conftest._is_service_manager_mutation_command(
        ["systemctl", "--user", "status", "hermes-gateway.service"]
    )
    assert not conftest._is_service_manager_mutation_command(
        ["launchctl", "print", "gui/501/ai.hermes.gateway"]
    )
    for verb in ("edit", "set-property", "reenable", "preset", "revert"):
        assert conftest._is_service_manager_mutation_command(
            ["systemctl", "--user", verb, "hermes-gateway.service"]
        ), verb


def test_bytes_argv_and_shell_payloads_are_classified_recursively():
    conftest = _guard_module()

    assert conftest._is_service_manager_mutation_command(
        [b"bash", b"-c", b"systemctl --user restart hermes-gateway.service"]
    )
    assert conftest._is_service_manager_mutation_command(
        ["cmd", "/c", "launchctl bootout gui/501/ai.hermes.gateway"]
    )
    assert conftest._is_detached_gateway_spawn(
        "Popen", ["powershell", "-Command", "hermes gateway run"]
    )


def test_shell_wrapper_variants_recurse_into_protected_payloads():
    conftest = _guard_module()
    encoded = base64.b64encode(
        "systemctl restart hermes-gateway".encode("utf-16-le")
    ).decode()

    assert conftest._is_service_manager_mutation_command(
        'bash -lc "systemctl restart hermes-gateway"'
    )
    assert conftest._is_service_manager_mutation_command(
        'cmd.exe /c "systemctl stop hermes-gateway"'
    )
    assert conftest._is_service_manager_mutation_command(
        'powershell -Command "Stop-Service hermes-gateway"'
    )
    assert conftest._is_service_manager_mutation_command(
        ["pwsh", "-EncodedCommand", encoded]
    )
    assert conftest._is_service_manager_mutation_command(
        ["pwsh", "-enc", "hermes-gateway"]
    )
    assert conftest._is_detached_gateway_spawn(
        "Popen", 'sh -c "hermes gateway run &"'
    )


def test_shell_wrapper_variants_leave_benign_payloads_unblocked():
    conftest = _guard_module()

    assert not conftest._is_service_manager_mutation_command(
        'bash -lc "echo hello"'
    )


def test_external_kill_commands_use_numeric_subtree_ownership():
    conftest = _guard_module()

    assert conftest._is_process_killer_command(
        ["kill", "-TERM", "987654321"], is_own_subtree=lambda _pid: False
    )
    assert conftest._is_process_killer_command(
        ["killpg", "987654321"], is_own_subtree=lambda _pid: False
    )
    assert conftest._is_process_killer_command(
        ["kill", "--", "-1"], is_own_subtree=lambda _pid: False
    )
    assert not conftest._is_process_killer_command(
        ["kill", "-TERM", "12345"], is_own_subtree=lambda pid: pid == 12345
    )


def test_kill_negative_targets_and_foreign_process_groups_are_blocked():
    conftest = _guard_module()

    assert conftest._is_process_killer_command(
        ["kill", "-9", "-1"], is_own_subtree=lambda _pid: False
    )
    assert conftest._is_process_killer_command(
        ["kill", "-9", "-987654321"], is_own_subtree=lambda _pid: False
    )
    assert conftest._is_process_killer_command(
        ["killpg", "987654321", "9"], is_own_subtree=lambda _pid: False
    )


def test_service_manager_verb_is_positional_and_read_only_forms_pass():
    conftest = _guard_module()

    assert conftest._is_service_manager_mutation_command(
        ["systemctl", "reenable", "hermes-gateway.service"]
    )
    assert conftest._is_service_manager_mutation_command(
        ["systemctl", "set-property", "hermes-gateway.service", "CPUQuota=1%"]
    )
    assert conftest._is_service_manager_mutation_command(
        ["systemctl", "--property", "status", "restart", "hermes-gateway.service"]
    )
    assert not conftest._is_service_manager_mutation_command(
        ["systemctl", "--user", "status", "hermes-gateway"]
    )
    assert not conftest._is_service_manager_mutation_command(
        ["systemctl", "show", "--property=Restart", "hermes-gateway"]
    )


def test_taskkill_image_filter_blocks_only_gateway_or_broad_python_images():
    conftest = _guard_module()

    assert conftest._is_process_killer_command(
        ["taskkill.exe", "/IM", "hermes-gateway.exe"],
        is_own_subtree=lambda _pid: False,
    )
    assert conftest._is_process_killer_command(
        ["taskkill", "/IM", "python*.exe"],
        is_own_subtree=lambda _pid: False,
    )
    assert not conftest._is_process_killer_command(
        ["taskkill.exe", "/IM", "notepad.exe"],
        is_own_subtree=lambda _pid: False,
    )
