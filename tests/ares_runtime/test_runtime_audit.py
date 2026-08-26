from __future__ import annotations

from pathlib import Path

from ares_runtime.runtime_audit import (
    ManagedRuntimeProcess,
    audit_process_snapshots,
    classify_runtime_role,
)


REVISION = "a" * 40
OTHER_REVISION = "b" * 40
ACTIVE_SOURCE = Path("/runtime/releases") / REVISION / "source"
ACTIVE_PYTHON = Path("/opt/python3.14")


def _process(
    *,
    pid: int = 10,
    role: str = "gateway",
    revision: str | None = REVISION,
    executable: Path | None = ACTIVE_PYTHON,
    cwd: Path | None = ACTIVE_SOURCE,
    deleted: tuple[str, ...] = (),
) -> ManagedRuntimeProcess:
    return ManagedRuntimeProcess(
        pid=pid,
        ppid=1,
        role=role,
        revision=revision,
        executable=executable,
        cwd=cwd,
        argv=(str(ACTIVE_SOURCE / ".venv/bin/python"), "-m", "hermes_cli.main", "gateway"),
        deleted_runtime_mappings=deleted,
    )


def test_role_classifier_uses_exact_runtime_entry_shapes() -> None:
    python = "/runtime/.venv/bin/python"

    assert classify_runtime_role((python, "-m", "ares_runtime.local_runtime", "doctor")) == "controller"
    assert classify_runtime_role((python, "-m", "hermes_cli.main", "--tui")) == "tui"
    assert classify_runtime_role((python, "-m", "tui_gateway.entry")) == "tui_gateway"
    assert classify_runtime_role((python, "-m", "hermes_cli.main", "gateway")) == "gateway"
    assert classify_runtime_role((python, "-m", "hermes_cli.main", "serve", "--port", "0")) == "desktop_backend"
    assert classify_runtime_role(
        (python, "-m", "hermes_cli.main", "--profile", "public", "serve", "--port", "0")
    ) == "profile_backend"
    assert classify_runtime_role((python, "/runtime/tools/mcp_stdio_watchdog.py", "--ppid", "10")) == "mcp_watchdog"
    assert classify_runtime_role(("python3", "/runtime/scripts/relay.py")) is None


def test_current_release_and_interpreter_are_coherent() -> None:
    report = audit_process_snapshots(
        [_process()],
        active_revision=REVISION,
        active_source=ACTIVE_SOURCE,
        expected_python=ACTIVE_PYTHON,
    )

    assert report.ok is True
    assert report.managed_count == 1
    assert report.stale == ()
    assert [process.pid for process in report.coherent] == [10]


def test_external_operator_workspace_is_not_runtime_source_drift() -> None:
    report = audit_process_snapshots(
        [_process(cwd=Path("/home/operator/project"))],
        active_revision=REVISION,
        active_source=ACTIVE_SOURCE,
        expected_python=ACTIVE_PYTHON,
    )

    assert report.ok is True
    assert report.stale == ()


def test_other_managed_release_cwd_is_source_drift() -> None:
    report = audit_process_snapshots(
        [
            _process(
                cwd=Path("/runtime/releases") / OTHER_REVISION / "source",
            )
        ],
        active_revision=REVISION,
        active_source=ACTIVE_SOURCE,
        expected_python=ACTIVE_PYTHON,
    )

    assert report.ok is False
    assert report.stale[0].reasons == ("source_mismatch",)


def test_stale_tui_detects_replaced_interpreter_and_deleted_extensions() -> None:
    stale_python = Path("/opt/python3.11")
    deleted = (
        "/runtime/releases/a/source/.venv/lib/python3.11/site-packages/jiter.so (deleted)",
    )
    report = audit_process_snapshots(
        [
            _process(
                pid=22,
                role="tui",
                executable=stale_python,
                deleted=deleted,
            )
        ],
        active_revision=REVISION,
        active_source=ACTIVE_SOURCE,
        expected_python=ACTIVE_PYTHON,
    )

    assert report.ok is False
    assert len(report.stale) == 1
    assert report.stale[0].process.pid == 22
    assert report.stale[0].reasons == (
        "interpreter_mismatch",
        "deleted_runtime_mapping",
    )


def test_old_release_process_is_stale_even_with_a_valid_old_interpreter() -> None:
    report = audit_process_snapshots(
        [
            _process(
                pid=33,
                revision=OTHER_REVISION,
                executable=Path("/runtime/releases")
                / OTHER_REVISION
                / "source/.venv/bin/python",
            )
        ],
        active_revision=REVISION,
        active_source=ACTIVE_SOURCE,
        expected_python=ACTIVE_PYTHON,
    )

    assert report.stale[0].reasons == (
        "release_mismatch",
        "interpreter_mismatch",
    )


def test_electron_process_is_checked_for_release_but_not_python_identity() -> None:
    report = audit_process_snapshots(
        [
            _process(
                pid=44,
                role="desktop",
                executable=ACTIVE_SOURCE / "apps/desktop/Ares",
            )
        ],
        active_revision=REVISION,
        active_source=ACTIVE_SOURCE,
        expected_python=ACTIVE_PYTHON,
    )

    assert report.ok is True
    assert [process.pid for process in report.coherent] == [44]


def test_unknown_release_identity_fails_closed() -> None:
    report = audit_process_snapshots(
        [_process(pid=55, revision=None)],
        active_revision=REVISION,
        active_source=ACTIVE_SOURCE,
        expected_python=ACTIVE_PYTHON,
    )

    assert report.stale[0].reasons == ("release_unknown",)
    assert "stale=1" in report.summary()


def test_empty_process_projection_fails_closed() -> None:
    report = audit_process_snapshots(
        [],
        active_revision=REVISION,
        active_source=ACTIVE_SOURCE,
        expected_python=ACTIVE_PYTHON,
    )

    assert report.ok is False
    assert report.summary() == "managed=0 coherent=0 stale=0 roles=none"
