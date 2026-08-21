"""Regression tests for path-aware systemd service staleness checks."""

from __future__ import annotations

import sys

import pytest


pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="systemd path comparison is POSIX-only"
)


def _unit(*lines: str) -> str:
    return "[Service]\n" + "\n".join(lines) + "\n"


def _is_current(monkeypatch, tmp_path, installed: str, expected: str) -> bool:
    from hermes_cli import gateway as gw

    unit_file = tmp_path / "hermes-gateway.service"
    unit_file.write_text(installed)
    monkeypatch.setattr(gw, "get_systemd_unit_path", lambda system=False: unit_file)
    monkeypatch.setattr(
        gw,
        "generate_systemd_unit",
        lambda system=False, run_as_user=None: expected,
    )
    return gw.systemd_unit_is_current(system=False)


def test_symlinked_and_resolved_generated_paths_are_current(tmp_path, monkeypatch):
    release = tmp_path / "hermes-agent.release"
    (release / "venv" / "bin").mkdir(parents=True)
    (release / "node_modules" / ".bin").mkdir(parents=True)
    (release / "venv" / "bin" / "python").write_text("")
    stable = tmp_path / "hermes-agent"
    stable.symlink_to(release, target_is_directory=True)

    installed = _unit(
        f"ExecStart={stable}/venv/bin/python -m hermes_cli.main gateway run",
        f"WorkingDirectory={stable}",
        f'Environment="PATH={stable}/venv/bin:{stable}/node_modules/.bin:/usr/bin"',
        f'Environment="VIRTUAL_ENV={stable}/venv"',
        f'Environment="HERMES_HOME={stable}"',
        f"ExecStopPost=-{stable}/venv/bin/python -m gateway.cgroup_cleanup",
    )
    expected = installed.replace(str(stable), str(release))

    assert _is_current(monkeypatch, tmp_path, installed, expected) is True
    assert _is_current(monkeypatch, tmp_path, expected, installed) is True


def test_different_existing_service_paths_are_stale(tmp_path, monkeypatch):
    first = tmp_path / "first" / "bin"
    second = tmp_path / "second" / "bin"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "python").write_text("")
    (second / "python").write_text("")

    installed = _unit(f"ExecStart={first}/python -m hermes_cli.main gateway run")
    expected = _unit(f"ExecStart={second}/python -m hermes_cli.main gateway run")

    assert _is_current(monkeypatch, tmp_path, installed, expected) is False


def test_missing_or_broken_service_paths_are_stale(tmp_path, monkeypatch):
    missing = tmp_path / "missing" / "python"
    broken = tmp_path / "broken-python"
    broken.symlink_to(missing)

    installed = _unit(f"ExecStart={broken} -m hermes_cli.main gateway run")
    expected = _unit(f"ExecStart={missing} -m hermes_cli.main gateway run")

    assert _is_current(monkeypatch, tmp_path, installed, expected) is False


def test_resolution_errors_leave_paths_unchanged(monkeypatch):
    from hermes_cli import gateway as gw

    def deny_resolution(self, strict=False):
        raise PermissionError(self)

    monkeypatch.setattr(gw.Path, "resolve", deny_resolution)

    assert gw._resolve_existing_service_path("/restricted/python") == "/restricted/python"


def test_non_path_exec_arguments_remain_significant(tmp_path, monkeypatch):
    executable = tmp_path / "python"
    executable.write_text("")
    installed = _unit(f"ExecStart={executable} -m hermes_cli.main gateway run --profile one")
    expected = _unit(f"ExecStart={executable} -m hermes_cli.main gateway run --profile two")

    assert _is_current(monkeypatch, tmp_path, installed, expected) is False


def test_unrecognized_environment_paths_are_not_normalized(tmp_path, monkeypatch):
    release = tmp_path / "release"
    release.mkdir()
    stable = tmp_path / "stable"
    stable.symlink_to(release, target_is_directory=True)

    installed = _unit(f'Environment="CUSTOM_ROOT={stable}"')
    expected = _unit(f'Environment="CUSTOM_ROOT={release}"')

    assert _is_current(monkeypatch, tmp_path, installed, expected) is False
