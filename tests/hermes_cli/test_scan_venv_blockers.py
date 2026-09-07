"""Tests for hermes_cli/_scan_venv_blockers.py.

Tests call the real production functions (``main``, ``_redact_sensitive_cmdline``).
Most patch the detector directly; one Windows-only regression exercises the
real process table using only subprocesses owned by that test.
"""

from __future__ import annotations

import builtins
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import hermes_cli._scan_venv_blockers as scanner
from hermes_cli._scan_venv_blockers import (
    _classify_local_preview_args,
    _executable_within,
    _hermes_cli_command,
    _is_pausable_gateway,
    _probe_fail_json,
    _redact_sensitive_cmdline,
    _terminate_safe_preview,
    main,
)


# ---------------------------------------------------------------------------
# main() — stdout, stderr, exit code (with patched detector)
# ---------------------------------------------------------------------------


def test_main_invalid_arguments_emit_one_fail_closed_json_document(capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        main([])

    captured = capsys.readouterr()
    assert raised.value.code == 1
    assert len(captured.out.splitlines()) == 1
    payload = json.loads(captured.out)
    assert payload["ok"] is False
    assert payload["blocked"] is True
    assert payload["reason"] == "invalid_arguments"
    assert payload["error"]["code"] == "invalid_arguments"
    assert "--root" in captured.err


def test_main_probe_exception_emits_one_fail_closed_json_document(
    monkeypatch, capsys
) -> None:
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setattr(
        scanner,
        "scan_venv_blockers",
        lambda _root: (_ for _ in ()).throw(PermissionError("probe denied")),
    )

    with pytest.raises(SystemExit) as raised:
        main(["--root", str(root)])

    captured = capsys.readouterr()
    assert raised.value.code == 1
    assert len(captured.out.splitlines()) == 1
    payload = json.loads(captured.out)
    assert payload["ok"] is False
    assert payload["blocked"] is True
    assert payload["reason"] == "probe_failed"
    assert payload["error"] == {
        "code": "probe_failed",
        "message": "probe denied",
    }
    assert captured.err.strip() == "probe denied"


def _psutil_fake() -> dict:
    """Return a sys.modules dict entry that makes psutil appear available."""
    return {
        "psutil": types.SimpleNamespace(
            Process=lambda *a: MagicMock(),
            NoSuchProcess=type("NoSuchProcess", (Exception,), {}),
        )
    }


def _detector_proc(pid, exe, name, cmdline=None, cwd="", *, ppid=None, parents=()):
    proc = MagicMock()
    proc.info = {
        "pid": pid,
        "ppid": ppid,
        "exe": exe,
        "name": name,
        "cmdline": cmdline or [],
        "cwd": cwd,
    }
    proc.parents.return_value = list(parents)
    proc.parent.return_value = parents[0] if parents else None
    proc.ppid.return_value = ppid or 0
    proc.exe.return_value = exe
    proc.cmdline.return_value = list(cmdline or [])
    return proc


def _detector_ancestor(pid: int, exe: Path, argv: list[str]):
    return types.SimpleNamespace(
        pid=pid,
        exe=lambda: str(exe),
        cmdline=lambda: list(argv),
    )


def test_strict_detector_excludes_only_self_and_keeps_venv_parent_candidate(
    monkeypatch, tmp_path: Path
) -> None:
    import os

    venv_python = str(tmp_path / "venv" / "Scripts" / "python.exe")
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda _attrs: iter(
            [
                _detector_proc(os.getpid(), venv_python, "python.exe"),
                _detector_proc(555, venv_python, "python.exe"),
            ]
        )
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    assert scanner._detect_target_venv_holders(tmp_path, strict=True) == [(555, "python.exe", "")]


def test_strict_detector_excludes_exact_immediate_hermes_console_shim(
    monkeypatch, tmp_path: Path
) -> None:
    import os

    shim = tmp_path / "venv" / "Scripts" / "hermes.exe"
    parent = types.SimpleNamespace(
        pid=555,
        exe=lambda: str(shim),
        cmdline=lambda: [str(shim), "update", "--preflight"],
    )
    current = types.SimpleNamespace(parent=lambda: parent)
    fake_psutil = types.SimpleNamespace(
        Process=lambda pid: current if int(pid) == os.getpid() else parent,
        process_iter=lambda _attrs: iter(
            [_detector_proc(555, str(shim), "hermes.exe", [str(shim), "update"])]
        ),
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    assert scanner._detect_target_venv_holders(tmp_path, strict=True) == []


def test_strict_detector_excludes_exact_immediate_python_venv_trampoline(
    monkeypatch, tmp_path: Path
) -> None:
    import os

    python = tmp_path / "venv" / "Scripts" / "python.exe"
    argv = [
        str(python),
        "-m",
        "hermes_cli.main",
        "update",
        "--preflight",
        "--json",
    ]
    parent = types.SimpleNamespace(
        pid=556,
        exe=lambda: str(python),
        cmdline=lambda: argv,
    )
    current = types.SimpleNamespace(parent=lambda: parent)
    fake_psutil = types.SimpleNamespace(
        Process=lambda pid: current if int(pid) == os.getpid() else parent,
        process_iter=lambda _attrs: iter(
            [_detector_proc(556, str(python), "python.exe", argv)]
        ),
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    assert scanner._detect_target_venv_holders(tmp_path, strict=True) == []


def test_strict_detector_excludes_exact_immediate_standalone_scanner_trampoline(
    monkeypatch, tmp_path: Path
) -> None:
    import os

    python = tmp_path / "venv" / "Scripts" / "python.exe"
    argv = [
        str(python),
        "-m",
        "hermes_cli._scan_venv_blockers",
        "--root",
        str(tmp_path),
    ]
    parent = types.SimpleNamespace(
        pid=558,
        exe=lambda: str(python),
        cmdline=lambda: argv,
    )
    current = types.SimpleNamespace(parent=lambda: parent)
    fake_psutil = types.SimpleNamespace(
        Process=lambda pid: current if int(pid) == os.getpid() else parent,
        process_iter=lambda _attrs: iter(
            [_detector_proc(558, str(python), "python.exe", argv)]
        ),
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    assert scanner._detect_target_venv_holders(tmp_path, strict=True) == []


def test_strict_detector_keeps_standalone_scanner_for_a_different_root(
    monkeypatch, tmp_path: Path
) -> None:
    import os

    python = tmp_path / "venv" / "Scripts" / "python.exe"
    argv = [
        str(python),
        "-m",
        "hermes_cli._scan_venv_blockers",
        "--root",
        str(tmp_path / "other-install"),
    ]
    parent = types.SimpleNamespace(
        pid=559,
        exe=lambda: str(python),
        cmdline=lambda: argv,
    )
    current = types.SimpleNamespace(parent=lambda: parent)
    fake_psutil = types.SimpleNamespace(
        Process=lambda pid: current if int(pid) == os.getpid() else parent,
        process_iter=lambda _attrs: iter(
            [_detector_proc(559, str(python), "python.exe", argv)]
        ),
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    assert scanner._detect_target_venv_holders(tmp_path, strict=True) == [(559, "python.exe", " ".join(argv))]


def test_strict_detector_keeps_immediate_python_venv_non_update_parent(
    monkeypatch, tmp_path: Path
) -> None:
    import os

    python = tmp_path / "venv" / "Scripts" / "python.exe"
    argv = [str(python), "-m", "hermes_cli.main", "serve"]
    parent = types.SimpleNamespace(
        pid=557,
        exe=lambda: str(python),
        cmdline=lambda: argv,
    )
    current = types.SimpleNamespace(parent=lambda: parent)
    fake_psutil = types.SimpleNamespace(
        Process=lambda pid: current if int(pid) == os.getpid() else parent,
        process_iter=lambda _attrs: iter(
            [_detector_proc(557, str(python), "python.exe", argv)]
        ),
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    assert scanner._detect_target_venv_holders(tmp_path, strict=True) == [(557, "python.exe", " ".join(argv))]


@pytest.mark.parametrize(
    "global_args",
    [
        ["--safe-mode"],
        ["--provider", "openrouter"],
        ["--ignore-user-config", "--yolo"],
        ["--"],
    ],
)
def test_strict_detector_excludes_current_shim_with_global_options_before_update(
    monkeypatch, tmp_path: Path, global_args: list[str]
) -> None:
    import os

    shim = tmp_path / "venv" / "Scripts" / "hermes.exe"
    argv = [str(shim), *global_args, "update", "--preflight"]
    parent = types.SimpleNamespace(
        pid=555,
        exe=lambda: str(shim),
        cmdline=lambda: argv,
    )
    current = types.SimpleNamespace(parent=lambda: parent)
    fake_psutil = types.SimpleNamespace(
        Process=lambda pid: current if int(pid) == os.getpid() else parent,
        process_iter=lambda _attrs: iter(
            [_detector_proc(555, str(shim), "hermes.exe", argv)]
        ),
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    assert scanner._detect_target_venv_holders(tmp_path, strict=True) == []


@pytest.mark.parametrize(
    "argv_tail",
    [
        ["--", "--", "update"],
        ["--", "chat", "update"],
        ["--update"],
    ],
)
def test_current_update_shim_rejects_ambiguous_end_of_options(
    argv_tail: list[str],
) -> None:

    assert not scanner._is_current_update_shim_argv(
        [r"C:\Hermes\venv\Scripts\hermes.exe", *argv_tail]
    )


def test_strict_detector_keeps_higher_target_venv_python_ancestor(
    monkeypatch, tmp_path: Path
) -> None:
    import os

    shell = types.SimpleNamespace(
        pid=444,
        exe=lambda: r"C:\Windows\System32\cmd.exe",
        cmdline=lambda: [r"C:\Windows\System32\cmd.exe"],
    )
    current = types.SimpleNamespace(parent=lambda: shell)
    venv_python = tmp_path / "venv" / "Scripts" / "python.exe"
    fake_psutil = types.SimpleNamespace(
        Process=lambda pid: current if int(pid) == os.getpid() else shell,
        process_iter=lambda _attrs: iter(
            [
                _detector_proc(
                    333,
                    str(venv_python),
                    "python.exe",
                    [str(venv_python), "-m", "agent.run"],
                )
            ]
        ),
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    assert scanner._detect_target_venv_holders(tmp_path, strict=True) == [(333, "python.exe", f"{venv_python} -m agent.run")]


def test_strict_detector_keeps_target_candidate_with_unreadable_exe(
    monkeypatch, tmp_path: Path
) -> None:

    fake_psutil = types.SimpleNamespace(
        process_iter=lambda _attrs: iter(
            [
                _detector_proc(
                    777,
                    None,
                    "python.exe",
                    ["python.exe", "-m", "hermes_cli.main", "serve"],
                    cwd=str(tmp_path),
                )
            ]
        )
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    assert scanner._detect_target_venv_holders(tmp_path, strict=True) == [(777, "python.exe", "python.exe -m hermes_cli.main serve")]


def test_strict_detector_fails_closed_when_python_identity_is_fully_unreadable(
    monkeypatch, tmp_path: Path
) -> None:

    proc = MagicMock()
    proc.info = {
        "pid": 778,
        "exe": None,
        "name": "python.exe",
        "cmdline": None,
        "cwd": None,
    }
    fake_psutil = types.SimpleNamespace(process_iter=lambda _attrs: iter([proc]))
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    with pytest.raises(RuntimeError, match=r"process 778 .* identity metadata was unreadable"):
        scanner._detect_target_venv_holders(tmp_path, strict=True)


def test_strict_detector_keeps_external_mcp_worker_with_live_target_wrapper(
    monkeypatch, tmp_path: Path
) -> None:

    module = "agent.transports.hermes_tools_mcp_server"
    wrapper_pid = 801
    worker_pid = 802
    wrapper_python = tmp_path / "venv" / "Scripts" / "python.exe"
    base_python = Path(r"C:\Python311\python.exe")
    argv_tail = ["-m", module]
    wrapper_argv = [str(wrapper_python), *argv_tail]
    worker_argv = [str(base_python), *argv_tail]
    wrapper_ancestor = _detector_ancestor(
        wrapper_pid, wrapper_python, wrapper_argv
    )
    worker = _detector_proc(
        worker_pid,
        str(base_python),
        "python.exe",
        worker_argv,
        cwd=r"C:\unrelated\workspace",
        ppid=wrapper_pid,
        parents=[wrapper_ancestor],
    )
    wrapper = _detector_proc(
        wrapper_pid,
        str(wrapper_python),
        "python.exe",
        wrapper_argv,
        cwd=r"C:\unrelated\workspace",
        ppid=1,
    )
    # Deliberately enumerate the worker first: discovery cannot depend on the
    # order returned by the Windows process table.
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda _attrs: iter([worker, wrapper])
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    matches = scanner._detect_target_venv_holders(tmp_path, strict=True)

    assert {pid for pid, _name, _cmdline in matches} == {
        wrapper_pid,
        worker_pid,
    }


def test_strict_detector_keeps_general_base_worker_for_same_venv_invocation(
    monkeypatch, tmp_path: Path
) -> None:

    wrapper_pid = 811
    worker_pid = 812
    wrapper_python = tmp_path / "venv" / "Scripts" / "python.exe"
    base_python = Path(r"C:\Python311\python.exe")
    argv_tail = [r"C:\tools\buzz_native_presence.py", "--watch"]
    wrapper_argv = [str(wrapper_python), *argv_tail]
    worker_argv = [str(base_python), *argv_tail]
    wrapper_ancestor = _detector_ancestor(
        wrapper_pid, wrapper_python, wrapper_argv
    )
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda _attrs: iter(
            [
                _detector_proc(
                    wrapper_pid,
                    str(wrapper_python),
                    "python.exe",
                    wrapper_argv,
                    ppid=1,
                ),
                _detector_proc(
                    worker_pid,
                    str(base_python),
                    "python.exe",
                    worker_argv,
                    cwd=r"C:\unrelated\workspace",
                    ppid=wrapper_pid,
                    parents=[wrapper_ancestor],
                ),
            ]
        )
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    matches = scanner._detect_target_venv_holders(tmp_path, strict=True)

    assert [pid for pid, _name, _cmdline in matches] == [wrapper_pid, worker_pid]


def test_strict_detector_excludes_external_mcp_worker_from_other_install(
    monkeypatch, tmp_path: Path
) -> None:

    module = "agent.transports.hermes_tools_mcp_server"
    foreign_root = tmp_path / "other-install"
    foreign_python = foreign_root / "venv" / "Scripts" / "python.exe"
    base_python = Path(r"C:\Python311\python.exe")
    wrapper_argv = [str(foreign_python), "-m", module]
    worker = _detector_proc(
        822,
        str(base_python),
        "python.exe",
        [str(base_python), "-m", module],
        cwd=r"C:\unrelated\workspace",
        ppid=821,
        parents=[_detector_ancestor(821, foreign_python, wrapper_argv)],
    )
    # Even an unreadable exact-MCP ancestry from another install must not make
    # this target's strict scan fail: the process table has no edge to this
    # target venv, so the live ancestry probe must not be attempted.
    worker.ppid.side_effect = PermissionError("foreign process denied")
    foreign_wrapper = _detector_proc(
        821,
        str(foreign_python),
        "python.exe",
        wrapper_argv,
        ppid=1,
    )
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda _attrs: iter([worker, foreign_wrapper])
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    assert scanner._detect_target_venv_holders(tmp_path, strict=True) == []
    worker.ppid.assert_not_called()


def test_strict_detector_does_not_rewalk_snapshot_proven_target_ancestry(
    monkeypatch, tmp_path: Path
) -> None:

    wrapper_pid = 831
    worker_pid = 832
    wrapper_python = tmp_path / "venv" / "Scripts" / "python.exe"
    base_python = Path(r"C:\Python311\python.exe")
    argv_tail = [r"C:\tools\buzz_native_presence.py", "--watch"]
    wrapper_argv = [str(wrapper_python), *argv_tail]
    worker = _detector_proc(
        worker_pid,
        str(base_python),
        "python.exe",
        [str(base_python), *argv_tail],
        ppid=wrapper_pid,
    )
    worker.ppid.side_effect = AssertionError("snapshot ppid must be reused")
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda _attrs: iter(
            [
                _detector_proc(
                    wrapper_pid,
                    str(wrapper_python),
                    "python.exe",
                    wrapper_argv,
                    ppid=1,
                ),
                worker,
            ]
        )
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    matches = scanner._detect_target_venv_holders(tmp_path, strict=True)

    assert [pid for pid, _name, _cmdline in matches] == [wrapper_pid, worker_pid]
    worker.ppid.assert_not_called()


def test_detector_uses_one_ppid_map_only_for_matching_external_invocation(
    monkeypatch, tmp_path: Path
) -> None:

    wrapper_pid = 841
    wrapper_python = tmp_path / "venv" / "Scripts" / "python.exe"
    base_python = Path(r"C:\Python311\python.exe")
    argv_tail = [r"C:\tools\buzz_native_presence.py", "--watch"]
    wrapper = _detector_proc(
        wrapper_pid,
        str(wrapper_python),
        "python.exe",
        [str(wrapper_python), *argv_tail],
        ppid=1,
    )
    worker = _detector_proc(
        842,
        str(base_python),
        "python.exe",
        [str(base_python), *argv_tail],
        cwd=r"C:\unrelated\workspace",
    )
    worker.ppid.side_effect = AssertionError("per-process ppid must not be read")
    unrelated = _detector_proc(
        843,
        str(base_python),
        "python.exe",
        [str(base_python), "-c", "import time; time.sleep(1)"],
    )
    unrelated.ppid.side_effect = AssertionError(
        "unrelated Python ancestry must not be queried"
    )
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda _attrs: iter([wrapper, worker, unrelated]),
        _ppid_map=lambda: {841: 1, 842: 841, 843: 1},
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    matches = scanner._detect_target_venv_holders(tmp_path, strict=True)

    assert [pid for pid, _name, _cmdline in matches] == [wrapper_pid, 842]
    worker.ppid.assert_not_called()
    unrelated.ppid.assert_not_called()


def test_strict_detector_fails_closed_when_parent_snapshot_is_unreadable(
    monkeypatch, tmp_path: Path
) -> None:

    wrapper_python = tmp_path / "venv" / "Scripts" / "python.exe"
    base_python = Path(r"C:\Python311\python.exe")
    argv_tail = [r"C:\tools\buzz_native_presence.py", "--watch"]
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda _attrs: iter(
            [
                _detector_proc(
                    851,
                    str(wrapper_python),
                    "python.exe",
                    [str(wrapper_python), *argv_tail],
                    ppid=1,
                ),
                _detector_proc(
                    852,
                    str(base_python),
                    "python.exe",
                    [str(base_python), *argv_tail],
                    cwd=r"C:\unrelated\workspace",
                ),
            ]
        ),
        _ppid_map=lambda: (_ for _ in ()).throw(PermissionError("denied")),
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    with pytest.raises(RuntimeError, match="parent process enumeration failed"):
        scanner._detect_target_venv_holders(tmp_path, strict=True)


def test_strict_detector_fails_when_candidate_is_absent_from_parent_snapshot(
    monkeypatch, tmp_path: Path
) -> None:

    wrapper_python = tmp_path / "venv" / "Scripts" / "python.exe"
    base_python = Path(r"C:\Python311\python.exe")
    argv_tail = [r"C:\tools\buzz_native_presence.py", "--watch"]
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda _attrs: iter(
            [
                _detector_proc(
                    861,
                    str(wrapper_python),
                    "python.exe",
                    [str(wrapper_python), *argv_tail],
                    ppid=1,
                ),
                _detector_proc(
                    862,
                    str(base_python),
                    "python.exe",
                    [str(base_python), *argv_tail],
                    cwd=r"C:\unrelated\workspace",
                ),
            ]
        )
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    with pytest.raises(
        RuntimeError,
        match=r"process 862 was absent from the parent snapshot",
    ):
        scanner._detect_target_venv_holders(
            tmp_path,
            strict=True,
            _parent_by_pid={861: 1},
        )






# ---------------------------------------------------------------------------
# _redact_sensitive_cmdline
# ---------------------------------------------------------------------------


def test_redact_long_flag_value_space_separated() -> None:
    """--token SECRET must preserve --token and emit --token <redacted>."""
    raw = "python.exe -m hermes_cli.main serve --token ghp_abc123 --host 10.0.0.1"
    result = _redact_sensitive_cmdline(raw)
    assert result == "python.exe -m hermes_cli.main serve --token <redacted>"
    assert "ghp_abc123" not in result




def test_redact_masking_failure_returns_fully_redacted() -> None:
    """A masking failure must collapse to '<redacted>' so PID and name still
    provide diagnostics rather than leaking an unmasked command line."""
    with patch.object(
        scanner,
        "_mask_secret_values",
        side_effect=RuntimeError("no redactor"),
    ):
        result = _redact_sensitive_cmdline("python.exe --token abc123")

    assert result == "<redacted>"


def test_redaction_never_imports_the_target_checkout(monkeypatch) -> None:
    """#104687 B3: the carrier is byte-identical here and runs against a root
    whose ``agent`` package is mid-rewrite, so redaction must be self-contained."""
    real_import = builtins.__import__

    def _refuse_checkout_import(name, *args, **kwargs):
        assert not name.startswith(("agent", "hermes_cli.", "hermes_constants")), name

        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _refuse_checkout_import)

    raw = "python.exe -m hermes_cli.main serve --host 127.0.0.1"
    assert _redact_sensitive_cmdline(raw) == raw


def test_redact_masks_bare_vendor_token_without_a_flag() -> None:
    raw = "python.exe -m worker --header ghp_0123456789abcdef"
    result = _redact_sensitive_cmdline(raw)
    assert "ghp_0123456789abcdef" not in result
    assert result == "python.exe -m worker --header <redacted>"


def test_redact_masks_assignment_and_url_credentials() -> None:
    raw = "python.exe -m worker HERMES_API_KEY=abc123 --url https://bob:hunter2@example.test/x"
    result = _redact_sensitive_cmdline(raw)
    assert "abc123" not in result
    assert "hunter2" not in result
    assert "HERMES_API_KEY=<redacted>" in result
    assert "https://<redacted>@example.test/x" in result


def test_redact_does_not_mask_identifier_lookalikes() -> None:
    raw = "python.exe -m worker --monkey=banana --keyboard=qwerty --directory C:\\site"
    assert _redact_sensitive_cmdline(raw) == raw


def test_redact_session_key() -> None:
    """--session-key <identifier> must redact the value and everything after."""
    raw = "python.exe -m tui_gateway.slash_worker --session-key 20260712-abcdef --model test"
    result = _redact_sensitive_cmdline(raw)
    assert result == "python.exe -m tui_gateway.slash_worker --session-key <redacted>"


def test_redact_normal_host_port_profile_remain() -> None:
    raw = "python.exe -m hermes_cli.main serve --host 10.0.0.1 --port 9119 --profile work"
    result = _redact_sensitive_cmdline(raw)
    assert "10.0.0.1" in result
    assert "9119" in result
    assert "work" in result


def test_redact_no_sensitive_flags_is_noop() -> None:
    raw = "python.exe -m hermes_cli.main serve --host 127.0.0.1"
    assert _redact_sensitive_cmdline(raw) == raw


def test_redact_empty_string() -> None:
    assert _redact_sensitive_cmdline("") == ""


def test_redact_short_flags_not_redacted() -> None:
    """Short flags -t (toolset), -p (profile), -k are NOT redacted."""
    raw = "python.exe -m hermes_cli.main serve -t web -p default -k somearg"
    result = _redact_sensitive_cmdline(raw)
    assert result == raw  # short flags pass through unchanged


def test_classify_local_preview_args_preserves_full_directory_label_and_port() -> None:
    args = [
        r"C:\Hermes\venv\Scripts\python.exe",
        "-m",
        "http.server",
        "8766",
        "--bind",
        "0.0.0.0",
        "--directory",
        r"C:\Projects\Example Preview",
    ]

    assert _classify_local_preview_args(args) == {
        "kind": "local-preview",
        "safeToStop": True,
        "label": "Example Preview",
        "port": 8766,
    }


def test_classify_local_preview_args_rejects_arbitrary_python_process() -> None:
    assert _classify_local_preview_args(["python.exe", "important-script.py"]) == {}


def test_classify_local_preview_args_rejects_module_flags_passed_to_a_script() -> None:
    assert _classify_local_preview_args(
        ["python.exe", "important-script.py", "-m", "http.server", "8765"]
    ) == {}


def test_terminate_safe_preview_revalidates_identity_and_exact_argv(tmp_path) -> None:
    class FakeProcess:
        def __init__(self, pid: int, *, created: float, args: list[str]) -> None:
            self.pid = pid
            self._created = created
            self._args = args
            self.terminated = False
            self.killed = False
            self._children: list[FakeProcess] = []

        def create_time(self) -> float:
            return self._created

        def cmdline(self) -> list[str]:
            return self._args

        def exe(self) -> str:
            return self.executable

        def children(self, *, recursive: bool) -> list[FakeProcess]:
            assert recursive is True
            return self._children

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

    venv = tmp_path / "venv"
    (venv / "Scripts").mkdir(parents=True)
    child = FakeProcess(200, created=10.0, args=["child.exe"])
    child.executable = str(venv / "Scripts" / "child.exe")
    parent = FakeProcess(100, created=1722798000.25, args=["python.exe", "-m", "http.server", "8766"])
    parent.executable = str(venv / "Scripts" / "python.exe")
    parent._children = [child]
    fake_psutil = types.SimpleNamespace(
        Process=lambda pid: parent if pid == 100 else child,
        wait_procs=lambda processes, timeout: (processes, []),
    )

    stopped, error = _terminate_safe_preview(
        100, 1722798000.25, venv, psutil_module=fake_psutil
    )

    assert stopped is True
    assert error is None
    assert parent.terminated is True
    assert child.terminated is True


def test_terminate_safe_preview_refuses_a_target_outside_the_target_venv(tmp_path) -> None:
    """#104687 H10: --terminate-safe was not root-scoped at all."""
    venv = tmp_path / "venv"
    (venv / "Scripts").mkdir(parents=True)
    foreign = tmp_path / "elsewhere"
    foreign.mkdir()

    process = MagicMock()
    process.create_time.return_value = 1722798000.25
    process.cmdline.return_value = ["python.exe", "-m", "http.server", "8766"]
    process.exe.return_value = str(foreign / "python.exe")
    fake_psutil = types.SimpleNamespace(Process=lambda _pid: process)

    stopped, error = _terminate_safe_preview(
        100, 1722798000.25, venv, psutil_module=fake_psutil
    )

    assert stopped is False
    assert error == "process is not running from the target root venv"
    process.terminate.assert_not_called()
    process.kill.assert_not_called()


def test_terminate_safe_preview_spares_a_child_tree_outside_the_target_venv(tmp_path) -> None:
    """#104687 H10: a preview server is not authorization to kill a foreign child."""
    venv = tmp_path / "venv"
    (venv / "Scripts").mkdir(parents=True)
    foreign = tmp_path / "elsewhere"
    foreign.mkdir()

    class FakeProcess:
        def __init__(self, pid, created, args, executable):
            self.pid = pid
            self._created = created
            self._args = args
            self.executable = executable
            self.terminated = False
            self._children = []

        def create_time(self):
            return self._created

        def cmdline(self):
            return self._args

        def exe(self):
            return self.executable

        def children(self, *, recursive):
            return self._children

        def terminate(self):
            self.terminated = True

    child = FakeProcess(200, 10.0, ["cmd.exe"], str(foreign / "cmd.exe"))
    parent = FakeProcess(
        100,
        1722798000.25,
        ["python.exe", "-m", "http.server", "8766"],
        str(venv / "Scripts" / "python.exe"),
    )
    parent._children = [child]
    fake_psutil = types.SimpleNamespace(
        Process=lambda pid: parent,
        wait_procs=lambda processes, timeout: (processes, []),
    )

    stopped, error = _terminate_safe_preview(
        100, 1722798000.25, venv, psutil_module=fake_psutil
    )

    assert stopped is True
    assert error is None
    assert parent.terminated is True
    assert child.terminated is False


def test_terminate_safe_preview_rejects_a_nonsense_identity(tmp_path) -> None:
    called = []
    fake_psutil = types.SimpleNamespace(Process=lambda pid: called.append(pid))

    for pid, created in ((0, 1.0), (-1, 1.0), (100, 0.0), (100, float("nan"))):
        stopped, error = _terminate_safe_preview(
            pid, created, tmp_path, psutil_module=fake_psutil
        )
        assert stopped is False
        assert error == "invalid process identity"

    assert called == []


def test_executable_within_treats_unreadable_identity_as_foreign(tmp_path) -> None:
    denied = MagicMock()
    denied.exe.side_effect = PermissionError("access denied")
    assert _executable_within(denied, tmp_path) is False

    empty = MagicMock()
    empty.exe.return_value = ""
    assert _executable_within(empty, tmp_path) is False


def test_terminate_safe_preview_refuses_reused_pid(tmp_path) -> None:
    process = MagicMock()
    process.create_time.return_value = 1722798999.0
    fake_psutil = types.SimpleNamespace(Process=lambda _pid: process)

    stopped, error = _terminate_safe_preview(
        100, 1722798000.25, tmp_path, psutil_module=fake_psutil
    )

    assert stopped is False
    assert error == "process identity changed"
    process.cmdline.assert_not_called()
    process.terminate.assert_not_called()


# ---------------------------------------------------------------------------
# _is_pausable_gateway — the gateway exemption
#
# `hermes-setup` always invokes `hermes update --yes --gateway`, whose
# `_pause_windows_gateways_for_update()` stops running gateways itself. The
# Desktop preflight must therefore not report gateway launcher/worker chains
# as blockers — doing so aborts the handoff before the component that can
# handle them ever runs.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cmdline",
    [
        # venv-side launcher, exactly as the scheduled task spawns it
        r"C:\Users\u\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe"
        " -m hermes_cli.main gateway run --replace",
        # uv-side worker re-running the same argv (quoted exe, double space)
        r'"C:\Users\u\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none\python.exe"'
        "  -m hermes_cli.main gateway run --replace",
        # profile-scoped gateway
        "python.exe -m hermes_cli.main --profile work gateway run",
        # a profile literally NAMED "gateway" — the profile value must not
        # shadow the subcommand token (the hand-rolled matcher regressed this)
        "python.exe -m hermes_cli.main --profile gateway gateway run",
        "python.exe -m hermes_cli.main -p gateway gateway run",
        # case variations survive
        "PYTHON.EXE -m hermes_cli.main GATEWAY RUN",
    ],
)
def test_is_pausable_gateway_accepts_gateway_run_chains(cmdline: str) -> None:
    assert _is_pausable_gateway(cmdline) is True


@pytest.mark.parametrize(
    "cmdline",
    [
        # desktop backend: no pause machinery downstream, must keep blocking
        "python.exe -m hermes_cli.main serve --host 127.0.0.1 --port 8756",
        # other gateway subcommands are not running gateways
        "python.exe -m hermes_cli.main gateway stop",
        "python.exe -m hermes_cli.main gateway status",
        "python.exe -m hermes_cli.main gateway install",
        # A bare gateway command is not proof that the operative action was run.
        "python.exe -m hermes_cli.main gateway",
        # operator REPL / stray script
        "python.exe",
        "python.exe myscript.py gateway run",  # not a hermes_cli.main invocation
        "",
    ],
)
def test_is_pausable_gateway_rejects_everything_else(cmdline: str) -> None:
    assert _is_pausable_gateway(cmdline) is False


def test_hermes_command_parser_is_exact_and_understands_python_options() -> None:
    assert (
        _hermes_cli_command(
            "python.exe -X utf8 -m hermes_cli.main serve --host 127.0.0.1"
        )
        == "serve"
    )
    assert (
        _hermes_cli_command(
            "python.exe -m hermes_cli.main --profile serve gateway run"
        )
        == "gateway"
    )
    assert (
        _hermes_cli_command(
            "python.exe -m agent.transports.hermes_tools_mcp_server"
        )
        is None
    )
    assert (
        _hermes_cli_command(
            "python.exe script.py -m hermes_cli.main serve"
        )
        is None
    )
    assert (
        _hermes_cli_command(
            "python.exe -m hermes_cli.main --unknown serve"
        )
        is None
    )


def test_gateway_classifier_preserves_live_argv_token_boundaries() -> None:
    argv = [
        "python.exe",
        "-m",
        "hermes_cli.main",
        "serve",
        "--title",
        "x gateway run",
    ]

    assert _hermes_cli_command(argv) == "serve"
    assert not _is_pausable_gateway(argv)


def test_strict_detector_keeps_exe_less_exact_mcp_candidate(
    monkeypatch, tmp_path: Path
) -> None:

    module = "agent.transports.hermes_tools_mcp_server"
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda _attrs: iter(
            [
                _detector_proc(
                    778,
                    None,
                    "python.exe",
                    ["python.exe", "-m", module],
                    cwd=str(tmp_path),
                )
            ]
        )
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    assert scanner._detect_target_venv_holders(tmp_path, strict=True) == [(778, "python.exe", f"python.exe -m {module}")]


def _run_main_with_detector(monkeypatch, capsys, matches):
    """Run main() with the process detector patched to return *matches*."""
    for name, mod in _psutil_fake().items():
        monkeypatch.setitem(sys.modules, name, mod)

    monkeypatch.setattr(
        scanner,
        "_detect_target_venv_holders",
        lambda root, *, strict: matches,
    )
    live = {
        int(pid): scanner._ProcessSnapshot(
            pid=int(pid),
            ppid=1,
            name=str(name),
            exe=scanner._tokens(cmdline)[0] if scanner._tokens(cmdline) else "python.exe",
            argv=tuple(scanner._tokens(cmdline)),
            created_at=100.0 + int(pid),
            process=MagicMock(),
        )
        for pid, name, cmdline in matches
    }
    monkeypatch.setattr(scanner, "_snapshot_for_pid", lambda pid: live[int(pid)])
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setattr(scanner, "_validated_root", lambda _root: (root, root / "venv"))
    with pytest.raises(SystemExit) as excinfo:
        main(["--root", str(root)])
    out = capsys.readouterr().out
    return excinfo.value.code, json.loads(out)


def test_probe_fail_json_is_unambiguous_failure() -> None:
    """A failed probe must not look like a clear scan (#83149).

    Humans and naive callers used to read ``blocked: false`` as "no holders"
    when psutil was missing after a gutted venv. The document must mark
    ``probe_failed`` and keep ``ok`` false.
    """
    data = json.loads(
        _probe_fail_json(
            message="psutil is not available: No module named 'psutil'"
        )
    )
    assert data["ok"] is False
    assert data["ready"] is False
    assert data["blocked"] is True
    assert data["reason"] == "probe_failed"
    assert data["processes"] == []
    assert data["error"]["code"] == "probe_failed"
    assert "psutil" in data["error"]["message"]


def test_main_psutil_missing_is_probe_failure_not_clear(monkeypatch, capsys):
    """Missing psutil exits non-zero and stays fail-closed — never a clear scan.

    #104687 H9: the scanner imports psutil from the very site-packages the
    update is about to rewrite, so an interrupted update can leave psutil
    half-written and every later attempt refuses identically. The refusal must
    stay fail-closed, but it carries its own repair command and a code distinct
    from a generic probe failure so the trap is recognisable in a log.
    """
    real_import = builtins.__import__

    def _no_psutil(name, *args, **kwargs):
        if name == "psutil" or name.startswith("psutil."):
            raise ImportError("DLL load failed while importing _psutil_windows")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_psutil)
    monkeypatch.delitem(sys.modules, "psutil", raising=False)

    root = Path(__file__).resolve().parents[2]
    with pytest.raises(SystemExit) as excinfo:
        main(["--root", str(root)])
    captured = capsys.readouterr()
    assert excinfo.value.code == 1
    data = json.loads(captured.out)
    assert data["ok"] is False
    assert data["ready"] is False
    assert data["blocked"] is True
    assert data["reason"] == "scanner_dependency_unavailable"
    assert data["error"]["code"] == "scanner_dependency_unavailable"
    assert "psutil" in captured.err.lower()
    assert "pip install --force-reinstall psutil" in data["error"]["message"]
    assert sys.executable in data["error"]["message"]


def test_main_exempts_gateway_chain_but_keeps_other_holders(monkeypatch, capsys):
    """A gateway launcher/worker pair alone must scan clear; a non-gateway
    holder alongside it must still block (and be the only reported PID)."""
    gateway_launcher = (
        12,
        "python.exe",
        r"C:\x\venv\Scripts\python.exe -m hermes_cli.main gateway run --replace",
    )
    gateway_worker = (
        34,
        "python.exe",
        r'"C:\u\uv\python\python.exe"  -m hermes_cli.main gateway run --replace',
    )
    stray_repl = (56, "python.exe", r"C:\x\venv\Scripts\python.exe")

    # Gateway chain only → clear
    code, data = _run_main_with_detector(
        monkeypatch, capsys, [gateway_launcher, gateway_worker]
    )
    assert code == 0
    assert data["ok"] is True
    assert data["blocked"] is False
    assert data["processes"] == []
    assert data["pausable_gateways"] == 2

    # Gateway chain + stray REPL → blocked, reporting only the REPL
    code, data = _run_main_with_detector(
        monkeypatch, capsys, [gateway_launcher, gateway_worker, stray_repl]
    )
    assert code == 0
    assert data["blocked"] is True
    assert [p["pid"] for p in data["processes"]] == [56]
    assert data["pausable_gateways"] == 2


def test_main_desktop_serve_backend_still_blocks(monkeypatch, capsys):
    """A `serve` backend with NO ledger identity must keep blocking —
    the updater's stop/relaunch rungs only own ledger-verified backends."""
    import hermes_cli.process_identity as pi

    monkeypatch.setattr(pi, "ledger_entries", lambda **kw: [])
    serve = (
        78,
        "python.exe",
        r"C:\x\venv\Scripts\python.exe -m hermes_cli.main serve --host 127.0.0.1",
    )
    code, data = _run_main_with_detector(monkeypatch, capsys, [serve])
    assert code == 0
    assert data["blocked"] is True
    assert [p["pid"] for p in data["processes"]] == [78]
    assert data["pausable_gateways"] == 0
    assert data["deferred_backends"] == 0


# ---------------------------------------------------------------------------
# _is_updater_owned_backend — the serve/dashboard deferral (#98336)
# ---------------------------------------------------------------------------


_SERVE_CMD = r"C:\x\venv\Scripts\python.exe -m hermes_cli.main serve --host 127.0.0.1"


def _patch_ledger(monkeypatch, entries, dead):
    import hermes_cli.process_identity as pi

    monkeypatch.setattr(pi, "ledger_entries", lambda **kw: entries)
    monkeypatch.setattr(pi, "spawner_is_dead", lambda entry: dead)


def test_updater_owned_backend_dead_spawner_is_deferred(monkeypatch, capsys):
    """A ledger-verified serve whose spawner is provably dead is exactly the
    orphan `_ledger_reapable_backend_pids` reaps — the scan must defer it
    instead of dead-ending the hand-off (#98336)."""
    _patch_ledger(
        monkeypatch,
        [{"pid": 78, "purpose": "serve", "spawner_pid": 4242, "port": 9119}],
        dead=True,
    )
    code, data = _run_main_with_detector(monkeypatch, capsys, [(78, "python.exe", _SERVE_CMD)])
    assert code == 0
    assert data["blocked"] is False
    assert data["processes"] == []
    assert data["deferred_backends"] == 1
    # Sanitized decision evidence (#98350): structured ledger identity only —
    # the deferral must explain itself without echoing the command line.
    assert data["deferred_backend_evidence"] == [
        {"pid": 78, "purpose": "serve", "port": 9119}
    ]
    assert "--host" not in json.dumps(data["deferred_backend_evidence"])


def test_updater_owned_backend_unrecorded_spawner_is_deferred(monkeypatch, capsys):
    """spawner unrecorded (None from spawner_is_dead) but ledger-registered:
    the manual-serve rung stops and relaunches it — defer."""
    _patch_ledger(monkeypatch, [{"pid": 78, "purpose": "serve"}], dead=None)
    code, data = _run_main_with_detector(monkeypatch, capsys, [(78, "python.exe", _SERVE_CMD)])
    assert data["blocked"] is False
    assert data["deferred_backends"] == 1


def test_updater_owned_backend_live_foreign_spawner_still_blocks(monkeypatch, capsys):
    """A serve whose recorded spawner is ALIVE and not in this scan's
    ancestry (another supervisor would respawn it) must keep blocking."""
    from hermes_cli import _scan_venv_blockers as scan_mod

    _patch_ledger(
        monkeypatch, [{"pid": 78, "purpose": "serve", "spawner_pid": 4242}], dead=False
    )
    monkeypatch.setattr(scan_mod, "_spawner_is_this_handoff_desktop", lambda entry: False)
    code, data = _run_main_with_detector(monkeypatch, capsys, [(78, "python.exe", _SERVE_CMD)])
    assert data["blocked"] is True
    assert [p["pid"] for p in data["processes"]] == [78]
    assert data["deferred_backends"] == 0


def test_updater_owned_backend_handoff_desktop_spawner_is_deferred(monkeypatch, capsys):
    """A serve owned by THIS hand-off's Desktop (the scan's own ancestor) is
    deferred: the Desktop exits before the updater runs, turning it into the
    dead-spawner orphan the ledger rung reaps (#98336 field case)."""
    from hermes_cli import _scan_venv_blockers as scan_mod

    _patch_ledger(
        monkeypatch, [{"pid": 78, "purpose": "serve", "spawner_pid": 4242}], dead=False
    )
    monkeypatch.setattr(scan_mod, "_spawner_is_this_handoff_desktop", lambda entry: True)
    code, data = _run_main_with_detector(monkeypatch, capsys, [(78, "python.exe", _SERVE_CMD)])
    assert data["blocked"] is False
    assert data["deferred_backends"] == 1


def test_updater_owned_backend_purpose_mismatch_blocks(monkeypatch, capsys):
    """Argv says serve but the ledger entry says a different purpose — the
    identity is inconsistent; fail closed and keep blocking."""
    _patch_ledger(
        monkeypatch, [{"pid": 78, "purpose": "mcp-helper", "spawner_pid": 4242}], dead=True
    )
    code, data = _run_main_with_detector(monkeypatch, capsys, [(78, "python.exe", _SERVE_CMD)])
    assert data["blocked"] is True
    assert data["deferred_backends"] == 0


def test_updater_owned_backend_non_backend_argv_never_deferred(monkeypatch, capsys):
    """A ledger entry cannot exempt a process whose argv is not a
    serve/dashboard subcommand (#90778 token rules: `kanban --preserve-cache`
    must not read as serve)."""
    _patch_ledger(
        monkeypatch, [{"pid": 78, "purpose": "serve", "spawner_pid": 4242}], dead=True
    )
    kanban = (
        78,
        "python.exe",
        r"C:\x\venv\Scripts\python.exe -m hermes_cli.main kanban --preserve-cache",
    )
    code, data = _run_main_with_detector(monkeypatch, capsys, [kanban])
    assert data["blocked"] is True
    assert data["deferred_backends"] == 0


def test_updater_owned_backend_ledger_failure_blocks(monkeypatch, capsys):
    """Ledger unreadable → unprovable → fail closed (pre-exemption behavior)."""
    import hermes_cli.process_identity as pi

    def _boom(**kw):
        raise RuntimeError("ledger unavailable")

    monkeypatch.setattr(pi, "ledger_entries", _boom)
    code, data = _run_main_with_detector(monkeypatch, capsys, [(78, "python.exe", _SERVE_CMD)])
    assert data["blocked"] is True
    assert data["deferred_backends"] == 0


def test_main_gateway_with_long_managed_runtime_path_is_exempt(monkeypatch, capsys):
    r"""Regression: the detector must hand the FULL cmdline to the exemption.

    Gateways launched via the managed-runtime interpreter carry a >120-char
    exe path (`.hermes-runtime\python\generation-...\cpython-3.11-...`).
    The old `cmdline_raw[:120]` truncation in the detector cut the cmdline
    before `-m hermes_cli.main gateway run`, so the exemption never matched
    and every Desktop update aborted with 'Update didn't finish'.
    Here the detector returns full cmdlines (post-fix contract); the scan
    must exempt the gateway and truncate only the *displayed* cmdline.
    """
    long_exe = (
        r'"C:\Users\u\AppData\Local\hermes\hermes-agent\.hermes-runtime\python'
        r"\generation-1785095035-66720-be29ea9c\cpython-3.11-windows-x86_64-none"
        r'\python.exe"'
    )
    assert len(long_exe) > 120  # the truncation point was inside the exe path
    gateway = (91, "python.exe", long_exe + "  -m hermes_cli.main gateway run --replace")
    code, data = _run_main_with_detector(monkeypatch, capsys, [gateway])
    assert code == 0
    assert data["blocked"] is False
    assert data["processes"] == []
    assert data["pausable_gateways"] == 1

    # A long-path NON-gateway holder still blocks, with cmdline truncated for display.
    stray = (92, "python.exe", long_exe + "  -m some_other_module --serve-forever")
    code, data = _run_main_with_detector(monkeypatch, capsys, [gateway, stray])
    assert data["blocked"] is True
    assert [p["pid"] for p in data["processes"]] == [92]
    assert len(data["processes"][0]["cmdline"]) <= 120


class _FakeProcess:
    def __init__(self, *, parents=()):
        self._parents = list(parents)
        self.kills = 0

    def parents(self):
        return list(self._parents)

    def kill(self):
        self.kills += 1


class _FakeAncestor:
    def __init__(
        self,
        name: str,
        exe: str,
        argv: list[str],
        *,
        pid: int = 10,
        created_at: float = 1.0,
    ):
        self.pid = pid
        self._created_at = created_at
        self._name = name
        self._exe = exe
        self._argv = argv

    def create_time(self):
        return self._created_at

    def name(self):
        return self._name

    def exe(self):
        return self._exe

    def cmdline(self):
        return list(self._argv)


def _snapshot(
    *,
    pid: int,
    ppid: int,
    exe: Path,
    argv: tuple[str, ...],
    created_at: float,
    process: _FakeProcess | None = None,
):
    return scanner._ProcessSnapshot(
        pid=pid,
        ppid=ppid,
        name="python.exe",
        exe=str(exe),
        argv=argv,
        created_at=created_at,
        process=process or _FakeProcess(),
    )


def test_snapshot_distinguishes_pid_reuse_from_exit(monkeypatch) -> None:
    no_such_process = type("NoSuchProcess", (Exception,), {})
    initial = types.SimpleNamespace(
        create_time=lambda: 100.0,
        cmdline=lambda: [r"C:\Python311\python.exe", "-m", "worker"],
        exe=lambda: r"C:\Python311\python.exe",
        name=lambda: "python.exe",
    )
    reused = types.SimpleNamespace(create_time=lambda: 100.001)
    processes = iter([initial, reused])
    fake_psutil = types.SimpleNamespace(
        Process=lambda _pid: next(processes),
        NoSuchProcess=no_such_process,
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    with pytest.raises(RuntimeError, match="changed generation"):
        scanner._snapshot_for_pid(20, parent_by_pid={20: 10})

    exit_reads = 0

    def _exited_after_identity(_pid):
        nonlocal exit_reads
        exit_reads += 1
        if exit_reads == 1:
            return initial
        raise no_such_process

    fake_psutil.Process = _exited_after_identity
    assert scanner._snapshot_for_pid(20, parent_by_pid={20: 10}) is None


def test_scan_keeps_identity_refresh_generation_change_as_hard_blocker(
    monkeypatch, tmp_path: Path
) -> None:

    root = tmp_path / "install"
    venv = root / "venv"
    module = "agent.transports.hermes_tools_mcp_server"
    initial = types.SimpleNamespace(
        create_time=lambda: 100.0,
        cmdline=lambda: [r"C:\Python311\python.exe", "-m", module],
        exe=lambda: r"C:\Python311\python.exe",
        name=lambda: "python.exe",
    )
    reused = types.SimpleNamespace(create_time=lambda: 100.001)
    processes = iter([initial, reused])
    parent_maps = MagicMock(return_value={20: 10})
    fake_psutil = types.SimpleNamespace(
        _ppid_map=parent_maps,
        Process=lambda _pid: next(processes),
        NoSuchProcess=type("NoSuchProcess", (Exception,), {}),
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(scanner, "_validated_root", lambda _root: (root, venv))
    monkeypatch.setattr(
        scanner,
        "_detect_target_venv_holders",
        lambda root, *, strict, **_kwargs: [
            (20, "python.exe", f"python.exe -m {module}")
        ],
    )

    result = scanner.scan_venv_blockers(root)

    assert result["mcp_bridges"] == []
    assert [entry["pid"] for entry in result["processes"]] == [20]
    assert result["processes"][0]["action"] == "refuse"
    assert parent_maps.call_count == 1


def test_terminate_refuses_pid_reused_during_identity_refresh(
    monkeypatch, tmp_path: Path
) -> None:
    root = tmp_path / "install"
    venv = root / "venv"
    initial = MagicMock()
    initial.create_time.return_value = 100.0
    initial.cmdline.return_value = [
        r"C:\Python311\python.exe",
        "-m",
        "agent.transports.hermes_tools_mcp_server",
    ]
    initial.exe.return_value = r"C:\Python311\python.exe"
    initial.ppid.return_value = 10
    initial.name.return_value = "python.exe"
    reused = types.SimpleNamespace(create_time=lambda: 100.001)
    processes = iter([initial, reused])
    fake_psutil = types.SimpleNamespace(
        Process=lambda _pid: next(processes),
        NoSuchProcess=type("NoSuchProcess", (Exception,), {}),
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(scanner, "_validated_root", lambda _root: (root, venv))

    assert not scanner.terminate_mcp_bridge(root, pid=20, created_at=100.0)
    initial.kill.assert_not_called()


def test_scan_reports_worker_before_wrapper_and_keeps_relationship(
    monkeypatch, tmp_path: Path
) -> None:

    root = tmp_path / "install"
    venv = root / "venv"
    module = "agent.transports.hermes_tools_mcp_server"
    wrapper = _snapshot(
        pid=10,
        ppid=1,
        exe=venv / "Scripts" / "python.exe",
        argv=(str(venv / "Scripts" / "python.exe"), "-m", module),
        created_at=100.0,
    )
    worker = _snapshot(
        pid=20,
        ppid=10,
        exe=root / ".hermes-runtime" / "python" / "generation" / "python.exe",
        argv=("python.exe", "-m", module),
        created_at=101.0,
    )
    parent_maps = MagicMock(return_value={10: 1, 20: 10})
    live_created_at = {10: 100.0, 20: 101.0}
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        types.SimpleNamespace(
            _ppid_map=parent_maps,
            Process=lambda pid: types.SimpleNamespace(
                create_time=lambda: live_created_at[int(pid)]
            ),
            NoSuchProcess=type("NoSuchProcess", (Exception,), {}),
        ),
    )
    monkeypatch.setattr(scanner, "_validated_root", lambda _root: (root, venv))
    monkeypatch.setattr(
        scanner,
        "_detect_target_venv_holders",
        lambda root, *, strict, **_kwargs: [
            (10, "python.exe", " ".join(wrapper.argv)),
            (20, "python.exe", " ".join(worker.argv)),
        ],
    )
    monkeypatch.setattr(
        scanner,
        "_snapshot_for_pid",
        lambda pid, **_kwargs: {10: wrapper, 20: worker}[pid],
    )
    owner_probe = MagicMock(return_value="codex")
    monkeypatch.setattr(scanner, "_owner_from_ancestry", owner_probe)

    result = scanner.scan_venv_blockers(root)

    assert [entry["pid"] for entry in result["mcp_bridges"]] == [20, 10]
    assert result["mcp_bridges"][0]["role"] == "mcp_bridge_worker"
    assert result["mcp_bridges"][0]["wrapper_pid"] == 10
    assert result["mcp_bridges"][1]["role"] == "mcp_bridge_wrapper"
    assert all(entry["owner"] == "codex" for entry in result["mcp_bridges"])
    assert owner_probe.call_count == 1
    assert parent_maps.call_count == 2


def test_scan_reports_only_host_proven_desktop_plugin_service_pairs(
    monkeypatch, tmp_path: Path
) -> None:
    """A detached Desktop plugin service is drainable only with its exact host.

    This is the real Desktop updater failure shape: the plugin's venv wrapper
    re-execs a managed-runtime worker, and the VBS service host would recreate
    both unless the updater drains the proven pair after consent.
    """

    root = tmp_path / "install"
    venv = root / "venv"
    service = root.parent / "desktop-plugins" / "tracker" / "service.py"
    wrapper = _snapshot(
        pid=10,
        ppid=5,
        exe=venv / "Scripts" / "python.exe",
        argv=(str(venv / "Scripts" / "python.exe"), str(service)),
        created_at=100.0,
    )
    worker = _snapshot(
        pid=20,
        ppid=10,
        exe=root / ".hermes-runtime" / "python" / "generation" / "python.exe",
        argv=(
            str(root / ".hermes-runtime" / "python" / "generation" / "python.exe"),
            str(service),
        ),
        created_at=101.0,
    )
    parent_maps = MagicMock(return_value={5: 1, 10: 5, 20: 10})
    live_created_at = {10: 100.0, 20: 101.0}
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        types.SimpleNamespace(
            _ppid_map=parent_maps,
            Process=lambda pid: types.SimpleNamespace(
                create_time=lambda: live_created_at[int(pid)]
            ),
            NoSuchProcess=type("NoSuchProcess", (Exception,), {}),
        ),
    )
    monkeypatch.setattr(scanner, "_validated_root", lambda _root: (root, venv))
    monkeypatch.setattr(
        scanner,
        "_detect_target_venv_holders",
        lambda root, *, strict, **_kwargs: [
            (10, "python.exe", " ".join(wrapper.argv)),
            (20, "python.exe", " ".join(worker.argv)),
        ],
    )
    monkeypatch.setattr(
        scanner,
        "_snapshot_for_pid",
        lambda pid, **_kwargs: {10: wrapper, 20: worker}[pid],
    )
    host_probe = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(scanner, "_desktop_plugin_service_host", host_probe)

    result = scanner.scan_venv_blockers(root)

    assert result["processes"] == []
    assert [entry["pid"] for entry in result["desktop_plugin_services"]] == [20, 10]
    assert result["desktop_plugin_services"][0]["role"] == "desktop_plugin_worker"
    assert result["desktop_plugin_services"][0]["wrapper_pid"] == 10
    assert result["desktop_plugin_services"][1]["action"] == "terminate_desktop_plugin_service"
    assert host_probe.call_count == 1


def test_scan_keeps_desktop_plugin_script_without_proven_service_host_blocked(
    monkeypatch, tmp_path: Path
) -> None:

    root = tmp_path / "install"
    venv = root / "venv"
    service = root.parent / "desktop-plugins" / "tracker" / "service.py"
    wrapper = _snapshot(
        pid=10,
        ppid=5,
        exe=venv / "Scripts" / "python.exe",
        argv=(str(venv / "Scripts" / "python.exe"), str(service)),
        created_at=100.0,
    )
    parent_maps = MagicMock(return_value={5: 1, 10: 5})
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        types.SimpleNamespace(
            _ppid_map=parent_maps,
            Process=lambda _pid: types.SimpleNamespace(create_time=lambda: 100.0),
            NoSuchProcess=type("NoSuchProcess", (Exception,), {}),
        ),
    )
    monkeypatch.setattr(scanner, "_validated_root", lambda _root: (root, venv))
    monkeypatch.setattr(
        scanner,
        "_detect_target_venv_holders",
        lambda root, *, strict, **_kwargs: [(10, "python.exe", " ".join(wrapper.argv))],
    )
    monkeypatch.setattr(scanner, "_snapshot_for_pid", lambda _pid, **_kwargs: wrapper)
    monkeypatch.setattr(scanner, "_desktop_plugin_service_host", lambda *_args: None)

    result = scanner.scan_venv_blockers(root)

    assert result["desktop_plugin_services"] == []
    assert [entry["pid"] for entry in result["processes"]] == [10]
    assert result["processes"][0]["action"] == "refuse"


def test_scan_owner_cache_is_bound_to_anchor_generation(
    monkeypatch, tmp_path: Path
) -> None:

    root = tmp_path / "install"
    venv = root / "venv"
    module = "agent.transports.hermes_tools_mcp_server"
    wrappers = {
        pid: _snapshot(
            pid=pid,
            ppid=10,
            exe=venv / "Scripts" / "python.exe",
            argv=(str(venv / "Scripts" / "python.exe"), "-m", module),
            created_at=created_at,
        )
        for pid, created_at in ((20, 100.0), (30, 200.0))
    }
    parent_maps = MagicMock(return_value={20: 10, 30: 10, 10: 1})
    live_created_at = {20: 100.0, 30: 200.0}
    fake_psutil = types.SimpleNamespace(
        _ppid_map=parent_maps,
        Process=lambda pid: types.SimpleNamespace(
            create_time=lambda: live_created_at[int(pid)]
        ),
        NoSuchProcess=type("NoSuchProcess", (Exception,), {}),
    )
    owner_probe = MagicMock(side_effect=["codex", "unknown"])
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(scanner, "_validated_root", lambda _root: (root, venv))
    monkeypatch.setattr(
        scanner,
        "_detect_target_venv_holders",
        lambda root, *, strict, **_kwargs: [
            (pid, "python.exe", " ".join(snapshot.argv))
            for pid, snapshot in wrappers.items()
        ],
    )
    monkeypatch.setattr(
        scanner,
        "_snapshot_for_pid",
        lambda pid, **_kwargs: wrappers[pid],
    )
    monkeypatch.setattr(scanner, "_owner_from_ancestry", owner_probe)

    result = scanner.scan_venv_blockers(root)

    assert [entry["owner"] for entry in result["mcp_bridges"]] == [
        "codex",
        "unknown",
    ]
    assert result["mcp_bridges"][1]["action"] == "refuse"
    assert owner_probe.call_count == 2


def test_scan_fails_closed_when_fresh_parent_snapshot_fails(
    monkeypatch, tmp_path: Path
) -> None:

    root = tmp_path / "install"
    venv = root / "venv"
    wrapper = _snapshot(
        pid=10,
        ppid=1,
        exe=venv / "Scripts" / "python.exe",
        argv=(str(venv / "Scripts" / "python.exe"), "-m", "worker"),
        created_at=100.0,
    )
    parent_maps = MagicMock(
        side_effect=[{10: 1}, PermissionError("fresh parent map denied")]
    )
    fake_psutil = types.SimpleNamespace(
        _ppid_map=parent_maps,
        NoSuchProcess=type("NoSuchProcess", (Exception,), {}),
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(scanner, "_validated_root", lambda _root: (root, venv))
    monkeypatch.setattr(
        scanner,
        "_detect_target_venv_holders",
        lambda root, *, strict, **_kwargs: [
            (10, "python.exe", " ".join(wrapper.argv))
        ],
    )
    monkeypatch.setattr(
        scanner,
        "_snapshot_for_pid",
        lambda _pid, **_kwargs: wrapper,
    )

    with pytest.raises(RuntimeError, match="parent process enumeration failed"):
        scanner.scan_venv_blockers(root)

    assert parent_maps.call_count == 2


def test_scan_keeps_generation_recheck_error_as_hard_blocker(
    monkeypatch, tmp_path: Path
) -> None:

    root = tmp_path / "install"
    venv = root / "venv"
    wrapper = _snapshot(
        pid=10,
        ppid=1,
        exe=venv / "Scripts" / "python.exe",
        argv=(str(venv / "Scripts" / "python.exe"), "-m", "worker"),
        created_at=100.0,
    )
    parent_maps = MagicMock(return_value={10: 1})
    fake_psutil = types.SimpleNamespace(
        _ppid_map=parent_maps,
        Process=lambda _pid: (_ for _ in ()).throw(PermissionError("denied")),
        NoSuchProcess=type("NoSuchProcess", (Exception,), {}),
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(scanner, "_validated_root", lambda _root: (root, venv))
    monkeypatch.setattr(
        scanner,
        "_detect_target_venv_holders",
        lambda root, *, strict, **_kwargs: [
            (10, "python.exe", " ".join(wrapper.argv))
        ],
    )
    monkeypatch.setattr(
        scanner,
        "_snapshot_for_pid",
        lambda _pid, **_kwargs: wrapper,
    )

    result = scanner.scan_venv_blockers(root)

    assert result["mcp_bridges"] == []
    assert [entry["pid"] for entry in result["processes"]] == [10]
    assert result["processes"][0]["action"] == "refuse"
    assert parent_maps.call_count == 2


def test_scan_does_not_offer_worker_after_pid_generation_changes(
    monkeypatch, tmp_path: Path
) -> None:

    root = tmp_path / "install"
    venv = root / "venv"
    module = "agent.transports.hermes_tools_mcp_server"
    wrapper = _snapshot(
        pid=10,
        ppid=1,
        exe=venv / "Scripts" / "python.exe",
        argv=(str(venv / "Scripts" / "python.exe"), "-m", module),
        created_at=100.0,
    )
    worker = _snapshot(
        pid=20,
        ppid=10,
        exe=Path(r"C:\Python311\python.exe"),
        argv=(r"C:\Python311\python.exe", "-m", module),
        created_at=201.0,
    )
    parent_maps = MagicMock(
        side_effect=[
            {10: 1, 20: 10},  # discovery generation
            {10: 1, 20: 10},  # fresh classification generation
        ]
    )

    def _detect(root, *, strict, _parent_by_pid=None):
        assert root == tmp_path / "install"
        assert strict is True
        parents = _parent_by_pid
        if parents is None:
            parents = parent_maps()
        assert parents == {10: 1, 20: 10}
        return [
            (10, "python.exe", " ".join(wrapper.argv)),
            (20, "python.exe", " ".join(worker.argv)),
        ]

    live_created_at = {10: 100.0, 20: 201.001}
    fake_psutil = types.SimpleNamespace(
        _ppid_map=parent_maps,
        Process=lambda pid: types.SimpleNamespace(
            create_time=lambda: live_created_at[int(pid)]
        ),
        NoSuchProcess=type("NoSuchProcess", (Exception,), {}),
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(scanner, "_validated_root", lambda _root: (root, venv))
    monkeypatch.setattr(scanner, "_detect_target_venv_holders", _detect)
    monkeypatch.setattr(
        scanner,
        "_snapshot_for_pid",
        lambda pid, **_kwargs: {10: wrapper, 20: worker}[pid],
    )
    monkeypatch.setattr(
        scanner,
        "_owner_from_ancestry",
        lambda _snapshot, **_kwargs: "codex",
    )

    result = scanner.scan_venv_blockers(root)

    assert all(entry["pid"] != 20 for entry in result["mcp_bridges"])
    assert [entry["pid"] for entry in result["processes"]] == [20]
    assert result["processes"][0]["action"] == "refuse"
    assert parent_maps.call_count == 2


def test_unreadable_target_gateway_is_a_hard_blocker_not_exempted(
    monkeypatch, tmp_path: Path
) -> None:

    root = tmp_path / "install"
    venv = root / "venv"
    monkeypatch.setattr(scanner, "_validated_root", lambda _root: (root, venv))
    monkeypatch.setattr(
        scanner,
        "_detect_target_venv_holders",
        lambda root, *, strict, **_kwargs: [
            (77, "python.exe", "python.exe -m hermes_cli.main gateway run")
        ],
    )
    monkeypatch.setattr(
        scanner,
        "_snapshot_for_pid",
        lambda _pid, **_kwargs: (_ for _ in ()).throw(
            PermissionError("access denied")
        ),
    )

    result = scanner.scan_venv_blockers(root)

    assert result["blocked"] is True
    assert result["pausable_gateways"] == 0
    assert [process["pid"] for process in result["processes"]] == [77]
    assert result["processes"][0]["actionability"] == "hard_block"


def test_terminate_refuses_managed_worker_after_wrapper_exits(
    monkeypatch, tmp_path: Path
) -> None:
    root = tmp_path / "install"
    venv = root / "venv"
    process = _FakeProcess()
    worker = _snapshot(
        pid=20,
        ppid=10,
        exe=root / ".hermes-runtime" / "python" / "generation" / "python.exe",
        argv=("python.exe", "-m", "agent.transports.hermes_tools_mcp_server"),
        created_at=101.0,
        process=process,
    )
    monkeypatch.setattr(scanner, "_validated_root", lambda _root: (root, venv))
    monkeypatch.setattr(scanner, "_snapshot_for_pid", lambda pid: worker)

    assert not scanner.terminate_mcp_bridge(root, pid=20, created_at=101.0)
    assert process.kills == 0


def test_terminate_refuses_reused_unreadable_ancestor_before_codex(
    monkeypatch, tmp_path: Path
) -> None:
    root = tmp_path / "install"
    venv = root / "venv"

    def unreadable_name():
        raise RuntimeError("ancestor metadata unreadable")

    unreadable_parent = types.SimpleNamespace(
        pid=10,
        create_time=lambda: 99.0,
        name=unreadable_name,
        exe=lambda: r"C:\tools\old-parent.exe",
        cmdline=lambda: ["old-parent.exe"],
    )
    codex = _FakeAncestor(
        "codex.exe",
        r"C:\tools\codex.exe",
        ["codex.exe"],
        pid=1,
        created_at=90.0,
    )
    process = _FakeProcess(parents=[unreadable_parent, codex])
    worker = _snapshot(
        pid=20,
        ppid=10,
        exe=venv / "Scripts" / "python.exe",
        argv=(
            str(venv / "Scripts" / "python.exe"),
            "-m",
            "agent.transports.hermes_tools_mcp_server",
        ),
        created_at=100.0,
        process=process,
    )
    live_ancestors = {
        10: types.SimpleNamespace(create_time=lambda: 99.001),
        1: types.SimpleNamespace(create_time=lambda: 90.0),
    }
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        types.SimpleNamespace(Process=lambda pid: live_ancestors[int(pid)]),
    )
    monkeypatch.setattr(scanner, "_validated_root", lambda _root: (root, venv))
    monkeypatch.setattr(
        scanner,
        "_snapshot_for_pid",
        lambda pid: worker if int(pid) == worker.pid else None,
    )

    assert not scanner.terminate_mcp_bridge(root, pid=20, created_at=100.0)
    assert process.kills == 0


def test_external_worker_termination_targets_worker_not_live_wrapper(
    monkeypatch, tmp_path: Path
) -> None:
    root = tmp_path / "install"
    venv = root / "venv"
    module = "agent.transports.hermes_tools_mcp_server"
    wrapper_process = _FakeProcess()
    wrapper = _snapshot(
        pid=10,
        ppid=1,
        exe=venv / "Scripts" / "python.exe",
        argv=(str(venv / "Scripts" / "python.exe"), "-m", module),
        created_at=100.0,
        process=wrapper_process,
    )
    wrapper_parent = types.SimpleNamespace(pid=10)
    worker_process = _FakeProcess(parents=[wrapper_parent])
    worker = _snapshot(
        pid=20,
        ppid=10,
        exe=Path(r"C:\Python311\python.exe"),
        argv=(r"C:\Python311\python.exe", "-m", module),
        created_at=101.0,
        process=worker_process,
    )
    monkeypatch.setattr(scanner, "_validated_root", lambda _root: (root, venv))
    monkeypatch.setattr(
        scanner,
        "_snapshot_for_pid",
        lambda pid: {10: wrapper, 20: worker}.get(pid),
    )
    monkeypatch.setattr(
        scanner,
        "_owner_from_ancestry",
        lambda _snapshot, **_kwargs: "codex",
    )

    assert scanner.terminate_mcp_bridge(root, pid=20, created_at=101.0)
    assert worker_process.kills == 1
    assert wrapper_process.kills == 0


def test_node_hosted_claude_ancestry_is_attributed_exactly(
    monkeypatch, tmp_path: Path
) -> None:
    claude = _FakeAncestor(
        "node.exe",
        r"C:\Program Files\nodejs\node.exe",
        [
            r"C:\Program Files\nodejs\node.exe",
            r"C:\Users\u\AppData\Roaming\npm\node_modules\@anthropic-ai\claude-code\cli.js",
        ],
    )
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        types.SimpleNamespace(Process=lambda _pid: claude),
    )
    snapshot = _snapshot(
        pid=20,
        ppid=10,
        exe=tmp_path / "python.exe",
        argv=("python.exe", "-m", "agent.transports.hermes_tools_mcp_server"),
        created_at=101.0,
        process=_FakeProcess(parents=[claude]),
    )

    assert scanner._owner_from_ancestry(snapshot) == "claude"
    record = scanner._mcp_record(
        snapshot, role="mcp_bridge_worker", wrapper_pid=None
    )
    assert record["actionable"] is True
    assert record["action"] == "terminate_exact_mcp"


@pytest.mark.parametrize(
    ("parent_created_at", "expected_owner"),
    [
        (99.0, "codex"),
        (100.0, "codex"),
        (100.005, "unknown"),
        (100.02, "unknown"),
    ],
)
def test_snapshot_parent_map_rejects_reused_owner_pid(
    monkeypatch,
    tmp_path: Path,
    parent_created_at: float,
    expected_owner: str,
) -> None:
    parent = types.SimpleNamespace(
        pid=10,
        create_time=lambda: parent_created_at,
        name=lambda: "codex.exe",
        exe=lambda: r"C:\tools\codex.exe",
        cmdline=lambda: ["codex.exe"],
    )
    fake_psutil = types.SimpleNamespace(Process=lambda _pid: parent)
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    snapshot = _snapshot(
        pid=20,
        ppid=10,
        exe=tmp_path / "python.exe",
        argv=("python.exe", "-m", "agent.transports.hermes_tools_mcp_server"),
        created_at=100.0,
    )

    assert (
        scanner._owner_from_ancestry(
            snapshot,
            parent_by_pid={20: 10, 10: 0},
        )
        == expected_owner
    )


def test_snapshot_parent_map_rejects_owner_generation_changed_during_read(
    monkeypatch, tmp_path: Path
) -> None:
    parent = types.SimpleNamespace(
        pid=10,
        create_time=lambda: 99.0,
        name=lambda: "codex.exe",
        exe=lambda: r"C:\tools\codex.exe",
        cmdline=lambda: ["codex.exe"],
    )
    reused = types.SimpleNamespace(create_time=lambda: 99.001)
    processes = iter([parent, reused])
    fake_psutil = types.SimpleNamespace(Process=lambda _pid: next(processes))
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    snapshot = _snapshot(
        pid=20,
        ppid=10,
        exe=tmp_path / "python.exe",
        argv=("python.exe", "-m", "agent.transports.hermes_tools_mcp_server"),
        created_at=100.0,
    )

    assert (
        scanner._owner_from_ancestry(
            snapshot,
            parent_by_pid={20: 10, 10: 0},
        )
        == "unknown"
    )


def test_live_parent_ancestry_rejects_generation_changed_during_read(
    monkeypatch, tmp_path: Path
) -> None:
    parent = _FakeAncestor(
        "codex.exe",
        r"C:\tools\codex.exe",
        ["codex.exe"],
        created_at=99.0,
    )
    reused = types.SimpleNamespace(create_time=lambda: 99.001)
    fake_psutil = types.SimpleNamespace(Process=lambda _pid: reused)
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    snapshot = _snapshot(
        pid=20,
        ppid=10,
        exe=tmp_path / "python.exe",
        argv=("python.exe", "-m", "agent.transports.hermes_tools_mcp_server"),
        created_at=100.0,
        process=_FakeProcess(parents=[parent]),
    )

    assert scanner._owner_from_ancestry(snapshot) == "unknown"


@pytest.mark.parametrize("created_at", [float("nan"), float("inf"), -1.0, 102.0])
def test_terminate_refuses_invalid_or_reused_process_identity(
    monkeypatch, tmp_path: Path, created_at: float
) -> None:
    root = tmp_path / "install"
    venv = root / "venv"
    process = _FakeProcess()
    record = {
        "created_at": 101.0,
        "owner": "codex",
        "role": "mcp_bridge_worker",
        "actionable": True,
        "action": "terminate_exact_mcp",
    }
    monkeypatch.setattr(scanner, "_validated_root", lambda _root: (root, venv))
    monkeypatch.setattr(
        scanner,
        "_live_mcp_bridge_process",
        lambda _root, _pid: (process, record),
    )

    assert not scanner.terminate_mcp_bridge(root, pid=20, created_at=created_at)
    assert process.kills == 0


def test_terminate_refuses_unknown_owner_even_for_exact_worker(
    monkeypatch, tmp_path: Path
) -> None:
    root = tmp_path / "install"
    venv = root / "venv"
    process = _FakeProcess()
    record = {
        "created_at": 101.0,
        "owner": "unknown",
        "role": "mcp_bridge_worker",
        "actionable": False,
        "action": "refuse",
    }
    monkeypatch.setattr(scanner, "_validated_root", lambda _root: (root, venv))
    monkeypatch.setattr(
        scanner,
        "_live_mcp_bridge_process",
        lambda _root, _pid: (process, record),
    )

    assert not scanner.terminate_mcp_bridge(root, pid=20, created_at=101.0)
    assert process.kills == 0


def test_terminate_rereads_and_refuses_changed_live_argv(
    monkeypatch, tmp_path: Path
) -> None:
    root = tmp_path / "install"
    venv = root / "venv"
    process = _FakeProcess()
    changed = _snapshot(
        pid=20,
        ppid=10,
        exe=root / ".hermes-runtime" / "python" / "generation" / "python.exe",
        argv=("python.exe", "-m", "hermes_cli.main", "serve"),
        created_at=101.0,
        process=process,
    )
    monkeypatch.setattr(scanner, "_validated_root", lambda _root: (root, venv))
    monkeypatch.setattr(scanner, "_snapshot_for_pid", lambda pid: changed)

    assert not scanner.terminate_mcp_bridge(root, pid=20, created_at=101.0)
    assert process.kills == 0


class _UnitMember(_FakeProcess):
    """A unit member that records kill order into a shared journal."""

    def __init__(self, label: str, journal: list[str], *, pid: int = 0, gone: bool = False):
        super().__init__()
        self.label = label
        self.journal = journal
        self.pid = pid
        self.gone = gone

    def kill(self):
        import psutil

        self.journal.append(self.label)
        if self.gone:
            raise psutil.NoSuchProcess(self.pid)
        self.kills += 1

    def cmdline(self):
        return ["C:\\Windows\\system32\\wscript.exe", "C:\\hermes\\desktop-plugins\\tracker\\service-host.vbs"]

    def create_time(self):
        return 90.0

    def cwd(self):
        return "C:\\hermes"


def _plugin_unit(journal: list[str], *, worker_gone: bool = False):
    host = _UnitMember("host", journal, pid=5)
    wrapper = _UnitMember("wrapper", journal, pid=10)
    worker = _UnitMember("worker", journal, pid=20, gone=worker_gone)
    wrapper_snapshot = _snapshot(
        pid=10,
        ppid=5,
        exe=Path("C:/hermes/install/venv/Scripts/python.exe"),
        argv=("python.exe", "C:\\hermes\\desktop-plugins\\tracker\\service.py"),
        created_at=100.0,
        process=wrapper,
    )
    worker_snapshot = _snapshot(
        pid=20,
        ppid=10,
        exe=Path("C:/hermes/install/.hermes-runtime/python/generation/python.exe"),
        argv=("python.exe", "C:\\hermes\\desktop-plugins\\tracker\\service.py"),
        created_at=101.0,
        process=worker,
    )
    return host, wrapper_snapshot, worker_snapshot


def test_terminate_desktop_plugin_service_stops_host_then_wrapper_then_worker_from_either_member(
    monkeypatch, tmp_path: Path
) -> None:
    """The supervisor dies first so nothing can respawn; one call stops the unit.

    Worker-first draining in two calls was the 2026-09-04 failure: the venv
    wrapper exited on its own when its child died, the wrapper call had
    nothing left to prove, the Windows Script Host loop survived and
    relaunched the service ten seconds later inside the updater's lock
    window.
    """
    root = tmp_path / "install"
    venv = root / "venv"
    journal: list[str] = []
    host, wrapper, worker = _plugin_unit(journal)
    worker_record = {
        "pid": 20,
        "created_at": 101.0,
        "owner": "desktop",
        "role": "desktop_plugin_worker",
        "actionable": True,
        "action": "terminate_desktop_plugin_service",
        "wrapper_pid": 10,
    }
    monkeypatch.setattr(scanner, "_validated_root", lambda _root: (root, venv))
    monkeypatch.setattr(
        scanner,
        "_desktop_plugin_service_unit",
        lambda _root, pid: (worker_record, host, wrapper, [worker]) if int(pid) == 20 else None,
    )

    outcome = scanner.terminate_desktop_plugin_service_unit(root, pid=20, created_at=101.0)

    assert outcome["terminated"] is True
    assert journal == ["host", "wrapper", "worker"]
    assert host.kills == 1 and wrapper.process.kills == 1 and worker.process.kills == 1
    assert outcome["host"] == {
        "pid": 5,
        "created_at": 90.0,
        "argv": host.cmdline(),
        "cwd": "C:\\hermes",
    }
    assert scanner.terminate_desktop_plugin_service(root, pid=20, created_at=101.0) is True


def test_terminate_desktop_plugin_service_counts_a_member_that_already_exited(
    monkeypatch, tmp_path: Path
) -> None:
    root = tmp_path / "install"
    venv = root / "venv"
    journal: list[str] = []
    host, wrapper, worker = _plugin_unit(journal, worker_gone=True)
    wrapper_record = {
        "pid": 10,
        "created_at": 100.0,
        "owner": "desktop",
        "role": "desktop_plugin_wrapper",
        "actionable": True,
        "action": "terminate_desktop_plugin_service",
    }
    monkeypatch.setattr(scanner, "_validated_root", lambda _root: (root, venv))
    monkeypatch.setattr(
        scanner,
        "_desktop_plugin_service_unit",
        lambda _root, pid: (wrapper_record, host, wrapper, [worker]),
    )

    outcome = scanner.terminate_desktop_plugin_service_unit(root, pid=10, created_at=100.0)

    assert outcome["terminated"] is True
    assert journal == ["host", "wrapper", "worker"]
    assert worker.process.kills == 0


def test_terminate_desktop_plugin_service_refuses_a_changed_or_unproven_unit(
    monkeypatch, tmp_path: Path
) -> None:
    root = tmp_path / "install"
    venv = root / "venv"
    journal: list[str] = []
    host, wrapper, worker = _plugin_unit(journal)
    record = {
        "pid": 10,
        "created_at": 100.0,
        "owner": "desktop",
        "role": "desktop_plugin_wrapper",
        "actionable": True,
        "action": "terminate_desktop_plugin_service",
    }
    monkeypatch.setattr(scanner, "_validated_root", lambda _root: (root, venv))
    monkeypatch.setattr(
        scanner,
        "_desktop_plugin_service_unit",
        lambda _root, pid: (record, host, wrapper, [worker]),
    )

    # create-time drift beyond tolerance = a different process generation
    assert scanner.terminate_desktop_plugin_service_unit(root, pid=10, created_at=100.5) == {
        "terminated": False,
        "host": None,
    }
    monkeypatch.setattr(scanner, "_desktop_plugin_service_unit", lambda _root, pid: None)
    assert scanner.terminate_desktop_plugin_service(root, pid=10, created_at=100.0) is False
    assert journal == []


def test_desktop_plugin_service_unit_resolves_wrapper_host_and_workers_from_a_worker(
    monkeypatch, tmp_path: Path
) -> None:
    root = tmp_path / "install"
    venv = root / "venv"
    service = root.parent / "desktop-plugins" / "tracker" / "service.py"
    host = _FakeProcess()
    wrapper_children: list = []
    wrapper_process = _FakeProcess()
    wrapper_process.children = lambda recursive=False: list(wrapper_children)
    wrapper = _snapshot(
        pid=10,
        ppid=5,
        exe=venv / "Scripts" / "python.exe",
        argv=(str(venv / "Scripts" / "python.exe"), str(service)),
        created_at=100.0,
        process=wrapper_process,
    )
    worker = _snapshot(
        pid=20,
        ppid=10,
        exe=root / ".hermes-runtime" / "python" / "generation" / "python.exe",
        argv=(str(root / ".hermes-runtime" / "python" / "generation" / "python.exe"), str(service)),
        created_at=101.0,
    )
    stray = _snapshot(
        pid=21,
        ppid=10,
        exe=root / ".hermes-runtime" / "python" / "generation" / "python.exe",
        argv=(str(root / ".hermes-runtime" / "python" / "generation" / "python.exe"), "-c", "pass"),
        created_at=102.0,
    )
    wrapper_children.extend([types.SimpleNamespace(pid=20), types.SimpleNamespace(pid=21)])
    worker_record = {
        "pid": 20,
        "created_at": 101.0,
        "owner": "desktop",
        "role": "desktop_plugin_worker",
        "actionable": True,
        "action": "terminate_desktop_plugin_service",
        "wrapper_pid": 10,
    }
    monkeypatch.setattr(
        scanner,
        "_live_desktop_plugin_service_process",
        lambda _root, pid: (worker.process, worker_record, host) if int(pid) == 20 else None,
    )
    monkeypatch.setattr(
        scanner,
        "_snapshot_for_pid",
        lambda pid, **_kwargs: {10: wrapper, 20: worker, 21: stray}[int(pid)],
    )

    unit = scanner._desktop_plugin_service_unit(root, 20)

    assert unit is not None
    record, unit_host, unit_wrapper, workers = unit
    assert record is worker_record
    assert unit_host is host
    assert unit_wrapper is wrapper
    assert [member.pid for member in workers] == [20]


def test_main_reports_the_stopped_supervisor_for_a_plugin_service_stop(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    root = tmp_path / "install"
    venv = root / "venv"
    host = {"pid": 5, "created_at": 90.0, "argv": ["wscript.exe", "host.vbs"], "cwd": None}
    monkeypatch.setattr(scanner, "_validated_root", lambda _root: (root, venv))
    monkeypatch.setattr(
        scanner,
        "terminate_desktop_plugin_service_unit",
        lambda _root, *, pid, created_at: {"terminated": True, "host": host},
    )

    with pytest.raises(SystemExit) as exit_info:
        scanner.main(
            ["--root", str(root), "--terminate-desktop-plugin-service", "20", "--created-at", "101.0"]
        )

    assert exit_info.value.code == 0
    document = json.loads(capsys.readouterr().out.strip())
    assert document["mode"] == "terminate_desktop_plugin_service"
    assert document["terminated"] is True
    assert document["host"] == host

    monkeypatch.setattr(scanner, "terminate_mcp_bridge", lambda _root, *, pid, created_at: True)
    with pytest.raises(SystemExit):
        scanner.main(["--root", str(root), "--terminate-mcp-bridge", "20", "--created-at", "101.0"])
    assert "host" not in json.loads(capsys.readouterr().out.strip())

def test_main_terminate_safe_preview_requires_root_and_validates_it(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """#104687 H10: the safe-preview stop used to dispatch before argument parsing.

    ``python scan-venv-blockers.py --terminate-safe <pid> <t>`` reached
    termination with no ``--root``, no root validation, and no containment.
    """
    calls: list[tuple] = []
    monkeypatch.setattr(
        scanner,
        "_terminate_safe_preview",
        lambda pid, created_at, venv_dir: calls.append((pid, created_at, venv_dir)) or (True, None),
    )

    # No --root at all: argparse refuses before anything is terminated.
    with pytest.raises(SystemExit) as exit_info:
        scanner.main(["--terminate-safe-preview", "20", "--created-at", "101.0"])

    assert exit_info.value.code == 1
    assert json.loads(capsys.readouterr().out)["reason"] == "invalid_arguments"
    assert calls == []

    # A root that does not validate is a fail-closed probe failure.
    with pytest.raises(SystemExit) as exit_info:
        scanner.main(
            [
                "--root",
                str(tmp_path / "not-an-install"),
                "--terminate-safe-preview",
                "20",
                "--created-at",
                "101.0",
            ]
        )

    assert exit_info.value.code == 1
    assert json.loads(capsys.readouterr().out)["reason"] == "probe_failed"
    assert calls == []


def test_main_terminate_safe_preview_passes_the_validated_venv(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    root = tmp_path / "install"
    venv = root / "venv"
    calls: list[tuple] = []
    monkeypatch.setattr(scanner, "_validated_root", lambda _root: (root, venv))
    monkeypatch.setattr(
        scanner,
        "_terminate_safe_preview",
        lambda pid, created_at, venv_dir: calls.append((pid, created_at, venv_dir)) or (True, None),
    )

    with pytest.raises(SystemExit) as exit_info:
        scanner.main(
            ["--root", str(root), "--terminate-safe-preview", "20", "--created-at", "101.0"]
        )

    assert exit_info.value.code == 0
    document = json.loads(capsys.readouterr().out.strip())
    assert document["mode"] == "terminate_safe_preview"
    assert document["terminated"] is True
    assert document["pid"] == 20
    assert calls == [(20, 101.0, venv)]


def test_main_refuses_more_than_one_terminate_flag(monkeypatch, tmp_path: Path, capsys) -> None:
    root = tmp_path / "install"
    monkeypatch.setattr(scanner, "_validated_root", lambda _root: (root, root / "venv"))
    stopped: list[int] = []
    monkeypatch.setattr(
        scanner,
        "_terminate_safe_preview",
        lambda pid, created_at, venv_dir: stopped.append(pid) or (True, None),
    )

    with pytest.raises(SystemExit) as exit_info:
        scanner.main(
            [
                "--root",
                str(root),
                "--terminate-safe-preview",
                "20",
                "--terminate-venv-holder",
                "21",
                "--created-at",
                "101.0",
            ]
        )

    assert exit_info.value.code == 1
    assert json.loads(capsys.readouterr().out)["reason"] == "invalid_arguments"
    assert stopped == []


def test_scanner_has_no_pre_parser_terminate_dispatch() -> None:
    """The ``__main__`` guard must not act on raw ``sys.argv`` before ``main()``."""
    for path in (
        Path(scanner.__file__),
        Path(scanner.__file__).resolve().parents[1]
        / "apps"
        / "desktop"
        / "resources"
        / "update-scanner"
        / "scan-venv-blockers.py",
    ):
        text = path.read_text(encoding="utf-8")
        assert "_terminate_safe_main" not in text, path
        assert 'sys.argv[1] == "--terminate-safe"' not in text, path


def test_terminate_venv_holder_stops_any_fresh_target_scan_match(
    monkeypatch, tmp_path: Path
) -> None:

    root = tmp_path / "install"
    venv = root / "venv"
    process = _FakeProcess()
    snapshot = _snapshot(
        pid=20,
        ppid=10,
        exe=venv / "Scripts" / "python.exe",
        argv=(str(venv / "Scripts" / "python.exe"), "user-script.py"),
        created_at=101.0,
        process=process,
    )
    monkeypatch.setattr(scanner, "_validated_root", lambda _root: (root, venv))
    monkeypatch.setattr(
        scanner,
        "_detect_target_venv_holders",
        lambda root, *, strict: [(20, "python.exe", "python.exe user-script.py")],
    )
    monkeypatch.setattr(scanner, "_snapshot_for_pid", lambda _pid: snapshot)

    assert scanner.terminate_venv_holder(root, pid=20, created_at=101.0)
    assert process.kills == 1


def test_terminate_venv_holder_refuses_a_recycled_or_no_longer_scanned_pid(
    monkeypatch, tmp_path: Path
) -> None:

    root = tmp_path / "install"
    venv = root / "venv"
    process = _FakeProcess()
    snapshot = _snapshot(
        pid=20,
        ppid=10,
        exe=venv / "Scripts" / "python.exe",
        argv=(str(venv / "Scripts" / "python.exe"), "user-script.py"),
        created_at=102.0,
        process=process,
    )
    monkeypatch.setattr(scanner, "_validated_root", lambda _root: (root, venv))
    monkeypatch.setattr(
        scanner,
        "_detect_target_venv_holders",
        lambda root, *, strict: [(20, "python.exe", "python.exe user-script.py")],
    )
    monkeypatch.setattr(scanner, "_snapshot_for_pid", lambda _pid: snapshot)

    assert not scanner.terminate_venv_holder(root, pid=20, created_at=101.0)
    assert process.kills == 0


@pytest.mark.windows_only
def test_native_scanner_cli_excludes_its_exact_venv_redirector() -> None:
    """Match the standalone subprocess that Desktop launches on Windows."""
    root = Path(scanner.__file__).resolve().parents[1]
    target_root, venv = scanner._validated_root(root)
    python = venv / "Scripts" / "python.exe"

    process = subprocess.Popen(
        [
            str(python),
            "-m",
            "hermes_cli._scan_venv_blockers",
            "--root",
            str(target_root),
        ],
        cwd=target_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    stdout, stderr = process.communicate(timeout=30)

    assert process.returncode == 0, stderr
    payload = json.loads(stdout)
    records = [
        *payload["processes"],
        *payload["mcp_bridges"],
        *payload["pausable_gateway_processes"],
    ]
    assert all(record["pid"] != process.pid for record in records)


@pytest.mark.windows_only
def test_native_scanner_preserves_identity_redacts_and_refuses_stale_identity(
    monkeypatch, tmp_path: Path
) -> None:
    """Exercise the Windows psutil boundary without touching a live runtime."""
    import psutil

    root = Path(scanner.__file__).resolve().parents[1]
    target_root, venv = scanner._validated_root(root)
    module = "agent.transports.hermes_tools_mcp_server"
    synthetic = tmp_path / "synthetic"
    transport = synthetic / "agent" / "transports"
    transport.mkdir(parents=True)
    (synthetic / "agent" / "__init__.py").write_text("", encoding="utf-8")
    (transport / "__init__.py").write_text("", encoding="utf-8")
    (transport / "hermes_tools_mcp_server.py").write_text(
        "import time\ntime.sleep(120)\n",
        encoding="utf-8",
    )

    secret = "native-scanner-secret-9f6f1a"
    hidden = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    children: list[subprocess.Popen[bytes]] = []
    try:
        bridge = subprocess.Popen(
            [sys.executable, "-m", module],
            cwd=synthetic,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=hidden,
        )
        children.append(bridge)
        secret_holder = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import time; time.sleep(120)",
                "--token",
                secret,
            ],
            cwd=synthetic,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=hidden,
        )
        children.append(secret_holder)

        def _redirector_worker_snapshot():
            try:
                descendants = psutil.Process(bridge.pid).children(recursive=True)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return None
            for descendant in descendants:
                try:
                    snapshot = scanner._snapshot_for_pid(descendant.pid)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                if (
                    snapshot is not None
                    and scanner.is_exact_mcp_module_argv(snapshot.argv)
                    and not scanner._within(snapshot.exe, venv)
                ):
                    return snapshot
            return None

        # CPython and uv both commonly use a venv-side redirector process on
        # Windows, but keep the assertion conditional for runtimes that truly
        # execute in-place. Give the redirector child a bounded startup window
        # so a fast first scan cannot mistake a not-yet-spawned child for an
        # in-place runtime.
        redirector_deadline = time.monotonic() + 2.0
        redirector_worker = None
        while time.monotonic() < redirector_deadline:
            redirector_worker = _redirector_worker_snapshot()
            if redirector_worker is not None:
                break
            assert bridge.poll() is None
            time.sleep(0.02)

        # Owner attribution has separate ancestry tests above. Keep this test
        # focused on the real Windows pid/exe/argv/create-time reads.
        monkeypatch.setattr(
            scanner,
            "_owner_from_ancestry",
            lambda _snapshot, **_kwargs: "codex",
        )

        deadline = time.monotonic() + 10.0
        payload = None
        bridge_record = None
        worker_record = None
        holder_record = None
        while time.monotonic() < deadline:
            assert bridge.poll() is None
            assert secret_holder.poll() is None
            payload = scanner.scan_venv_blockers(target_root)
            bridge_record = next(
                (
                    entry
                    for entry in payload["mcp_bridges"]
                    if entry["pid"] == bridge.pid
                ),
                None,
            )
            worker_record = next(
                (
                    entry
                    for entry in payload["mcp_bridges"]
                    if redirector_worker is not None
                    and entry["pid"] == redirector_worker.pid
                ),
                None,
            )
            holder_record = next(
                (
                    entry
                    for entry in payload["processes"]
                    if entry["pid"] == secret_holder.pid
                ),
                None,
            )
            if (
                bridge_record is not None
                and holder_record is not None
                and (redirector_worker is None or worker_record is not None)
            ):
                break
            time.sleep(0.05)

        assert payload is not None
        assert bridge_record is not None
        assert holder_record is not None

        bridge_snapshot = scanner._snapshot_for_pid(bridge.pid)
        holder_snapshot = scanner._snapshot_for_pid(secret_holder.pid)
        assert bridge_snapshot is not None
        assert holder_snapshot is not None
        assert bridge_snapshot.pid == bridge.pid
        assert scanner.is_exact_mcp_module_argv(bridge_snapshot.argv)
        assert scanner._within(bridge_snapshot.exe, venv)
        assert bridge_record["role"] == "mcp_bridge_wrapper"
        assert bridge_record["owner"] == "codex"
        assert bridge_record["actionable"] is True
        assert bridge_record["created_at"] == pytest.approx(
            bridge_snapshot.created_at, abs=0.01
        )

        if redirector_worker is not None:
            assert worker_record is not None
            assert worker_record["role"] == "mcp_bridge_worker"
            assert worker_record["wrapper_pid"] == bridge.pid
            assert worker_record["owner"] == "codex"
            assert worker_record["actionable"] is True
            assert worker_record["created_at"] == pytest.approx(
                redirector_worker.created_at, abs=0.01
            )
            bridge_order = next(
                index
                for index, entry in enumerate(payload["mcp_bridges"])
                if entry["pid"] == bridge.pid
            )
            worker_order = next(
                index
                for index, entry in enumerate(payload["mcp_bridges"])
                if entry["pid"] == redirector_worker.pid
            )
            assert worker_order < bridge_order

            # A stale identity cannot terminate either member of the pair.
            assert not scanner.terminate_mcp_bridge(
                target_root,
                pid=redirector_worker.pid,
                created_at=redirector_worker.created_at + 10.0,
            )
            assert psutil.pid_exists(redirector_worker.pid)
            assert bridge.poll() is None

        raw_holder_cmdline = " ".join(holder_snapshot.argv)
        assert secret in raw_holder_cmdline, "the OS snapshot must contain the fixture secret"
        redacted = scanner._redact_sensitive_cmdline(raw_holder_cmdline)
        assert secret not in redacted
        assert "--token <redacted>" in redacted
        assert secret not in json.dumps(payload)
        assert holder_record["created_at"] == pytest.approx(
            holder_snapshot.created_at, abs=0.01
        )

        stale_created_at = float(bridge_record["created_at"]) + 10.0
        assert not scanner.terminate_mcp_bridge(
            target_root,
            pid=bridge.pid,
            created_at=stale_created_at,
        )
        assert bridge.poll() is None, "a stale identity must not kill the live process"
        live_snapshot = scanner._snapshot_for_pid(bridge.pid)
        assert live_snapshot is not None
        assert live_snapshot.created_at == pytest.approx(
            bridge_snapshot.created_at, abs=0.01
        )
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)


def test_live_plugin_service_proof_skips_inaccessible_ancestor_above_host(
    monkeypatch, tmp_path: Path
) -> None:
    """2026-09-06 18:06Z: after an update the Desktop relaunches the plugin
    host itself, so the wrapper's ancestry reads wscript -> Hermes.exe ->
    WmiPrvSE.exe (session 0, OpenProcess denied). The terminate re-proof
    aborted on that ancestor, every unit stop reported ``not stopped`` and
    the update refused with the force-release dialog. Ancestors above the
    supervisor are irrelevant to the unit's identity: skip what cannot be
    opened, as the scan path already does, and keep the host proof separate.
    """
    root = tmp_path / "hermes-agent"
    venv = root / "venv"
    script = tmp_path / "desktop-plugins" / "llm-usage-tracker" / "service.py"
    host = types.SimpleNamespace(pid=10)  # wscript.exe service-host.vbs
    desktop = types.SimpleNamespace(pid=9)  # Hermes.exe that relaunched it
    protected = types.SimpleNamespace(pid=8)  # WmiPrvSE.exe, session 0

    wrapper = _snapshot(
        pid=20,
        ppid=10,
        exe=venv / "Scripts" / "python.exe",
        argv=(str(venv / "Scripts" / "python.exe"), str(script)),
        created_at=100.0,
        process=_FakeProcess(parents=[host, desktop, protected]),
    )
    managed_python = root / ".hermes-runtime" / "python" / "generation-1" / "python.exe"
    worker = _snapshot(
        pid=30,
        ppid=20,
        exe=managed_python,
        argv=(str(managed_python), str(script)),
        created_at=101.0,
        process=_FakeProcess(
            parents=[types.SimpleNamespace(pid=20), host, desktop, protected]
        ),
    )

    class AccessDenied(Exception):
        pass

    def snapshot_for_pid(pid, **_kwargs):
        pid = int(pid)
        if pid == 20:
            return wrapper
        if pid == 30:
            return worker
        if pid == 8:
            raise AccessDenied("[WinError 5] Access is denied: '(originated from OpenProcess)'")
        return None

    monkeypatch.setattr(scanner, "_snapshot_for_pid", snapshot_for_pid)
    monkeypatch.setattr(scanner, "_desktop_plugin_service_host", lambda _wrapper, _script: host)

    for pid, role in ((20, "desktop_plugin_wrapper"), (30, "desktop_plugin_worker")):
        live = scanner._live_desktop_plugin_service_process(root, pid)
        assert live is not None, f"pid {pid}: an unreadable ancestor above the host voided the proof"
        _process, record, proven_host = live
        assert record["role"] == role
        assert record["action"] == "terminate_desktop_plugin_service"
        assert proven_host is host


def test_live_plugin_service_proof_still_refuses_generation_change(
    monkeypatch, tmp_path: Path
) -> None:
    """The relaxation above must not swallow a real identity failure."""
    root = tmp_path / "hermes-agent"
    venv = root / "venv"
    script = tmp_path / "desktop-plugins" / "llm-usage-tracker" / "service.py"
    host = types.SimpleNamespace(pid=10)
    wrapper = _snapshot(
        pid=20,
        ppid=10,
        exe=venv / "Scripts" / "python.exe",
        argv=(str(venv / "Scripts" / "python.exe"), str(script)),
        created_at=100.0,
        process=_FakeProcess(parents=[host]),
    )

    def snapshot_for_pid(pid, **_kwargs):
        if int(pid) == 20:
            return wrapper
        raise scanner._ProcessGenerationChanged("pid reused during identity refresh")

    monkeypatch.setattr(scanner, "_snapshot_for_pid", snapshot_for_pid)
    monkeypatch.setattr(scanner, "_desktop_plugin_service_host", lambda _wrapper, _script: host)

    assert scanner._live_desktop_plugin_service_process(root, 20) is None


@pytest.mark.parametrize("module_kind", ["in-tree", "carrier"])
@pytest.mark.parametrize("name", ["Code.exe", "powershell.exe", "cmd.exe", "rg.exe"])
def test_argv_mention_of_venv_does_not_admit_editor_or_shell(monkeypatch, tmp_path, module_kind, name):
    module = scanner
    if module_kind == "carrier":
        import importlib.util
        carrier = Path(__file__).resolve().parents[2] / "apps/desktop/resources/update-scanner/scan-venv-blockers.py"
        spec = importlib.util.spec_from_file_location("test_carrier_name_gate", carrier)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
    external_exe = str(tmp_path.parent / "external" / name)
    mentioned = str(tmp_path / "venv" / "pyvenv.cfg")
    proc = _detector_proc(555, external_exe, name, [external_exe, mentioned, "hermes_cli.main"], cwd=str(tmp_path))
    monkeypatch.setitem(sys.modules, "psutil", types.SimpleNamespace(process_iter=lambda attrs: iter([proc])))
    assert module._detect_target_venv_holders(tmp_path, strict=True) == []


@pytest.fixture(params=["in-tree", "carrier"])
def scanner_copy(request, monkeypatch):
    if request.param == "in-tree":
        return scanner
    import importlib.util
    carrier = Path(__file__).resolve().parents[2] / "apps/desktop/resources/update-scanner/scan-venv-blockers.py"
    spec = importlib.util.spec_from_file_location("test_scanner_behavioral_parity", carrier)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("unrelated_invocation", ["plain", "venv-operand", "module-operand", "same-tail"])
def test_shared_managed_python_requires_target_invocation_or_wrapper(
    scanner_copy, monkeypatch, tmp_path, unrelated_invocation,
):
    root = tmp_path / "install"
    python = root / "venv" / "Scripts" / "python.exe"
    managed = root / ".hermes-runtime" / "python" / "generation" / "python.exe"
    tail = [str(tmp_path / "service.py")]
    unrelated_tail = {
        "plain": ["-c", "import time; time.sleep(10)"],
        "venv-operand": ["-c", "import time; time.sleep(10)", str(root / "venv" / "pyvenv.cfg")],
        "module-operand": ["-c", "import time; time.sleep(10)", "-m", "hermes_cli.main", "serve"],
        "same-tail": tail,
    }[unrelated_invocation]
    rows = [
        _detector_proc(10, str(python), "python.exe", [str(python), *tail], ppid=1),
        _detector_proc(20, str(managed), "python.exe", [str(managed), *tail], ppid=10),
        _detector_proc(30, str(managed), "python.exe", [str(managed), *unrelated_tail], cwd=str(root), ppid=1),
        # A base worker retaining the exact target-venv argv[0] is also a holder.
        _detector_proc(40, str(managed), "python.exe", [str(python), "-c", "pass"], ppid=1),
    ]
    monkeypatch.setitem(sys.modules, "psutil", types.SimpleNamespace(process_iter=lambda attrs: iter(rows)))
    matches = scanner_copy._detect_target_venv_holders(root, strict=True)
    assert {pid for pid, _name, _cmd in matches} == {10, 20, 40}


@pytest.mark.parametrize("host_proven", [True, False])
def test_shared_runtime_plugin_unit_keeps_its_separate_host_proof(
    scanner_copy, monkeypatch, tmp_path, host_proven,
):
    root = tmp_path / "install"
    venv = root / "venv"
    python = venv / "Scripts" / "python.exe"
    managed = root / ".hermes-runtime" / "python" / "generation" / "python.exe"
    script = tmp_path / "desktop-plugins" / "tracker" / "service.py"
    rows = [
        _detector_proc(10, str(python), "python.exe", [str(python), str(script)], ppid=1),
        _detector_proc(20, str(managed), "python.exe", [str(managed), str(script)], ppid=10),
        _detector_proc(30, str(managed), "python.exe", [str(managed), str(script)], ppid=1),
    ]
    snapshots = {
        row.info["pid"]: scanner_copy._ProcessSnapshot(
            row.info["pid"], row.info["ppid"], row.info["name"], row.info["exe"],
            tuple(row.info["cmdline"]), 100 + row.info["pid"], row,
        )
        for row in rows
    }
    monkeypatch.setitem(sys.modules, "psutil", types.SimpleNamespace(process_iter=lambda attrs: iter(rows)))
    monkeypatch.setattr(scanner_copy, "_validated_root", lambda supplied: (root, venv))
    monkeypatch.setattr(scanner_copy, "_snapshot_for_pid", lambda pid, **kwargs: snapshots[pid])
    monkeypatch.setattr(scanner_copy, "_desktop_plugin_service_host", lambda wrapper, candidate: object() if host_proven else None)
    result = scanner_copy.scan_venv_blockers(root)
    assert result["blocked"] is True
    if host_proven:
        assert {(row["pid"], row["role"]) for row in result["desktop_plugin_services"]} == {
            (10, "desktop_plugin_wrapper"), (20, "desktop_plugin_worker"),
        }
        assert result["processes"] == []
    else:
        assert result["desktop_plugin_services"] == []
        assert {row["pid"] for row in result["processes"]} == {10, 20}
        assert all(row["action"] == "refuse" for row in result["processes"])
