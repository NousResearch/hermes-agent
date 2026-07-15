import json
from datetime import datetime, timedelta, timezone

import pytest

import gateway.crash_diagnostics as cd


def test_restart_notice_formats_crash_cause_and_faulting_frame(monkeypatch):
    monkeypatch.setattr(
        cd,
        "recent_crashes",
        lambda **_kwargs: [
            {
                "process": "Python",
                "signal": "SIGABRT",
                "cause": "GPU error (Metal/MLX)",
                "backtrace": ["libmlx.dylib: mlx_abort"],
            }
        ],
    )

    assert cd.restart_notice(
        expected_pid=4242,
        since=datetime(2026, 7, 15, tzinfo=timezone.utc),
    ) == (
        "\n\nCrash cause: GPU error (Metal/MLX) (Python, SIGABRT)"
        "\nFaulting frame: libmlx.dylib: mlx_abort"
    )


def test_restart_notice_forwards_exact_crash_identity(monkeypatch):
    since = datetime(2026, 7, 15, 10, 30, 30, tzinfo=timezone.utc)
    captured = {}

    def _fake_recent_crashes(**kwargs):
        captured.update(kwargs)
        return [
            {
                "process": "python3.12",
                "signal": "SIGSEGV",
                "cause": "segmentation fault",
                "backtrace": [],
            }
        ]

    monkeypatch.setattr(cd, "recent_crashes", _fake_recent_crashes)

    notice = cd.restart_notice(expected_pid=4242, since=since)

    assert "Crash cause: segmentation fault" in notice
    assert captured == {
        "name_filter": "python",
        "expected_pid": 4242,
        "since": since,
        "limit": 1,
    }


def test_restart_notice_skips_lookup_without_complete_identity(monkeypatch):
    lookups = []

    def _unexpected_lookup(**kwargs):
        lookups.append(kwargs)
        return []

    monkeypatch.setattr(cd, "recent_crashes", _unexpected_lookup)

    assert cd.restart_notice() == ""
    assert cd.restart_notice(expected_pid=4242) == ""
    assert cd.restart_notice(since=datetime.now(timezone.utc)) == ""
    assert cd.restart_notice(expected_pid=4242, since=datetime(2026, 7, 15)) == ""
    assert lookups == []


def test_recent_crashes_swallows_reader_failure(monkeypatch):
    since = datetime(2026, 7, 15, tzinfo=timezone.utc)
    monkeypatch.setattr(cd.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        cd,
        "_macos_recent_crashes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    assert cd.recent_crashes(expected_pid=4242, since=since) == []


def test_recent_crashes_passes_exact_identity_to_platform_reader(monkeypatch):
    since = datetime(2026, 7, 15, 10, 30, 30, tzinfo=timezone.utc)
    captured = {}

    def _fake_linux_reader(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return []

    monkeypatch.setattr(cd.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cd, "_linux_recent_crashes", _fake_linux_reader)

    assert cd.recent_crashes(expected_pid=4242, since=since, limit=3) == []
    assert captured == {
        "args": ("python",),
        "kwargs": {"expected_pid": 4242, "since": since, "limit": 3},
    }


def test_parse_macos_ips_extracts_triggered_thread(tmp_path):
    report = tmp_path / "Python-2026-06-05.ips"
    report.write_text(
        "header\n"
        + json.dumps(
            {
                "pid": 4242,
                "procName": "Python",
                "captureTime": "2026-06-05 09:08:00.000 +0800",
                "exception": {"signal": "SIGABRT"},
                "usedImages": [{"name": "libsystem_kernel.dylib"}, {"name": "libmlx.dylib"}],
                "threads": [
                    {"frames": [{"imageIndex": 0, "symbol": "idle"}]},
                    {
                        "triggered": True,
                        "frames": [
                            {"imageIndex": 1, "symbol": "mlx_abort"},
                            {"imageIndex": 0, "symbol": "__pthread_kill"},
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    parsed = cd._parse_macos_ips(report)

    assert parsed is not None
    assert parsed["_pid"] == 4242
    assert parsed["process"] == "Python"
    assert parsed["signal"] == "SIGABRT"
    assert parsed["cause"] == "GPU error (Metal/MLX)"
    assert parsed["backtrace"][:2] == [
        "libmlx.dylib: mlx_abort",
        "libsystem_kernel.dylib: __pthread_kill",
    ]


def test_macos_reader_requires_matching_pid_after_exact_status_time(tmp_path, monkeypatch):
    since = datetime(2026, 7, 15, 10, 30, 30, tzinfo=timezone.utc)
    reports = tmp_path / "Library" / "Logs" / "DiagnosticReports"
    reports.mkdir(parents=True)

    def _write_report(name, *, pid, capture_time, signal):
        (reports / name).write_text(
            json.dumps(
                {
                    "pid": pid,
                    "procName": "Python",
                    "captureTime": capture_time,
                    "exception": {"signal": signal},
                    "threads": [],
                }
            ),
            encoding="utf-8",
        )

    _write_report(
        "Python-unrelated.ips",
        pid=9002,
        capture_time="2026-07-15 18:45:00.000 +0800",
        signal="SIGABRT",
    )
    _write_report(
        "Python-too-early.ips",
        pid=4242,
        capture_time="2026-07-15 18:30:29.000 +0800",
        signal="SIGABRT",
    )
    _write_report(
        "Python-matching.ips",
        pid=4242,
        capture_time="2026-07-15 18:31:00.000 +0800",
        signal="SIGSEGV",
    )
    monkeypatch.setattr(cd.Path, "home", classmethod(lambda _cls: tmp_path))
    original_glob = cd.Path.glob
    monkeypatch.setattr(
        cd.Path,
        "glob",
        lambda directory, pattern: original_glob(directory, pattern)
        if directory == reports
        else [],
    )

    records = cd._macos_recent_crashes(
        "python",
        expected_pid=4242,
        since=since,
        limit=5,
    )

    assert len(records) == 1
    assert records[0]["_pid"] == 4242
    assert records[0]["signal"] == "SIGSEGV"


def test_macos_reader_ignores_reports_without_capture_time(tmp_path, monkeypatch):
    reports = tmp_path / "Library" / "Logs" / "DiagnosticReports"
    reports.mkdir(parents=True)
    (reports / "Python-missing-time.ips").write_text(
        json.dumps(
            {
                "pid": 4242,
                "procName": "Python",
                "exception": {"signal": "SIGSEGV"},
                "threads": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cd.Path, "home", classmethod(lambda _cls: tmp_path))
    original_glob = cd.Path.glob
    monkeypatch.setattr(
        cd.Path,
        "glob",
        lambda directory, pattern: original_glob(directory, pattern)
        if directory == reports
        else [],
    )

    records = cd._macos_recent_crashes(
        "python",
        expected_pid=4242,
        since=datetime.now(timezone.utc) - timedelta(minutes=1),
        limit=5,
    )

    assert records == []


def test_linux_journal_parser_filters_to_python_crashes():
    text = "\n".join(
        [
            "2026-06-05T09:00:00Z host kernel: unrelated segfault in node",
            "2026-06-05T09:08:00Z host kernel: python[123]: segfault at 0 ip 0 sp 0 error 4",
        ]
    )

    records = cd._parse_journal_crashes(text, "python", 5)

    assert len(records) == 1
    assert records[0]["process"] == "python"
    assert records[0]["signal"] == "SIGSEGV"
    assert records[0]["cause"] == "segmentation fault"


def test_linux_journal_parser_keeps_oom_process_name():
    since = datetime(2026, 6, 5, 9, 7, tzinfo=timezone.utc)
    text = (
        "2026-06-05T09:08:00.000000+0000 host kernel: "
        "Out of memory: Killed process 4242 (python3.12) total-vm:1234kB"
    )

    records = cd._parse_journal_crashes(
        text,
        "python",
        5,
        expected_pid=4242,
        since=since,
    )

    assert len(records) == 1
    assert records[0]["_pid"] == 4242
    assert records[0]["process"] == "python3.12"
    assert records[0]["cause"] == "out of memory"


def test_parse_coredumpctl_info_backtrace_extracts_stack_frames():
    text = """
           PID: 1234 (python)
        Signal: 11 (SEGV)
   Stack trace of thread 1234:
            #0  0x000000 libnative.so crash_here
            #1  0x000001 python PyEval_EvalFrame

    Metadata after stack
    """

    assert cd._parse_coredumpctl_info_backtrace(text) == [
        "#0 0x000000 libnative.so crash_here",
        "#1 0x000001 python PyEval_EvalFrame",
    ]


def test_run_returns_empty_string_on_timeout(monkeypatch):
    def _raise_timeout(*_args, **_kwargs):
        raise TimeoutError("slow")

    monkeypatch.setattr(cd.subprocess, "run", _raise_timeout)

    assert cd._run(["fake"]) == ""


def test_parse_coredumpctl_json_array_with_lowercase_fields():
    # Real ``coredumpctl --json=short`` shape: one compact JSON array with
    # lowercase keys and the signal as a number (systemd format-table output).
    text = json.dumps(
        [
            {
                "time": 1765432100000000,
                "pid": 4242,
                "uid": 1000,
                "gid": 1000,
                "sig": 11,
                "corefile": "present",
                "exe": "/usr/bin/python3.12",
                "size": 123456,
            },
            {
                "time": 1765432000000000,
                "pid": 4100,
                "sig": 6,
                "corefile": "present",
                "exe": "/usr/bin/node",
                "size": 2222,
            },
        ]
    )

    records = cd._parse_coredumpctl_json(text, "python", 5)

    assert len(records) == 1
    assert records[0]["process"] == "python3.12"
    assert records[0]["signal"] == "SIGSEGV"
    assert records[0]["cause"] == "segmentation fault"
    assert records[0]["_pid"] == 4242


def test_coredump_rows_require_matching_pid_after_exact_status_time():
    since = datetime(2026, 7, 15, 10, 30, 30, tzinfo=timezone.utc)

    def _micros(value):
        return int(value.timestamp() * 1_000_000)

    text = json.dumps(
        [
            {
                "time": _micros(since + timedelta(minutes=20)),
                "pid": 9002,
                "sig": 6,
                "exe": "/usr/bin/python3.12",
            },
            {
                "time": _micros(since - timedelta(microseconds=1)),
                "pid": 4242,
                "sig": 6,
                "exe": "/usr/bin/python3.12",
            },
            {
                "time": _micros(since + timedelta(minutes=10)),
                "sig": 6,
                "exe": "/usr/bin/python3.12",
            },
            {
                "time": _micros(since + timedelta(minutes=1)),
                "pid": 4242,
                "sig": 11,
                "exe": "/usr/bin/python3.12",
            },
        ]
    )

    records = cd._parse_coredumpctl_json(
        text,
        "python",
        5,
        expected_pid=4242,
        since=since,
    )

    assert len(records) == 1
    assert records[0]["_pid"] == 4242
    assert records[0]["signal"] == "SIGSEGV"


def test_parse_coredumpctl_json_keeps_journal_field_fallback():
    text = "\n".join(
        [
            json.dumps(
                {
                    "COREDUMP_COMM": "python3",
                    "COREDUMP_SIGNAL_NAME": "SIGABRT",
                    "COREDUMP_PID": "77",
                }
            ),
            "not json",
        ]
    )

    records = cd._parse_coredumpctl_json(text, "python", 5)

    assert len(records) == 1
    assert records[0]["process"] == "python3"
    assert records[0]["signal"] == "SIGABRT"


def test_parse_coredumpctl_json_garbage_returns_empty():
    assert cd._parse_coredumpctl_json("No coredumps found.", "python", 5) == []
    assert cd._parse_coredumpctl_json("", "python", 5) == []


def test_signal_display_name_maps_numbers():
    assert cd._signal_display_name(11) == "SIGSEGV"
    assert cd._signal_display_name("6") == "SIGABRT"
    assert cd._signal_display_name("SIGKILL") == "SIGKILL"
    assert cd._signal_display_name(None) == ""


def test_linux_journalctl_fallback_scans_kernel_log(monkeypatch):
    commands = []
    since = datetime(2026, 6, 5, 9, 8, 0, 500_000, tzinfo=timezone.utc)

    def _fake_run(cmd):
        commands.append(cmd)
        return "\n".join(
            [
                "2026-06-05T09:08:00.750000+0000 host kernel: python[999]: segfault at 0 ip 0 sp 0 error 4",
                "2026-06-05T09:08:00.499999+0000 host kernel: python[123]: segfault at 0 ip 0 sp 0 error 4",
                "2026-06-05T09:08:00.750000+0000 host kernel: python[123]: segfault at 0 ip 0 sp 0 error 4",
            ]
        )

    monkeypatch.setattr(
        cd.shutil,
        "which",
        lambda name: None if name == "coredumpctl" else f"/usr/bin/{name}",
    )
    monkeypatch.setattr(cd, "_run", _fake_run)

    records = cd._linux_recent_crashes(
        "python",
        expected_pid=123,
        since=since,
        limit=5,
    )

    assert len(records) == 1
    assert records[0]["_pid"] == 123
    assert records[0]["signal"] == "SIGSEGV"
    assert commands and commands[0][0] == "journalctl"
    assert "-k" in commands[0]
    since_index = commands[0].index("--since") + 1
    assert commands[0][since_index] == "2026-06-05T09:08:00.500000+00:00"
    format_index = commands[0].index("-o") + 1
    assert commands[0][format_index] == "short-iso-precise"


def test_linux_coredumpctl_bounds_candidate_records(monkeypatch):
    commands = []
    since = datetime(2026, 6, 5, 9, 8, tzinfo=timezone.utc)

    def _fake_run(cmd):
        commands.append(cmd)
        if "info" in cmd:
            return ""
        return json.dumps(
            [
                {
                    "time": int((since + timedelta(seconds=1)).timestamp() * 1_000_000),
                    "pid": 123,
                    "sig": 11,
                    "exe": "/usr/bin/python3.12",
                }
            ]
        )

    monkeypatch.setattr(
        cd.shutil,
        "which",
        lambda name: "/usr/bin/coredumpctl" if name == "coredumpctl" else None,
    )
    monkeypatch.setattr(cd, "_run", _fake_run)

    records = cd._linux_recent_crashes(
        "python",
        expected_pid=123,
        since=since,
        limit=1,
    )

    assert len(records) == 1
    assert commands[0][0] == "coredumpctl"
    limit_index = commands[0].index("-n") + 1
    assert commands[0][limit_index] == str(cd._MAX_CANDIDATE_RECORDS)


def test_windows_reader_requires_matching_pid_after_exact_status_time(monkeypatch):
    since = datetime(2026, 7, 15, 10, 30, 30, tzinfo=timezone.utc)
    commands = []

    def _event(pid, when, exception_code):
        return {
            "TimeCreated": when,
            "ProviderName": "Application Error",
            "Message": (
                "Faulting application name: python.exe, version: 3.12.0\n"
                f"Faulting process id: 0x{pid:x}\n"
                f"Exception code: {exception_code}"
            ),
        }

    rows = [
        _event(9002, "2026-07-15T10:45:00Z", "0x80000003"),
        _event(4242, "2026-07-15T10:30:29Z", "0x80000003"),
        _event(
            4242,
            f"/Date({int(datetime(2026, 7, 15, 10, 31, tzinfo=timezone.utc).timestamp() * 1_000)})/",
            "0xc0000005",
        ),
    ]

    monkeypatch.setattr(cd.sys, "platform", "win32")
    monkeypatch.setattr(cd.shutil, "which", lambda _name: "C:/Windows/powershell.exe")

    def _fake_run(cmd):
        commands.append(cmd)
        return json.dumps(rows)

    monkeypatch.setattr(cd, "_run", _fake_run)

    records = cd._windows_recent_crashes(
        "python",
        expected_pid=4242,
        since=since,
        limit=5,
    )

    assert len(records) == 1
    assert records[0]["_pid"] == 4242
    assert records[0]["when"].startswith("/Date(")
    assert "2026-07-15T10:30:30+00:00" in commands[0][-1]
    assert "-MaxEvents 100" in commands[0][-1]


@pytest.mark.parametrize(
    "label",
    ["Faulting process id", "Faulting application process id"],
)
def test_extract_windows_faulting_pid_accepts_common_event_wording(label):
    assert cd._extract_windows_faulting_pid(f"{label}: 0x1092") == 4242


def test_parse_macos_ips_filters_on_header_bug_type(tmp_path):
    body = json.dumps(
        {"procName": "Python", "exception": {"signal": "SIGABRT"}, "threads": []}
    )

    # Hang report (bug_type 288) — same .ips extension, not a crash.
    hang = tmp_path / "Python-hang.ips"
    hang.write_text(json.dumps({"bug_type": "288"}) + "\n" + body, encoding="utf-8")
    assert cd._parse_macos_ips(hang) is None

    # Crash report (bug_type 309) parses normally.
    crash = tmp_path / "Python-crash.ips"
    crash.write_text(json.dumps({"bug_type": "309"}) + "\n" + body, encoding="utf-8")
    parsed = cd._parse_macos_ips(crash)
    assert parsed is not None
    assert parsed["signal"] == "SIGABRT"


def test_restart_notice_does_not_repeat_signal_as_cause(monkeypatch):
    monkeypatch.setattr(
        cd,
        "recent_crashes",
        lambda **_kwargs: [
            {"process": "Python", "signal": "SIGTRAP", "cause": "SIGTRAP", "backtrace": []}
        ],
    )

    assert cd.restart_notice(
        expected_pid=4242,
        since=datetime(2026, 7, 15, tzinfo=timezone.utc),
    ) == "\n\nCrash cause: SIGTRAP (Python)"
