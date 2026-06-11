import json

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

    assert cd.restart_notice() == (
        "\n\nCrash cause: GPU error (Metal/MLX) (Python, SIGABRT)"
        "\nFaulting frame: libmlx.dylib: mlx_abort"
    )


def test_recent_crashes_swallows_reader_failure(monkeypatch):
    monkeypatch.setattr(cd.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        cd,
        "_macos_recent_crashes",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    assert cd.recent_crashes() == []


def test_parse_macos_ips_extracts_triggered_thread(tmp_path):
    report = tmp_path / "Python-2026-06-05.ips"
    report.write_text(
        "header\n"
        + json.dumps(
            {
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
    assert parsed["process"] == "Python"
    assert parsed["signal"] == "SIGABRT"
    assert parsed["cause"] == "GPU error (Metal/MLX)"
    assert parsed["backtrace"][:2] == [
        "libmlx.dylib: mlx_abort",
        "libsystem_kernel.dylib: __pthread_kill",
    ]


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

    def _fake_run(cmd):
        commands.append(cmd)
        return "2026-06-05T09:08:00Z host kernel: python[123]: segfault at 0 ip 0 sp 0 error 4"

    monkeypatch.setattr(
        cd.shutil,
        "which",
        lambda name: None if name == "coredumpctl" else f"/usr/bin/{name}",
    )
    monkeypatch.setattr(cd, "_run", _fake_run)

    records = cd._linux_recent_crashes("python", 24, 5)

    assert len(records) == 1
    assert records[0]["signal"] == "SIGSEGV"
    assert commands and commands[0][0] == "journalctl"
    assert "-k" in commands[0]


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

    assert cd.restart_notice() == "\n\nCrash cause: SIGTRAP (Python)"
