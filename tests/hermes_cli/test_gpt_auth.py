from __future__ import annotations

import io
from collections import deque

from hermes_cli import gpt_auth


class _FakeProcess:
    def __init__(self, output: str, returncode: int = 0):
        self.stdout = io.StringIO(output)
        self.returncode = returncode
        self.killed = False
        self.waited = False

    def poll(self):
        return self.returncode if self.waited else None

    def wait(self, timeout=None):
        self.waited = True
        return self.returncode

    def kill(self):
        self.killed = True
        self.waited = True


class _FakeRunner:
    def __init__(self, status_results, process: _FakeProcess | None = None):
        self.status_results = deque(status_results)
        self.process = process
        self.run_calls = []
        self.popen_calls = []

    def run(self, argv, *, timeout):
        self.run_calls.append(tuple(argv))
        return self.status_results.popleft()

    def popen(self, argv):
        self.popen_calls.append(tuple(argv))
        if self.process is None:
            raise AssertionError("unexpected signin command")
        return self.process


def _status(code: int, text: str):
    return gpt_auth.CommandResult(returncode=code, stdout=text)


def test_already_authenticated_fast_path_does_not_start_device_auth():
    runner = _FakeRunner([_status(0, "Logged in using ChatGPT\n")])
    out = io.StringIO()
    err = io.StringIO()

    rc = gpt_auth.run([], runner=runner, stdout=out, stderr=err)

    assert rc == 0
    assert runner.run_calls == [("codex", "login", "status")]
    assert runner.popen_calls == []
    assert "already authenticated" in out.getvalue()
    assert err.getvalue() == ""


def test_run_uses_sys_argv_when_argv_is_none(monkeypatch):
    monkeypatch.setattr(
        gpt_auth.sys,
        "argv",
        ["gpt-auth", "--status-command", "custom status"],
    )
    runner = _FakeRunner([_status(0, "Logged in using ChatGPT\n")])

    rc = gpt_auth.run(None, runner=runner, stdout=io.StringIO(), stderr=io.StringIO())

    assert rc == 0
    assert runner.run_calls == [("custom", "status")]


def test_device_auth_happy_path_prints_only_code_and_url_then_verifies():
    process = _FakeProcess(
        "\n".join(
            [
                "access_token=secret-should-not-print",
                "Open https://auth.openai.com/codex/device",
                "Enter code ABCD-EFGH",
                "Login successful",
            ]
        )
        + "\n",
        returncode=0,
    )
    runner = _FakeRunner(
        [
            _status(1, "Not logged in\n"),
            _status(0, "Logged in using ChatGPT\n"),
        ],
        process,
    )
    out = io.StringIO()
    err = io.StringIO()

    rc = gpt_auth.run([], runner=runner, input_func=lambda prompt: "", stdout=out, stderr=err)

    assert rc == 0
    assert runner.popen_calls == [("codex", "login", "--device-auth")]
    shown = out.getvalue()
    assert "Device code: ABCD-EFGH" in shown
    assert "Verification URL: https://auth.openai.com/codex/device" in shown
    assert "HITL PAUSE" in shown
    assert "Codex CLI session verified." in shown
    assert "secret-should-not-print" not in shown
    assert err.getvalue() == ""


def test_denied_device_auth_exits_nonzero_with_specific_error():
    process = _FakeProcess(
        "\n".join(
            [
                "Open https://auth.openai.com/codex/device",
                "Enter code WXYZ-1234",
                "error: access_denied",
            ]
        )
        + "\n",
        returncode=1,
    )
    runner = _FakeRunner([_status(1, "Not logged in\n")], process)
    out = io.StringIO()
    err = io.StringIO()

    rc = gpt_auth.run([], runner=runner, input_func=lambda prompt: "", stdout=out, stderr=err)

    assert rc == 2
    assert "Device code: WXYZ-1234" in out.getvalue()
    assert "ERROR auth_denied" in err.getvalue()


def test_expired_device_code_exits_nonzero_with_specific_error():
    process = _FakeProcess(
        "\n".join(
            [
                "Open https://auth.openai.com/codex/device",
                "Enter code LMNO-9876",
                "error: expired_token",
            ]
        )
        + "\n",
        returncode=1,
    )
    runner = _FakeRunner([_status(1, "Not logged in\n")], process)
    out = io.StringIO()
    err = io.StringIO()

    rc = gpt_auth.run([], runner=runner, input_func=lambda prompt: "", stdout=out, stderr=err)

    assert rc == 2
    assert "Device code: LMNO-9876" in out.getvalue()
    assert "ERROR auth_code_expired" in err.getvalue()


def test_device_code_timeout_kills_process_before_partial_state():
    process = _FakeProcess("", returncode=0)
    runner = _FakeRunner([_status(1, "Not logged in\n")], process)
    out = io.StringIO()
    err = io.StringIO()

    rc = gpt_auth.run(
        ["--code-timeout-seconds", "0"],
        runner=runner,
        input_func=lambda prompt: "",
        stdout=out,
        stderr=err,
    )

    assert rc == 2
    assert process.killed is True
    assert out.getvalue() == ""
    assert "ERROR device_code_timeout" in err.getvalue()


def test_expired_output_is_classified_with_stable_error_code():
    err = gpt_auth._classify_signin_failure(1, ["device code expired"])
    assert err.code == "auth_code_expired"
