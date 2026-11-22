"""Pre-build macOS notarization credential preflight tests."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest

from tests.ci.desktop_release_roles import (
    DOWNLOADABLE_DISPATCHES,
    DRY_DISPATCH,
    ROOT,
    admitted,
    gate,
    load,
    native_builds,
)

_PEM_BEGIN = "-----BEGIN " + "PRIVATE KEY-----"
_PEM_END = "-----END " + "PRIVATE KEY-----"


def _synthetic_p8(*, newline: str = "\n") -> str:
    """Build a syntactically valid PKCS#8 DER-framed test fixture (not a real key)."""
    der = bytes([0x30, 0x26]) + b"\x02\x01\x00" + (b"\x04" * 35)
    body = base64.b64encode(der).decode("ascii")
    return f"{_PEM_BEGIN}{newline}{body[:40]}{newline}{body[40:]}{newline}{_PEM_END}{newline}"


def _step_by_name(job: dict, name: str) -> tuple[int, dict]:
    matches = [(i, step) for i, step in enumerate(job["steps"]) if step.get("name") == name]
    assert len(matches) == 1, f"expected one step named {name!r}, found {len(matches)}"
    return matches[0]


def _preflight_step(job: dict) -> tuple[int, dict]:
    matches = [
        (i, step)
        for i, step in enumerate(job["steps"])
        if "check_macos_notary_auth" in step.get("run", "")
    ]
    assert len(matches) == 1, f"expected one notary auth preflight step, found {len(matches)}"
    return matches[0]


def test_darwin_legs_verify_notary_credentials_after_writing_key_and_before_building():
    jobs = load()["jobs"]
    legs = native_builds(jobs)
    darwin_jobs = [jobs[name] for (target, _mode), name in legs.items() if target.startswith("darwin-")]
    assert darwin_jobs, "walk broken: no darwin build legs found"

    for job in darwin_jobs:
        write_idx, _write_step = _step_by_name(job, "Write App Store Connect key for notarytool")
        preflight_idx, preflight = _preflight_step(job)
        build_indices = [
            i
            for i, step in enumerate(job["steps"])
            if "scripts/bundles/desktop.py" in step.get("run", "") and "--prepared " in step.get("run", "")
        ]
        assert build_indices, "walk broken: no darwin desktop build step found"
        assert preflight_idx == write_idx + 1, "preflight must run immediately after writing apple-api-key.p8"
        assert preflight_idx < build_indices[0], "preflight must run before building"

        for dispatch, inputs in DOWNLOADABLE_DISPATCHES.items():
            assert gate(preflight["if"], inputs, {}, job_if=False), f"preflight skipped for {dispatch}"
        assert not gate(preflight["if"], DRY_DISPATCH, {}, job_if=False), "preflight ran on dry build"

        step_env = preflight.get("env") or {}
        assert step_env.get("APPLE_API_KEY_ID") == "${{ secrets.APPLE_API_KEY_ID }}"
        assert step_env.get("APPLE_API_ISSUER") == "${{ secrets.APPLE_API_ISSUER }}"
        assert "APPLE_API_KEY" not in step_env, "step env must not shadow APPLE_API_KEY from GITHUB_ENV"


def test_validate_p8_pem_accepts_pkcs8_and_rejects_malformed_keys():
    from scripts.ci.check_macos_notary_auth import validate_p8_pem

    assert validate_p8_pem(_synthetic_p8()) is None
    assert validate_p8_pem(_synthetic_p8(newline="\r\n")) is None

    for bad in (
        "",
        "   \n\t ",
        "not-a-pem-key",
        f"{_PEM_BEGIN}\n{_PEM_END}\n",
        f"{_PEM_END}\nabcd\n{_PEM_BEGIN}\n",
        f"-----BEGIN ENCRYPTED PRIVATE KEY-----\n{base64.b64encode(b'0' * 40).decode()}\n-----END ENCRYPTED PRIVATE KEY-----\n",
        f"{_PEM_BEGIN}\n%%%not-valid-base64%%%\n{_PEM_END}\n",
        f"{_PEM_BEGIN}\n{base64.b64encode(b'not-der-sequence-payload-at-all-1234567890').decode()}\n{_PEM_END}\n",
    ):
        assert validate_p8_pem(bad) is not None, f"expected rejection for {bad!r}"


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (
            "Error: HTTP status code: 401. Unauthenticated. Ensure that all authentication arguments are correct.",
            "auth_failed",
        ),
        (
            "Error: HTTP status code: 401. Invalid credentials. Username or password is incorrect.",
            "auth_failed",
        ),
        (
            "Error: HTTP status code: 403.\n"
            "A required agreement is missing or has expired.\n"
            "Ensure your team has signed the necessary legal agreements and that they are not expired.",
            "auth_failed",
        ),
        ("Error: HTTP status code: 403. Forbidden.", "auth_failed"),
        ("Error: HTTPError(statusCode: Optional(401), error: unauthorized)", "auth_failed"),
        ("Error: HTTPError(statusCode: Optional(403), error: forbidden)", "auth_failed"),
        ("Error: The value 'not-a-uuid' is invalid for '--issuer <issuer>'", "auth_failed"),
        ("Error: Invalid option: ***", "auth_failed"),
        ("Error: Failed to parse private key (CryptoKit.CryptoKitError error 1.)", "auth_failed"),
        (
            "Error: HTTP status code: 500.\n"
            "Internal server error.\n"
            "Error communicating with authentication service.\n"
            "Please try again at a later time.",
            "transient",
        ),
        # A 5xx status or a transport failure wins over auth-sounding text in the same output.
        ("Error: HTTP status code: 500. Unauthenticated.", "transient"),
        ("Error: HTTP status code: 502. Bad Gateway.", "transient"),
        ("Error: HTTP status code: 503. Service Unavailable.", "transient"),
        ("Error: HTTP status code: 504. Gateway Timeout.", "transient"),
        ("Error: HTTP status code: 429. Too Many Requests.", "transient"),
        (
            'Error: HTTPError(statusCode: nil, error: Error Domain=NSURLErrorDomain Code=-1001 "The request timed out.")',
            "transient",
        ),
        ('Error Domain=NSURLErrorDomain Code=-1005 "The network connection was lost."', "transient"),
        ('Error Domain=NSURLErrorDomain Code=-1009 "The Internet connection appears to be offline."', "transient"),
        ('Error Domain=NSURLErrorDomain Code=-1005 "The network connection was lost." Invalid credentials.', "transient"),
        ("xcrun: error: unexpected transient portal hiccup", "transient"),
    ],
)
def test_classify_notarytool_failure_distinguishes_auth_from_transient(output: str, expected: str):
    from scripts.ci.check_macos_notary_auth import PreflightStatus, classify_notarytool_failure

    status, reason = classify_notarytool_failure(output)
    assert status is PreflightStatus(expected), (output, status, reason)
    assert reason


def test_check_notary_auth_handles_success_timeout_oserror_and_redacts_secrets(tmp_path: Path):
    from scripts.ci.check_macos_notary_auth import PreflightStatus, check_notary_auth

    key_file = tmp_path / "apple-api-key.p8"
    pem = _synthetic_p8()
    key_file.write_text(pem, encoding="utf-8")
    key_id = "FAKEKEY999"
    issuer = "11111111-2222-3333-4444-555555555555"
    seen_calls: list[tuple[list[str], float]] = []

    def ok_runner(argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
        seen_calls.append((argv, timeout))
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps({"history": [{"id": "a"}, {"id": "b"}], "message": "ok"}),
            stderr="",
        )

    ok = check_notary_auth(str(key_file), key_id, issuer, timeout=12.0, runner=ok_runner)
    assert ok.status is PreflightStatus.OK
    assert "2" in ok.message
    assert seen_calls == [
        (
            [
                "xcrun",
                "notarytool",
                "history",
                "--key",
                str(key_file),
                "--key-id",
                key_id,
                "--issuer",
                issuer,
                "--output-format",
                "json",
            ],
            12.0,
        )
    ]

    def leaking_401_runner(argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            argv,
            1,
            stdout="",
            stderr=f"Error: HTTP status code: 401. Unauthenticated. key-id={key_id} issuer={issuer}\n{pem}",
        )

    failed = check_notary_auth(str(key_file), key_id, issuer, runner=leaking_401_runner)
    assert failed.status is PreflightStatus.AUTH_FAILED
    combined = f"{failed.message}\n{failed.detail}"
    assert key_id not in combined
    assert issuer not in combined
    assert _PEM_BEGIN not in combined

    def timeout_runner(argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(argv, timeout)

    timed_out = check_notary_auth(str(key_file), key_id, issuer, timeout=5.0, runner=timeout_runner)
    assert timed_out.status is PreflightStatus.TRANSIENT

    def missing_tool_runner(argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("xcrun not found")

    missing_tool = check_notary_auth(str(key_file), key_id, issuer, runner=missing_tool_runner)
    assert missing_tool.status is PreflightStatus.TRANSIENT

    bad_key = tmp_path / "bad.p8"
    bad_key.write_text("not a pem", encoding="utf-8")
    called = False

    def must_not_run(argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
        nonlocal called
        called = True
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    bad = check_notary_auth(str(bad_key), key_id, issuer, runner=must_not_run)
    assert bad.status is PreflightStatus.AUTH_FAILED
    assert not called

    absent = check_notary_auth(str(tmp_path / "missing.p8"), key_id, issuer, runner=must_not_run)
    assert absent.status is PreflightStatus.AUTH_FAILED
    assert not called


@pytest.mark.platforms("posix")
def test_workflow_write_and_preflight_steps_replay_end_to_end(tmp_path: Path):
    jobs = load()["jobs"]
    release_job = jobs[native_builds(jobs)[("darwin-arm64", "release")]]
    _write_idx, write_step = _step_by_name(release_job, "Write App Store Connect key for notarytool")
    _preflight_idx, preflight_step = _preflight_step(release_job)

    tools = tmp_path / "bin"
    tools.mkdir()
    (tools / "python").write_text(
        f"#!/bin/sh\nexec {shlex.quote(sys.executable)} \"$@\"\n",
        encoding="utf-8",
    )
    (tools / "python").chmod(0o755)

    xcrun_log = tmp_path / "xcrun-argv.json"
    xcrun_driver = tools / "fake_xcrun.py"
    xcrun_driver.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['FAKE_XCRUN_LOG']).write_text(json.dumps(sys.argv[1:]), encoding='utf-8')\n"
        "sys.stdout.write(os.environ.get('FAKE_XCRUN_STDOUT', ''))\n"
        "sys.stderr.write(os.environ.get('FAKE_XCRUN_STDERR', ''))\n"
        "raise SystemExit(int(os.environ.get('FAKE_XCRUN_RC', '0')))\n",
        encoding="utf-8",
    )
    (tools / "xcrun").write_text(
        f"#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(xcrun_driver))} \"$@\"\n",
        encoding="utf-8",
    )
    (tools / "xcrun").chmod(0o755)

    key_id = "FAKEKEY999"
    issuer = "11111111-2222-3333-4444-555555555555"
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()

    def run_steps(
        *,
        pem_text: str,
        xcrun_rc: int = 0,
        xcrun_stdout: str = "",
        xcrun_stderr: str = "",
        cwd: Path = ROOT,
        build_commit: str = "",
    ) -> subprocess.CompletedProcess[str]:
        xcrun_log.unlink(missing_ok=True)
        github_env = tmp_path / "github_env"
        github_env.write_text("", encoding="utf-8")
        base_env = {
            "PATH": str(tools) + os.pathsep + os.environ["PATH"],
            "RUNNER_TEMP": str(runner_temp),
            "GITHUB_ENV": str(github_env),
            "FAKE_XCRUN_LOG": str(xcrun_log),
            "FAKE_XCRUN_RC": str(xcrun_rc),
            "FAKE_XCRUN_STDOUT": xcrun_stdout,
            "FAKE_XCRUN_STDERR": xcrun_stderr,
            "HERMES_BUILD_COMMIT": build_commit,
        }
        write_proc = subprocess.run(
            ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", write_step["run"]],
            cwd=cwd,
            env={**base_env, "APPLE_API_KEY_P8": pem_text},
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            check=False,
        )
        assert write_proc.returncode == 0, write_proc.stdout + write_proc.stderr
        exported = dict(
            line.split("=", 1)
            for line in github_env.read_text(encoding="utf-8-sig").splitlines()
            if "=" in line
        )
        return subprocess.run(
            ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", preflight_step["run"]],
            cwd=cwd,
            env={
                **base_env,
                **exported,
                "APPLE_API_KEY_ID": key_id,
                "APPLE_API_ISSUER": issuer,
            },
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            check=False,
        )

    ok = run_steps(pem_text=_synthetic_p8(), xcrun_stdout=json.dumps({"history": [{"id": "1"}]}))
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert json.loads(xcrun_log.read_text(encoding="utf-8-sig")) == [
        "notarytool",
        "history",
        "--key",
        str(runner_temp / "apple-api-key.p8"),
        "--key-id",
        key_id,
        "--issuer",
        issuer,
        "--output-format",
        "json",
    ]

    auth_401 = run_steps(
        pem_text=_synthetic_p8(),
        xcrun_rc=1,
        xcrun_stderr=(
            f"Error: HTTP status code: 401. Unauthenticated. key={key_id} issuer={issuer}"
        ),
    )
    assert auth_401.returncode != 0
    assert "::error::" in auth_401.stdout + auth_401.stderr
    assert key_id not in auth_401.stdout + auth_401.stderr
    assert issuer not in auth_401.stdout + auth_401.stderr

    auth_403 = run_steps(
        pem_text=_synthetic_p8(),
        xcrun_rc=1,
        xcrun_stderr="Error: HTTP status code: 403. A required agreement is missing or has expired.",
    )
    assert auth_403.returncode != 0
    assert "::error::" in auth_403.stdout + auth_403.stderr

    transient_500 = run_steps(
        pem_text=_synthetic_p8(),
        xcrun_rc=1,
        xcrun_stderr=(
            "Error: HTTP status code: 500. Internal server error. "
            "Error communicating with authentication service."
        ),
    )
    assert transient_500.returncode == 0, transient_500.stdout + transient_500.stderr
    assert "::warning::" in transient_500.stdout + transient_500.stderr

    transient_nsurl = run_steps(
        pem_text=_synthetic_p8(),
        xcrun_rc=1,
        xcrun_stderr='Error Domain=NSURLErrorDomain Code=-1005 "The network connection was lost."',
    )
    assert transient_nsurl.returncode == 0, transient_nsurl.stdout + transient_nsurl.stderr
    assert "::warning::" in transient_nsurl.stdout + transient_nsurl.stderr

    bad_pem = run_steps(pem_text="corrupted-not-a-p8-key")
    assert bad_pem.returncode != 0
    assert "::error::" in bad_pem.stdout + bad_pem.stderr
    assert not xcrun_log.exists(), "invalid PEM must fail before invoking xcrun"

    # Channel builds prefer the trusted controller checkout over the target branch checkout.
    channel_workspace = tmp_path / "channel-workspace"
    controller_dir = channel_workspace / ".build/channel-controller/scripts/ci"
    controller_dir.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/ci/check_macos_notary_auth.py", controller_dir / "check_macos_notary_auth.py")
    # The admitted checkout's own helper passes without probing, so only the controller's can fail here.
    workspace_helper = channel_workspace / "scripts/ci/check_macos_notary_auth.py"
    workspace_helper.parent.mkdir(parents=True)
    workspace_helper.write_text("raise SystemExit(0)\n", encoding="utf-8")
    channel_401 = run_steps(
        pem_text=_synthetic_p8(),
        xcrun_rc=1,
        xcrun_stderr="Error: HTTP status code: 401. Unauthenticated.",
        cwd=channel_workspace,
    )
    assert channel_401.returncode == 1, channel_401.stdout + channel_401.stderr
    assert "HTTP 401" in channel_401.stdout + channel_401.stderr
    assert xcrun_log.exists(), "expected the controller's helper to probe Apple, not the workspace's"

    # A release candidate's admitted commit can predate the helper; the controller checkout supplies it.
    release_workspace = tmp_path / "older-release-source"
    release_workspace.mkdir()
    for step in release_job["steps"]:
        checkout = step.get("with", {})
        if checkout.get("ref") == "${{ github.sha }}" and gate(
            step["if"], DOWNLOADABLE_DISPATCHES["candidate"], admitted(["validate"]), job_if=False,
        ):
            dest = release_workspace / checkout["path"] / "scripts/ci/check_macos_notary_auth.py"
            dest.parent.mkdir(parents=True)
            shutil.copy2(ROOT / "scripts/ci/check_macos_notary_auth.py", dest)
    older_release = run_steps(pem_text=_synthetic_p8(), cwd=release_workspace)
    assert older_release.returncode == 0, older_release.stdout + older_release.stderr
    assert xcrun_log.exists(), "a release of an older commit still needs the auth probe"

    # Older commit builds lacking the helper warn and continue; release checkouts fail closed.
    older_checkout = tmp_path / "older-checkout"
    older_checkout.mkdir()
    older_commit = run_steps(pem_text=_synthetic_p8(), cwd=older_checkout, build_commit="a" * 40)
    assert older_commit.returncode == 0
    assert "::warning::" in older_commit.stdout + older_commit.stderr

    missing_on_release = run_steps(pem_text=_synthetic_p8(), cwd=older_checkout, build_commit="")
    assert missing_on_release.returncode != 0
    assert "::error::" in missing_on_release.stdout + missing_on_release.stderr
