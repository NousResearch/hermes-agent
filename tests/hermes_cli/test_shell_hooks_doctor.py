"""Regression tests for #69257: `hermes hooks doctor` false-positive
on inline ``sh -c '...'`` shell hooks.

The doctor command previously reported inline shell commands as
"script missing or not executable" because ``_command_script_path()``
mistook the inline body (which can contain ``/`` characters from
redirections or paths) for a script path. The fix introduces a
sentinel value for inline-shell invocations and routes them through
``script_is_executable`` as a "healthy by construction" case.
"""

from __future__ import annotations

import os

import sys

import pytest

from agent.shell_hooks import (
    _INLINE_SHELL_SENTINEL,
    _command_script_path,
    _is_inline_shell_command,
    script_is_executable,
)


@pytest.mark.parametrize(
    "command",
    [
        "sh -c 'echo hi'",
        "bash -c 'echo hi'",
        "bash -c 'cmd 2>/dev/null'",  # the original false-positive trigger
        "dash -c 'echo'",
        "zsh -c 'echo'",
        "sudo sh -c 'echo hi'",
        "sudo bash -c 'cmd 2>/dev/null'",
        "env FOO=bar bash -c 'echo $FOO'",
    ],
)
def test_inline_shell_command_is_detected(command):
    """Every inline-shell invocation should be detected by the heuristic."""
    parts = command.split() if " " in command else [command]
    # just exercise the public function
    assert _command_script_path(command) == _INLINE_SHELL_SENTINEL


@pytest.mark.parametrize(
    "command",
    [
        "sh -c 'echo hi'",
        "bash -c 'cmd 2>/dev/null'",
        "sudo bash -c 'echo hi'",
    ],
)
def test_inline_shell_script_is_executable(command):
    """Inline shell hooks are healthy by definition — no script file to verify."""
    assert script_is_executable(command) is True


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows test environments typically lack /bin/bash and bash on PATH; covered by POSIX CI",
)
@pytest.mark.parametrize(
    "command",
    [
        "/bin/bash -c 'complex'",
        "env FOO=bar bash -c 'echo $FOO'",
    ],
)
def test_inline_shell_script_is_executable_path_form(command):
    """Same as above but for paths not on Windows PATH by default."""
    assert script_is_executable(command) is True


def test_file_based_hook_still_resolves(tmp_path):
    """A real file-based hook still resolves to its script path (no regression)."""
    script = tmp_path / "hook.py"
    script.write_text("#!/usr/bin/env python3\nprint()\n")
    script.chmod(0o644)

    command = f"python3 {script}"
    resolved = _command_script_path(command)
    # On Windows shlex treats backslashes as escape; verify resolved
    # ends with the script's basename regardless of path separator quirks.
    assert resolved.endswith("hook.py")
    assert resolved != _INLINE_SHELL_SENTINEL


def test_bare_script_path_still_resolves(tmp_path):
    """A bare ``/path/hook.sh`` invocation still resolves (no regression)."""
    script = tmp_path / "hook.sh"
    script.write_text("#!/bin/sh\necho hi\n")
    script.chmod(0o755)

    command = str(script)
    resolved = _command_script_path(command)
    assert resolved.endswith("hook.sh")
    assert resolved != _INLINE_SHELL_SENTINEL


def test_unknown_command_without_script_still_uses_first_token():
    """A plain command (no interpreter, no script extension) still returns
    the first token — the original fallback in the existing code."""
    assert _command_script_path("my-tool --flag") == "my-tool"


def test_sentinel_is_distinct_from_real_paths():
    """The sentinel must never collide with a real filesystem path."""
    assert _INLINE_SHELL_SENTINEL == "<inline-shell>"
    # No real path should start with the sentinel prefix in a sane system.
    assert not os.path.exists("/" + _INLINE_SHELL_SENTINEL)


def test_interpreter_without_dash_c_is_not_inline():
    """`bash script.sh` (no -c) is not inline — it's a normal script invocation."""
    parts = ["bash", "/tmp/some.sh"]
    assert _is_inline_shell_command(parts) is False


@pytest.mark.parametrize(
    "command",
    [
        "bash -lc 'echo hi'",
        "bash -lic 'echo'",
        "zsh -fc 'echo'",
        "sh -cl 'echo'",
        "bash -clpe 'echo $0'",  # last char 'e' not 'c' → reject
    ],
)
def test_bundled_c_flag_is_inline(command):
    """Bundled flags ending in ``c`` (``-lc``, ``-fc``, ``-lic``, ``-clpe``)
    are all treated as ``c``-flag bundles. This covers the common
    ``bash -lc '...'`` and ``zsh -fc '...'`` shapes called out in
    the review."""
    from shlex import split as _split
    parts = _split(command)
    if command.endswith("-clpe 'echo $0'"):
        # -clpe ends in 'e', should NOT match
        assert _is_inline_shell_command(parts) is False
    else:
        assert _is_inline_shell_command(parts) is True
        assert _command_script_path(command) == _INLINE_SHELL_SENTINEL


@pytest.mark.skipif(sys.platform == "win32", reason="Windows PATH typically lacks /bin/bash; covered by POSIX CI")
def test_inline_shell_requires_existing_interpreter():
    """An inline shell using a non-existent interpreter must be reported unhealthy.

    Per review: ``shutil.which()`` (with /bin fallback for POSIX) must
    confirm the interpreter is reachable before marking the inline
    hook as healthy. A typo like ``bsh -c '...'`` should NOT be reported
    as a working hook.
    """
    assert script_is_executable("bsh -c 'echo hi'") is False, (
        "A non-existent interpreter (bsh) must NOT be reported as a healthy inline shell hook"
    )
    assert script_is_executable("nonesuch-shell -c 'echo hi'") is False


class _FakeSpec:
    """Minimal stand-in for the real hook spec dataclass used by doctor."""
    def __init__(self, command, event="pre_tool_call"):
        self.command = command
        self.event = event


def _capture_doctor(spec, allowlist_entry=None):
    """Run ``_doctor_one`` and return its printed output.

    ``_doctor_one`` calls ``shell_hooks.script_is_executable`` and
    ``shell_hooks.allowlist_entry_for`` on the passed module.  We
    pass the real ``agent.shell_hooks`` for the former (its functions
    are exactly what we're testing) and accept an injected
    ``allowlist_entry`` for the latter so each test controls the
    allowlist side independently.
    """
    import io
    import contextlib
    from unittest.mock import patch
    from agent import shell_hooks as real_shell_hooks
    from hermes_cli.hooks import _doctor_one

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        with patch.object(
            real_shell_hooks, "allowlist_entry_for",
            return_value=allowlist_entry,
        ):
            problems = _doctor_one(spec, shell_hooks=real_shell_hooks)
    return problems, buf.getvalue()


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows test environment typically lacks sh on PATH; covered by POSIX CI",
)
def test_doctor_reports_inline_shell_distinctly():
    """Per review: ``_doctor_one`` should print ``inline shell command``
    for ``sh -c '...'`` hooks, NOT ``script exists and is executable``
    (which implies there is a script file).
    """
    problems, output = _capture_doctor(_FakeSpec("sh -c 'echo hi'"))
    assert problems == 0, "inline shell hook should not be a problem"
    assert "inline shell command" in output, (
        "doctor must print 'inline shell command' for sh -c '...'"
    )
    assert "script exists and is executable" not in output, (
        "doctor must NOT print the file-script success line for inline shells"
    )


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Test relies on real shell availability for the inline path; covered by POSIX CI",
)
def test_doctor_reports_file_based_hook_normally(tmp_path):
    """File-based hooks (the existing path) still print the original
    'script exists and is executable' message — no regression."""
    script = tmp_path / "hook.py"
    script.write_text("#!/usr/bin/env python3\nprint()\n")
    script.chmod(0o755)

    problems, output = _capture_doctor(_FakeSpec(f"python3 {script}"))
    assert problems == 0
    assert "script exists and is executable" in output


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows sh availability complicates the inline-detection branch",
)
def test_doctor_still_flags_missing_script(tmp_path):
    """A real file-based hook with a missing script must STILL be flagged.
    The unallowlisted safety gate is unchanged — ``hermes hooks doctor``
    never executes untrusted hooks; it only inspects them.
    """
    missing = tmp_path / "no-such-hook.sh"  # does not exist
    problems, output = _capture_doctor(_FakeSpec(str(missing)))
    assert problems == 1, "missing script must remain a problem"
    assert "script missing or not executable" in output
