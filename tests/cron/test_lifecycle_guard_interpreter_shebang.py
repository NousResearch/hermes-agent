"""Regression tests for the shebang-based interpreter exemption in the lifecycle walk.

The POSIX referenced-script walk applied to non-shell interpreter sources (suffixless
shebang-launched Python/Node/Ruby tools) tokenizes string literals into bogus
executed-script candidates: ``print("/dev/null" in sys.argv[1:])`` leaves a bare
``/dev/null`` token that fails closed on the character device. See #125378.
"""

from __future__ import annotations

from pathlib import Path

import cron.lifecycle_guard as lifecycle_guard
from cron.lifecycle_guard import GatewayLifecycleBlocked, check_gateway_lifecycle, scan_gateway_lifecycle


def _write_executable(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)
    return path


def test_suffixless_interpreter_script_with_dev_null_literal_is_allowed(tmp_path):
    """The exact #125378 repro: a shebang-launched python tool whose source contains a
    quoted ``/dev/null`` literal must not fail closed as an executed character device."""
    tool = _write_executable(
        tmp_path / "demo-tool",
        '#!/usr/bin/env python3\nimport sys\nprint("/dev/null" in sys.argv[1:])\n',
    )

    unsafe, refusal = scan_gateway_lifecycle(f"{tool} go")

    assert not unsafe
    assert refusal is None


def test_env_shebang_interpreter_script_is_also_allowed(tmp_path):
    """/usr/bin/env-prefixed shebangs resolve to the same interpreter exemption."""
    tool = _write_executable(
        tmp_path / "env-tool",
        '#!/usr/bin/env python3\nimport sys\nprint("/dev/null" in sys.argv[1:])\n',
    )

    unsafe, refusal = scan_gateway_lifecycle(f"{tool} --check")

    assert not unsafe
    assert refusal is None


def test_interpreter_script_text_still_blocks_literal_lifecycle_commands(tmp_path):
    """The exemption skips the shell-semantics WALK only; the direct lifecycle regex
    still scans the full text, so a literal gateway-restart string stays blocked."""
    tool = _write_executable(
        tmp_path / "bad-tool",
        '#!/usr/bin/env python3\nimport subprocess\n'
        'subprocess.run(["hermes", "gateway", "restart"])\n',
    )

    unsafe, _ = scan_gateway_lifecycle(f"{tool} go")

    assert unsafe


def test_shell_shebang_script_keeps_the_full_walk(tmp_path):
    """A #!/bin/sh script whose quoted ``/dev/null`` lands at segment COMMAND position
    (the exact promotion shape from #125378) still fails closed: for shell sources the
    walk (and its device fail-closed) must apply — the exemption is interpreter-specific."""
    script = _write_executable(
        tmp_path / "demo.sh",
        '#!/bin/sh\ntrue; "/dev/null" --filter\n',
    )

    unsafe, refusal = scan_gateway_lifecycle(f"{script} x")

    assert unsafe
    assert refusal is not None and "/dev/null" in refusal


def test_check_gateway_lifecycle_allows_suffixless_interpreter_cron_script(tmp_path):
    """The cron-job shape from #125378: prompt + suffixless python script with the
    device-naming string literal — the .py-suffix exemption generalized to content."""
    tool = _write_executable(
        tmp_path / "reviewer-write-check",
        '#!/usr/bin/env python3\nimport sys\nprint("/dev/null" in sys.argv[1:])\n',
    )

    check_gateway_lifecycle("nightly review pass", str(tool))


def test_check_gateway_lifecycle_blocks_lifecycle_literal_in_interpreter_script(tmp_path):
    tool = _write_executable(
        tmp_path / "bad-tool",
        '#!/usr/bin/env python3\n# runs: hermes gateway restart\n',
    )

    try:
        check_gateway_lifecycle("nightly", str(tool))
        raise AssertionError("literal lifecycle command in interpreter script was allowed")
    except GatewayLifecycleBlocked:
        pass
