"""Tests for inline interpreter execution (-c/-e) masking in lifecycle_guard (#91433)."""

import pytest
from cron.lifecycle_guard import (
    _mask_data_sink_arguments,
    contains_gateway_lifecycle_command_or_referenced_script as guard,
)


def test_benign_print_payload_is_masked():
    cmd = "python3 -c \"print('hermes gateway restart')\""
    masked = _mask_data_sink_arguments(cmd)
    assert "hermes gateway restart" not in masked
    assert guard(cmd) is False


def test_benign_print_with_flags_between_binary_and_c():
    cmd = "python3 -u -c \"print('hermes gateway restart')\""
    masked = _mask_data_sink_arguments(cmd)
    assert "hermes gateway restart" not in masked
    assert guard(cmd) is False


def test_versioned_and_other_interpreters_benign_masked():
    interpreters = [
        "python3.11",
        "pypy3",
        "deno eval",
        "bun -e",
        "node -e",
        "ruby -e",
        "perl -e",
    ]
    for interp in interpreters:
        flag = "" if " " in interp else "-c "
        fn = "console.log" if any(k in interp for k in ("node", "bun", "deno")) else "print"
        cmd = f"{interp} {flag}\"{fn}('hermes gateway restart')\""
        masked = _mask_data_sink_arguments(cmd)
        assert "hermes gateway restart" not in masked, f"Failed for interpreter {interp}"


def test_dangerous_os_system_not_masked():
    cmd = "python3 -c \"import os; os.system('hermes gateway restart')\""
    masked = _mask_data_sink_arguments(cmd)
    # Dangerous markers prevent masking
    assert "hermes gateway restart" in masked
    assert guard(cmd) is True


def test_dangerous_shutil_rmtree_not_masked():
    cmd = "python3 -c \"import shutil; shutil.rmtree('/data')\""
    masked = _mask_data_sink_arguments(cmd)
    assert "shutil" in masked


def test_dangerous_ruby_delete_not_masked():
    cmd = "ruby -e 'File.delete(\"/etc/important\")'"
    masked = _mask_data_sink_arguments(cmd)
    assert "File.delete" in masked


def test_dangerous_perl_unlink_not_masked():
    cmd = "perl -e 'unlink glob \"/tmp/*\"'"
    masked = _mask_data_sink_arguments(cmd)
    assert "unlink" in masked


def test_dangerous_node_rmsync_not_masked():
    cmd = 'node -e \'require("fs").rmSync("/", {recursive:true})\''
    masked = _mask_data_sink_arguments(cmd)
    assert "rmSync" in masked
