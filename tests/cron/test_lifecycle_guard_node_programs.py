"""Directly executed non-shell programs on a remote backend (#125396 review).

Adapted from jlxq0's regression file in PR comment 6234081373; the final test
records the retained shared-read-budget policy rather than requiring fewer reads.

A ~215 KB minified node CLI behind ``#!/usr/bin/env node`` was refused as a gateway restart:
the walk read it as shell and ran out of remote reads. The program must be allowed, a literal
restart inside it must still block, and a shell running it must still walk it as shell.
"""

import pytest

import cron.lifecycle_guard as lifecycle_guard

_MANY_PATHS = "".join(f"/nonexistent/p{i}\n" for i in range(200))

_REMOTE_FILES = {
    "/x/node-many": "#!/usr/bin/env node\n" + _MANY_PATHS,
    "/x/node-restart": "#!/usr/bin/env node\nrequire('child_process').execSync('hermes gateway restart')\n",
    # One minified line longer than the default 64 KiB line budget.
    "/x/node-minified": "#!/usr/bin/env node\n" + "a=1;" * 17_000 + "require('child_process').execSync('hermes gateway restart')\n",
    "/x/sh-restart": "#!/bin/sh\nhermes gateway restart\n",
    "/x/sh-many": "#!/bin/sh\n" + _MANY_PATHS,
    # A node shebang over shell content: a shell running it ignores the shebang.
    "/x/node-shebang-runs-sh": "#!/usr/bin/env node\n/x/sh-restart\n",
    "/x/wrapper.sh": "#!/bin/sh\nbash /x/node-shebang-runs-sh\n",
}


@pytest.fixture
def remote():
    """A remote backend that serves _REMOTE_FILES and records every read."""
    reads: list[str] = []

    def read(path: str) -> str:
        reads.append(path)
        return _REMOTE_FILES.get(path, "")

    def blocked(command: str) -> bool:
        reads.clear()
        return lifecycle_guard.contains_gateway_lifecycle_command_or_referenced_script(
            command, read_remote_script=read
        )

    blocked.reads = reads
    return blocked


def test_sudo_wrapped_node_program_is_allowed(remote):
    assert remote("sudo -n -u svc /x/node-many sync-status") is False


def test_literal_restart_inside_node_program_still_blocked(remote):
    assert remote("/x/node-restart") is True


def test_shell_script_restart_still_blocked(remote):
    assert remote("/x/sh-restart") is True


def test_shell_script_with_the_same_paths_still_fails_closed(remote):
    assert remote("/x/sh-many") is True


@pytest.mark.parametrize("command", [
    "bash /x/node-shebang-runs-sh",
    "sh /x/node-shebang-runs-sh",
    "dash /x/node-shebang-runs-sh",
    "zsh /x/node-shebang-runs-sh",
    "ksh /x/node-shebang-runs-sh",
    "bash -e /x/node-shebang-runs-sh",
    "/bin/bash /x/node-shebang-runs-sh",
    "sudo -n bash /x/node-shebang-runs-sh",
    "source /x/node-shebang-runs-sh",
    ". /x/node-shebang-runs-sh",
    "/x/node-shebang-runs-sh; bash /x/node-shebang-runs-sh",
    "/x/wrapper.sh",
    "/x/node-shebang-runs-sh; /x/wrapper.sh",
    "/x/wrapper.sh; /x/node-shebang-runs-sh",
])
def test_shell_run_file_is_walked_whatever_its_shebang(remote, command):
    assert remote(command) is True


# --- the two findings -----------------------------------------------------------------------


def test_unscannable_node_program_text_still_fails_closed(remote, monkeypatch):
    """The exemption relaxes the reference walk, not the scan of the program's own text."""
    monkeypatch.setattr(lifecycle_guard, "_MAX_LIFECYCLE_SCAN_LINE_BYTES", 16)
    assert remote("/x/node-restart") is True


def test_minified_line_over_the_default_line_budget_still_fails_closed(remote):
    """Same case at default limits: one minified line longer than 64 KiB."""
    assert remote("/x/node-minified") is True


def test_allowed_node_program_reference_reads_remain_bounded(remote, caplog):
    """Keep nested-script detection; incidental references may spend the shared read cap."""
    assert remote("/x/node-many sync-status") is False
    assert len(remote.reads) == lifecycle_guard._MAX_LIFECYCLE_SCAN_REMOTE_READS
    assert "scan budget exhausted" not in caplog.text
