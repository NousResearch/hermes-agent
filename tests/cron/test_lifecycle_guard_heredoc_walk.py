"""Inert-heredoc masking in the referenced-script walk (regression for #110422).

The direct lifecycle scan masks provably-inert heredoc bodies
(``strip_inert_heredoc_bodies``), but the referenced-script and ``-c`` payload
walks in ``_contains_unsafe_gateway_action`` ran on the unmasked command: a
path (or an ``sh -c`` payload) inside such a body is never shell-executed, so
walking it was a pure false positive — e.g. a >1 MiB path mentioned in a
``python3 - <<'PY'`` body failed closed and hard-blocked an innocent command.
The walks now use the same masked view as the direct scan.
"""

from __future__ import annotations

from cron.lifecycle_guard import (
    contains_gateway_lifecycle_command_or_referenced_script,
)

guard = contains_gateway_lifecycle_command_or_referenced_script


def _big_file(tmp_path):
    """A regular file over the 1 MiB referenced-script cap."""
    path = tmp_path / "big_blob.bin"
    path.write_bytes(b"\0" * (2 * 1024 * 1024))
    return path


def test_inert_heredoc_body_path_not_walked_as_script(tmp_path):
    """A >1 MiB path inside a provably-inert heredoc body is not a script reference."""
    big = _big_file(tmp_path)
    command = f"python3 - <<'PY'\n{big}\nPY"
    assert guard(command, cwd=str(tmp_path)) is False


def test_inert_heredoc_body_sh_c_prose_not_extracted(tmp_path):
    """An `sh -c` line inside an inert body is printed data, not an executed payload."""
    command = 'cat <<\'EOF\'\nsh -c "hermes gateway restart"\nEOF'
    assert guard(command, cwd=str(tmp_path)) is False


def test_unquoted_heredoc_body_path_still_walked(tmp_path):
    """Unquoted delimiter = expansion-capable = still visible to the walk, fail-closed."""
    big = _big_file(tmp_path)
    command = f"cat > /tmp/x <<EOF\n{big}\nEOF"
    assert guard(command, cwd=str(tmp_path)) is True


def test_script_reference_outside_heredoc_still_blocked(tmp_path):
    """Masking must not neuter the walk: a real referenced script still blocks."""
    script = tmp_path / "restart.sh"
    script.write_text("#!/bin/sh\nhermes gateway restart\n")
    assert guard(str(script), cwd=str(tmp_path)) is True
