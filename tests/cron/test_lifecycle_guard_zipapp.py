r"""Python zipapps are archives, not scripts, for the gateway lifecycle guard (#78833).

A zipapp is a shebang line glued to a zip archive. ``_has_binary_magic`` let any
``#!`` file through as "interpreted, never binary", so a pip-installed ``yt-dlp``
(~3 MB) fell to the oversized fail-closed branch in
``_read_referenced_script_unlocked`` and ``yt-dlp --version`` was refused as a
gateway-lifecycle command.

The size limit itself is unchanged: an oversized *script* must still fail closed,
and forged ``PK\x03\x04`` magic must not buy a skip.
"""

from __future__ import annotations

import io
import os
import zipfile

import cron.lifecycle_guard as lifecycle_guard

guard = lifecycle_guard.contains_gateway_lifecycle_command_or_referenced_script

SHEBANG = b"#!/usr/bin/env python3\n"
OVERSIZED = lifecycle_guard._MAX_REFERENCED_SCRIPT_BYTES + 1


def _write_executable(path, payload: bytes):
    path.write_bytes(payload)
    os.chmod(path, 0o755)
    return path


def _zipapp_bytes(payload_size: int) -> bytes:
    """A real zipapp: shebang + a zip whose central directory is intact.

    The member is incompressible random data so the archive genuinely exceeds
    the byte cap rather than deflating back under it.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("__main__.py", "print('hello')\n")
        archive.writestr("payload.bin", os.urandom(payload_size))
    return SHEBANG + buffer.getvalue()


def test_oversized_zipapp_is_not_a_lifecycle_refusal(tmp_path):
    """The regression: ``yt-dlp --version`` must run."""
    tool = _write_executable(tmp_path / "yt-dlp", _zipapp_bytes(OVERSIZED))
    assert tool.stat().st_size > lifecycle_guard._MAX_REFERENCED_SCRIPT_BYTES

    assert guard(f"{tool} --version", cwd=str(tmp_path)) is False


def test_small_zipapp_is_not_scanned_as_shell_text(tmp_path):
    """Size is not what identifies a zipapp; the archive structure is."""
    tool = _write_executable(tmp_path / "tiny-app", _zipapp_bytes(16))

    assert guard(f"{tool} --version", cwd=str(tmp_path)) is False


def test_oversized_shell_script_still_fails_closed(tmp_path):
    """The oversized branch is untouched for files that really are scripts."""
    padding = b"# padding\n" * ((OVERSIZED // 10) + 1)
    script = _write_executable(
        tmp_path / "big.sh",
        b"#!/bin/bash\n" + padding + b"hermes gateway restart\n",
    )
    assert script.stat().st_size > lifecycle_guard._MAX_REFERENCED_SCRIPT_BYTES

    assert guard(f"bash {script}", cwd=str(tmp_path)) is True


def test_forged_zip_magic_without_central_directory_still_fails_closed(tmp_path):
    """Magic alone must not buy a skip: ``zipfile.is_zipfile`` is the real check."""
    forged = _write_executable(
        tmp_path / "forged",
        SHEBANG + b"PK\x03\x04" + b"A" * OVERSIZED,
    )
    assert not zipfile.is_zipfile(forged)

    assert guard(f"bash {forged}", cwd=str(tmp_path)) is True
