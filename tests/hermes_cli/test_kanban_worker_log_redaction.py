"""Worker log redaction + filesystem-hygiene defense (t_ad15582c).

Kanban worker logs captured raw subprocess stdout/stderr; live Infisical
machine-identity client secrets and Cloudflare tokens leaked into plaintext
on disk in a 0664 file under a 0775 board subdir. The fix is two-pronged:

1. ``hermes_cli.kanban_db_dispatch._RedactingLog`` wraps the worker stdout
   file and scrubs every line through ``agent.redact.redact_sensitive_text``
   before the bytes hit disk.
2. ``_ensure_owner_only_perms`` chmods the per-task log to 0600 and the log
   dir to 0700 at open/rotate time, so a future umask drift can't reopen
   the group-readable hole.
"""

import io
import os
import stat

import pytest

from hermes_cli import kanban_db_dispatch as kbd


SECRET_HEX = "3e76e00d4a836650f2723570176598c9c7de641425adb6c2d245b634d62eb811"
CFUT_TOKEN = "cfut_abcdefghijklmnopqrstuvwxyz0123456789ABCD"
SAFE_SHA = "9d1d30c28f77958ef106cbfbae26ee008133feeca4ee1966b1b37d42fee256a2"


def _sink():
    return io.BytesIO()


def test_redacting_log_scrubs_64hex_with_credential_keyword():
    sink = _sink()
    log = kbd._RedactingLog(sink)
    log.write(f"client-secret: `{SECRET_HEX}`\n")
    log.close()
    out = sink.getvalue().decode()
    assert SECRET_HEX not in out, out
    # masked form: at least the hex starts-with prefix survives in truncated form
    assert "3e76e" in out, out


def test_redacting_log_scrubs_cfut_token():
    sink = _sink()
    log = kbd._RedactingLog(sink)
    log.write(f"Authorization: Bearer {CFUT_TOKEN}\n")
    log.close()
    out = sink.getvalue().decode()
    assert CFUT_TOKEN not in out


def test_redacting_log_scrubs_env_assignment():
    sink = _sink()
    log = kbd._RedactingLog(sink)
    log.write(f"INFISICAL_UNIVERSAL_AUTH_CLIENT_SECRET={SECRET_HEX}\n")
    log.close()
    out = sink.getvalue().decode()
    assert SECRET_HEX not in out
    assert "INFISICAL_UNIVERSAL_AUTH_CLIENT_SECRET=***" in out


def test_redacting_log_keeps_safe_sha256_research_fingerprint():
    """A bare sha256 of research content is NOT credential-shaped and must
    survive the filter — redaction must not mangle legitimate audit trails."""
    sink = _sink()
    log = kbd._RedactingLog(sink)
    log.write(f"# during research (sha256 {SAFE_SHA})\n")
    log.close()
    out = sink.getvalue().decode()
    assert SAFE_SHA in out, out


def test_redacting_log_handles_partial_line_buffering():
    """A credential that straddles two writes must still be caught when the
    trailing newline finally arrives."""
    sink = _sink()
    log = kbd._RedactingLog(sink)
    # split mid-token, no newline yet -> buffered, secret still in buffer
    log.write(f"PASSWORD={SECRET_HEX[:32]}")
    assert sink.getvalue() == b""  # nothing flushed yet
    log.write(SECRET_HEX[32:] + "\n")
    out = sink.getvalue().decode()
    assert SECRET_HEX not in out


def test_redacting_log_flushes_partial_tail_on_close():
    sink = _sink()
    log = kbd._RedactingLog(sink)
    log.write(f"client-secret: `{SECRET_HEX[:32]}`")
    log.write(SECRET_HEX[32:] + "`")
    log.close()  # no newline ever -> tail must still be scrubbed + written
    out = sink.getvalue().decode()
    assert SECRET_HEX not in out


def test_redacting_log_survives_redactor_exception():
    """A redactor crash must not silently kill the log — the line falls
    through unscrubbed so the audit trail stays intact."""
    import agent.redact as _real

    def _boom(*_a, **_kw):
        raise RuntimeError("redactor unavailable")

    sink = _sink()
    log = kbd._RedactingLog(sink)
    real = _real.redact_sensitive_text
    _real.redact_sensitive_text = _boom
    try:
        log.write(f"PASSWORD=hunter2\n")
    finally:
        _real.redact_sensitive_text = real
    log.close()
    # the line survived (unscrubbed) — the alternative (dropping it) hides evidence
    assert b"hunter2" in sink.getvalue()


def test_ensure_owner_only_perms_chmods_to_0600_0700(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    log_path = log_dir / "t_demo.log"
    log_path.write_text("hi")

    # start with the leaky defaults (the pre-fix state)
    log_dir.chmod(0o775)
    log_path.chmod(0o664)

    kbd._ensure_owner_only_perms(log_dir, log_path)

    dir_mode = stat.S_IMODE(log_dir.stat().st_mode)
    file_mode = stat.S_IMODE(log_path.stat().st_mode)
    assert dir_mode == 0o700, oct(dir_mode)
    assert file_mode == 0o600, oct(file_mode)


def test_ensure_owner_only_perms_is_idempotent(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    log_path = log_dir / "t_demo.log"
    log_path.write_text("hi")
    kbd._ensure_owner_only_perms(log_dir, log_path)
    # second call must not raise or change anything
    kbd._ensure_owner_only_perms(log_dir, log_path)
    assert stat.S_IMODE(log_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(log_path.stat().st_mode) == 0o600
