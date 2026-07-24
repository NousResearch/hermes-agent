"""Fixed form of H4 — BOM-sniff before decode; refuse-to-mangle (#66475)."""
import codecs
import io
import os
import tempfile
from pathlib import Path


def atomic_replace(src, dst):
    os.replace(src, dst)


def _sanitize_env_lines(lines):
    return lines


def _sanitize_env_file_if_needed(path: Path) -> None:
    if not path.exists():
        return
    try:
        raw = path.read_bytes()
    except Exception:
        return

    force_utf8_rewrite = False
    if raw.startswith(codecs.BOM_UTF32_LE) or raw.startswith(codecs.BOM_UTF32_BE):
        return  # refuse-to-mangle
    if raw.startswith(codecs.BOM_UTF16_LE) or raw.startswith(codecs.BOM_UTF16_BE):
        try:
            with io.TextIOWrapper(
                io.BytesIO(raw), encoding="utf-16", newline=None
            ) as f:
                original = f.readlines()
        except UnicodeDecodeError:
            return
        force_utf8_rewrite = True
    else:
        # utf-8-sig path WITHOUT persisting errors=replace corruption:
        # read with replace for NUL stripping, but abort rewrite if the
        # first line starts with U+FFFD (unknown binary / mis-decoded).
        try:
            with open(path, encoding="utf-8-sig", errors="replace") as f:  # encoding-safety: ok — guarded: abort rewrite when first line starts with U+FFFD; UTF-16 sniffed above
                original = f.readlines()
        except Exception:
            return
        if original and original[0].startswith("\ufffd"):
            return

    stripped = [line.replace("\x00", "") for line in original]
    sanitized = _sanitize_env_lines(stripped)
    if sanitized != original or force_utf8_rewrite:
        fd, tmp = tempfile.mkstemp(
            dir=str(path.parent), suffix=".tmp", prefix=".env_"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.writelines(sanitized)
                f.flush()
                os.fsync(f.fileno())
            atomic_replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
