"""``hermes debug`` debug tools for Hermes Agent.

Currently supports:
    hermes debug share    Upload debug report (system info + logs) to a
                          paste service and print a shareable URL.
                          By default, log content is run through
                          ``agent.redact.redact_sensitive_text`` with
                          ``force=True`` before upload so credentials in
                          ``~/.hermes/logs/*.log`` are not leaked into
                          the public paste service. Pass ``--no-redact``
                          to disable.
"""

import io
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from hermes_constants import get_hermes_home
from utils import atomic_replace

logger = logging.getLogger(__name__)

# Banner prepended to upload-bound log content when redaction is enabled.
# Visible in the public paste so reviewers know the content was sanitized.
# Kept short; the trailing newline guarantees the banner sits on its own line.
_REDACTION_BANNER = (
    "[hermes debug share: log content redacted at upload time. "
    "run with --no-redact to disable]\n"
)

_EMAIL_ADDRESS_RE = re.compile(
    r"(?<![A-Za-z0-9._%+-])"
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    r"(?![A-Za-z0-9._%+-])"
)

# Structured-PII patterns for upload-bound log content (debug share). These
# catch PII that regex can mask reliably without shredding ordinary log noise
# (timestamps, ids, counts). NOTE: bare personal NAMES are deliberately NOT
# covered — they are not reliably regex-detectable, so the real protection for
# names is HERMES_DEBUG_LOCAL_DEFAULT=1 (no upload by default). E.164 phones are
# already handled upstream by agent.redact.redact_sensitive_text.
_PII_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    (re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)"),
     "[REDACTED_PHONE]"),
    (re.compile(
        r"\b\d{1,6}\s+(?:[NSEW]\.?\s+)?[A-Za-z0-9.'\-]+(?:\s+[A-Za-z0-9.'\-]+){0,4}\s+"
        r"(?:St|Street|Ave|Avenue|Rd|Road|Dr|Drive|Ln|Lane|Blvd|Boulevard|Way|Ct|Court|"
        r"Cir|Circle|Pl|Place|Pkwy|Parkway|Hwy|Highway|Ter|Terrace|Trail|Loop)\b\.?",
        re.I), "[REDACTED_ADDRESS]"),
    (re.compile(r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b"), "[REDACTED_ZIP]"),
]


def _redact_pii_text(text: str) -> str:
    """Mask structured PII (SSN, US/NA phones, street addresses, ZIP) in
    upload-bound text. Over-redaction is the safe failure mode here."""
    if not text:
        return text
    for rx, repl in _PII_PATTERNS:
        text = rx.sub(repl, text)
    return text


# ---------------------------------------------------------------------------
# Paste services — try paste.rs first, dpaste.com as fallback.
# ---------------------------------------------------------------------------

_PASTE_RS_URL = "https://paste.rs/"
_DPASTE_COM_URL = "https://dpaste.com/api/"
_GIST_API_URL = "https://api.github.com/gists"

# Maximum bytes to read from a single log file for upload.
# paste.rs caps at ~1 MB; we stay under that with headroom.
_MAX_LOG_BYTES = 512_000

# Auto-delete pastes after this many seconds. Default 15 min; override with
# HERMES_DEBUG_AUTO_DELETE_SECONDS (was a hard-coded 6h — long public exposure).
try:
    _AUTO_DELETE_SECONDS = int(os.environ.get("HERMES_DEBUG_AUTO_DELETE_SECONDS", "900"))
except ValueError:
    _AUTO_DELETE_SECONDS = 900


# ---------------------------------------------------------------------------
# Pending-deletion tracking (replaces the old fork-and-sleep subprocess).
# ---------------------------------------------------------------------------

def _pending_file() -> Path:
    """Path to ``~/.hermes/pastes/pending.json``.

    Each entry: ``{"url": "...", "expire_at": <unix_ts>}``.  Scheduled
    DELETEs used to be handled by spawning a detached Python process per
    paste that slept for 6 hours; those accumulated forever if the user
    ran ``hermes debug share`` repeatedly.

    Deletion is now driven by the gateway's cron ticker
    (``gateway/run.py::_start_cron_ticker``) which calls
    ``_sweep_expired_pastes`` once per hour.  ``hermes debug share`` also
    runs an opportunistic sweep on entry as a fallback for CLI-only users
    who never start the gateway.
    """
    return get_hermes_home() / "pastes" / "pending.json"


def _load_pending() -> list[dict]:
    path = _pending_file()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            # Filter to well-formed entries only
            return [
                e for e in data
                if isinstance(e, dict) and "url" in e and "expire_at" in e
            ]
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    return []


def _save_pending(entries: list[dict]) -> None:
    path = _pending_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        atomic_replace(tmp, path)
    except OSError:
        # Non-fatal — worst case the user has to run ``hermes debug delete``
        # manually.
        pass


def _record_pending(urls: list[str], delay_seconds: int = _AUTO_DELETE_SECONDS) -> None:
    """Record *urls* for deletion at ``now + delay_seconds``.

    Only paste.rs URLs are recorded (dpaste.com auto-expires).  Entries
    are merged into any existing pending.json.
    """
    paste_rs_urls = [u for u in urls if _extract_paste_id(u)]
    if not paste_rs_urls:
        return

    entries = _load_pending()
    # Dedupe by URL: keep the later expire_at if same URL appears twice
    by_url: dict[str, float] = {e["url"]: float(e["expire_at"]) for e in entries}
    expire_at = time.time() + delay_seconds
    for u in paste_rs_urls:
        by_url[u] = max(expire_at, by_url.get(u, 0.0))
    merged = [{"url": u, "expire_at": ts} for u, ts in by_url.items()]
    _save_pending(merged)


def _record_pending_gist(gist_id: str, ref_url: str = "",
                         delay_seconds: int = _AUTO_DELETE_SECONDS) -> None:
    """Record a secret gist for deletion at ``now + delay_seconds``.

    Deduped by ``gist_id`` (the 3 file raw URLs share one gist — one DELETE
    revokes all of them). Entries carry ``kind="gist"`` so the sweep dispatches
    to :func:`delete_gist` instead of :func:`delete_paste`.
    """
    if not gist_id:
        return
    entries = _load_pending()
    expire_at = time.time() + delay_seconds
    for e in entries:
        if e.get("kind") == "gist" and e.get("gist_id") == gist_id:
            try:
                e["expire_at"] = max(float(e.get("expire_at", 0)), expire_at)
            except (TypeError, ValueError):
                e["expire_at"] = expire_at
            _save_pending(entries)
            return
    entries.append({
        "url": ref_url or f"{_GIST_API_URL}/{gist_id}",
        "expire_at": expire_at,
        "kind": "gist",
        "gist_id": gist_id,
    })
    _save_pending(entries)


def _sweep_expired_pastes(now: Optional[float] = None) -> tuple[int, int]:
    """Synchronously DELETE any pending pastes whose ``expire_at`` has passed.

    Returns ``(deleted, remaining)``.  Best-effort: failed deletes stay in
    the pending file and will be retried on the next sweep.  Silent —
    intended to be called from every ``hermes debug`` invocation with
    minimal noise.
    """
    entries = _load_pending()
    if not entries:
        return (0, 0)

    current = time.time() if now is None else now
    deleted = 0
    remaining: list[dict] = []

    for entry in entries:
        try:
            expire_at = float(entry.get("expire_at", 0))
        except (TypeError, ValueError):
            continue  # drop malformed entries
        if expire_at > current:
            remaining.append(entry)
            continue

        try:
            if entry.get("kind") == "gist":
                if delete_gist(entry.get("gist_id", "")):
                    deleted += 1
                    continue
            elif delete_paste(entry.get("url", "")):
                deleted += 1
                continue
        except Exception:
            # Network hiccup, 404 (already gone), etc. — drop the entry
            # after a grace period; don't retry forever.
            pass

        # Retain failed deletes for up to 24h past expiration, then give up.
        if expire_at + 86400 > current:
            remaining.append(entry)
        else:
            deleted += 1  # count as reaped (paste.rs will GC eventually)

    if deleted:
        _save_pending(remaining)

    return (deleted, len(remaining))


def _best_effort_sweep_expired_pastes() -> None:
    """Attempt pending-paste cleanup without letting /debug fail offline."""
    try:
        _sweep_expired_pastes()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Privacy / delete helpers
# ---------------------------------------------------------------------------

_PRIVACY_NOTICE = """\
⚠️  This will upload the following to a public paste service:
  • System info (OS, Python version, Hermes version, provider, which API keys
    are configured — NOT the actual keys)
  • Recent log lines (agent.log, errors.log, gateway.log — may contain
    conversation fragments and file paths)
  • Full agent.log and gateway.log (up to 512 KB each — likely contains
    conversation content, tool outputs, and file paths)

Pastes auto-delete after ~15 minutes.
"""

_GATEWAY_PRIVACY_NOTICE = (
    "⚠️ **Privacy notice:** This uploads system info + recent log tails "
    "(may contain conversation fragments) to a public paste service. "
    "Full logs are NOT included from the gateway — use `hermes debug share` "
    "from the CLI for full log uploads.\n"
    "Pastes auto-delete after ~15 minutes."
)


def _extract_paste_id(url: str) -> Optional[str]:
    """Extract the paste ID from a paste.rs or dpaste.com URL.

    Returns the ID string, or None if the URL doesn't match a known service.
    """
    url = url.strip().rstrip("/")
    for prefix in ("https://paste.rs/", "http://paste.rs/"):
        if url.startswith(prefix):
            return url[len(prefix):]
    return None


def _extract_gist_id(url: str) -> Optional[str]:
    """Extract the gist id from a gist URL (raw or html). None if not a gist.

    Raw URL: ``https://gist.githubusercontent.com/<user>/<id>/raw/<sha>/<file>``
    HTML URL: ``https://gist.github.com/<user>/<id>``
    The id is the second non-empty path segment and is hex.
    """
    try:
        parts = urllib.parse.urlsplit(url.strip())
    except Exception:
        return None
    if parts.netloc not in ("gist.githubusercontent.com", "gist.github.com"):
        return None
    segs = [s for s in parts.path.split("/") if s]
    if len(segs) >= 2 and re.fullmatch(r"[0-9a-fA-F]+", segs[1]):
        return segs[1]
    return None


def delete_gist(gist_id: str) -> bool:
    """Delete a secret gist by id via the authenticated GitHub API.

    Returns True on success (204). Unlike paste.rs, gist DELETE requires the
    same ``HERMES_DEBUG_GIST_TOKEN`` (classic PAT, ``gist`` scope) used to
    create it. Deleting the gist revokes ALL of its file raw URLs at once.
    """
    if not gist_id:
        raise ValueError("delete_gist: empty gist_id")
    token = os.environ.get("HERMES_DEBUG_GIST_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HERMES_DEBUG_GIST_TOKEN is not set")
    target = f"{_GIST_API_URL}/{gist_id}"
    req = urllib.request.Request(
        target, method="DELETE",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "hermes-agent/debug-share",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return 200 <= resp.status < 300


def delete_paste(url: str) -> bool:
    """Delete a paste from paste.rs.  Returns True on success.

    Only paste.rs supports unauthenticated DELETE.  dpaste.com pastes
    expire automatically but cannot be deleted via API.
    """
    paste_id = _extract_paste_id(url)
    if not paste_id:
        raise ValueError(
            f"Cannot delete: only paste.rs URLs are supported.  Got: {url}"
        )

    target = f"{_PASTE_RS_URL}{paste_id}"
    req = urllib.request.Request(
        target, method="DELETE",
        headers={"User-Agent": "hermes-agent/debug-share"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return 200 <= resp.status < 300


def _schedule_auto_delete(urls: list[str], delay_seconds: int = _AUTO_DELETE_SECONDS):
    """Record *urls* for deletion ``delay_seconds`` from now.

    Previously this spawned a detached Python subprocess per call that slept
    for 6 hours and then issued DELETE requests.  Those subprocesses leaked —
    every ``hermes debug share`` invocation added ~20 MB of resident Python
    interpreters that never exited until the sleep completed.

    The replacement is stateless: we append to ``~/.hermes/pastes/pending.json``
    and the gateway's cron ticker sweeps expired entries once per hour.
    ``hermes debug share`` also runs an opportunistic sweep as a fallback
    for CLI-only users.  If neither runs again, paste.rs's own retention
    policy handles cleanup.
    """
    _record_pending(urls, delay_seconds=delay_seconds)


def _schedule_gist_auto_delete(urls: list[str],
                               delay_seconds: int = _AUTO_DELETE_SECONDS) -> Optional[str]:
    """Schedule deletion of the secret gist behind *urls* (its file raw URLs).

    All raw URLs from one ``/debug`` share a single gist id; we extract it from
    the first parseable URL and record ONE pending entry. The gateway cron
    ticker's hourly sweep then issues the authenticated DELETE once the entry
    expires. Returns the gist id (or None if no URL looked like a gist).
    """
    for u in urls:
        gid = _extract_gist_id(u)
        if gid:
            _record_pending_gist(gid, ref_url=u, delay_seconds=delay_seconds)
            return gid
    return None


def _delete_hint(url: str) -> str:
    """Return a one-liner delete command for the given paste URL."""
    paste_id = _extract_paste_id(url)
    if paste_id:
        return f"hermes debug delete {url}"
    # dpaste.com — no API delete, expires on its own.
    return "(auto-expires per dpaste.com policy)"


def _upload_paste_rs(content: str) -> str:
    """Upload to paste.rs.  Returns the paste URL.

    paste.rs accepts a plain POST body and returns the URL directly.
    """
    data = content.encode("utf-8")
    req = urllib.request.Request(
        _PASTE_RS_URL, data=data, method="POST",
        headers={
            "Content-Type": "text/plain; charset=utf-8",
            "User-Agent": "hermes-agent/debug-share",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        url = resp.read().decode("utf-8").strip()
    if not url.startswith("http"):
        raise ValueError(f"Unexpected response from paste.rs: {url[:200]}")
    return url


def _upload_dpaste_com(content: str, expiry_days: int = 7) -> str:
    """Upload to dpaste.com.  Returns the paste URL.

    dpaste.com uses multipart form data.
    """
    boundary = "----HermesDebugBoundary9f3c"

    def _field(name: str, value: str) -> str:
        return (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n'
            f"\r\n"
            f"{value}\r\n"
        )

    body = (
        _field("content", content)
        + _field("syntax", "text")
        + _field("expiry_days", str(expiry_days))
        + f"--{boundary}--\r\n"
    ).encode("utf-8")

    req = urllib.request.Request(
        _DPASTE_COM_URL, data=body, method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "hermes-agent/debug-share",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        url = resp.read().decode("utf-8").strip()
    if not url.startswith("http"):
        raise ValueError(f"Unexpected response from dpaste.com: {url[:200]}")
    return url


def upload_to_pastebin(content: str, expiry_days: int = 7) -> str:
    """Upload *content* to a paste service, trying paste.rs then dpaste.com.

    Returns the paste URL on success, raises on total failure.
    """
    errors: list[str] = []

    # Try paste.rs first (simple, fast)
    try:
        return _upload_paste_rs(content)
    except Exception as exc:
        errors.append(f"paste.rs: {exc}")

    # Fallback: dpaste.com (supports expiry)
    try:
        return _upload_dpaste_com(content, expiry_days=expiry_days)
    except Exception as exc:
        errors.append(f"dpaste.com: {exc}")

    raise RuntimeError(
        "Failed to upload to any paste service:\n  " + "\n  ".join(errors)
    )


def _gist_enabled() -> bool:
    """True when a GitHub token is configured for private-gist debug uploads."""
    return bool(os.environ.get("HERMES_DEBUG_GIST_TOKEN", "").strip())


def _local_default() -> bool:
    """When HERMES_DEBUG_LOCAL_DEFAULT is truthy, `hermes debug share` prints
    locally and never uploads to a public pastebin."""
    return os.environ.get("HERMES_DEBUG_LOCAL_DEFAULT", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def upload_to_gist(files: dict, description: str = "hermes debug share",
                   public: bool = False) -> dict:
    """Upload {filename: content} to a SECRET GitHub gist.

    Returns {filename: raw_url}. Requires env HERMES_DEBUG_GIST_TOKEN — a
    classic PAT with the 'gist' scope (fine-grained PATs are not reliably
    accepted by the gists API and return 401).
    """
    token = os.environ.get("HERMES_DEBUG_GIST_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HERMES_DEBUG_GIST_TOKEN is not set")
    payload = json.dumps({
        "description": description,
        "public": bool(public),
        "files": {name: {"content": (content or "(empty)")}
                  for name, content in files.items()},
    }).encode("utf-8")
    req = urllib.request.Request(
        _GIST_API_URL, data=payload, method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
            "User-Agent": "hermes-agent/debug-share",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    out = {name: f["raw_url"]
           for name, f in data.get("files", {}).items()
           if f and f.get("raw_url")}
    if not out:
        raise ValueError("Gist API returned no file URLs")
    return out


# ---------------------------------------------------------------------------
# Log file reading
# ---------------------------------------------------------------------------


@dataclass
class LogSnapshot:
    """Single-read snapshot of a log file used by debug-share."""

    path: Optional[Path]
    tail_text: str
    full_text: Optional[str]


def _primary_log_path(log_name: str) -> Optional[Path]:
    """Where *log_name* would live if present. Doesn't check existence."""
    from hermes_cli.logs import LOG_FILES

    filename = LOG_FILES.get(log_name)
    return (get_hermes_home() / "logs" / filename) if filename else None


def _resolve_log_path(log_name: str) -> Optional[Path]:
    """Find the log file for *log_name*, falling back to the .1 rotation.

    Returns the first non-empty candidate (primary, then .1), or None.
    Callers distinguish 'empty primary' from 'truly missing' via
    :func:`_primary_log_path`.
    """
    primary = _primary_log_path(log_name)
    if primary is None:
        return None

    if primary.exists() and primary.stat().st_size > 0:
        return primary

    rotated = primary.parent / f"{primary.name}.1"
    if rotated.exists() and rotated.stat().st_size > 0:
        return rotated

    return None


def _redact_log_text(text: str) -> str:
    """Run ``redact_sensitive_text`` with ``force=True`` over upload-bound text.

    Uses ``force=True`` so redaction fires regardless of the operator's
    ``security.redact_secrets`` setting. The local on-disk log file is
    not modified; only the in-memory copy headed for the public paste
    service is sanitized. Returns the redacted text (or the original
    when empty / non-string).
    """
    if not text:
        return text
    from agent.redact import redact_sensitive_text

    text = redact_sensitive_text(text, force=True)
    text = _EMAIL_ADDRESS_RE.sub("[REDACTED_EMAIL]", text)
    return _redact_pii_text(text)


def _capture_log_snapshot(
    log_name: str,
    *,
    tail_lines: int,
    max_bytes: int = _MAX_LOG_BYTES,
    redact: bool = True,
) -> LogSnapshot:
    """Capture a log once and derive summary/full-log views from it.

    The report tail and standalone log upload must come from the same file
    snapshot. Otherwise a rotation/truncate between reads can make the report
    look newer than the uploaded ``agent.log`` paste.

    When ``redact`` is True (the default), both ``tail_text`` and
    ``full_text`` are run through ``_redact_log_text`` so the snapshot
    returned is upload-safe. The on-disk log file is never modified.
    Pass ``redact=False`` to capture original log content (used by
    ``hermes debug share --no-redact``).
    """
    log_path = _resolve_log_path(log_name)
    if log_path is None:
        primary = _primary_log_path(log_name)
        tail = "(file empty)" if primary and primary.exists() else "(file not found)"
        return LogSnapshot(path=None, tail_text=tail, full_text=None)

    try:
        size = log_path.stat().st_size
        if size == 0:
            # race: file was truncated between _resolve_log_path and stat
            return LogSnapshot(path=log_path, tail_text="(file empty)", full_text=None)

        with open(log_path, "rb") as f:
            if size <= max_bytes:
                raw = f.read()
                truncated = False
            else:
                # Read from the end until we have enough bytes for the
                # standalone upload and enough newline context to render the
                # summary tail from the same snapshot.
                chunk_size = 8192
                pos = size
                chunks: list[bytes] = []
                total = 0
                newline_count = 0

                while pos > 0 and (total < max_bytes or newline_count <= tail_lines + 1) and total < max_bytes * 2:
                    read_size = min(chunk_size, pos)
                    pos -= read_size
                    f.seek(pos)
                    chunk = f.read(read_size)
                    chunks.insert(0, chunk)
                    total += len(chunk)
                    newline_count += chunk.count(b"\n")
                    chunk_size = min(chunk_size * 2, 65536)

                raw = b"".join(chunks)
                truncated = pos > 0

        full_raw = raw
        if truncated and len(full_raw) > max_bytes:
            cut = len(full_raw) - max_bytes
            # Check whether the cut lands exactly on a line boundary.  If the
            # byte just before the cut position is a newline the first retained
            # byte starts a complete line and we should keep it.  Only drop a
            # partial first line when we're genuinely mid-line.
            on_boundary = cut > 0 and full_raw[cut - 1 : cut] == b"\n"
            full_raw = full_raw[cut:]
            if not on_boundary and b"\n" in full_raw:
                full_raw = full_raw.split(b"\n", 1)[1]

        all_text = raw.decode("utf-8", errors="replace")
        tail_text = "".join(all_text.splitlines(keepends=True)[-tail_lines:]).rstrip("\n")

        full_text = full_raw.decode("utf-8", errors="replace")
        if truncated:
            full_text = f"[... truncated — showing last ~{max_bytes // 1024}KB ...]\n{full_text}"

        if redact:
            tail_text = _redact_log_text(tail_text)
            full_text = _redact_log_text(full_text)

        return LogSnapshot(path=log_path, tail_text=tail_text, full_text=full_text)
    except Exception as exc:
        return LogSnapshot(path=log_path, tail_text=f"(error reading: {exc})", full_text=None)


def _capture_default_log_snapshots(
    log_lines: int, *, redact: bool = True
) -> dict[str, LogSnapshot]:
    """Capture all logs used by debug-share exactly once.

    ``redact`` is forwarded to each ``_capture_log_snapshot`` call so all
    captured logs share the same redaction policy for a given run.
    """
    errors_lines = min(log_lines, 100)
    return {
        "agent": _capture_log_snapshot(
            "agent", tail_lines=log_lines, redact=redact
        ),
        "errors": _capture_log_snapshot(
            "errors", tail_lines=errors_lines, redact=redact
        ),
        "gateway": _capture_log_snapshot(
            "gateway", tail_lines=errors_lines, redact=redact
        ),
    }


# ---------------------------------------------------------------------------
# Debug report collection
# ---------------------------------------------------------------------------

def _capture_dump() -> str:
    """Run ``hermes dump`` and return its stdout as a string."""
    from hermes_cli.dump import run_dump

    class _FakeArgs:
        show_keys = False

    old_stdout = sys.stdout
    sys.stdout = capture = io.StringIO()
    try:
        run_dump(_FakeArgs())
    except SystemExit:
        pass
    finally:
        sys.stdout = old_stdout

    return capture.getvalue()


def collect_debug_report(
    *,
    log_lines: int = 200,
    dump_text: str = "",
    log_snapshots: Optional[dict[str, LogSnapshot]] = None,
) -> str:
    """Build the summary debug report: system dump + log tails.

    Parameters
    ----------
    log_lines
        Number of recent lines to include per log file.
    dump_text
        Pre-captured dump output.  If empty, ``hermes dump`` is run
        internally.

    Returns the report as a plain-text string ready for upload.
    """
    buf = io.StringIO()

    if not dump_text:
        dump_text = _capture_dump()
    buf.write(dump_text)

    if log_snapshots is None:
        log_snapshots = _capture_default_log_snapshots(log_lines)

    # ── Recent log tails (summary only) ──────────────────────────────────
    buf.write("\n\n")
    buf.write(f"--- agent.log (last {log_lines} lines) ---\n")
    buf.write(log_snapshots["agent"].tail_text)
    buf.write("\n\n")

    errors_lines = min(log_lines, 100)
    buf.write(f"--- errors.log (last {errors_lines} lines) ---\n")
    buf.write(log_snapshots["errors"].tail_text)
    buf.write("\n\n")

    buf.write(f"--- gateway.log (last {errors_lines} lines) ---\n")
    buf.write(log_snapshots["gateway"].tail_text)
    buf.write("\n")

    return buf.getvalue()


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------

def run_debug_share(args):
    """Collect debug report + full logs, upload each, print URLs."""
    _best_effort_sweep_expired_pastes()

    log_lines = getattr(args, "lines", 200)
    expiry = getattr(args, "expire", 7)
    local_only = getattr(args, "local", False) or _local_default()
    redact = not getattr(args, "no_redact", False)

    if not local_only:
        print(_PRIVACY_NOTICE)

    print("Collecting debug report...")

    # Capture dump once — prepended to every paste for context.
    # The dump is already redacted at extract time via dump.py:_redact;
    # log_snapshots are redacted by _capture_default_log_snapshots when
    # redact=True so credentials never reach the public paste service.
    dump_text = _capture_dump()
    log_snapshots = _capture_default_log_snapshots(log_lines, redact=redact)

    if redact:
        logger.info(
            "hermes debug share: applied force-mode redaction to log snapshots before upload"
        )

    report = collect_debug_report(
        log_lines=log_lines,
        dump_text=dump_text,
        log_snapshots=log_snapshots,
    )
    agent_log = log_snapshots["agent"].full_text
    gateway_log = log_snapshots["gateway"].full_text

    # Prepend dump header to each full log so every paste is self-contained.
    if agent_log:
        agent_log = dump_text + "\n\n--- full agent.log ---\n" + agent_log
    if gateway_log:
        gateway_log = dump_text + "\n\n--- full gateway.log ---\n" + gateway_log

    # Visible banner so reviewers reading the public paste know redaction
    # was applied at upload time. Banner is omitted under --no-redact.
    if redact:
        report = _REDACTION_BANNER + report
        if agent_log:
            agent_log = _REDACTION_BANNER + agent_log
        if gateway_log:
            gateway_log = _REDACTION_BANNER + gateway_log

    if local_only:
        print(report)
        if agent_log:
            print(f"\n\n{'=' * 60}")
            print("FULL agent.log")
            print(f"{'=' * 60}\n")
            print(agent_log)
        if gateway_log:
            print(f"\n\n{'=' * 60}")
            print("FULL gateway.log")
            print(f"{'=' * 60}\n")
            print(gateway_log)
        return

    print("Uploading...")
    urls: dict[str, str] = {}
    failures: list[str] = []

    # 1. Summary report (required)
    try:
        urls["Report"] = upload_to_pastebin(report, expiry_days=expiry)
    except RuntimeError as exc:
        print(f"\nUpload failed: {exc}", file=sys.stderr)
        print("\nFull report printed below — copy-paste it manually:\n")
        print(report)
        sys.exit(1)

    # 2. Full agent.log (optional)
    if agent_log:
        try:
            urls["agent.log"] = upload_to_pastebin(agent_log, expiry_days=expiry)
        except Exception as exc:
            failures.append(f"agent.log: {exc}")

    # 3. Full gateway.log (optional)
    if gateway_log:
        try:
            urls["gateway.log"] = upload_to_pastebin(gateway_log, expiry_days=expiry)
        except Exception as exc:
            failures.append(f"gateway.log: {exc}")

    # Print results
    label_width = max(len(k) for k in urls)
    print(f"\nDebug report uploaded:")
    for label, url in urls.items():
        print(f"  {label:<{label_width}}  {url}")

    if failures:
        print(f"\n  (failed to upload: {', '.join(failures)})")

    # Schedule auto-deletion after 6 hours
    _schedule_auto_delete(list(urls.values()))
    print(f"\n⏱  Pastes will auto-delete in ~15 minutes.")

    # Manual delete fallback
    print(f"To delete now:  hermes debug delete <url>")

    print(f"\nShare these links with the Hermes team for support.")


def run_debug_delete(args):
    """Delete one or more paste URLs uploaded by /debug."""
    urls = getattr(args, "urls", [])
    if not urls:
        print("Usage: hermes debug delete <url> [<url> ...]")
        print("  Deletes paste.rs pastes uploaded by 'hermes debug share'.")
        return

    for url in urls:
        try:
            ok = delete_paste(url)
            if ok:
                print(f"  ✓ Deleted: {url}")
            else:
                print(f"  ✗ Failed to delete: {url} (unexpected response)")
        except ValueError as exc:
            print(f"  ✗ {exc}")
        except Exception as exc:
            print(f"  ✗ Could not delete {url}: {exc}")


def run_debug(args):
    """Route debug subcommands."""
    # Opportunistic sweep of expired pastes on every ``hermes debug`` call.
    # Replaces the old per-paste sleeping subprocess that used to leak as
    # one orphaned Python interpreter per scheduled deletion.  Silent and
    # best-effort — any failure is swallowed so ``hermes debug`` stays
    # reliable even when offline.
    try:
        _sweep_expired_pastes()
    except Exception:
        pass

    subcmd = getattr(args, "debug_command", None)
    if subcmd == "share":
        run_debug_share(args)
    elif subcmd == "delete":
        run_debug_delete(args)
    else:
        # Default: show help
        print("Usage: hermes debug <command>")
        print()
        print("Commands:")
        print("  share    Upload debug report to a paste service and print URL")
        print("  delete   Delete a previously uploaded paste")
        print()
        print("Options (share):")
        print("  --lines N    Number of log lines to include (default: 200)")
        print("  --expire N   Paste expiry in days (default: 7)")
        print("  --local      Print report locally instead of uploading")
        print("  --no-redact  Disable upload-time secret redaction (default: redact)")
        print()
        print("Options (delete):")
        print("  <url> ...    One or more paste URLs to delete")
