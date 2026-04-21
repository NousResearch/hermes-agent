"""``hermes debug`` — debug tools for Hermes Agent.

Currently supports:
    hermes debug share    Upload debug report (system info + logs) to a
                          paste service and print a shareable URL.
"""

import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

from hermes_constants import get_hermes_home


# ---------------------------------------------------------------------------
# Paste services — try paste.rs first, dpaste.com as fallback.
# ---------------------------------------------------------------------------

_PASTE_RS_URL = "https://paste.rs/"
_DPASTE_COM_URL = "https://dpaste.com/api/"

# Maximum bytes to read from a single log file for upload.
# paste.rs caps at ~1 MB; we stay under that with headroom.
_MAX_LOG_BYTES = 512_000

# Auto-delete pastes after this many seconds (6 hours).
_AUTO_DELETE_SECONDS = 21600


# ---------------------------------------------------------------------------
# Pending-deletion tracking (replaces the old fork-and-sleep subprocess).
# ---------------------------------------------------------------------------

def _pending_file() -> Path:
    """Path to ``~/.hermes/pastes/pending.json``.

    Each entry: ``{"url": "...", "expire_at": <unix_ts>}``.  Scheduled
    DELETEs used to be handled by spawning a detached Python process per
    paste that slept for 6 hours; those accumulated forever if the user
    ran ``hermes debug share`` repeatedly.  We now persist the schedule
    to disk and sweep expired entries on the next debug invocation.
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
        os.replace(tmp, path)
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

        url = entry.get("url", "")
        try:
            if delete_paste(url):
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

Pastes auto-delete after 6 hours.
"""

_GATEWAY_PRIVACY_NOTICE = (
    "⚠️ **Privacy notice:** This uploads system info + recent log tails "
    "(may contain conversation fragments) to a public paste service. "
    "Full logs are NOT included from the gateway — use `hermes debug share` "
    "from the CLI for full log uploads.\n"
    "Pastes auto-delete after 6 hours."
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
    and rely on opportunistic sweeps (``_sweep_expired_pastes``) called from
    every ``hermes debug`` invocation.  If the user never runs ``hermes debug``
    again, paste.rs's own retention policy handles cleanup.
    """
    _record_pending(urls, delay_seconds=delay_seconds)


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


# ---------------------------------------------------------------------------
# Log file reading
# ---------------------------------------------------------------------------

def _resolve_log_path(log_name: str) -> Optional[Path]:
    """Find the log file for *log_name*, falling back to the .1 rotation.

    Returns the path if found, or None.
    """
    from hermes_cli.logs import LOG_FILES

    filename = LOG_FILES.get(log_name)
    if not filename:
        return None

    log_dir = get_hermes_home() / "logs"
    primary = log_dir / filename
    if primary.exists() and primary.stat().st_size > 0:
        return primary

    # Fall back to the most recent rotated file (.1).
    rotated = log_dir / f"{filename}.1"
    if rotated.exists() and rotated.stat().st_size > 0:
        return rotated

    return None


def _read_log_tail(log_name: str, num_lines: int) -> str:
    """Read the last *num_lines* from a log file, or return a placeholder."""
    from hermes_cli.logs import _read_last_n_lines

    log_path = _resolve_log_path(log_name)
    if log_path is None:
        return "(file not found)"

    try:
        lines = _read_last_n_lines(log_path, num_lines)
        return "".join(lines).rstrip("\n")
    except Exception as exc:
        return f"(error reading: {exc})"


def _read_full_log(log_name: str, max_bytes: int = _MAX_LOG_BYTES) -> Optional[str]:
    """Read a log file for standalone upload.

    Returns the file content (last *max_bytes* if truncated), or None if the
    file doesn't exist or is empty.
    """
    log_path = _resolve_log_path(log_name)
    if log_path is None:
        return None

    try:
        size = log_path.stat().st_size
        if size == 0:
            return None

        if size <= max_bytes:
            return log_path.read_text(encoding="utf-8", errors="replace")

        # File is larger than max_bytes — read the tail.
        with open(log_path, "rb") as f:
            f.seek(size - max_bytes)
            # Skip partial line at the seek point.
            f.readline()
            content = f.read().decode("utf-8", errors="replace")
        return f"[... truncated — showing last ~{max_bytes // 1024}KB ...]\n{content}"
    except Exception:
        return None


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


def collect_debug_report(*, log_lines: int = 200, dump_text: str = "") -> str:
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

    # ── Recent log tails (summary only) ──────────────────────────────────
    buf.write("\n\n")
    buf.write(f"--- agent.log (last {log_lines} lines) ---\n")
    buf.write(_read_log_tail("agent", log_lines))
    buf.write("\n\n")

    errors_lines = min(log_lines, 100)
    buf.write(f"--- errors.log (last {errors_lines} lines) ---\n")
    buf.write(_read_log_tail("errors", errors_lines))
    buf.write("\n\n")

    buf.write(f"--- gateway.log (last {errors_lines} lines) ---\n")
    buf.write(_read_log_tail("gateway", errors_lines))
    buf.write("\n")

    return buf.getvalue()


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------

def run_debug_share(args):
    """Collect debug report + full logs, upload each, print URLs."""
    log_lines = getattr(args, "lines", 200)
    expiry = getattr(args, "expire", 7)
    local_only = getattr(args, "local", False)

    if not local_only:
        print(_PRIVACY_NOTICE)

    print("Collecting debug report...")

    # Capture dump once — prepended to every paste for context.
    dump_text = _capture_dump()

    report = collect_debug_report(log_lines=log_lines, dump_text=dump_text)
    agent_log = _read_full_log("agent")
    gateway_log = _read_full_log("gateway")

    # Prepend dump header to each full log so every paste is self-contained.
    if agent_log:
        agent_log = dump_text + "\n\n--- full agent.log ---\n" + agent_log
    if gateway_log:
        gateway_log = dump_text + "\n\n--- full gateway.log ---\n" + gateway_log

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
    print(f"\n⏱  Pastes will auto-delete in 6 hours.")

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
        print()
        print("Options (delete):")
        print("  <url> ...    One or more paste URLs to delete")
