"""``hermes debug`` debug tools for Hermes Agent."""

import contextlib
import datetime
import gzip
import io
import json
import logging
import re
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from hermes_constants import get_hermes_home
from utils import atomic_replace

logger = logging.getLogger(__name__)

# Prepended to upload-bound content when redaction is enabled so paste reviewers know.
_REDACTION_BANNER = (
    "[hermes debug share: log content redacted at upload time. "
    "run with --no-redact to disable]\n")
_EMAIL_ADDRESS_RE = re.compile(
    r"(?<![A-Za-z0-9._%+-])"
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    r"(?![A-Za-z0-9._%+-])")
_PASTE_RS_URL = "https://paste.rs/"  # primary; dpaste.com is the fallback
_DPASTE_COM_URL = "https://dpaste.com/api/"
_USER_AGENT = "hermes-agent/debug-share"
_MAX_LOG_BYTES = 512_000  # per log file for upload (paste.rs caps at ~1 MB)
_AUTO_DELETE_SECONDS = 21600  # 6 hours

# Pending-deletion tracking: the gateway cron ticker calls ``_sweep_expired_pastes`` hourly and
# ``hermes debug`` sweeps on entry (CLI-only users). Replaced a fork-and-sleep subprocess that
# leaked ~20 MB per share.


def _pending_file() -> Path:
    return get_hermes_home() / "pastes" / "pending.json"


def _load_pending() -> list[dict]:
    try:
        data = json.loads(_pending_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    return [e for e in data if isinstance(e, dict) and {"url", "expire_at"} <= e.keys()]


def _save_pending(entries: list[dict]) -> None:
    path = _pending_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        atomic_replace(tmp, path)
    except OSError:
        pass  # non-fatal — worst case the user runs ``hermes debug delete`` manually


def _sweep_expired_pastes(now: Optional[float] = None) -> tuple[int, int]:
    """Synchronously DELETE pending pastes whose ``expire_at`` has passed → (deleted, remaining).

    Best-effort and silent: failed deletes stay pending for the next sweep, up to 24h past
    expiration (paste.rs GCs them eventually).
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
            gone = delete_paste(entry.get("url", ""))
        except Exception:
            gone = False  # network hiccup, 404 (already gone), ...
        if gone or expire_at + 86400 <= current:
            deleted += 1  # deleted, or given up on → count as reaped
        else:
            remaining.append(entry)
    if deleted:
        _save_pending(remaining)
    return (deleted, len(remaining))


def _best_effort_sweep_expired_pastes() -> None:
    """Pending-paste cleanup that never lets /debug fail offline."""
    with contextlib.suppress(Exception):
        _sweep_expired_pastes()


_PRIVACY_NOTICE = """\
⚠️  This will upload system info + logs to a PUBLIC paste service.

Cryptographic secrets (API keys, tokens, passwords) are redacted before
upload, but the following personal data is NOT redacted and will be public:
  • Your display name and persistent platform user ID
  • Verbatim content of your recent messages (prompts, responses, tool output)
  • Local filesystem paths
  • Any other PII present in the logs

The resulting URL is public to anyone who has the link. Pastes auto-delete
after 6 hours, but may be archived by third parties in the meantime.

Use --local to view the report without uploading.
"""

_GATEWAY_PRIVACY_NOTICE = (
    "⚠️ **Privacy notice:** This uploads system info + recent log tails "
    "(may contain conversation fragments) to a public paste service. "
    "Full logs are NOT included from the gateway — use `hermes debug share` "
    "from the CLI for full log uploads.\n"
    "Pastes auto-delete after 6 hours.")


def _extract_paste_id(url: str) -> Optional[str]:
    """Paste ID from a paste.rs URL (dpaste.com pastes have no deletable ID)."""
    url = url.strip().rstrip("/")
    for prefix in ("https://paste.rs/", "http://paste.rs/"):
        if url.startswith(prefix):
            return url[len(prefix):]
    return None


def delete_paste(url: str) -> bool:
    """Delete a paste.rs paste (the only service with unauthenticated DELETE). True on success."""
    paste_id = _extract_paste_id(url)
    if not paste_id:
        raise ValueError(f"Cannot delete: only paste.rs URLs are supported.  Got: {url}")
    req = urllib.request.Request(f"{_PASTE_RS_URL}{paste_id}", method="DELETE",
                                 headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return 200 <= resp.status < 300


def _schedule_auto_delete(urls: list[str], delay_seconds: int = _AUTO_DELETE_SECONDS):
    """Record paste.rs *urls* for deletion ``delay_seconds`` from now (merged into pending.json)."""
    paste_rs_urls = [u for u in urls if _extract_paste_id(u)]
    if not paste_rs_urls:
        return
    # Dedupe by URL, keeping the later expire_at.
    by_url: dict[str, float] = {e["url"]: float(e["expire_at"]) for e in _load_pending()}
    expire_at = time.time() + delay_seconds
    for u in paste_rs_urls:
        by_url[u] = max(expire_at, by_url.get(u, 0.0))
    _save_pending([{"url": u, "expire_at": ts} for u, ts in by_url.items()])


def _post_paste(service: str, endpoint: str, body: bytes, content_type: str) -> str:
    """POST *body* to a paste service and return the paste URL it echoes back."""
    req = urllib.request.Request(endpoint, data=body, method="POST",
                                 headers={"Content-Type": content_type, "User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        url = resp.read().decode("utf-8").strip()
    if not url.startswith("http"):
        raise ValueError(f"Unexpected response from {service}: {url[:200]}")
    return url


def _upload_paste_rs(content: str) -> str:
    return _post_paste("paste.rs", _PASTE_RS_URL, content.encode("utf-8"), "text/plain; charset=utf-8")


def _upload_dpaste_com(content: str, expiry_days: int = 7) -> str:
    boundary = "----HermesDebugBoundary9f3c"
    fields = (("content", content), ("syntax", "text"), ("expiry_days", str(expiry_days)))
    body = ("".join(f'--{boundary}\r\nContent-Disposition: form-data; name="{n}"\r\n\r\n{v}\r\n'
                    for n, v in fields) + f"--{boundary}--\r\n").encode("utf-8")
    return _post_paste("dpaste.com", _DPASTE_COM_URL, body,
                       f"multipart/form-data; boundary={boundary}")


def upload_to_pastebin(content: str, expiry_days: int = 7) -> str:
    """Upload *content* to a paste service, trying paste.rs then dpaste.com."""
    errors: list[str] = []
    for service, upload in (
        ("paste.rs", lambda: _upload_paste_rs(content)),
        ("dpaste.com", lambda: _upload_dpaste_com(content, expiry_days=expiry_days))):
        try:
            return upload()
        except Exception as exc:
            errors.append(f"{service}: {exc}")
    raise RuntimeError("Failed to upload to any paste service:\n  " + "\n  ".join(errors))


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


# Logs written by a client process, invisible to a remote/docker/SSH backend running `debug
# share`; a bare "(file not found)" would read as "the app logged nothing" and misdirect triage.
_CLIENT_SIDE_LOGS = {
    "desktop": (
        "written by Hermes Desktop on the machine running the app, not by this "
        "backend. If the desktop connects to a remote/docker/SSH backend, collect "
        "it on that client machine")}


def _missing_log_note(log_name: str) -> str:
    """Explain a missing log instead of stating a bare absence."""
    reason = _CLIENT_SIDE_LOGS.get(log_name)
    if reason is None:
        return "(file not found)"
    primary = _primary_log_path(log_name)
    return f"(not on this host: {reason}{f' — expected at {primary}' if primary else ''})"


def _resolve_log_path(log_name: str) -> Optional[Path]:
    """First non-empty candidate for *log_name* (primary, then the .1 rotation), or None."""
    primary = _primary_log_path(log_name)
    if primary is None:
        return None
    for candidate in (primary, primary.parent / f"{primary.name}.1"):
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def _redact_log_text(text: str) -> str:
    """``redact_sensitive_text(force=True)`` + email scrub — fires regardless of the operator's
    ``security.redact_secrets`` setting; only the in-memory upload copy is sanitized."""
    if not text:
        return text
    from agent.redact import redact_sensitive_text
    text = redact_sensitive_text(text, force=True)
    return _EMAIL_ADDRESS_RE.sub("[REDACTED_EMAIL]", text)


def _read_tail_bytes(
    log_path: Path, size: int, max_bytes: int, tail_lines: int) -> tuple[bytes, bool]:
    """Whole file, or (oversized) a backwards read holding ``max_bytes`` for the full upload AND
    enough newlines for the summary tail from the same snapshot → (raw, truncated)."""
    with open(log_path, "rb") as f:
        if size <= max_bytes:
            return f.read(), False
        chunk_size = 8192
        pos = size
        chunks: list[bytes] = []
        total = 0
        newline_count = 0
        while (pos > 0 and total < max_bytes * 2
               and (total < max_bytes or newline_count <= tail_lines + 1)):
            read_size = min(chunk_size, pos)
            pos -= read_size
            f.seek(pos)
            chunk = f.read(read_size)
            chunks.insert(0, chunk)
            total += len(chunk)
            newline_count += chunk.count(b"\n")
            chunk_size = min(chunk_size * 2, 65536)
        return b"".join(chunks), pos > 0


def _capture_log_snapshot(
    log_name: str, *, tail_lines: int, max_bytes: int = _MAX_LOG_BYTES, redact: bool = True,
) -> LogSnapshot:
    """Capture a log once and derive the summary tail and full-log views from that single read
    (a rotation between two reads would make the report look newer than the uploaded log)."""
    log_path = _resolve_log_path(log_name)
    if log_path is None:
        primary = _primary_log_path(log_name)
        tail = "(file empty)" if primary and primary.exists() else _missing_log_note(log_name)
        return LogSnapshot(path=None, tail_text=tail, full_text=None)

    try:
        size = log_path.stat().st_size
        if size == 0:  # truncated between _resolve_log_path and stat
            return LogSnapshot(path=log_path, tail_text="(file empty)", full_text=None)
        raw, truncated = _read_tail_bytes(log_path, size, max_bytes, tail_lines)
        full_raw = raw
        if truncated and len(full_raw) > max_bytes:
            cut = len(full_raw) - max_bytes
            # Drop a partial first line only when the cut lands genuinely mid-line.
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


# Logs the debug report tails, in output order. ``agent`` gets the full ``--lines`` budget;
# the rest are capped at 100 lines. Every log but ``errors`` is also uploaded in full.
_REPORT_LOGS = ("agent", "errors", "gateway", "gui", "desktop")
_FULL_LOGS = ("agent", "gateway", "gui", "desktop")


def _tail_budget(name: str, log_lines: int) -> int:
    return log_lines if name == "agent" else min(log_lines, 100)


def _capture_default_log_snapshots(
    log_lines: int, *, redact: bool = True) -> dict[str, LogSnapshot]:
    """Capture all logs used by debug-share exactly once."""
    return {
        name: _capture_log_snapshot(name, tail_lines=_tail_budget(name, log_lines), redact=redact)
        for name in _REPORT_LOGS}


def _capture_dump() -> str:
    """Run ``hermes dump`` and return its stdout as a string."""
    from hermes_cli.dump import run_dump
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture), contextlib.suppress(SystemExit):
        run_dump(SimpleNamespace(show_keys=False))
    return capture.getvalue()


def collect_debug_report(
    *, log_lines: int = 200, dump_text: str = "",
    log_snapshots: Optional[dict[str, LogSnapshot]] = None) -> str:
    """Build the summary debug report (system dump + log tails) as upload-ready text.

    ``dump_text`` is pre-captured dump output; when empty, ``hermes dump`` is run internally.
    """
    buf = io.StringIO()
    buf.write(dump_text or _capture_dump())
    if log_snapshots is None:
        log_snapshots = _capture_default_log_snapshots(log_lines)
    # In-process sanitiser heal counters: populated only inside a process that ran agent turns
    # (gateway /debug share); a fresh CLI's errors.log tail carries the same escalation lines.
    with contextlib.suppress(Exception):
        # See #96870.
        from agent.agent_runtime_helpers import get_sanitizer_heal_stats
        heal_stats = get_sanitizer_heal_stats()
        if heal_stats:
            buf.write("\n\n--- transcript sanitiser heal counters ---\n")
            for sess, st in sorted(heal_stats.items()):
                buf.write(f"session {sess}: {st['heal_events']} heal events, "
                          f"{st['messages_healed']} messages healed, escalated={st['escalated']}\n")
    buf.write("\n")
    for name in _REPORT_LOGS:
        buf.write(f"\n--- {name}.log (last {_tail_budget(name, log_lines)} lines) ---\n"
                  f"{log_snapshots[name].tail_text}\n")
    return buf.getvalue()


# Nous-S3 envelope format id; the discord-support viewer keys off it.
_NOUS_BUNDLE_FORMAT = "hermes-debug-share/1"


def collect_share_bundle(log_lines: int = 200, redact: bool = True) -> dict[str, str]:
    """Collect the debug report + full logs as a label→text mapping.

    The dump header is prepended to each full log so every file is self-contained, and the
    redaction banner is prepended when ``redact`` is True.
    """
    dump_text = _capture_dump()
    log_snapshots = _capture_default_log_snapshots(log_lines, redact=redact)
    report = collect_debug_report(log_lines=log_lines, dump_text=dump_text,
                                  log_snapshots=log_snapshots)
    banner = _REDACTION_BANNER if redact else ""
    bundle: dict[str, str] = {"report": banner + report}
    for name in _FULL_LOGS:
        if full := log_snapshots[name].full_text:
            bundle[f"{name}.log"] = banner + dump_text + f"\n\n--- full {name}.log ---\n" + full
    return bundle


def build_nous_bundle(bundle: dict[str, str], redact: bool = True) -> bytes:
    """Gzip a :func:`collect_share_bundle` mapping into the Nous envelope (shape parsed by the
    discord-support viewer — keep it stable)."""
    envelope = {"format": _NOUS_BUNDLE_FORMAT, "redacted": bool(redact),
                "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "files": bundle}
    return gzip.compress(json.dumps(envelope).encode("utf-8"))


@dataclass
class DebugShareResult:
    """Outcome of a ``debug share`` upload, so non-CLI callers can render real links."""
    urls: dict  # label -> paste URL (e.g. {"Report": "...", "agent.log": "..."})
    failures: list  # human-readable "label: error" strings for optional uploads
    redacted: bool  # whether force-mode redaction was applied before upload
    auto_delete_seconds: int  # how long until the pastes auto-delete
    report: str = ""  # the summary report text (kept for local fallback)


def build_debug_share(
    *, log_lines: int = 200, expiry: int = 7, redact: bool = True) -> DebugShareResult:
    """Collect the debug report + full logs, upload each, return the URLs.

    Shared by ``hermes debug share`` and the dashboard ``POST /api/ops/debug-share``. Blocking
    network I/O — callers inside an event loop must run it in a worker thread.
    """
    _best_effort_sweep_expired_pastes()
    bundle = collect_share_bundle(log_lines=log_lines, redact=redact)
    if redact:
        logger.info(
            "hermes debug share: applied force-mode redaction to log snapshots before upload")
    report = bundle["report"]
    failures: list[str] = []
    # The summary report is required (raises so callers can fall back); full logs are optional.
    urls = {"Report": upload_to_pastebin(report, expiry_days=expiry)}
    for label, content in bundle.items():
        if label == "report":
            continue
        try:
            urls[label] = upload_to_pastebin(content, expiry_days=expiry)
        except Exception as exc:
            failures.append(f"{label}: {exc}")
    _schedule_auto_delete(list(urls.values()))
    return DebugShareResult(urls=urls, failures=failures, redacted=redact,
                            auto_delete_seconds=_AUTO_DELETE_SECONDS, report=report)


def _confirm_upload(args) -> bool:
    """Gate the actual upload: ``--yes`` proceeds unprompted, else ask an interactive [y/N]."""
    if bool(getattr(args, "yes", False)):
        return True
    if not sys.stdin.isatty():
        print("ERROR: Non-interactive mode requires --yes to confirm upload.\n"
              "       This prevents accidental exposure of personal data.\n"
              "       Use --local to view the report without uploading.", file=sys.stderr)
        sys.exit(1)
    try:
        answer = input("Upload debug report? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = ""
    if answer in ("y", "yes"):
        return True
    print("Aborted.")
    return False


def run_debug_share(args):
    """Collect debug report + full logs, upload each, print URLs."""
    log_lines = getattr(args, "lines", 200)
    expiry = getattr(args, "expire", 7)
    redact = not getattr(args, "no_redact", False)

    if getattr(args, "local", False):
        # Same collector as the upload path so the output matches exactly; no network I/O.
        _best_effort_sweep_expired_pastes()
        print("Collecting debug report...")
        bundle = collect_share_bundle(log_lines=log_lines, redact=redact)
        print(bundle["report"])
        for label, body in bundle.items():
            if label != "report":
                print(f"\n\n{'=' * 60}\nFULL {label}\n{'=' * 60}\n\n{body}")
        return

    if getattr(args, "nous", False):
        return _run_debug_share_nous(args, log_lines=log_lines, redact=redact)
    print(_PRIVACY_NOTICE)
    if not _confirm_upload(args):
        return
    print("Collecting debug report...\nUploading...")
    try:
        result = build_debug_share(log_lines=log_lines, expiry=expiry, redact=redact)
    except RuntimeError as exc:
        print(f"\nUpload failed: {exc}", file=sys.stderr)
        print("\nRun `hermes debug share --local` to print the report instead.\n")
        sys.exit(1)
    label_width = max(len(k) for k in result.urls)
    print("\nDebug report uploaded:")
    for label, url in result.urls.items():
        print(f"  {label:<{label_width}}  {url}")
    if result.failures:
        print(f"\n  (failed to upload: {', '.join(result.failures)})")
    print(f"\n⏱  Pastes will auto-delete in {result.auto_delete_seconds // 3600} hours.\n"
          "To delete now:  hermes debug delete <url>\n"
          "\nShare these links with the Hermes team for support.")


_NOUS_PRIVACY_NOTICE = """\
⚠️  --nous: This uploads your debug bundle to Nous-INTERNAL storage (AWS S3),
    NOT a public paste service. The following is included:
  • System info (OS, Python/Hermes version, provider, which API keys are
    configured — NOT the actual keys)
  • Full agent.log, gateway.log, and desktop.log (up to 512 KB each — likely
    contains conversation content, tool outputs, and file paths)

  • The bundle is viewable only by Nous staff (and allowlisted Discord mods)
    via a Google-login-gated viewer.
  • It is NOT a public paste — there is no public URL to the contents.
  • It auto-deletes after 14 days.
"""


def _run_debug_share_nous(args, *, log_lines: int, redact: bool) -> None:
    """``hermes debug share --nous``: gzip the same bundle into the Nous envelope → Nous-S3."""
    from hermes_cli.diagnostics_upload import share_to_nous
    print(_NOUS_PRIVACY_NOTICE)
    if not _confirm_upload(args):
        return
    if not redact:
        print("⚠️  --no-redact is set: secrets in your logs will NOT be redacted before upload.\n")
    print("Collecting debug report...")
    _best_effort_sweep_expired_pastes()
    bundle = collect_share_bundle(log_lines=log_lines, redact=redact)
    if redact:
        logger.info("hermes debug share --nous: applied force-mode redaction before upload")
    print("Uploading to Nous diagnostics storage...")
    try:
        res = share_to_nous(build_nous_bundle(bundle, redact=redact))
    except Exception as exc:
        print(f"\nNous upload failed: {exc}\n"
              "\nThe Nous diagnostics service may be unavailable or not yet provisioned.\n"
              "Run `hermes debug share --local` to print the report instead, "
              "or `hermes debug share` to upload to a public paste service.\n", file=sys.stderr)
        sys.exit(1)
    view_url = res.get("viewUrl") or res.get("view_url")
    expires_at = res.get("expiresAt") or res.get("expires_at")
    print("\nDebug bundle uploaded to Nous (private):")
    print(f"  View URL  {view_url}" if view_url
          else f"  (no view URL returned; upload id: {res.get('id', '?')})")
    print(f"\n⏱  Auto-deletes at {expires_at} (14-day retention)." if expires_at
          else "\n⏱  Auto-deletes after 14 days.")
    print("\nShare this private link with the Nous team — only Nous staff "
          "(via Google login) can open it.\n"
          "\nPick up the discussion in:\n"
          "  GitHub Issues        https://github.com/NousResearch/hermes-agent/issues\n"
          "  Nous Portal Support  https://portal.nousresearch.com/help\n"
          "  Discord              https://discord.gg/NousResearch")


def run_debug_delete(args):
    """Delete one or more paste URLs uploaded by /debug."""
    urls = getattr(args, "urls", [])
    if not urls:
        print("Usage: hermes debug delete <url> [<url> ...]\n"
              "  Deletes paste.rs pastes uploaded by 'hermes debug share'.")
        return
    for url in urls:
        try:
            if delete_paste(url):
                print(f"  ✓ Deleted: {url}")
            else:
                print(f"  ✗ Failed to delete: {url} (unexpected response)")
        except ValueError as exc:
            print(f"  ✗ {exc}")
        except Exception as exc:
            print(f"  ✗ Could not delete {url}: {exc}")


def run_debug(args):
    """Route debug subcommands (sweeping expired pastes opportunistically on every call)."""
    _best_effort_sweep_expired_pastes()
    handler = {"share": run_debug_share, "delete": run_debug_delete}.get(
        getattr(args, "debug_command", None))
    if handler is None:
        print(_DEBUG_USAGE)
    else:
        handler(args)


_DEBUG_USAGE = """\
Usage: hermes debug <command>

Commands:
  share    Upload debug report to a paste service and print URL
  delete   Delete a previously uploaded paste

Options (share):
  --lines N    Number of log lines to include (default: 200)
  --expire N   Paste expiry in days (default: 7)
  --local      Print report locally instead of uploading
  --nous       Upload to Nous-internal storage (private, staff-only,
               auto-deletes in 14 days) instead of a public paste
  --no-redact  Disable upload-time secret redaction (default: redact)

Options (delete):
  <url> ...    One or more paste URLs to delete"""
