#!/usr/bin/env python3
"""
Source-of-truth watcher for the Hermes fork checkout at HERMES_SRC_DIR.

Polls the cloned repo every WATCHER_POLL_SEC. When `git status --porcelain`
shows uncommitted work, starts a debounce timer; each subsequent change
extends it up to a hard ceiling. When the timer expires, runs:

    git add -A
    git commit -m "auto: <ts> — <files>"
    git push origin main

Respects .gitignore via git status. Never resets or destroys local work.

Why this exists: Hermes can autonomously create new skills and edit
existing ones (skill-authoring tools, agent-driven self-improvement).
Without a watcher, those changes live only on the persistent volume and
get out of sync with the GitHub fork — so the next image rebuild has no
record of them, and a volume reset wipes them entirely. The watcher
captures every disk change to source within a few minutes.

Heartbeat: writes a timestamp to HEARTBEAT_PATH every poll so a
separate health check can detect a stuck/crashed watcher.

Required env: nothing. Pushes use the credential helper configured by
the entrypoint (writes ~/.git-credentials from $GITHUB_TOKEN).

Ported from GordonClaw's source_watcher.py with path adjustments.
"""

import hashlib
import html
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

SRC_DIR = os.environ.get("HERMES_SRC_DIR", "/opt/data/repo")
HEARTBEAT_PATH = os.environ.get(
    "SOURCE_WATCHER_HEARTBEAT", "/opt/data/source-watcher-heartbeat"
)

WATCHER_POLL_SEC = int(os.environ.get("SOURCE_WATCHER_POLL_SEC", "10"))
WATCHER_DEBOUNCE_SEC = int(os.environ.get("SOURCE_WATCHER_DEBOUNCE_SEC", "180"))
WATCHER_CEILING_SEC = int(os.environ.get("SOURCE_WATCHER_CEILING_SEC", "600"))
WATCHER_PUSH_RETRIES = int(os.environ.get("SOURCE_WATCHER_PUSH_RETRIES", "3"))

COMMIT_SUBJECT_FILE_LIMIT = 6
BLOCKED_LOG_PATH = os.environ.get(
    "HERMES_WATCHER_BLOCKED_LOG", "/opt/data/logs/watcher-blocked.log"
)
SKIP_LINT = os.environ.get("HERMES_WATCHER_SKIP_LINT", "").lower() in (
    "1", "true", "yes", "on"
)

# Dedup state: when lint blocks the same files with the same ruff output
# on consecutive polls, only Telegram-notify once. Resets when the lint
# eventually passes (or files change). Prevents notification floods if a
# broken file sits in the working tree overnight.
_last_lint_block_signature: str | None = None


def _lint_block_signature(files: list[str], ruff_output: str) -> str:
    h = hashlib.sha256()
    h.update("\n".join(sorted(files)).encode("utf-8", errors="replace"))
    h.update(b"\n--\n")
    h.update(ruff_output.encode("utf-8", errors="replace"))
    return h.hexdigest()


# Telegram notification config — used to alert the operator when a push
# touches code that requires a restart (Bucket 2) or rebuild (Bucket 3).
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_HOME_CHANNEL = os.environ.get("TELEGRAM_HOME_CHANNEL", "").strip()


def log(msg: str) -> None:
    print(f"[source-watcher] {msg}", flush=True)


def heartbeat() -> None:
    try:
        with open(HEARTBEAT_PATH, "w") as f:
            f.write(datetime.now(tz=timezone.utc).isoformat())
    except OSError as err:
        log(f"heartbeat write failed: {err}")


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", SRC_DIR, *args],
        capture_output=True,
        text=True,
        check=check,
    )


def porcelain_status() -> list[str]:
    try:
        result = git("status", "--porcelain")
    except subprocess.CalledProcessError as err:
        log(f"git status failed: {err.stderr.strip()}")
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def changed_files(porcelain_lines: list[str]) -> list[str]:
    out = []
    for line in porcelain_lines:
        body = line[3:] if len(line) > 3 else line
        if " -> " in body:
            body = body.split(" -> ", 1)[1]
        out.append(body)
    return out


def make_commit_message(files: list[str]) -> str:
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    if len(files) <= COMMIT_SUBJECT_FILE_LIMIT:
        subject = f"auto: {ts} — {' '.join(files)}"
    else:
        head = " ".join(files[:COMMIT_SUBJECT_FILE_LIMIT])
        more = len(files) - COMMIT_SUBJECT_FILE_LIMIT
        subject = f"auto: {ts} — {head} (+{more} more)"
    body = "\n".join(files)
    return f"{subject}\n\nFiles:\n{body}\n"

def staged_python_files() -> list[str]:
    """Return staged Python files that would be included in the commit."""
    try:
        result = git(
            "diff",
            "--cached",
            "--name-only",
            "--diff-filter=ACMR",
            "--",
            "*.py",
        )
    except subprocess.CalledProcessError as err:
        log(f"git diff --cached failed: {err.stderr.strip()}")
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def _ruff_command() -> list[str] | None:
    """Find a usable ruff command across system and Hermes venv interpreters."""
    candidates = [
        [sys.executable, "-m", "ruff"],
        ["/opt/hermes/.venv/bin/python", "-m", "ruff"],
        ["/opt/data/repo/.venv/bin/python", "-m", "ruff"],
    ]
    exe = shutil.which("ruff")
    if exe:
        candidates.append([exe])

    for cmd in candidates:
        if cmd[0].startswith("/") and not os.path.exists(cmd[0]):
            continue
        probe = subprocess.run(
            [*cmd, "--version"],
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0:
            return cmd
    return None


def _safe_diff_for_log() -> str:
    try:
        return git("diff", "--cached", check=False).stdout
    except Exception as err:
        return f"<failed to capture staged diff: {err}>"


def append_blocked_lint_log(files: list[str], ruff_output: str, diff: str) -> None:
    """Append full lint failure context for a skipped watcher commit."""
    try:
        os.makedirs(os.path.dirname(BLOCKED_LOG_PATH), exist_ok=True)
        ts = datetime.now(tz=timezone.utc).isoformat()
        with open(BLOCKED_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write("=" * 80 + "\n")
            fh.write(f"timestamp: {ts}\n")
            fh.write("files:\n")
            for path in files:
                fh.write(f"  - {path}\n")
            fh.write("\nruff output:\n")
            fh.write(ruff_output.rstrip() + "\n")
            fh.write("\ndiff that would have been committed:\n")
            fh.write(diff.rstrip() + "\n")
    except Exception as err:
        log(f"failed to append blocked lint log: {err}")


def format_lint_blocked_notification(files: list[str], ruff_output: str) -> str:
    listed = files[:8]
    file_lines = "\n  ".join(f"<code>{html.escape(f)}</code>" for f in listed)
    if len(files) > len(listed):
        file_lines += f"\n  <i>(+{len(files) - len(listed)} more)</i>"
    summary = "ruff F821/syntax gate failed; commit was skipped"
    for line in ruff_output.splitlines():
        if line.strip() and ("F821" in line or "E999" in line):
            summary = line.strip()[:180]
            break
    return (
        "🚫 <b>Watcher blocked a broken-code commit</b>\n"
        f"{html.escape(summary)}\n\n"
        f"Files:\n  {file_lines}\n\n"
        f"Full details: <code>{html.escape(BLOCKED_LOG_PATH)}</code>"
    )


def lint_staged_python_files() -> bool:
    """Gate commits on ruff F821; ruff parser errors still catch syntax failures."""
    global _last_lint_block_signature
    files = staged_python_files()
    if not files:
        _last_lint_block_signature = None
        return True
    if SKIP_LINT:
        log(
            "HERMES_WATCHER_SKIP_LINT is set; skipping ruff F821 gate "
            f"for {len(files)} Python file(s)"
        )
        _last_lint_block_signature = None
        return True

    ruff = _ruff_command()
    if not ruff:
        output = "ruff is not installed or not available to source_watcher"
        sig = _lint_block_signature(files, output)
        if sig != _last_lint_block_signature:
            _last_lint_block_signature = sig
            diff = _safe_diff_for_log()
            append_blocked_lint_log(files, output, diff)
            notify_telegram(format_lint_blocked_notification(files, output))
            log("REFUSING to commit — ruff is unavailable")
        else:
            log("REFUSING to commit — ruff is unavailable (same as last cycle, suppressing duplicate notification)")
        return False

    # Ruff 0.15 removed E999 as a selectable rule. Syntax failures are still
    # emitted as parser errors ("invalid-syntax") even with only F821 selected.
    cmd = [*ruff, "check", "--select", "F821", "--no-fix", *files]
    result = subprocess.run(
        cmd,
        cwd=SRC_DIR,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        log(f"ruff F821 gate passed for {len(files)} Python file(s)")
        _last_lint_block_signature = None
        return True

    output = (result.stdout or "") + (result.stderr or "")
    sig = _lint_block_signature(files, output)
    if sig != _last_lint_block_signature:
        _last_lint_block_signature = sig
        diff = _safe_diff_for_log()
        append_blocked_lint_log(files, output, diff)
        notify_telegram(format_lint_blocked_notification(files, output))
        log("REFUSING to commit — ruff F821 gate failed")
    else:
        log("REFUSING to commit — ruff F821 gate failed (same as last cycle, suppressing duplicate notification)")
    # Unstage the failed files so the next poll cycle starts from a clean
    # index. Without this, a deletion of the broken file from the worktree
    # creates a staged-add + worktree-deletion mismatch that wedges the
    # watcher (porcelain ambiguous, git add -A reconciles to empty staging,
    # git commit then fails with "nothing to commit" forever).
    try:
        git("reset", "HEAD", "--", *files, check=False)
    except Exception as err:
        log(f"unstage of failed-lint files failed (continuing): {err}")
    return False



# ---------------------------------------------------------------------------
# Bucket classification + Telegram notification
# ---------------------------------------------------------------------------
# After a successful push, classify the changed files into action buckets
# and notify the operator via Telegram for Bucket 2 (restart needed) or
# Bucket 3 (rebuild needed). Bucket 1 changes (skills/docs/on-demand
# scripts) need no notification — they're already live via symlinks.
#
# The watcher does NOT autonomously trigger Railway. The operator reads
# the notification, then asks Hermes to invoke self-restart or
# self-rebuild — which keeps human approval on every cost-incurring or
# downtime-incurring action.

# File-path → bucket mapping. First match wins; Bucket 3 is checked
# before Bucket 2 (rebuild covers everything restart would cover).
_BUCKET_3_PREFIXES = (
    "Dockerfile",
    "docker/",
    "pyproject.toml",
    "package.json",
    "package-lock.json",
    "uv.lock",
    "requirements",   # requirements.txt, requirements-*.txt
    "MANIFEST.in",
    "setup.py",
    "setup.cfg",
)

_BUCKET_2_PREFIXES = (
    "tools/",
    "gateway/",
    "cron/",
    "scripts/source_watcher.py",
    "scripts/cost_report_daemon.py",
    "hermes_",          # hermes_state.py, hermes_constants.py, hermes_logging.py, ...
    "hermes_cli/",
    "cli.py",
    "run_agent.py",
    "mcp_serve.py",
)


def classify_bucket(files: list[str]) -> int:
    """Return the highest action bucket touched by the file list.

    3 = rebuild required (Dockerfile / deps)
    2 = restart required (gateway / tools / long-running daemons)
    1 = no action needed (skills / docs / on-demand scripts)
    """
    bucket = 1
    for f in files:
        for pat in _BUCKET_3_PREFIXES:
            if f == pat or f.startswith(pat):
                return 3
        for pat in _BUCKET_2_PREFIXES:
            if f == pat or f.startswith(pat):
                bucket = max(bucket, 2)
                break
    return bucket


def notify_telegram(message: str) -> None:
    """Send a message to TELEGRAM_HOME_CHANNEL. Defensive — never raises."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_HOME_CHANNEL:
        log("telegram not configured; skipping notification")
        return
    try:
        import json as _json  # noqa: F401  (kept for local tooling)
        import urllib.parse
        import urllib.request
        data = urllib.parse.urlencode({
            "chat_id": TELEGRAM_HOME_CHANNEL,
            "text": message,
            "parse_mode": "HTML",
        }).encode("utf-8")
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data=data,
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            r.read()
    except Exception as err:
        # Notification failure must not break the push loop.
        log(f"telegram notify failed: {err}")


def format_bucket_notification(bucket: int, files: list[str]) -> str:
    """Build the operator-facing Telegram message for a Bucket 2 or 3 push."""
    listed = files[:8]
    file_lines = "\n  ".join(f"<code>{f}</code>" for f in listed)
    if len(files) > len(listed):
        file_lines += f"\n  <i>(+{len(files) - len(listed)} more)</i>"
    if bucket == 2:
        return (
            "🔄 <b>Restart needed</b>\n"
            "Hermes edited code that requires a container restart "
            "(~30s, no rebuild cost):\n"
            f"  {file_lines}\n\n"
            "Tell me to restart and I'll invoke the self-restart skill."
        )
    if bucket == 3:
        return (
            "🔨 <b>Rebuild needed</b>\n"
            "Hermes edited image-baked code or dependencies — "
            "requires a Railway image rebuild (~$0.11, ~10 min):\n"
            f"  {file_lines}\n\n"
            "Tell me to rebuild and I'll invoke the self-rebuild skill."
        )
    return ""


_SENSITIVE_NAME_RE = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD)", re.IGNORECASE)
_MIN_SECRET_LEN = 16  # ignore short env values to avoid false positives


def scan_for_secret_leaks(files: list[str]) -> list[tuple[str, str]]:
    """Refuse-to-commit defense.

    Walks the changed-file list and checks whether any file's content
    contains the literal value of an env var whose name suggests a
    secret (KEY/TOKEN/SECRET/PASSWORD). Catches the failure mode where
    the bot writes a credential into a doc/skill file by mistake — the
    case that just exposed all our keys publicly via origin/main.

    Returns a list of (file_path, env_var_name) tuples. Empty = clean.
    """
    sensitive = [
        (k, v.strip())
        for k, v in os.environ.items()
        if _SENSITIVE_NAME_RE.search(k) and len(v.strip()) >= _MIN_SECRET_LEN
    ]
    if not sensitive:
        return []
    leaks: list[tuple[str, str]] = []
    for rel in files:
        path = os.path.join(SRC_DIR, rel)
        if not os.path.isfile(path):
            continue  # deleted or directory
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                content = fh.read()
        except OSError:
            continue
        for name, val in sensitive:
            if val and val in content:
                leaks.append((rel, name))
    return leaks


def commit_and_push() -> bool:
    files = changed_files(porcelain_status())
    if not files:
        log("commit_and_push called but tree is clean; skipping")
        return True

    # Defensive index reset: clear any leftover staging from a prior cycle
    # whose lint failed (or any other reason commit_and_push exited early).
    # Without this, a once-blocked file stays staged forever, and a
    # later worktree deletion creates a staged-add + worktree-deletion
    # mismatch that wedges the watcher entirely. git reset (no args)
    # is index-only — it does not touch the working tree, so we don't
    # lose any unsynced edits. Idempotent and cheap.
    try:
        git("reset", check=False)
    except Exception as err:
        log(f"defensive git reset failed (continuing): {err}")

    leaks = scan_for_secret_leaks(files)
    if leaks:
        leak_lines = "\n".join(
            f"  <code>{path}</code> contains <code>${name}</code>"
            for path, name in leaks[:8]
        )
        more = ""
        if len(leaks) > 8:
            more = f"\n  <i>(+{len(leaks) - 8} more)</i>"
        log(
            f"REFUSING to commit — {len(leaks)} secret leak(s) detected: "
            + ", ".join(f"{p}:${n}" for p, n in leaks[:5])
        )
        notify_telegram(
            "🚫 <b>Watcher refused to commit — secret leak detected</b>\n"
            f"{leak_lines}{more}\n\n"
            "Fix the file(s) so they no longer contain the literal "
            "secret value, then the watcher will retry on the next poll. "
            "If the secret has already been committed in earlier history, "
            "rotate it immediately."
        )
        return False

    try:
        git("add", "-A")
    except subprocess.CalledProcessError as err:
        log(f"git add failed: {err.stderr.strip()}")
        return False

    if not lint_staged_python_files():
        return False

    msg = make_commit_message(files)
    try:
        git("commit", "--no-verify", "-m", msg)
    except subprocess.CalledProcessError as err:
        log(f"git commit failed: {err.stderr.strip()}")
        return False

    for attempt in range(1, WATCHER_PUSH_RETRIES + 1):
        try:
            git("push", "origin", "main")
            log(f"committed + pushed {len(files)} file(s)")
            # Classify and notify the operator if the push touched
            # Bucket 2 (restart needed) or Bucket 3 (rebuild needed).
            try:
                bucket = classify_bucket(files)
                if bucket >= 2:
                    msg = format_bucket_notification(bucket, files)
                    if msg:
                        notify_telegram(msg)
                        log(f"notified operator: bucket {bucket} change")
                else:
                    log("bucket 1 change — no notification")
            except Exception as exc:
                log(f"bucket-notify hook crashed (non-fatal): {exc}")
            return True
        except subprocess.CalledProcessError as err:
            stderr = (err.stderr or "").strip()
            log(
                f"git push attempt {attempt}/{WATCHER_PUSH_RETRIES} failed: {stderr}"
            )
            if "non-fast-forward" in stderr or "rejected" in stderr:
                log("non-fast-forward; attempting pull --rebase origin main")
                try:
                    git("fetch", "origin", "main")
                    git("rebase", "origin/main")
                    log("rebase succeeded; retrying push")
                    continue
                except subprocess.CalledProcessError as rebase_err:
                    log(
                        f"rebase failed ({rebase_err.stderr.strip()}); "
                        f"aborting and bailing"
                    )
                    try:
                        git("rebase", "--abort", check=False)
                    except Exception:
                        pass
                    return False
            if attempt < WATCHER_PUSH_RETRIES:
                time.sleep(5 * attempt)
    log("git push gave up; commit is local, will push next time tree is dirty")
    return False


def commit_now() -> bool:
    return commit_and_push()


def main() -> None:
    if not os.path.isdir(os.path.join(SRC_DIR, ".git")):
        log(f"FATAL: {SRC_DIR} is not a git repo; exiting")
        sys.exit(1)

    log(
        f"started; src={SRC_DIR} poll={WATCHER_POLL_SEC}s "
        f"debounce={WATCHER_DEBOUNCE_SEC}s ceiling={WATCHER_CEILING_SEC}s"
    )

    first_dirty_at: float | None = None
    last_change_at: float | None = None
    last_porcelain: list[str] = []

    while True:
        try:
            heartbeat()
            porcelain = porcelain_status()
            now = time.monotonic()

            if not porcelain:
                if first_dirty_at is not None:
                    log("tree clean again; clearing debounce")
                first_dirty_at = None
                last_change_at = None
                last_porcelain = []
            else:
                if first_dirty_at is None:
                    first_dirty_at = now
                    last_change_at = now
                    last_porcelain = porcelain
                    log(f"detected {len(porcelain)} change(s); debounce started")
                elif porcelain != last_porcelain:
                    last_change_at = now
                    last_porcelain = porcelain
                    log(f"changes still in flux ({len(porcelain)} files); debounce extended")

                quiet_for = now - (last_change_at or now)
                age = now - first_dirty_at
                if quiet_for >= WATCHER_DEBOUNCE_SEC or age >= WATCHER_CEILING_SEC:
                    if age >= WATCHER_CEILING_SEC and quiet_for < WATCHER_DEBOUNCE_SEC:
                        log(
                            f"hit {WATCHER_CEILING_SEC}s ceiling with edits "
                            f"still arriving; committing anyway"
                        )
                    commit_and_push()
                    first_dirty_at = None
                    last_change_at = None
                    last_porcelain = []

            time.sleep(WATCHER_POLL_SEC)
        except Exception:
            log(f"watcher loop error:\n{traceback.format_exc()}")
            time.sleep(WATCHER_POLL_SEC)


if __name__ == "__main__":
    main()
