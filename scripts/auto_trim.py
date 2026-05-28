#!/usr/bin/env python3
"""
auto_trim.py — Production context compression engine for Hermes Agent.

Replaces the broken auto-trim.sh (which had a fatal heredoc-in-substitution bug).

Reads context blocks from bridge/signals/context-status.json, trims them to fit
within a configurable token budget using a priority-based strategy:

  T0 (identity)     — Never touched
  T1 (task)         — Never touched
  T2 (high-import)  — Never touched
  T3 (semantic)     — Compress if over budget
  T4 (background)   — Compress if over budget
  T5 (tool-output)  — Delete first
  T6 (conversation) — Delete first

Archives trimmed blocks to logs/archive/ before discarding.
Writes results to bridge/signals/responses/ for downstream consumers.

Trim-suppression (file-based signals for standalone CLI operation):
  - bridge/signals/pause-trim          — When present, all trimming is suspended.
  - bridge/signals/protected-blocks.json — List of block IDs to skip during trimming.

Usage:
    python3 auto_trim.py                  # Auto-detect via trigger file
    python3 auto_trim.py --dry-run        # Analyse only, don't modify anything
    python3 auto_trim.py --target 80000   # Override token budget
    python3 auto_trim.py --model qwen3:8b # Override compression model
    python3 auto_trim.py --validate       # Validate inputs and exit
    python3 auto_trim.py --pause          # Pause trimming (create signal file)
    python3 auto_trim.py --resume         # Resume trimming (remove signal file)
    python3 auto_trim.py --protect BLOCK_ID [BLOCK_ID ...]   # Protect block(s)
    python3 auto_trim.py --unprotect BLOCK_ID [BLOCK_ID ...] # Unprotect block(s)

Environment variables (or defaults):
    OLLAMA_HOST        http://localhost:11434
    TRIM_MODEL         qwen3:8b
    TRIM_THRESHOLD     100000  (tokens — triggers trim when exceeded)
    TARGET_TOKENS      60000   (tokens — target after trim)
    WORKSPACE          <auto-detected>
    DRY_RUN            0 or 1  (force dry-run mode)
    MAX_PAUSE_SECONDS  3600    (auto-resume after this duration, 0 = disabled)
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# ─── Logging ──────────────────────────────────────────────────────────────────

LOG_FMT = "[auto_trim] %(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(format=LOG_FMT, level=logging.INFO, stream=sys.stderr)
log = logging.getLogger("auto_trim")

# ─── Configuration ────────────────────────────────────────────────────────────

def _resolve_base_dir() -> Path:
    """Resolve the pipeline root directory dynamically.

    Supports multiple layouts:
      - scripts/ layout:  auto_trim.py in scripts/,  run_bridge.py in pipeline root
      - flat layout:      auto_trim.py at pipeline root (e.g. Linux production)
      - linux_prod/ or linux_production/: deployment mirrors
      - WORKSPACE env var override

    Walks upward from the script's directory looking for run_bridge.py,
    then checks known deployment directories, then falls back to env.
    """
    script_dir = Path(__file__).resolve().parent

    # Walk upward from script_dir (handles nested dirs like scripts/,
    # linux_prod/, linux_production/, etc.)
    current = script_dir
    for _ in range(8):  # reasonable upward limit
        if (current / "run_bridge.py").exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent

    # Check known deployment mirror directories under ~/.hermes
    home_hermes = Path.home() / ".hermes"
    for mirror in ("linux_prod", "linux_production"):
        candidate = home_hermes / mirror
        if candidate.is_dir() and (candidate / "run_bridge.py").exists():
            return candidate

    # Fallback: WORKSPACE env var
    ws = os.environ.get("WORKSPACE", "")
    if ws and Path(ws).is_dir():
        return Path(ws)

    # Last resort: script_dir itself (tests may override module vars)
    return script_dir


BASE_DIR = _resolve_base_dir()
WORKSPACE = Path(os.environ.get("WORKSPACE", str(BASE_DIR)))
BRIDGE_DIR = WORKSPACE / "bridge"
SIGNALS_DIR = BRIDGE_DIR / "signals"
ARCHIVE_DIR = WORKSPACE / "logs" / "archive"

# Ensure directories exist at import time (safe: no-op if present)
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
SIGNALS_DIR.mkdir(parents=True, exist_ok=True)

# Ollama settings
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
TARGET_MODEL = os.environ.get("TRIM_MODEL", "qwen3:8b")

# Token budget — the threshold that *triggers* trimming, and the *target* to reach
TRIM_THRESHOLD_TOKENS = int(os.environ.get("TRIM_THRESHOLD", "80000"))
TARGET_TOKENS = int(os.environ.get("TARGET_TOKENS", "40000"))

# Safety floor — never delete below this many blocks regardless of budget
MIN_BLOCKS_KEPT = 3

# Auto-resume ceiling — trim suppression cannot last longer than this (0 = disabled)
MAX_PAUSE_SECONDS = int(os.environ.get("MAX_PAUSE_SECONDS", "3600"))

# Runtime mode flags (set by CLI or env)
DRY_RUN = bool(int(os.environ.get("DRY_RUN", "0")))
if "--dry-run" in sys.argv:
    DRY_RUN = True

# Signal file paths
PAUSE_SIGNAL = SIGNALS_DIR / "pause-trim"
PROTECTED_SIGNAL = SIGNALS_DIR / "protected-blocks.json"


# ─── Pause / Protect signal I/O ──────────────────────────────────────────────

def _is_trimming_paused() -> bool:
    """Check if the pause-trim signal file exists.

    Side-effect-free: does NOT auto-resume. Call auto_resume_if_expired()
    separately when you want expired pauses to be lifted.
    """
    return PAUSE_SIGNAL.exists()


def auto_resume_if_expired() -> bool:
    """If a pause signal exists and has expired its ceiling, remove it.

    Returns True if the pause was lifted (or was already absent).
    When MAX_PAUSE_SECONDS == 0 (no ceiling), any pause is immediately
    considered expired and is removed.
    """
    if not PAUSE_SIGNAL.exists():
        return True
    if MAX_PAUSE_SECONDS == 0:
        resume_trimming(reason="auto-resume: MAX_PAUSE_SECONDS=0 (no ceiling)")
        return True
    try:
        age = time.time() - PAUSE_SIGNAL.stat().st_mtime
        if age > MAX_PAUSE_SECONDS:
            log.warning(
                "Pause signal is %ds old (max %ds) — auto-resuming",
                int(age), MAX_PAUSE_SECONDS,
            )
            resume_trimming(reason="auto-resume: MAX_PAUSE_SECONDS exceeded")
            return True
    except OSError:
        pass
    return False


def _get_protected_blocks() -> set[str]:
    """Read the set of protected block IDs from signal file."""
    if not PROTECTED_SIGNAL.exists():
        return set()
    try:
        data = json.loads(PROTECTED_SIGNAL.read_text())
        if isinstance(data, dict) and "protected" in data:
            return set(data["protected"])
        if isinstance(data, list):
            return set(data)
        return set()
    except (json.JSONDecodeError, OSError) as e:
        log.warning("Corrupt protected-blocks.json: %s", e)
        return set()


def pause_trimming(reason: str = "") -> dict:
    """Create the pause-trim signal file to suspend all trimming."""
    PAUSE_SIGNAL.parent.mkdir(parents=True, exist_ok=True)
    payload = {"reason": reason or "manual", "paused_at": datetime.now().isoformat()}
    PAUSE_SIGNAL.write_text(json.dumps(payload, indent=2))
    log.info("Trimming paused (reason: %s)", reason or "manual")
    return {"status": "paused", "reason": reason or "manual"}


def resume_trimming(reason: str = "") -> dict:
    """Remove the pause-trim signal file to resume trimming."""
    if PAUSE_SIGNAL.exists():
        PAUSE_SIGNAL.unlink()
        log.info("Trimming resumed (reason: %s)", reason or "manual")
    return {"status": "resumed", "reason": reason or "manual"}


def set_block_protected(block_id: str, protected: bool = True) -> dict:
    """Add or remove a block ID from the protected set."""
    current = _get_protected_blocks()
    if protected:
        current.add(block_id)
    else:
        current.discard(block_id)

    PROTECTED_SIGNAL.parent.mkdir(parents=True, exist_ok=True)
    PROTECTED_SIGNAL.write_text(
        json.dumps({"protected": sorted(current)}, indent=2, default=str)
    )
    log.info("Block %s: protected=%s", block_id, protected)
    return {"block_id": block_id, "protected": protected}


# ─── Ollama API ──────────────────────────────────────────────────────────────

def query_ollama(model: str, prompt: str, max_tokens: int = 2048) -> Optional[str]:
    """
    Send a prompt to Ollama and return the response text.
    Returns None on any error (caller must handle).
    """
    import requests  # lazy import — not required at module level

    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0.3},
    }
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response", "").strip()
        if not text:
            log.warning("Ollama returned empty response for model=%s", model)
            return None
        return text
    except requests.exceptions.ConnectionError:
        log.error("Cannot connect to Ollama at %s", OLLAMA_HOST)
        return None
    except requests.exceptions.Timeout:
        log.error("Ollama request timed out after %ds", 120)
        return None
    except Exception as e:
        log.error("Ollama query failed: %s", e)
        return None


def count_tokens(text: str) -> int:
    """
    Rough token count suitable for budget enforcement.
    Uses max(word_count, char_count/4) to handle CJK and code.
    """
    if not text:
        return 0
    word_count = len(text.split())
    char_estimate = len(text) // 4
    return max(word_count, char_estimate)


# ─── Compression logic ───────────────────────────────────────────────────────

COMPRESSION_PROMPT = """\
You are a context compression engine. Given a conversation summary,
produce a dense 2-3 sentence summary that preserves ALL facts, decisions,
and action items. Do not add interpretation. Return ONLY the compressed
summary, no preamble, no tags.

Input: {text}
Compressed summary:"""


def compress_block(text: str, model: str) -> Optional[dict]:
    """
    Compress a single text block via Ollama.
    Returns dict with result or None on failure.
    """
    if not text or len(text.strip()) < 20:
        return None

    # Truncate input to 4000 chars to avoid overwhelming the model
    truncated = text[:4000]
    # Use .replace() instead of .format() to avoid crashes if
    # content contains literal curly braces (e.g. JSX, dicts)
    prompt = COMPRESSION_PROMPT.replace("{text}", truncated)
    result = query_ollama(model, prompt)

    if result is None:
        return {"status": "error", "error": "empty or failed response from model"}

    orig_tokens = count_tokens(text)
    comp_tokens = count_tokens(result)
    saved = max(0, orig_tokens - comp_tokens)

    return {
        "status": "ok",
        "original_tokens": orig_tokens,
        "compressed_text": result,
        "compressed_tokens": comp_tokens,
        "saved_tokens": saved,
        "ratio": round(comp_tokens / max(orig_tokens, 1), 3),
    }


def _archive_block(block: dict) -> Optional[Path]:
    """
    Archive a block to disk before deletion. Returns archive path or None.
    """
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    block_id = block.get("id", "unknown")
    # Sanitize filename
    safe_id = "".join(c if c.isalnum() or c in "_-" else "_" for c in str(block_id))
    archive_path = ARCHIVE_DIR / f"trimmed_{ts}_{safe_id}.json"
    try:
        archive_path.write_text(
            json.dumps(block, indent=2, default=str, ensure_ascii=False)
        )
        return archive_path
    except OSError as e:
        log.error("Failed to archive block %s: %s", block_id, e)
        return None


def _block_priority(block: dict) -> int:
    """Extract priority from block (0=highest, 6=lowest)."""
    pri = block.get("priority", 6)
    if isinstance(pri, int):
        return pri
    if isinstance(pri, float):
        return int(pri)
    return 6


def _response_base(**overrides) -> dict:
    """Build a response dict with sensible defaults, merged with call-site overrides.

    Returns the full 16-field response schema.  Each caller only needs to
    pass the fields that differ from defaults — the rest are filled in
    automatically.  This eliminates duplication across the four return paths
    in trim_context().
    """
    base = {
        "status": "ok",
        "action": "none",
        "reason": "",
        "tokens_before": 0,
        "tokens_after": 0,
        "tokens_saved": 0,
        "blocks_deleted": 0,
        "blocks_compressed": 0,
        "blocks_remaining": 0,
        "remaining_blocks": [],
        "compression_ratio": 0.0,
        "overage_resolved": True,
        "remaining_overage_tokens": 0,
        "dry_run": DRY_RUN,
        "paused": _is_trimming_paused(),
        "protected_blocks": sorted(_get_protected_blocks()),
    }
    base.update(overrides)
    return base


# ─── Core trim logic ─────────────────────────────────────────────────────────

def trim_context(
    context_blocks: list[dict],
    budget: int = TARGET_TOKENS,
    threshold: int = TRIM_THRESHOLD_TOKENS,
) -> dict:
    """
    Trim a list of context blocks to fit within a token budget.

    Strategy (two-phase):
      Phase 1 — Evict: Delete T5/T6 blocks (tool output, raw conversation)
                until budget is met or no more evictable blocks.
      Phase 2 — Compress: Compress T3/T4 blocks (semantic, background)
                until budget is met or no more compressible blocks.

    Never touches T0 (identity), T1 (task), T2 (high-import).
    Never reduces below MIN_BLOCKS_KEPT blocks.
    Protected blocks (from signal file) are skipped regardless of tier.

    Args:
        context_blocks: List of dicts, each with 'id', 'content', 'priority'
        budget: Target token count after trimming
        threshold: Only trim if total exceeds this

    Returns:
        Dict with trim statistics and updated blocks list.
    """
    if not context_blocks:
        return _response_base(reason="empty context")

    # ── Check pause signal ────────────────────────────────────────
    auto_resume_if_expired()
    if _is_trimming_paused():
        log.info("Trimming is paused — skipping")
        tokens_now = sum(count_tokens(b.get("content", "")) for b in context_blocks)
        return _response_base(
            status="paused",
            reason="trimming is paused (see bridge/signals/pause-trim)",
            tokens_before=tokens_now,
            tokens_after=tokens_now,
            blocks_remaining=len(context_blocks),
            remaining_blocks=context_blocks,
            compression_ratio=1.0,
            paused=True,
        )

    # Load protected block IDs from signal file
    protected_ids = _get_protected_blocks()

    # Validate schema of each block (work on copies to avoid mutating caller's dicts)
    blocks = [dict(b) for b in context_blocks]
    for i, block in enumerate(blocks):
        if "content" not in block:
            log.warning("Block index %d missing 'content' key — treating as 0 tokens", i)
        if "id" not in block:
            block["id"] = f"block_{i}"
        if "priority" not in block:
            block["priority"] = 6  # default: low priority

    total_tokens = sum(
        count_tokens(b.get("content", "")) for b in blocks
    )

    # Check if trimming is needed at all
    if total_tokens <= threshold:
        return _response_base(
            reason=f"total ({total_tokens}) below threshold ({threshold})",
            tokens_before=total_tokens,
            tokens_after=total_tokens,
            blocks_remaining=len(context_blocks),
            remaining_blocks=context_blocks,
            compression_ratio=1.0,
        )

    overage = total_tokens - budget
    deleted = 0
    compressed = 0
    saved = 0
    remaining: list[dict] = []

    # ── Phase 1: Evict low-priority blocks (T5, T6) ──
    total_blocks = len(context_blocks)

    for i, block in enumerate(context_blocks):
        priority = _block_priority(block)
        block_tokens = count_tokens(block.get("content", ""))
        block_id = block.get("id", "")

        # Skip protected blocks regardless of tier
        if block_id in protected_ids:
            log.debug("Skipping protected block %s (pri=%d)", block_id, priority)
            remaining.append(block)
            continue

        if priority >= 5 and overage > 0:
            # MIN_BLOCKS_KEPT safety: never delete if it would leave us
            # below the floor.  Count blocks that will still exist after
            # this deletion (already-kept + every unprocessed block).
            blocks_after = len(remaining) + (total_blocks - i - 1)
            if blocks_after < MIN_BLOCKS_KEPT:
                log.debug(
                    "MIN_BLOCKS_KEPT floor (%d) — keeping block %s (pri=%d)",
                    MIN_BLOCKS_KEPT, block_id, priority,
                )
                remaining.append(block)
                continue

            # Archive before deleting
            if DRY_RUN:
                log.info(
                    "[DRY RUN] Would evict block %s (pri=%d, %d tokens)",
                    block_id, priority, block_tokens,
                )
                remaining.append(block)  # keep in output for dry-run
            else:
                archive_path = _archive_block(block)
                if archive_path:
                    log.info(
                        "Evicted block %s (pri=%d, %d tokens) → %s",
                        block_id, priority, block_tokens, archive_path,
                    )
                deleted += 1
                saved += block_tokens
                overage -= block_tokens
            continue

        remaining.append(block)

    # ── Phase 2: Compress medium-priority blocks (T3, T4) ──
    if overage > 0 and not DRY_RUN:
        for block in remaining:
            priority = _block_priority(block)
            block_id = block.get("id", "")

            # Skip protected blocks
            if block_id in protected_ids:
                continue

            if priority in (3, 4) and overage > 0 and not block.get("_was_compressed"):
                result = compress_block(block["content"], TARGET_MODEL)
                if result and result.get("status") == "ok":
                    compressed += 1
                    token_saved = result["saved_tokens"]
                    saved += token_saved
                    overage -= token_saved
                    block["content"] = result["compressed_text"]
                    block["compressed"] = True
                    block["compressed_at"] = datetime.now().isoformat()
                    block["original_tokens"] = result["original_tokens"]
                    block["compressed_tokens"] = result["compressed_tokens"]
                    log.info(
                        "Compressed block %s (pri=%d): %d → %d tokens (saved %d)",
                        block_id, priority,
                        result["original_tokens"], result["compressed_tokens"],
                        token_saved,
                    )

    # ── Safety: ensure we keep at least MIN_BLOCKS_KEPT ──
    actual_kept = len(remaining)

    tokens_after = total_tokens - saved
    resolved = overage <= 0
    reason_text = (
        "within budget" if resolved
        else f"overage of {overage} tokens remains after trimming"
    )

    if not resolved:
        log.warning(
            "Could not fully resolve overage: %d tokens still over budget "
            "(deleted=%d, compressed=%d, saved=%d)",
            overage, deleted, compressed, saved,
        )

    return _response_base(
        status="ok" if resolved else "partial",
        action="trimmed" if (deleted + compressed) > 0 else "none",
        reason=reason_text,
        tokens_before=total_tokens,
        tokens_after=tokens_after,
        tokens_saved=saved,
        blocks_deleted=deleted,
        blocks_compressed=compressed,
        blocks_remaining=actual_kept,
        remaining_blocks=remaining,
        compression_ratio=round(tokens_after / max(total_tokens, 1), 3),
        overage_resolved=resolved,
        remaining_overage_tokens=max(0, overage),
        protected_blocks=sorted(protected_ids),
    )


# ─── IPC signal handling ─────────────────────────────────────────────────────

def write_signal(filename: str, data: dict) -> Path:
    """Write a signal file for the bridge pipeline to consume."""
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    path = SIGNALS_DIR / filename
    path.write_text(
        json.dumps(data, indent=2, default=str, ensure_ascii=False)
    )
    log.info("Wrote signal: %s", path)
    return path


def read_latest_telegram() -> dict:
    """Read the latest Telegram message from bridge signal file."""
    path = BRIDGE_DIR / "latest-from-telegram.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Cannot read telegram signal %s: %s", path, e)
            return {}
    return {}


def read_context_status() -> Optional[dict]:
    """Read the current context status from bridge signals."""
    path = SIGNALS_DIR / "context-status.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        log.error("Corrupt context-status.json: %s", e)
        return None


# ─── Validation ───────────────────────────────────────────────────────────────

def validate_inputs() -> tuple[bool, list[str]]:
    """
    Validate all inputs before a trim run.
    Returns (ok, list_of_errors).
    """
    errors: list[str] = []

    # Check Ollama connectivity
    try:
        import requests
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if resp.status_code != 200:
            errors.append(f"Ollama returned HTTP {resp.status_code}")
        else:
            models = resp.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            if TARGET_MODEL not in model_names:
                errors.append(
                    f"Trim model '{TARGET_MODEL}' not found in Ollama. "
                    f"Available: {', '.join(model_names)}"
                )
    except Exception as e:
        errors.append(f"Cannot reach Ollama at {OLLAMA_HOST}: {e}")

    # Check workspace directories
    for d, label in [
        (BRIDGE_DIR, "bridge/"),
        (SIGNALS_DIR, "bridge/signals/"),
        (ARCHIVE_DIR, "logs/archive/"),
    ]:
        if not d.exists():
            errors.append(f"Missing directory: {d} ({label})")

    # Check context-status.json exists
    ctx = read_context_status()
    if ctx is None:
        errors.append("No readable context-status.json found")
    elif "blocks" not in ctx:
        errors.append("context-status.json missing 'blocks' key")

    # Check trim model isn't an absurd size for trimming
    if TARGET_TOKENS > TRIM_THRESHOLD_TOKENS:
        errors.append(
            f"TARGET_TOKENS ({TARGET_TOKENS}) > TRIM_THRESHOLD_TOKENS "
            f"({TRIM_THRESHOLD_TOKENS}) — budget is nonsensical"
        )

    return (len(errors) == 0, errors)


# ─── CLI signal parsers ──────────────────────────────────────────────────────

def parse_trigger_signal() -> tuple[str, int]:
    """
    Check for a trigger signal file and return (mode, target_tokens).
    If no signal file exists, returns defaults.
    """
    trigger = SIGNALS_DIR / "trigger-trim.json"
    if not trigger.exists():
        return ("auto", TARGET_TOKENS)

    try:
        signal = json.loads(trigger.read_text())
        trigger.unlink()  # consume the signal
        mode = signal.get("mode", "auto")
        target = int(signal.get("target_tokens", TARGET_TOKENS))
        log.info("Trigger signal received: mode=%s, target=%d", mode, target)
        return (mode, max(target, 1000))  # floor of 1000 tokens
    except (json.JSONDecodeError, OSError, ValueError) as e:
        log.error("Bad trigger signal: %s", e)
        return ("auto", TARGET_TOKENS)


def handle_cli_pause_resume() -> bool:
    """
    Handle --pause, --resume, --protect, --unprotect CLI flags.
    Returns True if the operation was handled (and main() should exit after).
    """
    args = sys.argv[1:]

    if "--pause" in args:
        reason = ""
        idx = args.index("--pause")
        if idx + 1 < len(args) and not args[idx + 1].startswith("--"):
            reason = args[idx + 1]
        result = pause_trimming(reason=reason)
        print(f"✅ Trimming paused: {result['reason']}")
        print(f"   Signal file: {PAUSE_SIGNAL}")
        print(f"   Auto-resume after: {MAX_PAUSE_SECONDS}s" if MAX_PAUSE_SECONDS else "   Auto-resume: disabled")
        return True

    if "--resume" in args:
        result = resume_trimming(reason="manual resume")
        print(f"✅ Trimming resumed")
        print(f"   Signal file removed: {PAUSE_SIGNAL}")
        return True

    if "--protect" in args:
        idx = args.index("--protect")
        block_ids = []
        i = idx + 1
        while i < len(args) and not args[i].startswith("--"):
            block_ids.append(args[i])
            i += 1
        if not block_ids:
            print("❌ --protect requires at least one block ID")
            return True
        for bid in block_ids:
            result = set_block_protected(bid, protected=True)
            print(f"✅ Protected: {bid}")
        print(f"   Signal file: {PROTECTED_SIGNAL}")
        return True

    if "--unprotect" in args:
        idx = args.index("--unprotect")
        block_ids = []
        i = idx + 1
        while i < len(args) and not args[i].startswith("--"):
            block_ids.append(args[i])
            i += 1
        if not block_ids:
            print("❌ --unprotect requires at least one block ID")
            return True
        for bid in block_ids:
            result = set_block_protected(bid, protected=False)
            print(f"🔓 Unprotected: {bid}")
        return True

    if "--pause-status" in args:
        paused = _is_trimming_paused()
        protected = _get_protected_blocks()
        print(f"Trimming paused: {paused}")
        if paused and PAUSE_SIGNAL.exists():
            age = time.time() - PAUSE_SIGNAL.stat().st_mtime
            print(f"  Signal age: {int(age)}s (max: {MAX_PAUSE_SECONDS}s)")
            if MAX_PAUSE_SECONDS > 0:
                print(f"  Auto-resume in: {max(0, MAX_PAUSE_SECONDS - int(age))}s")
        print(f"Protected blocks: {sorted(protected) if protected else '(none)'}")
        return True

    return False


# ─── Main entry point ────────────────────────────────────────────────────────

def main() -> int:
    """Main entry point. Returns 0 on success, 1 on failure."""

    # Handle pause/resume/protect CLI flags first
    if handle_cli_pause_resume():
        return 0

    # CLI overrides
    if "--validate" in sys.argv:
        ok, errors = validate_inputs()
        paused = _is_trimming_paused()
        protected = _get_protected_blocks()

        print("=" * 50)
        print("  CONTEXT TRIMMER — VALIDATION")
        print("=" * 50)
        if ok:
            print("✅ All inputs valid")
        else:
            print("❌ Validation failed:")
            for err in errors:
                print(f"   • {err}")

        print(f"\n  OLLAMA_HOST:       {OLLAMA_HOST}")
        print(f"  TRIM_MODEL:        {TARGET_MODEL}")
        print(f"  TRIM_THRESHOLD:    {TRIM_THRESHOLD_TOKENS} tokens")
        print(f"  TARGET_TOKENS:     {TARGET_TOKENS} tokens")
        print(f"  DRY_RUN:           {DRY_RUN}")
        print(f"  WORKSPACE:         {WORKSPACE}")
        print(f"  ARCHIVE_DIR:       {ARCHIVE_DIR}")
        print(f"  MAX_PAUSE_SECONDS: {MAX_PAUSE_SECONDS}")
        print(f"  Trimming paused:   {paused}")
        if paused:
            print(f"  Pause signal:      {PAUSE_SIGNAL}")
        print(f"  Protected blocks:  {sorted(protected) if protected else '(none)'}")
        print(f"  Protected signal:  {PROTECTED_SIGNAL}")

        return 0 if ok else 1

    log.info("Starting context trimmer — dry_run=%s", DRY_RUN)

    # Parse trigger signal
    mode, target = parse_trigger_signal()
    log.info("Mode: %s, target budget: %d tokens", mode, target)

    # Read context
    ctx = read_context_status()
    if ctx is None:
        log.info("No context-status.json found — nothing to trim")
        return 0

    blocks = ctx.get("blocks", [])
    if not blocks:
        log.info("Context has no blocks — nothing to trim")
        return 0

    log.info(
        "Processing %d blocks, budget=%d tokens, mode=%s",
        len(blocks), target, mode,
    )

    # Run trim (pause/expiry handled inside trim_context)
    result = trim_context(blocks, budget=target, threshold=TRIM_THRESHOLD_TOKENS)

    # Determine response path
    responses_dir = SIGNALS_DIR / "responses"
    responses_dir.mkdir(parents=True, exist_ok=True)
    response_file = responses_dir / f"trim_{int(time.time())}.json"
    response_file.write_text(
        json.dumps(result, indent=2, default=str, ensure_ascii=False)
    )
    log.info("Result written to %s", response_file)

    # Log summary
    log.info(
        "Trim complete: saved=%d tokens, deleted=%d blocks, "
        "compressed=%d blocks, ratio=%.3f, resolved=%s, paused=%s",
        result.get("tokens_saved", 0),
        result.get("blocks_deleted", 0),
        result.get("blocks_compressed", 0),
        result.get("compression_ratio", 0),
        result.get("overage_resolved", False),
        result.get("paused", False),
    )

    # Write to Memory Palace if it exists
    palace = WORKSPACE / "wiki" / "MEMORY-PALACE.md"
    if palace.exists():
        try:
            entry = (
                f"\n---\n"
                f"### 📦 Auto-Trim — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"> Mode: {mode} | Saved: {result.get('tokens_saved', 0)} tokens | "
                f"Deleted: {result.get('blocks_deleted')} | "
                f"Compressed: {result.get('blocks_compressed')} | "
                f"Status: {result.get('status')} | "
                f"Paused: {result.get('paused', False)} | "
                f"Protected: {len(result.get('protected_blocks', []))} blocks\n"
            )
            with open(palace, "a", encoding="utf-8") as f:
                f.write(entry)
        except OSError as e:
            log.warning("Could not write to Memory Palace: %s", e)

    log.info("Done.")
    return 0 if result.get("overage_resolved", True) else 1


if __name__ == "__main__":
    sys.exit(main())