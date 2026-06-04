"""
drumbeat draft generation — turn top-velocity Slice A candidates into offline
Markdown drafts for Hafs to judge manually.

PIVOTED VERSION with:
- Source-grounding gate: Fetch article content before drafting
- Meta-preamble stripping: Remove "I don't have", "No browser available", etc.
- Quality-gate checks: Run deterministic checks from content-social.yaml
- Optional style support: --style argument for theme variations
- Approval-first: Only create drafts, never post directly

Raw drafting is text-only. The post-draft writer stage owns final-text,
audit, image brief/prompt creation, and optional image generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import textwrap
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from _db import bootstrap, connect
from theme_gen import DrumbeatError, load_style_entries

ROOT = Path("/home/ubuntu/.hermes/drumbeat")
PROMPTS_DIR = ROOT / "prompts"
THEME = PROMPTS_DIR / "theme.md"
THEME_META = PROMPTS_DIR / "theme.meta.json"
DRAFTS_DIR = ROOT / "drafts"
IMAGES_DIR = ROOT / "images"
PAUSED_SENTINEL = ROOT / ".paused"
BROADCAST = Path("/home/ubuntu/.hermes/scripts/cron-telegram-broadcast.sh")
QUALITY_GATES_YAML = Path("/home/ubuntu/.hermes/quality-gates/rubrics/content-social.yaml")
DEFAULT_K = 3
HERMES_TIMEOUT_SECONDS = 240
FETCH_TIMEOUT_SECONDS = 120
MAX_THEME_CHARS = 16000
MAX_SUMMARY_CHARS = 1800
MAX_SOURCE_CHARS = 8000

# Meta-preamble patterns to strip
META_PATTERNS = [
    r"^I don't have.*?(?:\n|$)",
    r"^I'll draft based on.*?(?:\n|$)",
    r"^Now I have the content.*?(?:\n|$)",
    r"^No browser available.*?(?:\n|$)",
    r"^Based on the (title|URL|summary).*?(?:\n|$)",
    r"^Since I (don't have|can't access).*?(?:\n|$)",
    r"^Let me draft.*?(?:\n|$)",
    r"^Here's a draft.*?(?:\n|$)",
    r"^I've drafted.*?(?:\n|$)",
]


@dataclass(frozen=True)
class Candidate:
    id: int
    source: str
    source_id: str
    url: str
    title: str
    summary: str
    raw_json: str
    fetched_at: int
    engagement_velocity: float
    points: int
    age_hours: float


@dataclass(frozen=True)
class QualityGateResult:
    passed: bool
    failures: list[str]


def ensure_skip_reason_column(conn) -> None:
    """Ensure skip_reason column exists in candidates table (idempotent)."""
    try:
        # Check if column exists
        cursor = conn.execute("PRAGMA table_info(candidates)")
        columns = [row[1] for row in cursor.fetchall()]
        if "skip_reason" not in columns:
            conn.execute("ALTER TABLE candidates ADD COLUMN skip_reason TEXT")
            conn.commit()
    except Exception as exc:
        # If ALTER fails, column might already exist or other issue - continue
        pass


def check_gates() -> tuple[bool, str]:
    """Check all prerequisites before generating draft."""
    # Gate 1: Not paused
    if PAUSED_SENTINEL.exists():
        return False, "drumbeat paused (sentinel file exists)"

    # Gate 2: Database integrity
    if not ROOT.exists():
        return False, "drumbeat root directory missing"

    # Gate 3: LLM CLI available (explicitly add ~/.local/bin to PATH for subprocess)
    env = os.environ.copy()
    env["PATH"] = f"/home/ubuntu/.local/bin:{env.get('PATH', '')}"
    try:
        subprocess.run(
            ["hermes", "--version"],
            capture_output=True,
            timeout=5,
            check=False,
            env=env
        )
    except FileNotFoundError:
        return False, "hermes CLI not found on PATH"
    except Exception as exc:
        return False, f"hermes CLI check failed: {exc}"

    # Gate 4: Theme prompt exists and is recent (not stale)
    if not THEME.exists():
        return False, "theme prompt missing (run theme_gen.py first)"

    try:
        theme_age_hours = (time.time() - THEME.stat().st_mtime) / 3600
        if theme_age_hours > 168:  # 7 days
            return False, f"theme prompt stale (age={theme_age_hours:.1f}h > 168h threshold)"
    except Exception:
        pass

    return True, ""


def validate_prereqs() -> str:
    # Fail on missing/empty voice prior before touching candidates or drafting.
    load_style_entries()
    if not THEME.exists() or THEME.stat().st_size < 200:
        raise DrumbeatError(
            f"Missing generated theme prompt at {THEME}. "
            "Next action: run `cd /home/ubuntu/.hermes/drumbeat/scripts && python3 theme_gen.py` after seeding style-refs.md."
        )
    theme = THEME.read_text(encoding="utf-8").strip()
    if not theme:
        raise DrumbeatError(f"Generated theme prompt is empty at {THEME}; rerun scripts/theme_gen.py.")
    return theme[:MAX_THEME_CHARS]


def prompt_version() -> str:
    if THEME_META.exists():
        try:
            meta = json.loads(THEME_META.read_text(encoding="utf-8"))
            version = meta.get("prompt_version")
            if version:
                return str(version)
        except Exception:
            pass
    digest = hashlib.sha1(THEME.read_bytes()).hexdigest()[:12] if THEME.exists() else "missing"
    return f"sha1:{digest}"


def pick_candidates(conn, k: int) -> list[Candidate]:
    rows = conn.execute(
        """SELECT id, source, source_id, url, title, summary, raw_json, fetched_at,
                  engagement_velocity, points, age_hours
           FROM candidates
           WHERE status = 'new'
           ORDER BY engagement_velocity DESC
           LIMIT ?""",
        (k,),
    ).fetchall()
    return [Candidate(*row) for row in rows]


def fetch_article_content(url: str) -> tuple[bool, str]:
    """
    Fetch article content using hermes with web toolset.
    Returns (success, content_or_error_reason).
    """
    # Use hermes with web tool to fetch article content
    prompt = f"Fetch and extract the main article text from this URL: {url}\nReturn only the article content, no commentary."
    cmd = [
        "hermes", "-z", prompt,
        "--model", "gemini-2.5-pro",
        "--provider", "google",
        "--accept-hooks",
        "--ignore-rules"
    ]

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=FETCH_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return False, f"fetch timeout after {FETCH_TIMEOUT_SECONDS}s"
    except FileNotFoundError:
        return False, "hermes CLI not found"
    except Exception as exc:
        return False, f"fetch error: {exc}"

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()[:500]
        return False, f"fetch failed: {stderr or 'unknown error'}"

    content = (proc.stdout or "").strip()

    # Check for common failure patterns
    if len(content) < 100:
        return False, "fetched content too short (< 100 chars)"

    if "error" in content.lower()[:200] and "fetch" in content.lower()[:200]:
        return False, "fetch returned error message"

    # Truncate to reasonable length
    content = content[:MAX_SOURCE_CHARS]
    return True, content


def build_draft_prompt(theme: str, candidate: Candidate, source_content: str, style: str | None = None) -> str:
    """
    Build draft prompt with source content and optional style parameter.
    """
    summary = candidate.summary.strip()[:MAX_SUMMARY_CHARS]

    style_instruction = ""
    if style and style.strip():
        style_instruction = f"\n\nSTYLE OVERRIDE: Apply the '{style}' style variation to this draft."

    return f"""You are Drumbeat drafting an offline LinkedIn/social post candidate for Hafs.

Use this Drumbeat theme prompt as the style contract:

--- THEME START ---
{theme}
--- THEME END ---
{style_instruction}

Candidate signal:
- Source: {candidate.source}
- Source ID: {candidate.source_id}
- Title: {candidate.title}
- URL: {candidate.url}
- Engagement velocity: {candidate.engagement_velocity:.2f}
- Points: {candidate.points}
- Age hours: {candidate.age_hours:.2f}
- Summary: {summary or '(none)'}

--- SOURCE CONTENT START ---
{source_content}
--- SOURCE CONTENT END ---

Write one post draft only.
Hard constraints:
- Output ONLY the post body, no YAML/front matter, no code fence, no analysis.
- Ground all claims in the SOURCE CONTENT above, not just the title or metadata.
- Do not claim Hafs personally tested, built, funded, or met anyone unless supported by the source content.
- Do not invent metrics, quotes, screenshots, images, or external facts.
- Do not include Telegram approval-card language, buttons, or image prompts.
- Keep it concise enough for LinkedIn: roughly 120-260 words unless the theme clearly calls for shorter.
- If the source is thin, frame it as a signal/question rather than overclaiming.
- DO NOT make any API calls or attempt to post to LinkedIn/social media directly. This is DRAFT ONLY.
- DO NOT include meta-commentary like "I don't have the article", "No browser available", or "Based on the title".
"""


def strip_meta_preamble(text: str) -> str:
    """
    Strip meta-preamble patterns from generated post text.
    Removes lines before "---" separator and common meta phrases.
    """
    # First, check for "---" separator and take content after it
    if "---" in text:
        parts = text.split("---", 1)
        if len(parts) > 1:
            text = parts[1].strip()

    # Apply regex patterns to remove meta-commentary
    for pattern in META_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)

    # Remove multiple consecutive blank lines
    text = re.sub(r"\n\n\n+", "\n\n", text)

    return text.strip()


def load_quality_gates() -> dict:
    """Load quality gates configuration from YAML."""
    if not QUALITY_GATES_YAML.exists():
        return {"criteria": []}

    try:
        import yaml
        content = QUALITY_GATES_YAML.read_text(encoding="utf-8")
        gates = yaml.safe_load(content)
        return gates or {"criteria": []}
    except ImportError:
        # pyyaml not installed - graceful fallback
        print("warning: pyyaml not available, skipping quality gate checks", file=sys.stderr)
        return {"criteria": []}
    except Exception as exc:
        print(f"warning: could not load quality gates: {exc}", file=sys.stderr)
        return {"criteria": []}


def check_quality_gates(post_text: str, candidate: Candidate) -> QualityGateResult:
    """
    Run deterministic quality gate checks from content-social.yaml.
    Returns QualityGateResult with pass/fail status and failure reasons.
    """
    gates = load_quality_gates()
    failures = []

    for criterion in gates.get("criteria", []):
        checks = criterion.get("checks", [])
        criterion_id = criterion.get("id", "unknown")

        for check in checks:
            if check.get("type") != "deterministic":
                continue

            pattern = check.get("pattern", "")
            fail_if_present = check.get("fail_if_present", False)
            reason = check.get("reason", "pattern check failed")

            if not pattern:
                continue

            # Check if pattern matches in post text
            match = re.search(pattern, post_text, re.IGNORECASE)

            if fail_if_present and match:
                failures.append(f"[{criterion_id}] {reason}")

    passed = len(failures) == 0
    return QualityGateResult(passed=passed, failures=failures)


def call_hermes(prompt: str) -> str:
    """Generate draft text using hermes CLI."""
    cmd = [
        "hermes", "-z", prompt,
        "--model", "gemini-2.5-pro",
        "--provider", "google",
        "--accept-hooks",
        "--ignore-rules"
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=HERMES_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as exc:
        raise DrumbeatError("Hermes CLI not found on PATH; cannot draft. Install/fix hermes, then rerun scripts/draft.py.") from exc
    except subprocess.TimeoutExpired as exc:
        raise DrumbeatError(f"Hermes draft generation timed out after {HERMES_TIMEOUT_SECONDS}s; rerun scripts/draft.py when the LLM path is healthy.") from exc

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()[:1200]
        stdout = (proc.stdout or "").strip()[:1200]
        detail = stderr or stdout or f"return code {proc.returncode}"
        raise DrumbeatError(f"Hermes draft generation failed: {detail}")

    output = (proc.stdout or "").strip()
    output = re.sub(r"^```(?:markdown|md|text)?\s*", "", output.strip(), flags=re.IGNORECASE)
    output = re.sub(r"\s*```$", "", output.strip())

    if len(output) < 80:
        raise DrumbeatError("Hermes returned an unexpectedly short draft; not writing a draft file.")

    return output.strip()


def draft_id_for(candidate: Candidate, generated_at: int, text: str) -> str:
    raw = f"{candidate.source_id}{generated_at}{text[:400]}"
    digest = hashlib.sha1(raw.encode()).hexdigest()[:16]
    return f"d_{candidate.id}_{generated_at}_{digest}"


def draft_markdown(draft_id: str, candidate: Candidate, post_text: str, prompt_version: str, generated_at: int) -> str:
    """Format draft as Markdown with YAML front matter."""
    summary = candidate.summary[:300] if candidate.summary else ""
    return f"""---
draft_id: {draft_id}
candidate_id: {candidate.id}
source: {candidate.source}
source_id: {candidate.source_id}
url: {candidate.url}
title: {candidate.title}
summary: {summary}
engagement_velocity: {candidate.engagement_velocity:.2f}
points: {candidate.points}
age_hours: {candidate.age_hours:.2f}
prompt_version: {prompt_version}
generated_at: {generated_at}
---

{post_text}
"""


def write_draft(conn, candidate: Candidate, post_text: str, version: str) -> tuple[Path, str, Path | None]:
    """Write draft file and database entry. Returns (path, draft_id, image_path)."""
    # Strip meta-preamble
    cleaned_text = strip_meta_preamble(post_text)

    # Check quality gates - BLOCKING
    gate_result = check_quality_gates(cleaned_text, candidate)
    if not gate_result.passed:
        failure_summary = "; ".join(gate_result.failures)
        raise DrumbeatError(f"Quality gate FAILED for candidate {candidate.id}: {failure_summary}")

    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = int(time.time())
    draft_id = draft_id_for(candidate, generated_at, cleaned_text)
    path = DRAFTS_DIR / f"{draft_id}.md"
    path.write_text(draft_markdown(draft_id, candidate, cleaned_text, version, generated_at), encoding="utf-8")

    image_path = None
    image_path_str = None

    conn.execute(
        """
        INSERT INTO drafts (id, candidate_id, post_text, raw_text, prompt_version, generated_at, status, image_path)
        VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
        """,
        (draft_id, candidate.id, cleaned_text, cleaned_text, version, generated_at, image_path_str),
    )
    return path, draft_id, image_path


def send_digest(paths: list[Path]) -> bool:
    """
    Send notification digest to Team Ops. This is NOT direct API posting to
    LinkedIn/social media — it's only internal notification via Telegram.
    """
    if not paths:
        return True
    if not BROADCAST.exists() or not BROADCAST.is_file():
        print(f"digest warning: broadcast wrapper missing at {BROADCAST}", file=sys.stderr)
        return False
    rels = [str(p.relative_to(ROOT)) for p in paths]
    message = f"✏️ drumbeat drafted {len(paths)} posts: " + ", ".join(rels)
    try:
        proc = subprocess.run(
            [str(BROADCAST), message],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        print(f"digest warning: broadcast subprocess error: {exc}", file=sys.stderr)
        return False
    if proc.returncode != 0:
        print(f"digest warning: broadcast failed rc={proc.returncode}: {(proc.stderr or proc.stdout)[:300]}", file=sys.stderr)
        return False
    return True


def log_run(run_id: str, started: int, status: str, notes: str) -> None:
    try:
        with connect() as conn:
            conn.execute(
                """INSERT INTO run_log (run_id, phase, started_at, finished_at, status, notes)
                   VALUES (?, 'draft', ?, ?, ?, ?)""",
                (run_id, started, int(time.time()), status, notes[:500]),
            )
            conn.commit()
    except Exception as exc:
        print(f"warning: could not write draft run_log: {exc}", file=sys.stderr)


def skip_candidate(conn, candidate: Candidate, reason: str) -> None:
    """Mark candidate as skipped with reason in database."""
    try:
        conn.execute(
            """UPDATE candidates SET status = 'skipped', skip_reason = ? WHERE id = ?""",
            (reason, candidate.id)
        )
        conn.commit()
    except Exception as exc:
        print(f"warning: could not mark candidate {candidate.id} as skipped: {exc}", file=sys.stderr)


def run(k: int = DEFAULT_K, send_file_digest: bool = False, style: str | None = None) -> tuple[list[Path], int]:
    run_id = uuid.uuid4().hex[:12]
    started = int(time.time())

    # Go/No-Go gate check
    should_proceed, skip_reason = check_gates()
    if not should_proceed:
        msg = f"[{run_id}] draft generation skipped: {skip_reason}"
        print(msg)
        log_run(run_id, started, "skip", skip_reason)
        return [], 0  # Exit 0 (success) — this is rollback/no-op, not a failure

    try:
        bootstrap()

        # Ensure skip_reason column exists (idempotent migration)
        with connect() as conn:
            ensure_skip_reason_column(conn)

        theme = validate_prereqs()
        version = prompt_version()
        with connect() as conn:
            candidates = pick_candidates(conn, k)
            if not candidates:
                print(f"[{run_id}] draft skipped: no eligible candidates")
                log_run(run_id, started, "skip", "no eligible candidates")
                return [], 0

            created: list[Path] = []
            for candidate in candidates:
                # SOURCE-GROUNDING GATE: Fetch article content first
                fetch_success, content_or_reason = fetch_article_content(candidate.url)

                if not fetch_success:
                    print(f"[{run_id}] skipping candidate {candidate.id}: {content_or_reason}")
                    skip_candidate(conn, candidate, f"source fetch failed: {content_or_reason}")
                    continue

                # Now we have source content, proceed with drafting
                source_content = content_or_reason
                try:
                    post_text = call_hermes(build_draft_prompt(theme, candidate, source_content, style))
                    path, draft_id, image_path = write_draft(conn, candidate, post_text, version)
                    created.append(path)
                    img_status = "ok" if image_path else "skipped"
                    print(f"[{run_id}] draft {draft_id}: text ok, image {img_status}")
                    # Mark candidate as seen after successful draft
                    conn.execute("UPDATE candidates SET status = 'seen' WHERE id = ?", (candidate.id,))
                except DrumbeatError as exc:
                    # Quality gate failure or other draft error - skip this candidate
                    print(f"[{run_id}] skipping candidate {candidate.id}: {exc}")
                    skip_candidate(conn, candidate, str(exc))
                    continue

                # Commit after every file so a later LLM failure does not orphan readable drafts.
                conn.commit()

        digest_ok = send_digest(created) if send_file_digest else True
        rels = ", ".join(str(p.relative_to(ROOT)) for p in created)
        status = "ok" if digest_ok else "error"
        style_note = f" style={style}" if style else ""
        log_run(run_id, started, status, f"created={len(created)} digest={'yes' if send_file_digest else 'no'} files={rels}{style_note}")
        print(f"[{run_id}] drafted {len(created)} posts: {rels}")
        if send_file_digest:
            print(f"[{run_id}] file digest {'sent' if digest_ok else 'failed'}")
        return created, 0 if digest_ok else 1
    except DrumbeatError as exc:
        print(f"draft error: {exc}", file=sys.stderr)
        log_run(run_id, started, "error", str(exc))
        return [], 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate offline drumbeat drafts from top candidate velocities")
    parser.add_argument("-k", "--count", type=int, default=DEFAULT_K, help=f"number of candidates to draft (default: {DEFAULT_K})")
    parser.add_argument("--send-digest", action="store_true", help="send only the file-path digest to Team Ops via cron-safe wrapper")
    parser.add_argument("--style", type=str, default=None, help="optional style override (e.g., deadpan_systems_parable)")
    args = parser.parse_args(argv)
    if args.count < 1:
        print("draft error: --count must be >= 1", file=sys.stderr)
        return 64
    _, rc = run(k=args.count, send_file_digest=args.send_digest, style=args.style)
    return rc


if __name__ == "__main__":
    sys.exit(main())
