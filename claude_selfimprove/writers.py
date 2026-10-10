"""Safe writes to Claude Code's global configuration files.

Every write in this module goes through the same sequence:

1. **Protected-content check** — refuse to touch anything the pipeline does
   not own. CLAUDE.md is never fully owned (only its managed block is);
   rule and skill files must carry the pipeline's ownership marker before
   this module will overwrite them, and a missing or damaged marker stops
   the write rather than guessing.
2. **Format validation** — the freshly rendered content must look like
   what it claims to be (balanced markers, well-formed frontmatter) before
   it touches disk.
3. **Secret scan** — a last-chance check for anything credential-shaped,
   independent of the redaction already applied upstream (defense in
   depth, see ``redact.find_secret_leftovers``).
4. **Backup** — the target's exact prior state (or "did not exist") is
   recorded before any write, so :func:`rollback_backup` can always
   restore it.
5. **Atomic write** — temp file + fsync + rename; a crash mid-write never
   leaves a half-written target file.
6. **Post-write verification** — the file is re-read and re-validated
   immediately after writing. If that fails, the write rolls itself back
   automatically before returning failure — the target is never left in a
   state this module didn't intend.
7. **Audit** — every attempt (success, refusal, or rollback) is logged.

CLAUDE.md itself is a human-authored file this pipeline shares with
George. Only the text between two marker comments belongs to the pipeline;
everything else in the file is preserved byte-for-byte. Rule and skill
files are either fully pipeline-owned (created by this pipeline, carrying
its ownership marker) or fully off-limits (anything else) — there is no
partial-ownership case for those.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import atomic, audit, paths, redact

CLAUDE_MD_BEGIN_MARKER = (
    "<!-- BEGIN claude-selfimprove MANAGED SECTION "
    "(auto-generated — edit via the pipeline, not by hand) -->"
)
CLAUDE_MD_END_MARKER = "<!-- END claude-selfimprove MANAGED SECTION -->"

RULE_OWNERSHIP_MARKER = "<!-- managed-by: claude-selfimprove-pipeline -->"

SKILL_MANAGED_BY_LINE = "  managed_by: claude-selfimprove-pipeline"


@dataclass(frozen=True)
class WriteResult:
    success: bool
    path: Path | None
    reason: str
    backup_meta_path: Path | None = None
    rolled_back: bool = False


# --- Backups ------------------------------------------------------------


def _backup_meta_path_for(target: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    slug = str(target).replace("/", "_").strip("_")
    return paths.backups_dir() / f"{stamp}__{slug}.json"


def backup_before_write(target: Path, *, reason: str) -> Path:
    """Snapshot *target*'s current state (content, or "did not exist").

    Returns the path to the backup metadata file, which :func:`rollback_backup`
    consumes. Raises ``OSError`` if the disk itself is unwritable — callers
    inside this module (``_finalize_write``) turn that into a refused write
    rather than proceeding without a safety net.
    """
    existed = target.exists()
    content_before = target.read_text(encoding="utf-8") if existed else None
    meta = {
        "target_path": str(target),
        "existed_before": existed,
        "content_before": content_before,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reason": reason,
    }
    meta_path = _backup_meta_path_for(target)
    atomic.atomic_write_json(meta_path, meta)
    return meta_path


def rollback_backup(backup_meta_path: Path) -> WriteResult:
    """Restore a target file to the state recorded in *backup_meta_path*."""
    try:
        meta = json.loads(Path(backup_meta_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return WriteResult(False, None, f"unreadable backup metadata: {exc}")

    target = Path(meta["target_path"])
    try:
        if meta["existed_before"]:
            atomic.atomic_write_text(target, meta["content_before"])
        elif target.exists():
            target.unlink()
    except OSError as exc:
        return WriteResult(False, target, f"rollback failed: {exc}")

    audit.record("rollback", target=str(target), backup=str(backup_meta_path))
    return WriteResult(True, target, "rolled back", rolled_back=True)


def list_backups() -> list[dict]:
    """Every backup, most recent first.

    Sorted by filename, not by the ``timestamp`` field: the filename embeds
    a microsecond-precision creation stamp (see ``_backup_meta_path_for``),
    while ``timestamp`` is second-precision — two backups made in the same
    second would otherwise tie and fall back to an arbitrary order.
    """
    entries = []
    for f in sorted(paths.backups_dir().glob("*.json"), reverse=True):
        try:
            meta = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        meta["_backup_path"] = str(f)
        entries.append(meta)
    return entries


def latest_backup_for(target: Path) -> dict | None:
    for meta in list_backups():
        if meta.get("target_path") == str(target):
            return meta
    return None


# --- Shared write sequence -------------------------------------------------


def _finalize_write(
    target: Path, content: str, *, reason: str, validate_fn
) -> WriteResult:
    leftovers = redact.find_secret_leftovers(content)
    if leftovers:
        audit.record("write_refused_secret_scan", target=str(target), patterns=leftovers)
        return WriteResult(False, target, f"secret scan found: {', '.join(leftovers)}")

    ok, format_reason = validate_fn(content)
    if not ok:
        audit.record("write_refused_format_invalid", target=str(target), reason=format_reason)
        return WriteResult(False, target, f"format validation failed: {format_reason}")

    try:
        backup_meta_path = backup_before_write(target, reason=reason)
    except OSError as exc:
        audit.record("write_refused_backup_failed", target=str(target), error=str(exc))
        return WriteResult(False, target, f"backup failed, refusing to write: {exc}")

    try:
        atomic.atomic_write_text(target, content, mode=0o644)
    except OSError as exc:
        audit.record("write_failed", target=str(target), error=str(exc))
        return WriteResult(False, target, f"write failed: {exc}", backup_meta_path)

    # Post-write verification: re-read and re-validate. A mismatch here
    # means something wrote a different file than intended (e.g. a
    # concurrent process) — roll back rather than leave it.
    try:
        on_disk = target.read_text(encoding="utf-8")
    except OSError as exc:
        rollback_backup(backup_meta_path)
        audit.record("write_rolled_back_unreadable", target=str(target), error=str(exc))
        return WriteResult(False, target, f"post-write read failed: {exc}", backup_meta_path, True)

    ok, verify_reason = validate_fn(on_disk)
    if not ok or on_disk != content:
        rollback_backup(backup_meta_path)
        audit.record("write_rolled_back_verify_failed", target=str(target), reason=verify_reason)
        return WriteResult(False, target, f"post-write verification failed: {verify_reason}", backup_meta_path, True)

    audit.record("write_succeeded", target=str(target), reason=reason)
    return WriteResult(True, target, "written", backup_meta_path)


# --- CLAUDE.md managed block -------------------------------------------------


def _validate_claude_md(content: str) -> tuple[bool, str]:
    begin_count = content.count(CLAUDE_MD_BEGIN_MARKER)
    end_count = content.count(CLAUDE_MD_END_MARKER)
    if begin_count != end_count:
        return False, f"unbalanced markers ({begin_count} begin, {end_count} end)"
    if begin_count > 1:
        return False, "more than one managed block"
    if begin_count == 1:
        begin_idx = content.index(CLAUDE_MD_BEGIN_MARKER)
        end_idx = content.index(CLAUDE_MD_END_MARKER)
        if end_idx < begin_idx:
            return False, "end marker precedes begin marker"
    return True, ""


def render_managed_block(entries: list[dict], *, max_chars: int = 1500) -> str:
    """Render the full managed block (with markers) from scratch.

    *entries* are ``{"title": ..., "body": ...}`` dicts, already
    redacted. Regenerated wholesale every write rather than patched
    incrementally — far simpler to keep correct, and the whole point of a
    marker-delimited block is that it is safe to fully replace.

    Entries are kept in the order given until the running total would
    exceed *max_chars*; anything past that point is dropped rather than
    truncating an entry's text mid-sentence (defense in depth — routing
    should already have kept the caller from handing this more than fits).
    """
    lines = [CLAUDE_MD_BEGIN_MARKER, ""]
    used = 0
    for entry in entries:
        title = str(entry.get("title") or "").strip()
        body = str(entry.get("body") or "").strip()
        block_text = f"- **{title}**: {body}" if title else f"- {body}"
        if used + len(block_text) > max_chars:
            break
        lines.append(block_text)
        used += len(block_text)
    lines.append("")
    lines.append(CLAUDE_MD_END_MARKER)
    return "\n".join(lines)


def write_claude_md_block(entries: list[dict]) -> WriteResult:
    """Regenerate the managed block in ``~/.claude/CLAUDE.md`` from *entries*.

    Refuses (does not write) when the file has exactly one of the two
    markers — that means the block was damaged by something other than
    this pipeline, and guessing how to repair it risks destroying
    George's own content around it.
    """
    target = paths.claude_md_path()
    current = target.read_text(encoding="utf-8") if target.exists() else ""

    begin_count = current.count(CLAUDE_MD_BEGIN_MARKER)
    end_count = current.count(CLAUDE_MD_END_MARKER)
    if begin_count != end_count or begin_count > 1:
        audit.record(
            "write_refused_protected_content",
            target=str(target),
            reason=f"damaged markers ({begin_count} begin, {end_count} end)",
        )
        return WriteResult(
            False, target,
            f"CLAUDE.md managed markers are damaged ({begin_count} begin, {end_count} end) — refusing to write",
        )

    new_block = render_managed_block(entries)

    if begin_count == 1:
        begin_idx = current.index(CLAUDE_MD_BEGIN_MARKER)
        end_idx = current.index(CLAUDE_MD_END_MARKER) + len(CLAUDE_MD_END_MARKER)
        new_content = current[:begin_idx] + new_block + current[end_idx:]
    else:
        separator = "\n\n" if current and not current.endswith("\n\n") else ""
        if current and not current.endswith("\n"):
            separator = "\n" + separator
        new_content = current + separator + new_block + "\n"

    return _finalize_write(
        target, new_content, reason="claude_md_block update", validate_fn=_validate_claude_md
    )


def read_managed_block_body() -> str:
    """The current managed block's inner content, or "" if absent."""
    target = paths.claude_md_path()
    if not target.exists():
        return ""
    content = target.read_text(encoding="utf-8")
    if CLAUDE_MD_BEGIN_MARKER not in content or CLAUDE_MD_END_MARKER not in content:
        return ""
    start = content.index(CLAUDE_MD_BEGIN_MARKER) + len(CLAUDE_MD_BEGIN_MARKER)
    end = content.index(CLAUDE_MD_END_MARKER)
    return content[start:end].strip()


# --- Rule files -----------------------------------------------------------


def _validate_rule_content(content: str) -> tuple[bool, str]:
    if not content.strip():
        return False, "empty content"
    if not content.startswith(RULE_OWNERSHIP_MARKER):
        return False, "missing ownership marker"
    return True, ""


def _is_pipeline_owned_rule(path: Path) -> bool:
    if not path.exists():
        return True  # nothing there yet — safe to create
    try:
        head = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return head.startswith(RULE_OWNERSHIP_MARKER)


def write_rule_file(path: Path, *, title: str, body: str) -> WriteResult:
    if not _is_pipeline_owned_rule(path):
        audit.record("write_refused_protected_content", target=str(path), reason="not pipeline-owned")
        return WriteResult(False, path, "refusing to overwrite a rule file the pipeline does not own")

    content = f"{RULE_OWNERSHIP_MARKER}\n\n# {title.strip()}\n\n{body.strip()}\n"
    return _finalize_write(path, content, reason="rule update", validate_fn=_validate_rule_content)


# --- Skill files ------------------------------------------------------------


def _validate_skill_content(content: str) -> tuple[bool, str]:
    if not content.startswith("---\n"):
        return False, "missing frontmatter opening delimiter"
    rest = content[4:]
    if "\n---\n" not in rest and not rest.rstrip().endswith("---"):
        return False, "missing frontmatter closing delimiter"
    if "name:" not in content.split("---", 2)[1]:
        return False, "missing name field"
    if "description:" not in content.split("---", 2)[1]:
        return False, "missing description field"
    if SKILL_MANAGED_BY_LINE.strip() not in content:
        return False, "missing ownership marker"
    return True, ""


def _is_pipeline_owned_skill(path: Path) -> bool:
    if not path.exists():
        return True
    try:
        head = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return SKILL_MANAGED_BY_LINE.strip() in head


def render_skill_content(*, slug: str, title: str, body: str, canonical_key: str) -> str:
    description = redact.truncate(body, 150)
    return (
        "---\n"
        f"name: {slug}\n"
        f"description: {description}\n"
        "metadata:\n"
        f"{SKILL_MANAGED_BY_LINE}\n"
        f"  canonical_key: {canonical_key}\n"
        "---\n\n"
        f"# {title.strip()}\n\n"
        f"{body.strip()}\n"
    )


def write_skill_file(path: Path, *, slug: str, title: str, body: str, canonical_key: str) -> WriteResult:
    if not _is_pipeline_owned_skill(path):
        audit.record("write_refused_protected_content", target=str(path), reason="not pipeline-owned")
        return WriteResult(False, path, "refusing to overwrite a skill file the pipeline does not own")

    content = render_skill_content(slug=slug, title=title, body=body, canonical_key=canonical_key)
    return _finalize_write(path, content, reason="skill update", validate_fn=_validate_skill_content)
