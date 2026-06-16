"""Shared handoff-document command helpers for the CLI.

This module extends the existing CLI ``/handoff`` surface with a document-based
workflow while preserving the older platform-transfer behavior handled in
``CLICommandsMixin._handle_handoff_command``.

Supported document modes:

  /handoff inline [mission...]
  /handoff save [path] [mission...]
  /handoff consume <path>

The implementation is deliberately deterministic for the MVP: it builds a
compact markdown handoff from the current session's local context without
calling the model or introducing a new core tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
from typing import Any


_HANDOFF_SECTIONS = (
    "Purpose of next session",
    "Current status",
    "Relevant artifacts",
    "Constraints and non-goals",
    "Exact first prompt",
    "Success criteria",
)


@dataclass
class HandoffDocument:
    markdown: str
    suggested_filename: str
    next_mode: str


@dataclass
class HandoffCommandResult:
    text: str
    saved_path: str | None = None
    agent_seed: str | None = None


def _strip_leading_command(cmd: str) -> str:
    text = (cmd or "").strip()
    if not text:
        return ""
    if text.startswith("/"):
        parts = text.split(None, 1)
        return parts[1].strip() if len(parts) > 1 else ""
    return text


def _split_handoff_tokens(cmd: str) -> list[str]:
    raw = _strip_leading_command(cmd)
    if raw.lower().startswith("handoff"):
        raw = raw[len("handoff"):].lstrip()
    if not raw:
        return []
    try:
        return shlex.split(raw)
    except ValueError:
        return raw.split()


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", (text or "").strip().lower())
    slug = re.sub(r"-+", "-", slug).strip("-._")
    return slug or "session-handoff"


def _looks_like_path(token: str) -> bool:
    if not token:
        return False
    if token.startswith(("/", "./", "../", "~/")):
        return True
    if "/" in token:
        return True
    if token.lower().endswith((".md", ".markdown", ".txt")):
        return True
    return False


def parse_handoff_args(cmd: str) -> tuple[str, str | None, str | None]:
    """Parse `/handoff` document-mode args.

    Returns `(mode, path, mission)` for the document subcommands only.
    Raises ValueError for malformed document-mode invocations.
    """

    tokens = _split_handoff_tokens(cmd)
    if not tokens:
        raise ValueError("Usage: /handoff [inline|save|consume] ... or /handoff <platform>")

    mode = tokens[0].lower()
    rest = tokens[1:]

    if mode == "inline":
        mission = " ".join(rest).strip() or None
        return mode, None, mission

    if mode == "save":
        path = None
        mission_tokens = rest
        if rest and _looks_like_path(rest[0]):
            path = rest[0]
            mission_tokens = rest[1:]
        mission = " ".join(mission_tokens).strip() or None
        return mode, path, mission

    if mode == "consume":
        if not rest:
            raise ValueError("Usage: /handoff consume <path>")
        path = rest[0]
        return mode, path, None

    raise ValueError(f"Unsupported handoff document mode: {mode}")


def default_handoff_path(hermes_home: Path, slug: str, now: datetime | None = None) -> Path:
    now = now or datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%d-%H%M")
    return hermes_home / "sessions" / "handoffs" / f"handoff-{stamp}-{slug}.md"


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, dict):
                txt = part.get("text") or part.get("content") or ""
                if isinstance(txt, str) and txt.strip():
                    chunks.append(txt.strip())
            elif isinstance(part, str) and part.strip():
                chunks.append(part.strip())
        return "\n".join(chunks).strip()
    if content is None:
        return ""
    return str(content).strip()


def _extract_recent_facts(conversation_history: list[dict[str, Any]], limit: int = 3) -> list[str]:
    facts: list[str] = []
    for msg in reversed(conversation_history or []):
        role = msg.get("role") or "message"
        if role not in {"user", "assistant"}:
            continue
        text = _content_to_text(msg.get("content"))
        if not text:
            continue
        text = re.sub(r"\s+", " ", text)
        facts.append(f"{role}: {text[:160]}{'...' if len(text) > 160 else ''}")
        if len(facts) >= limit:
            break
    return list(reversed(facts))


def _extract_artifacts(conversation_history: list[dict[str, Any]], workdir: str | None, session_id: str | None) -> dict[str, list[str] | str | None]:
    text = "\n".join(_content_to_text(m.get("content")) for m in conversation_history or [])
    abs_paths = sorted(set(re.findall(r"(?:/[^\s`'\"]+)+", text)))[:5]
    urls = sorted(set(re.findall(r"https?://[^\s)\]>]+", text)))[:5]
    return {
        "workdir": workdir or "",
        "files": abs_paths,
        "sessions": [session_id] if session_id else [],
        "services": urls,
        "other": [],
    }


def _last_user_message(conversation_history: list[dict[str, Any]]) -> str:
    for msg in reversed(conversation_history or []):
        if msg.get("role") == "user":
            text = _content_to_text(msg.get("content"))
            if text:
                return text
    return "continue this work"


def build_handoff_document(
    *,
    mission: str | None,
    conversation_history: list[dict[str, Any]],
    session_id: str | None,
    workdir: str | None,
    suggested_skills: list[str] | None = None,
    suggested_toolsets: list[str] | None = None,
) -> HandoffDocument:
    explicit_mission = (mission or "").strip() or None
    fallback = _last_user_message(conversation_history)
    objective = explicit_mission or fallback
    title = objective[:80].rstrip(" .:-")
    slug = _slugify(title)
    artifacts = _extract_artifacts(conversation_history, workdir, session_id)
    facts = _extract_recent_facts(conversation_history)
    skills = suggested_skills or ["handoff-execution", "hermes-agent"]
    toolsets = suggested_toolsets or ["file", "terminal", "session_search"]

    current_status = facts or [
        "This session has relevant context that should be continued in a fresh thread.",
        f"Current session id: {session_id}" if session_id else "Current session id is not available.",
        f"Working directory: {workdir}" if workdir else "No explicit working directory was detected.",
    ]

    exact_prompt = (
        f"Continue this task in a fresh session: {objective}. "
        "First restate the mission in one sentence, then validate the listed artifacts, "
        "respect the stated constraints, and proceed directly toward the success criteria."
    )

    file_refs = list(artifacts.get("files") or [])
    session_refs = list(artifacts.get("sessions") or [])
    service_refs = list(artifacts.get("services") or [])
    other_refs = list(artifacts.get("other") or [])

    lines = [
        f"# Handoff: {title}",
        "",
        "## Purpose of next session",
        f"Continue this task in a fresh session: {objective}.",
        "",
        "## Current status",
        *[f"- {item}" for item in current_status],
        "",
        "## Relevant artifacts",
        f"- workdir: {artifacts['workdir'] or '<not specified>'}",
        "- files:",
        *([f"  - {p}" for p in file_refs] or ["  - <none referenced in recent context>"]),
        "- sessions:",
        *([f"  - {s}" for s in session_refs] or ["  - <current session only>"]),
        "- services / endpoints:",
        *([f"  - {u}" for u in service_refs] or ["  - <none referenced>"]),
        "- other identifiers:",
        *([f"  - {o}" for o in other_refs] or ["  - <none captured>"]),
        "",
        "## Constraints and non-goals",
        "- Keep the next session narrow and artifact-first.",
        "- Do not reconstruct the full prior transcript unless a listed artifact is insufficient.",
        "- Non-goal: avoid unnecessary re-planning if the task can move directly into execution.",
        "",
        "## Recommended skills",
        *[f"- {skill}" for skill in skills],
        "",
        "## Recommended toolsets",
        *[f"- {toolset}" for toolset in toolsets],
        "",
        "## Exact first prompt",
        exact_prompt,
        "",
        "## Success criteria",
        f"- [ ] The new session clearly restates and advances this mission: {objective}",
        "- [ ] The new session validates the most relevant referenced artifacts before heavy execution",
        "- [ ] The new session produces a concrete deliverable or verified next result",
        "",
    ]
    markdown = "\n".join(lines)
    return HandoffDocument(
        markdown=markdown,
        suggested_filename=f"{default_handoff_path(Path('.'), slug).name}",
        next_mode="fresh session",
    )


def parse_handoff_markdown(text: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for idx, section in enumerate(_HANDOFF_SECTIONS):
        pattern = rf"^## {re.escape(section)}\n"
        m = re.search(pattern, text, flags=re.MULTILINE)
        if not m:
            raise ValueError(f"Missing required section: {section}")
        start = m.end()
        next_starts = []
        for later in _HANDOFF_SECTIONS[idx + 1:]:
            n = re.search(rf"^## {re.escape(later)}\n", text[start:], flags=re.MULTILINE)
            if n:
                next_starts.append(start + n.start())
        end = min(next_starts) if next_starts else len(text)
        parsed[section] = text[start:end].strip()
    return parsed


def build_handoff_consume_seed(parsed: dict[str, str], source_path: str) -> str:
    payload = {
        "source_path": source_path,
        "purpose": parsed.get("Purpose of next session", ""),
        "current_status": parsed.get("Current status", ""),
        "relevant_artifacts": parsed.get("Relevant artifacts", ""),
        "constraints": parsed.get("Constraints and non-goals", ""),
        "first_prompt": parsed.get("Exact first prompt", ""),
        "success_criteria": parsed.get("Success criteria", ""),
    }
    compact_json = json.dumps(payload, ensure_ascii=False)
    return (
        "You are continuing from a saved handoff document. "
        "First restate the mission in one sentence. Then validate the referenced artifacts before heavy execution, "
        "respect the listed constraints and non-goals, and proceed using the handoff's exact first prompt or an operational equivalent. "
        "Track completion against the listed success criteria. "
        f"Handoff payload: {compact_json}"
    )


def _format_inline_result(doc: HandoffDocument) -> str:
    return (
        f"{doc.markdown}\n"
        f"Suggested filename: {doc.suggested_filename}\n"
        f"Suggested next-step mode: {doc.next_mode}"
    )


def handle_handoff_document_command(
    *,
    cmd: str,
    conversation_history: list[dict[str, Any]],
    session_id: str | None,
    workdir: str | None,
    hermes_home: Path,
) -> HandoffCommandResult:
    mode, path_arg, mission = parse_handoff_args(cmd)

    if mode == "consume":
        source = Path(path_arg or "").expanduser()
        if not source.is_absolute():
            source = source.resolve()
        if not source.exists():
            return HandoffCommandResult(text=f"(._.) Handoff file not found: {source}")
        try:
            text = source.read_text(encoding="utf-8")
            parsed = parse_handoff_markdown(text)
            seed = build_handoff_consume_seed(parsed, str(source))
        except Exception as exc:
            return HandoffCommandResult(text=f"(._.) Could not consume handoff: {exc}")
        return HandoffCommandResult(
            text=(
                f"Loaded handoff from: {source}\n"
                "Queued it as the next agent turn."
            ),
            saved_path=str(source),
            agent_seed=seed,
        )

    doc = build_handoff_document(
        mission=mission,
        conversation_history=conversation_history,
        session_id=session_id,
        workdir=workdir,
    )

    if mode == "inline":
        return HandoffCommandResult(text=_format_inline_result(doc))

    if mode == "save":
        slug = _slugify((mission or _last_user_message(conversation_history))[:80])
        destination = (
            Path(path_arg).expanduser() if path_arg else default_handoff_path(hermes_home, slug)
        )
        if not destination.is_absolute():
            destination = destination.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(doc.markdown, encoding="utf-8")
        return HandoffCommandResult(
            text=(
                f"Saved handoff to: {destination}\n"
                f"Suggested next-step mode: {doc.next_mode}"
            ),
            saved_path=str(destination),
        )

    raise AssertionError(f"Unhandled handoff mode: {mode}")
