"""Obsidian-backed Study OS tools."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except Exception:  # pragma: no cover - PyYAML is a project dependency.
    yaml = None  # type: ignore[assignment]


WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
MAX_LIST_LIMIT = 500


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False)


def _ok(data: Any, warnings: list[str] | None = None) -> str:
    return _json({"ok": True, "data": data, "warnings": warnings or []})


def _err(code: str, message: str, details: dict[str, Any] | None = None) -> str:
    error: dict[str, Any] = {"code": code, "message": message}
    if details:
        error["details"] = details
    return _json({"ok": False, "error": error, "warnings": []})


def _get_env_value(name: str) -> str | None:
    try:
        from hermes_cli.config import get_env_value

        value = get_env_value(name)
        return value.strip() if isinstance(value, str) and value.strip() else None
    except Exception:
        value = os.environ.get(name)
        return value.strip() if value and value.strip() else None


def resolve_vault_path(raw: str | None = None) -> Path:
    candidate = (raw or "").strip() or _get_env_value("OBSIDIAN_VAULT_PATH")
    if not candidate:
        candidate = "~/Documents/Obsidian Vault"
    path = Path(candidate).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Obsidian vault not found: {path}")
    return path


def _safe_relative_path(vault: Path, rel: str | None) -> Path:
    if not rel or not str(rel).strip():
        return vault
    raw = Path(str(rel).strip())
    if raw.is_absolute():
        candidate = raw.expanduser().resolve()
    else:
        candidate = (vault / raw).resolve()
    try:
        candidate.relative_to(vault)
    except ValueError as exc:
        raise ValueError(f"Path escapes vault: {rel}") from exc
    return candidate


def _study_dir(vault: Path) -> Path:
    path = (vault / ".StudyOS").resolve()
    try:
        path.relative_to(vault)
    except ValueError as exc:  # defensive; should be impossible.
        raise ValueError(".StudyOS path escapes vault") from exc
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _append_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        existing = _read_text(path)
        sep = "" if existing.endswith("\n") else "\n"
        _write_text(path, existing + sep + content)
    else:
        _write_text(path, content)


def _parse_frontmatter(raw: str) -> tuple[dict[str, Any], str, str | None]:
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, raw, None
    end_idx = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end_idx = idx
            break
    if end_idx is None:
        return {}, "\n".join(lines[1:]), "Missing closing --- in frontmatter"
    fm_text = "\n".join(lines[1:end_idx])
    body = "\n".join(lines[end_idx + 1 :])
    if not fm_text.strip():
        return {}, body, None
    if yaml is None:
        return {}, body, "PyYAML unavailable; frontmatter not parsed"
    try:
        parsed = yaml.safe_load(fm_text) or {}
    except Exception as exc:
        return {}, body, f"Failed to parse frontmatter: {exc}"
    if not isinstance(parsed, dict):
        return {}, body, "Frontmatter is not a mapping"
    return parsed, body, None


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    return [str(value)]


def _strip_wikilink(value: str) -> str:
    text = value.strip()
    if text.startswith("[[") and text.endswith("]]"):
        text = text[2:-2]
    if "|" in text:
        text = text.split("|", 1)[0]
    if "#" in text:
        text = text.split("#", 1)[0]
    return text.strip()


def _clean_body_for_links(body: str) -> str:
    return CODE_BLOCK_RE.sub("", body)


def _extract_wikilinks(body: str) -> list[str]:
    links = []
    seen = set()
    for match in WIKILINK_RE.finditer(_clean_body_for_links(body)):
        target = match.group(1).strip()
        if not target or "://" in target or target in seen:
            continue
        seen.add(target)
        links.append(target)
    return links


def _extract_headings(body: str) -> list[dict[str, Any]]:
    headings = []
    clean = CODE_BLOCK_RE.sub("", body)
    for match in HEADING_RE.finditer(clean):
        headings.append({"level": len(match.group(1)), "text": match.group(2).strip()})
    return headings


def _excerpt(body: str, limit: int = 260) -> str:
    clean_lines = []
    for line in CODE_BLOCK_RE.sub("", body).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        clean_lines.append(stripped)
    text = " ".join(clean_lines)
    return text[:limit] + ("..." if len(text) > limit else "")


def _layer_from(path: Path, vault: Path, fm: dict[str, Any]) -> str:
    typ = str(fm.get("type") or "").strip()
    if typ:
        return typ
    rel = path.relative_to(vault).as_posix()
    if "/examples/" in f"/{rel}" or rel.startswith("examples/"):
        return "example"
    if "Box/题型/" in rel or "Box/题型\\" in rel:
        return "pattern"
    if "/Box/" in f"/{rel}" or rel.startswith("Box/"):
        return "concept"
    return "note"


def parse_note(path: Path, vault: Path, include_body: bool = False) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    raw = _read_text(path)
    fm, body, fm_warning = _parse_frontmatter(raw)
    if fm_warning:
        warnings.append(f"{path.relative_to(vault).as_posix()}: {fm_warning}")
    headings = _extract_headings(body)
    title = str(fm.get("title") or (headings[0]["text"] if headings else path.stem))
    data: dict[str, Any] = {
        "path": path.relative_to(vault).as_posix(),
        "basename": path.name,
        "title": title,
        "layer": _layer_from(path, vault, fm),
        "frontmatter": fm,
        "tags": _as_list(fm.get("tags")),
        "concepts": [_strip_wikilink(v) for v in _as_list(fm.get("concepts"))],
        "patterns": [_strip_wikilink(v) for v in _as_list(fm.get("patterns"))],
        "aliases": _as_list(fm.get("aliases")),
        "headings": headings,
        "wikilinks": _extract_wikilinks(body),
        "excerpt": _excerpt(body),
        "size": path.stat().st_size,
        "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
    }
    if include_body:
        data["body"] = body
    return data, warnings


def _iter_markdown_notes(
    vault: Path,
    *,
    folder: str | None = None,
    file_glob: str | None = None,
    include_study_os: bool = False,
) -> Iterable[Path]:
    root = _safe_relative_path(vault, folder)
    if not root.exists():
        return []
    pattern = (file_glob or "**/*.md").strip() or "**/*.md"
    paths = sorted(root.glob(pattern) if root.is_dir() else [root])
    out = []
    for path in paths:
        if not path.is_file() or path.suffix.lower() != ".md":
            continue
        rel = path.resolve().relative_to(vault).as_posix()
        if not include_study_os and (rel == ".StudyOS" or rel.startswith(".StudyOS/")):
            continue
        out.append(path.resolve())
    return out


def _matches_note(note: dict[str, Any], *, query: str | None, tag: str | None, layer: str | None) -> bool:
    if layer and note.get("layer") != layer:
        return False
    if tag:
        wanted = tag.strip().lstrip("#")
        tags = {t.lstrip("#") for t in note.get("tags", [])}
        if wanted not in tags:
            return False
    if query:
        q = query.casefold()
        haystacks = [
            str(note.get("path", "")),
            str(note.get("title", "")),
            str(note.get("excerpt", "")),
            " ".join(note.get("aliases", [])),
            " ".join(note.get("concepts", [])),
            " ".join(note.get("patterns", [])),
            " ".join(note.get("wikilinks", [])),
        ]
        if not any(q in h.casefold() for h in haystacks):
            return False
    return True


def _find_note(vault: Path, note_ref: str, include_study_os: bool = False) -> tuple[Path | None, list[Path]]:
    ref = (note_ref or "").strip()
    if not ref:
        return None, []
    try:
        direct = _safe_relative_path(vault, ref)
        if direct.is_file():
            return direct, []
        if direct.with_suffix(".md").is_file():
            return direct.with_suffix(".md"), []
    except ValueError:
        raise
    ref_clean = _strip_wikilink(ref)
    matches = []
    for path in _iter_markdown_notes(vault, include_study_os=include_study_os):
        data, _warnings = parse_note(path, vault, include_body=False)
        candidates = {
            data["path"],
            Path(data["path"]).with_suffix("").as_posix(),
            data["basename"],
            Path(data["basename"]).stem,
            data["title"],
            *data.get("aliases", []),
        }
        if ref_clean in candidates:
            matches.append(path)
    if len(matches) == 1:
        return matches[0], []
    return None, matches


def _limit_from(args: dict[str, Any], default: int = 100) -> int:
    try:
        limit = int(args.get("limit", default))
    except Exception:
        limit = default
    return max(1, min(limit, MAX_LIST_LIMIT))


def _today_iso() -> str:
    return date.today().isoformat()


def _parse_date(value: Any, default: date | None = None) -> date:
    if not value:
        return default or date.today()
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def _md_list(items: list[str]) -> str:
    return ", ".join(f"[[{item}]]" for item in items if item)


def _record_field(name: str, value: Any) -> str:
    if isinstance(value, list):
        rendered = ", ".join(str(v) for v in value if str(v).strip())
    else:
        rendered = str(value or "").strip()
    return f"- {name}: {rendered or '-'}"


def _read_study_files(vault: Path, subdir: str) -> list[tuple[Path, str]]:
    root = _study_dir(vault) / subdir
    if not root.exists():
        return []
    return [(path, _read_text(path)) for path in sorted(root.glob("*.md")) if path.is_file()]


def _date_in_range(value: str, start: date, end: date) -> bool:
    try:
        parsed = _parse_date(value)
    except Exception:
        return False
    return start <= parsed <= end


def _collect_error_records(vault: Path, start: date, end: date) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for path, text in _read_study_files(vault, "errors"):
        current: dict[str, str] | None = None
        for line in text.splitlines():
            heading = re.match(r"^###\s+(\d{4}-\d{2}-\d{2})\s+(.+)$", line)
            if heading:
                if current:
                    records.append(current)
                current = {"date": heading.group(1), "title": heading.group(2).strip(), "file": path.name}
                continue
            if current and line.startswith("- ") and ":" in line:
                key, value = line[2:].split(":", 1)
                current[key.strip().lower()] = value.strip()
        if current:
            records.append(current)
    return [r for r in records if _date_in_range(r.get("date", ""), start, end)]


def _collect_review_tasks(vault: Path, start: date, end: date) -> list[str]:
    path = _study_dir(vault) / "review_tasks.md"
    if not path.exists():
        return []
    tasks = []
    for line in _read_text(path).splitlines():
        if not line.startswith("- ["):
            continue
        due_match = re.search(r"due:(\d{4}-\d{2}-\d{2})", line)
        if due_match and not _date_in_range(due_match.group(1), start, end):
            continue
        tasks.append(line)
    return tasks


def handle_study_list_notes(args: dict[str, Any], **_kwargs) -> str:
    try:
        vault = resolve_vault_path(args.get("vault_path"))
        limit = _limit_from(args)
        notes = []
        warnings: list[str] = []
        for path in _iter_markdown_notes(
            vault,
            folder=args.get("folder"),
            file_glob=args.get("file_glob"),
            include_study_os=bool(args.get("include_study_os", False)),
        ):
            note, note_warnings = parse_note(path, vault, include_body=False)
            warnings.extend(note_warnings)
            if not _matches_note(note, query=args.get("query"), tag=args.get("tag"), layer=args.get("layer")):
                continue
            notes.append(note)
            if len(notes) >= limit:
                break
        return _ok({"vault_path": str(vault), "count": len(notes), "notes": notes}, warnings)
    except Exception as exc:
        return _err("LIST_NOTES_FAILED", str(exc))


def handle_study_read_note(args: dict[str, Any], **_kwargs) -> str:
    try:
        vault = resolve_vault_path(args.get("vault_path"))
        note_ref = str(args.get("note") or args.get("path") or "").strip()
        path, matches = _find_note(vault, note_ref, include_study_os=bool(args.get("include_study_os", False)))
        if matches:
            return _err(
                "NOTE_AMBIGUOUS",
                f"More than one note matched {note_ref!r}",
                {"matches": [p.relative_to(vault).as_posix() for p in matches[:20]]},
            )
        if not path:
            return _err("NOTE_NOT_FOUND", f"Note not found: {note_ref}")
        note, warnings = parse_note(path, vault, include_body=bool(args.get("include_body", False)))
        return _ok({"vault_path": str(vault), "note": note}, warnings)
    except Exception as exc:
        return _err("READ_NOTE_FAILED", str(exc))


def handle_study_extract_concepts(args: dict[str, Any], **_kwargs) -> str:
    try:
        vault = resolve_vault_path(args.get("vault_path"))
        limit = _limit_from(args, default=50)
        refs = _as_list(args.get("notes") or args.get("note"))
        paths: list[Path] = []
        if refs:
            ambiguous: dict[str, list[str]] = {}
            missing = []
            for ref in refs:
                path, matches = _find_note(vault, ref)
                if path:
                    paths.append(path)
                elif matches:
                    ambiguous[ref] = [p.relative_to(vault).as_posix() for p in matches[:20]]
                else:
                    missing.append(ref)
            if ambiguous or missing:
                return _err("NOTE_RESOLUTION_FAILED", "Some notes could not be resolved", {"ambiguous": ambiguous, "missing": missing})
        else:
            for path in _iter_markdown_notes(vault, folder=args.get("folder"), file_glob=args.get("file_glob")):
                note, _warnings = parse_note(path, vault, include_body=False)
                if _matches_note(note, query=args.get("query"), tag=args.get("tag"), layer=args.get("layer")):
                    paths.append(path)
                if len(paths) >= limit:
                    break

        concepts: Counter[str] = Counter()
        patterns: Counter[str] = Counter()
        tags: Counter[str] = Counter()
        candidates: Counter[str] = Counter()
        notes = []
        warnings: list[str] = []
        for path in paths[:limit]:
            note, note_warnings = parse_note(path, vault, include_body=False)
            warnings.extend(note_warnings)
            for item in note.get("concepts", []):
                concepts[item] += 1
            for item in note.get("patterns", []):
                patterns[item] += 1
            for item in note.get("tags", []):
                tags[item] += 1
            for item in note.get("wikilinks", []):
                candidates[_strip_wikilink(item)] += 1
            for heading in note.get("headings", []):
                text = str(heading.get("text", "")).strip()
                if 2 <= len(text) <= 40:
                    candidates[text] += 1
            notes.append({"path": note["path"], "title": note["title"], "layer": note["layer"]})

        return _ok(
            {
                "vault_path": str(vault),
                "notes": notes,
                "concepts": concepts.most_common(),
                "patterns": patterns.most_common(),
                "tags": tags.most_common(),
                "candidate_concepts": candidates.most_common(50),
            },
            warnings,
        )
    except Exception as exc:
        return _err("EXTRACT_CONCEPTS_FAILED", str(exc))


def handle_study_log_error(args: dict[str, Any], **_kwargs) -> str:
    try:
        vault = resolve_vault_path(args.get("vault_path"))
        occurred = _parse_date(args.get("occurred_on"), default=date.today())
        title = str(args.get("title") or args.get("source_note") or "学习错误").strip()
        concepts = [_strip_wikilink(v) for v in _as_list(args.get("concepts"))]
        patterns = [_strip_wikilink(v) for v in _as_list(args.get("patterns"))]
        source = str(args.get("source_note") or "").strip()
        block = "\n".join(
            [
                f"### {occurred.isoformat()} {title}",
                _record_field("Source", source),
                _record_field("Subject", args.get("subject")),
                _record_field("Concepts", _md_list(concepts)),
                _record_field("Patterns", _md_list(patterns)),
                _record_field("Cause", args.get("cause") or "未分类"),
                _record_field("Severity", args.get("severity") or "medium"),
                _record_field("Next action", args.get("next_action")),
                "",
                str(args.get("detail") or "").strip() or "（未填写细节）",
                "",
            ]
        )
        path = _study_dir(vault) / "errors" / f"{occurred:%Y-%m}.md"
        if not path.exists():
            _write_text(path, f"# Study OS Error Log {occurred:%Y-%m}\n\n")
        _append_text(path, block)
        return _ok({"vault_path": str(vault), "path": path.relative_to(vault).as_posix(), "title": title})
    except Exception as exc:
        return _err("LOG_ERROR_FAILED", str(exc))


def handle_study_create_review_task(args: dict[str, Any], **_kwargs) -> str:
    try:
        vault = resolve_vault_path(args.get("vault_path"))
        due = _parse_date(args.get("due_date"), default=date.today() + timedelta(days=1))
        title = str(args.get("title") or args.get("source_note") or "复习任务").strip()
        priority = str(args.get("priority") or "medium").strip()
        status = str(args.get("status") or "todo").strip()
        concepts = [_strip_wikilink(v) for v in _as_list(args.get("concepts"))]
        patterns = [_strip_wikilink(v) for v in _as_list(args.get("patterns"))]
        source = str(args.get("source_note") or "").strip()
        review_level = args.get("review_level", "")
        reason = str(args.get("reason") or "").strip()
        line = (
            f"- [ ] {title} "
            f"due:{due.isoformat()} priority:{priority} status:{status} "
            f"review_level:{review_level if review_level != '' else '-'} "
            f"source:{source or '-'} "
            f"concepts:{';'.join(concepts) or '-'} "
            f"patterns:{';'.join(patterns) or '-'}"
        )
        if reason:
            line += f" reason:{reason}"
        path = _study_dir(vault) / "review_tasks.md"
        if not path.exists():
            _write_text(path, "# Study OS Review Tasks\n\n")
        _append_text(path, line + "\n")
        return _ok({"vault_path": str(vault), "path": path.relative_to(vault).as_posix(), "title": title, "due_date": due.isoformat()})
    except Exception as exc:
        return _err("CREATE_REVIEW_TASK_FAILED", str(exc))


def handle_study_generate_weekly_report(args: dict[str, Any], **_kwargs) -> str:
    try:
        vault = resolve_vault_path(args.get("vault_path"))
        today = date.today()
        default_start = today - timedelta(days=today.weekday())
        start = _parse_date(args.get("start_date"), default=default_start)
        end = _parse_date(args.get("end_date"), default=start + timedelta(days=6))
        if end < start:
            return _err("INVALID_DATE_RANGE", "end_date must be on or after start_date")
        errors = _collect_error_records(vault, start, end)
        tasks = _collect_review_tasks(vault, start, end)
        causes = Counter(r.get("cause", "未分类") or "未分类" for r in errors)
        severities = Counter(r.get("severity", "medium") or "medium" for r in errors)
        week = start.isocalendar()
        report_path = _study_dir(vault) / "reports" / f"{week.year}-W{week.week:02d}.md"
        lines = [
            f"# Study OS Weekly Report {week.year}-W{week.week:02d}",
            "",
            f"- Range: {start.isoformat()} to {end.isoformat()}",
            f"- Errors logged: {len(errors)}",
            f"- Review tasks in range: {len(tasks)}",
            "",
            "## Error Causes",
            "",
        ]
        if causes:
            lines.extend(f"- {cause}: {count}" for cause, count in causes.most_common())
        else:
            lines.append("- No errors logged.")
        lines.extend(["", "## Severity", ""])
        if severities:
            lines.extend(f"- {severity}: {count}" for severity, count in severities.most_common())
        else:
            lines.append("- No severity data.")
        lines.extend(["", "## Error Records", ""])
        if errors:
            lines.extend(f"- {r.get('date')} {r.get('title')} ({r.get('cause', '未分类')})" for r in errors)
        else:
            lines.append("- No error records in this range.")
        lines.extend(["", "## Review Tasks", ""])
        if tasks:
            lines.extend(tasks)
        else:
            lines.append("- No review tasks in this range.")
        lines.extend(["", "## Next Focus", "", "- Prioritize repeated causes and overdue review tasks.", ""])
        _write_text(report_path, "\n".join(lines))
        return _ok(
            {
                "vault_path": str(vault),
                "path": report_path.relative_to(vault).as_posix(),
                "error_count": len(errors),
                "task_count": len(tasks),
                "causes": causes.most_common(),
            }
        )
    except Exception as exc:
        return _err("GENERATE_WEEKLY_REPORT_FAILED", str(exc))


def handle_study_export_anki_candidates(args: dict[str, Any], **_kwargs) -> str:
    try:
        vault = resolve_vault_path(args.get("vault_path"))
        limit = _limit_from(args, default=30)
        include_errors = bool(args.get("include_errors", True))
        candidates: list[dict[str, str]] = []

        for path in _iter_markdown_notes(vault, folder=args.get("folder"), file_glob=args.get("file_glob")):
            note, _warnings = parse_note(path, vault, include_body=False)
            if not _matches_note(note, query=args.get("query"), tag=args.get("tag"), layer=args.get("layer")):
                continue
            concepts = ", ".join(note.get("concepts", [])[:3])
            front = f"{note['title']} 的核心辨析点是什么？"
            back = f"候选来源：[[{Path(note['path']).with_suffix('').as_posix()}]]"
            if concepts:
                back += f"\n关联概念：{concepts}"
            candidates.append({"front": front, "back": back, "tags": "StudyOS Obsidian"})
            if len(candidates) >= limit:
                break

        if include_errors and len(candidates) < limit:
            start = _parse_date(args.get("start_date"), default=date.today() - timedelta(days=30))
            end = _parse_date(args.get("end_date"), default=date.today())
            for record in _collect_error_records(vault, start, end):
                front = f"错因复盘：{record.get('title', '学习错误')} 的错误原因是什么？"
                back = f"错因：{record.get('cause', '未分类')}\n下一步：{record.get('next action', '-')}"
                candidates.append({"front": front, "back": back, "tags": "StudyOS 错题"})
                if len(candidates) >= limit:
                    break

        exported = date.today().isoformat()
        path = _study_dir(vault) / "anki_candidates" / f"{exported}.md"
        lines = [
            f"# Study OS Anki Candidates {exported}",
            "",
            "These are candidates. Review before moving them into source notes or importing with obsidian-to-anki.",
            "",
        ]
        for idx, card in enumerate(candidates, start=1):
            lines.extend(
                [
                    f"## Candidate {idx}",
                    "",
                    "START",
                    "问答题",
                    f"正面: {card['front']}",
                    f"背面: {card['back']}",
                    f"Tags: {card['tags']}",
                    "END",
                    "",
                ]
            )
        _write_text(path, "\n".join(lines))
        return _ok({"vault_path": str(vault), "path": path.relative_to(vault).as_posix(), "count": len(candidates)})
    except Exception as exc:
        return _err("EXPORT_ANKI_CANDIDATES_FAILED", str(exc))


_VAULT_PROP = {
    "type": "string",
    "description": "Absolute Obsidian vault path. Defaults to OBSIDIAN_VAULT_PATH, then ~/Documents/Obsidian Vault.",
}


STUDY_LIST_NOTES_SCHEMA = {
    "description": "List Markdown notes in an Obsidian vault with metadata extracted from YAML frontmatter.",
    "parameters": {
        "type": "object",
        "properties": {
            "vault_path": _VAULT_PROP,
            "folder": {"type": "string", "description": "Optional vault-relative folder to search."},
            "file_glob": {"type": "string", "description": "Glob under folder, default **/*.md."},
            "query": {"type": "string", "description": "Case-insensitive query over path, title, excerpt, links, concepts, and patterns."},
            "tag": {"type": "string", "description": "Tag filter, with or without leading #."},
            "layer": {"type": "string", "description": "Layer/type filter such as concept, pattern, example, or note."},
            "limit": {"type": "integer", "description": "Maximum notes to return, capped at 500."},
            "include_study_os": {"type": "boolean", "description": "Include .StudyOS generated files."},
        },
    },
}


STUDY_READ_NOTE_SCHEMA = {
    "description": "Read one Obsidian note by vault-relative path, basename, title, or alias.",
    "parameters": {
        "type": "object",
        "properties": {
            "vault_path": _VAULT_PROP,
            "note": {"type": "string", "description": "Vault-relative path, basename, title, alias, or wikilink target."},
            "path": {"type": "string", "description": "Alias for note."},
            "include_body": {"type": "boolean", "description": "Include Markdown body content."},
            "include_study_os": {"type": "boolean", "description": "Allow reading .StudyOS generated files."},
        },
        "required": ["note"],
    },
}


STUDY_EXTRACT_CONCEPTS_SCHEMA = {
    "description": "Extract concepts, patterns, tags, wikilinks, headings, and candidate concepts from Obsidian notes.",
    "parameters": {
        "type": "object",
        "properties": {
            "vault_path": _VAULT_PROP,
            "notes": {"type": "array", "items": {"type": "string"}, "description": "Specific notes to inspect."},
            "note": {"type": "string", "description": "Single note to inspect."},
            "folder": {"type": "string"},
            "file_glob": {"type": "string"},
            "query": {"type": "string"},
            "tag": {"type": "string"},
            "layer": {"type": "string"},
            "limit": {"type": "integer"},
        },
    },
}


STUDY_LOG_ERROR_SCHEMA = {
    "description": "Append a learning mistake record under <vault>/.StudyOS/errors/YYYY-MM.md.",
    "parameters": {
        "type": "object",
        "properties": {
            "vault_path": _VAULT_PROP,
            "title": {"type": "string"},
            "source_note": {"type": "string"},
            "subject": {"type": "string"},
            "concepts": {"type": "array", "items": {"type": "string"}},
            "patterns": {"type": "array", "items": {"type": "string"}},
            "cause": {"type": "string", "description": "Mistake cause category, e.g. concept_confusion, condition_missed, calculation, abstraction_level."},
            "severity": {"type": "string", "description": "low, medium, high, or user-defined."},
            "next_action": {"type": "string"},
            "detail": {"type": "string"},
            "occurred_on": {"type": "string", "description": "YYYY-MM-DD. Defaults to today."},
        },
        "required": ["title"],
    },
}


STUDY_CREATE_REVIEW_TASK_SCHEMA = {
    "description": "Append a second-pass review task under <vault>/.StudyOS/review_tasks.md.",
    "parameters": {
        "type": "object",
        "properties": {
            "vault_path": _VAULT_PROP,
            "title": {"type": "string"},
            "source_note": {"type": "string"},
            "due_date": {"type": "string", "description": "YYYY-MM-DD. Defaults to tomorrow."},
            "priority": {"type": "string"},
            "status": {"type": "string"},
            "review_level": {"type": "integer"},
            "concepts": {"type": "array", "items": {"type": "string"}},
            "patterns": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string"},
        },
        "required": ["title"],
    },
}


STUDY_GENERATE_WEEKLY_REPORT_SCHEMA = {
    "description": "Generate a weekly Study OS report from .StudyOS errors and review tasks.",
    "parameters": {
        "type": "object",
        "properties": {
            "vault_path": _VAULT_PROP,
            "start_date": {"type": "string", "description": "YYYY-MM-DD. Defaults to this week's Monday."},
            "end_date": {"type": "string", "description": "YYYY-MM-DD. Defaults to six days after start_date."},
        },
    },
}


STUDY_EXPORT_ANKI_CANDIDATES_SCHEMA = {
    "description": "Export review-worthy Anki candidate blocks under <vault>/.StudyOS/anki_candidates/YYYY-MM-DD.md.",
    "parameters": {
        "type": "object",
        "properties": {
            "vault_path": _VAULT_PROP,
            "folder": {"type": "string"},
            "file_glob": {"type": "string"},
            "query": {"type": "string"},
            "tag": {"type": "string"},
            "layer": {"type": "string"},
            "limit": {"type": "integer"},
            "include_errors": {"type": "boolean"},
            "start_date": {"type": "string", "description": "YYYY-MM-DD for included errors. Defaults to 30 days ago."},
            "end_date": {"type": "string", "description": "YYYY-MM-DD for included errors. Defaults to today."},
        },
    },
}
