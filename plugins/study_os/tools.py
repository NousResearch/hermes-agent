"""Obsidian-backed Study OS tools."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from plugins.study_os.schemas import (
    DEFAULT_PROMPT_POLICY,
    PROJECT_ID_RE,
    SCHEDULE_ID_RE,
    validate_study_project,
    validate_study_schedule,
)

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
        cwd = os.getcwd()
        if (Path(cwd) / ".obsidian").exists() or (Path(cwd) / "Box").exists():
            candidate = cwd
    if not candidate:
        candidate = "~/Documents/Obsidian Vault"
    path = Path(candidate).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(
            f"Obsidian vault not found: {path}\n"
            "Set OBSIDIAN_VAULT_PATH env var or pass vault_path explicitly."
        )
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


def _study_projects_dir(vault: Path) -> Path:
    path = _study_dir(vault) / "projects"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _validate_project_id(project_id: Any) -> str:
    value = str(project_id or "").strip()
    if not PROJECT_ID_RE.match(value):
        raise ValueError("project_id must match ^[a-z0-9][a-z0-9-]{2,63}$")
    return value


def _validate_schedule_id(schedule_id: Any) -> str:
    value = str(schedule_id or "").strip()
    if not SCHEDULE_ID_RE.match(value):
        raise ValueError("schedule_id must match ^[a-z0-9][a-z0-9-]{2,79}$")
    return value


def _slugify(value: str, fallback: str = "decision") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:60] or fallback


def _project_dir(vault: Path, project_id: str) -> Path:
    project_id = _validate_project_id(project_id)
    path = (_study_projects_dir(vault) / project_id).resolve()
    try:
        path.relative_to(_study_projects_dir(vault))
    except ValueError as exc:
        raise ValueError(f"Project path escapes vault: {project_id}") from exc
    path.mkdir(parents=True, exist_ok=True)
    return path


def _active_project_path(vault: Path) -> Path:
    return _study_projects_dir(vault) / "active.json"


def _read_json_file(path: Path) -> dict[str, Any]:
    data = json.loads(_read_text(path))
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return data


def _resolve_project_id(vault: Path, project_id: Any = None) -> str:
    if project_id:
        return _validate_project_id(project_id)
    active_path = _active_project_path(vault)
    if not active_path.exists():
        raise FileNotFoundError("No active StudyOS project selected")
    active = _read_json_file(active_path)
    return _validate_project_id(active.get("project_id"))


def _project_manifest_path(vault: Path, project_id: str) -> Path:
    return _project_dir(vault, project_id) / "manifest.json"


def _read_project_manifest(vault: Path, project_id: Any = None) -> dict[str, Any]:
    resolved_id = _resolve_project_id(vault, project_id)
    path = _project_manifest_path(vault, resolved_id)
    if not path.exists():
        raise FileNotFoundError(f"StudyOS project not found: {resolved_id}")
    data = _read_json_file(path)
    ok, validated = validate_study_project(data)
    if not ok:
        raise ValueError("; ".join(validated))
    return validated


def _schedule_dir(vault: Path, project_id: str) -> Path:
    path = _project_dir(vault, project_id) / "schedules"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _schedule_path(vault: Path, project_id: str, schedule_id: str) -> Path:
    return _schedule_dir(vault, project_id) / f"{_validate_schedule_id(schedule_id)}.json"


def _decisions_dir(vault: Path, project_id: str) -> Path:
    path = _project_dir(vault, project_id) / "decisions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _learning_records_dir(vault: Path, project_id: str) -> Path:
    path = _project_dir(vault, project_id) / "learning-records"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _lessons_dir(vault: Path, project_id: str) -> Path:
    path = _project_dir(vault, project_id) / "lessons"
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


def _note_subject(note: dict[str, Any]) -> str | None:
    """Return the top-level course folder for a note, when it has one."""
    parts = Path(str(note.get("path") or "")).parts
    if len(parts) < 2 or parts[0].startswith("."):
        return None
    return parts[0]


_CN_NORMALIZE_RE = re.compile(r"[的与和之]")

def _normalize_cn(text: str) -> str:
    """Strip common Chinese particles for fuzzy concept-name matching."""
    return _CN_NORMALIZE_RE.sub("", text.casefold())


def _matches_note(
    note: dict[str, Any],
    *,
    query: str | None,
    tag: str | None,
    layer: str | None,
    search_body: bool = False,
    normalize: bool = False,
) -> bool:
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
        if search_body:
            haystacks.append(str(note.get("body", "")))
        haystacks_lower = [h.casefold() for h in haystacks]
        # Primary: strict substring match
        if any(q in h for h in haystacks_lower):
            return True
        # Secondary: Chinese-normalized fallback (strips 的与和之)
        if normalize:
            q_norm = _normalize_cn(q)
            if q_norm and any(q_norm in _normalize_cn(h) for h in haystacks_lower):
                return True
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
        search_body = bool(args.get("search_body", False))
        normalize = bool(args.get("normalize", False))
        include_body = search_body and bool(args.get("query"))
        notes = []
        warnings: list[str] = []
        for path in _iter_markdown_notes(
            vault,
            folder=args.get("folder"),
            file_glob=args.get("file_glob"),
            include_study_os=bool(args.get("include_study_os", False)),
        ):
            note, note_warnings = parse_note(path, vault, include_body=include_body)
            warnings.extend(note_warnings)
            if not _matches_note(
                note,
                query=args.get("query"),
                tag=args.get("tag"),
                layer=args.get("layer"),
                search_body=search_body,
                normalize=normalize,
            ):
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


def _cluster_errors(errors: list[dict[str, str]]) -> dict[str, Any]:
    """Cluster errors by (cause × concept) to detect repeated patterns.

    Returns a dict with:
      - ``pairs``: list of (cause, concept, count), sorted by count desc
      - ``repeated``: pairs with count ≥ 3 (same mistake on same concept)
      - ``deep_confusion``: concepts appearing with ≥2 different causes
    """
    cause_concept: Counter[tuple[str, str]] = Counter()
    concept_causes: dict[str, set[str]] = {}
    for r in errors:
        cause = (r.get("cause") or "未分类").strip()
        raw = (r.get("concepts") or "").strip()
        names = [c.strip() for c in raw.replace("[[", "").replace("]]", "").split(",") if c.strip()]
        for c_name in names:
            cause_concept[(cause, c_name)] += 1
            concept_causes.setdefault(c_name, set()).add(cause)

    pairs = [{"cause": c, "concept": n, "count": cnt} for (c, n), cnt in cause_concept.most_common()]
    repeated = [p for p in pairs if p["count"] >= 3]
    deep = [
        {"concept": name, "causes": sorted(causes), "cause_count": len(causes)}
        for name, causes in concept_causes.items()
        if len(causes) >= 2
    ]
    return {"pairs": pairs, "repeated": repeated, "deep_confusion": deep}


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
        clusters = _cluster_errors(errors)

        week = start.isocalendar()
        report_path = _study_dir(vault) / "reports" / f"{week.year}-W{week.week:02d}.md"
        lines = [
            f"# Study OS Weekly Report {week.year}-W{week.week:02d}",
            "",
            f"- Range: {start.isoformat()} to {end.isoformat()}",
            f"- Errors logged: {len(errors)}",
            f"- Review tasks in range: {len(tasks)}",
            "",
        ]

        lines.extend(["## Error Causes", ""])
        if causes:
            lines.extend(f"- {cause}: {count}" for cause, count in causes.most_common())
        else:
            lines.append("- No errors logged.")

        lines.extend(["", "## Error Patterns (cause × concept)", ""])
        if clusters["pairs"]:
            lines.append("| Cause | Concept | Count |")
            lines.append("|-------|---------|-------|")
            lines.extend(f"| {p['cause']} | [[{p['concept']}]] | {p['count']} |" for p in clusters["pairs"][:20])
        else:
            lines.append("- No clustered errors.")

        if clusters["repeated"]:
            lines.extend(["", "## ⚠️ Repeated Patterns (≥3 occurrences, same cause + concept)", ""])
            for p in clusters["repeated"]:
                lines.append(
                    f"- **{p['cause']}** on **[[{p['concept']}]]** — {p['count']} 次. "
                    f"建议：检查 /Box 中 `[[{p['concept']}]]` 概念卡是否清晰，"
                    f"创建专项复习任务重做相关例题。"
                )

        if clusters["deep_confusion"]:
            lines.extend(["", "## 🔴 Deep Confusion (same concept, multiple causes)", ""])
            for d in clusters["deep_confusion"]:
                causes_str = ", ".join(d["causes"])
                lines.append(
                    f"- **[[{d['concept']}]]** 出现 {d['cause_count']} 种不同错因：{causes_str}. "
                    f"这可能表明该概念的多个侧面都未掌握，建议从定义层重新梳理。"
                )

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

        lines.extend([
            "", "## Next Focus", "",
            "- 优先处理 Repeated Patterns 中的概念。",
            "- Deep Confusion 的概念建议回到 /Box 重新梳理定义层。",
            "- Overdue 复习任务优先于新任务。",
            "",
        ])
        _write_text(report_path, "\n".join(lines))
        return _ok(
            {
                "vault_path": str(vault),
                "path": report_path.relative_to(vault).as_posix(),
                "error_count": len(errors),
                "task_count": len(tasks),
                "causes": causes.most_common(),
                "clusters": clusters,
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
    "description": "List Markdown notes in an Obsidian vault with metadata extracted from YAML frontmatter. Use search_body=true to also search full body text. Use normalize=true for fuzzy Chinese matching (strips 的与和之).",
    "parameters": {
        "type": "object",
        "properties": {
            "vault_path": _VAULT_PROP,
            "folder": {"type": "string", "description": "Optional vault-relative folder to search."},
            "file_glob": {"type": "string", "description": "Glob under folder, default **/*.md."},
            "query": {"type": "string", "description": "Case-insensitive query over path, title, excerpt, links, concepts, patterns, and wikilinks. With search_body=true also searches full body text."},
            "tag": {"type": "string", "description": "Tag filter, with or without leading #."},
            "layer": {"type": "string", "description": "Layer/type filter such as concept, pattern, example, or note."},
            "limit": {"type": "integer", "description": "Maximum notes to return, capped at 500."},
            "include_study_os": {"type": "boolean", "description": "Include .StudyOS generated files."},
            "search_body": {"type": "boolean", "description": "Also search full note body text (not just metadata). Set true when the concept name might appear deep in body."},
            "normalize": {"type": "boolean", "description": "Fallback fuzzy match by stripping common Chinese particles (的与和之) after strict match fails. Use for concept-name variations like 导数定义 vs 导数的定义."},
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


# ---------------------------------------------------------------------------
# Spaced Repetition (Ebbinghaus) helpers
# ---------------------------------------------------------------------------

# Base Ebbinghaus intervals (days) indexed by review_count.
# review_count 0 = first review, 1 = second, etc.
_EBBINGHAUS_BASE = [1, 2, 4, 7, 15, 30, 60, 120]

# review_level → interval weight multiplier.
# Lower mastery → review more frequently; higher mastery → longer intervals.
_REVIEW_LEVEL_WEIGHT = {0: 0.5, 1: 0.7, 2: 1.0, 3: 1.3, 4: 1.6, 5: 2.5}


def _upsert_frontmatter_field(path: Path, field: str, value: Any) -> None:
    """Add or update a single YAML frontmatter field in-place.

    Uses string-level manipulation to avoid reformatting the entire YAML block.
    Date values are written as ISO strings; booleans as ``true``/``false``;
    integers and strings as-is.
    """
    raw = _read_text(path)
    lines = raw.splitlines()

    if isinstance(value, bool):
        serialized = "true" if value else "false"
    elif isinstance(value, date):
        serialized = value.isoformat()
    elif isinstance(value, datetime):
        serialized = value.strftime("%Y-%m-%d")
    else:
        serialized = str(value)

    if not lines or lines[0].strip() != "---":
        new_content = f"---\n{field}: {serialized}\n---\n\n{raw}"
        _write_text(path, new_content)
        return

    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        return

    field_re = re.compile(rf"^{re.escape(field)}\s*:.*$")
    for i in range(1, end_idx):
        if field_re.match(lines[i]):
            lines[i] = f"{field}: {serialized}"
            _write_text(path, "\n".join(lines) + "\n")
            return

    lines.insert(end_idx, f"{field}: {serialized}")
    _write_text(path, "\n".join(lines) + "\n")


def _calculate_next_review(
    review_count: int,
    review_level: int,
    passed: bool,
) -> tuple[int, date]:
    """Return (new_review_count, next_review_date) using Ebbinghaus + review_level weight.

    On pass: increment review_count, calculate interval from base curve × level weight.
    On fail: reset review_count to 0, next review in 1 day.
    """
    if not passed:
        return 0, date.today() + timedelta(days=1)

    new_count = min(review_count + 1, len(_EBBINGHAUS_BASE) - 1)
    base = _EBBINGHAUS_BASE[review_count] if review_count < len(_EBBINGHAUS_BASE) else _EBBINGHAUS_BASE[-1]
    weight = _REVIEW_LEVEL_WEIGHT.get(review_level, 1.0)
    interval = max(1, int(base * weight))
    return new_count, date.today() + timedelta(days=interval)


def _read_review_state(note: dict[str, Any]) -> dict[str, Any]:
    """Extract spaced-repetition fields from parsed note frontmatter."""
    fm = note.get("frontmatter", {})
    return {
        "review_count": int(fm.get("review_count", 0)),
        "last_reviewed_at": str(fm.get("last_reviewed_at", "")),
        "next_review_at": str(fm.get("next_review_at", "")),
    }


def _is_due(note: dict[str, Any], today: date) -> bool:
    """Check if a note is due for review.

    Due when: next_review_at <= today, OR never reviewed (no next_review_at).
    Only applies to example-layer notes.
    """
    if note.get("layer") != "example":
        return False
    state = _read_review_state(note)
    next_at = state["next_review_at"]
    if not next_at:
        return True
    try:
        return _parse_date(next_at) <= today
    except Exception:
        return True


def _review_filter_values(args: dict[str, Any], key: str) -> set[str]:
    """Read a review selector as normalized non-empty strings.

    Single-value selectors retain their compact form while multi-selectors use
    an array.  Silently ignoring malformed selector values would make a review
    scope broader than the user requested, so reject those at the boundary.
    """
    value = args.get(key)
    if value is None:
        return set()
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
        raise ValueError(f"{key} must be a string or an array of strings")
    return {item.strip().casefold() for item in values if item.strip()}


def _review_filter_ints(args: dict[str, Any], key: str) -> set[int]:
    value = args.get(key)
    if value is None:
        return set()
    values = [value] if isinstance(value, int) and not isinstance(value, bool) else value
    if not isinstance(values, list) or not all(isinstance(item, int) and not isinstance(item, bool) for item in values):
        raise ValueError(f"{key} must be an integer or an array of integers")
    if any(item < 0 or item > 5 for item in values):
        raise ValueError(f"{key} values must be between 0 and 5")
    return set(values)


# ---------------------------------------------------------------------------
# study_due_reviews
# ---------------------------------------------------------------------------

def handle_study_due_reviews(args: dict[str, Any], **_kwargs) -> str:
    """List review examples using explicit, composable queue selectors."""
    try:
        vault = resolve_vault_path(args.get("vault_path"))
        limit = _limit_from(args, default=30)
        today = date.today()
        raw_folder = args.get("folder")
        folder = str(raw_folder).strip() if raw_folder is not None else None
        subjects_filter = _review_filter_values(args, "subjects")
        if args.get("subject") is not None:
            subjects_filter.update(_review_filter_values(args, "subject"))
        notes_filter = _review_filter_values(args, "notes")
        notes_filter.update(_review_filter_values(args, "paths"))
        tags_filter = _review_filter_values(args, "tags")
        concepts_filter = _review_filter_values(args, "concepts")
        difficulties_filter = _review_filter_values(args, "difficulties")
        review_levels_filter = _review_filter_ints(args, "review_levels")
        match_mode = str(args.get("match") or "any").strip().lower()
        if match_mode not in {"any", "all"}:
            raise ValueError("match must be 'any' or 'all'")
        review_state = str(args.get("review_state") or "due").strip().lower()
        if review_state not in {"due", "all", "new", "reviewed"}:
            raise ValueError("review_state must be due, all, new, or reviewed")
        sort_by = str(args.get("sort") or "priority").strip().lower()
        if sort_by not in {"priority", "oldest", "newest", "difficulty_asc", "difficulty_desc", "title"}:
            raise ValueError("sort must be priority, oldest, newest, difficulty_asc, difficulty_desc, or title")
        min_level = args.get("min_review_level")
        max_level = args.get("max_review_level")
        if min_level is not None and (not isinstance(min_level, int) or isinstance(min_level, bool) or not 0 <= min_level <= 5):
            raise ValueError("min_review_level must be an integer from 0 to 5")
        if max_level is not None and (not isinstance(max_level, int) or isinstance(max_level, bool) or not 0 <= max_level <= 5):
            raise ValueError("max_review_level must be an integer from 0 to 5")
        if min_level is not None and max_level is not None and min_level > max_level:
            raise ValueError("min_review_level cannot exceed max_review_level")

        due: list[dict[str, Any]] = []
        subjects: set[str] = set()
        warnings: list[str] = []
        # A StudyOS vault may contain one course or multiple top-level course
        # folders (for example OS/examples and 计组/examples).  Scanning only a
        # root-level examples/ directory silently empties the review queue for
        # the latter layout.  An explicitly supplied folder remains scoped.
        for path in _iter_markdown_notes(vault, folder=folder):
            note, note_warnings = parse_note(path, vault, include_body=False)
            warnings.extend(note_warnings)
            if note.get("layer") != "example":
                continue
            note_subject = _note_subject(note)
            if note_subject:
                subjects.add(note_subject)

            state = _read_review_state(note)
            fm = note.get("frontmatter", {})
            rl = int(fm.get("review_level", 0))
            is_due = _is_due(note, today)
            if review_state == "due" and not is_due:
                continue
            if review_state == "new" and state["review_count"] != 0:
                continue
            if review_state == "reviewed" and state["review_count"] == 0:
                continue

            note_tags = {str(tag).lstrip("#").casefold() for tag in note.get("tags", [])}
            note_concepts = {str(concept).casefold() for concept in note.get("concepts", [])}
            subject_matches = {
                value
                for value in subjects_filter
                if value == (note_subject or "").casefold()
                or value in note_tags
                or any(value in concept for concept in note_concepts)
            }
            if subjects_filter and (subject_matches != subjects_filter if match_mode == "all" else not subject_matches):
                continue
            if notes_filter and note["path"].casefold() not in notes_filter:
                continue
            if tags_filter and (not tags_filter <= note_tags if match_mode == "all" else not tags_filter & note_tags):
                continue
            concept_matches = {
                value for value in concepts_filter if any(value in concept for concept in note_concepts)
            }
            if concepts_filter and (concept_matches != concepts_filter if match_mode == "all" else not concept_matches):
                continue
            difficulty = str(fm.get("difficulty") or "").casefold()
            if difficulties_filter and difficulty not in difficulties_filter:
                continue
            if review_levels_filter and rl not in review_levels_filter:
                continue
            if min_level is not None and rl < min_level:
                continue
            if max_level is not None and rl > max_level:
                continue

            due.append({
                "path": note["path"],
                "title": note["title"],
                "review_level": rl,
                "review_count": state["review_count"],
                "last_reviewed_at": state["last_reviewed_at"] or None,
                "next_review_at": state["next_review_at"] or None,
                "concepts": note.get("concepts", []),
                "tags": note.get("tags", []),
                "difficulty": fm.get("difficulty"),
                "subject": _note_subject(note),
            })

        difficulty_rank = {"easy": 1, "medium": 2, "hard": 3}
        if sort_by == "priority":
            due.sort(key=lambda item: (item["review_level"], item["last_reviewed_at"] or "0000-00-00", item["path"]))
        elif sort_by == "oldest":
            due.sort(key=lambda item: (item["last_reviewed_at"] or "0000-00-00", item["path"]))
        elif sort_by == "newest":
            due.sort(key=lambda item: (item["last_reviewed_at"] or "0000-00-00", item["path"]), reverse=True)
        elif sort_by.startswith("difficulty_"):
            due.sort(
                key=lambda item: (difficulty_rank.get(str(item["difficulty"]).casefold(), 2), item["path"]),
                reverse=sort_by == "difficulty_desc",
            )
        else:
            due.sort(key=lambda item: (item["title"].casefold(), item["path"]))
        return _ok({
            "vault_path": str(vault),
            "date": today.isoformat(),
            "count": min(len(due), limit),
            "subjects": sorted(subjects),
            "due": due[:limit],
            "selection": {
                "review_state": review_state,
                "sort": sort_by,
                "match": match_mode,
            },
        }, warnings)
    except Exception as exc:
        return _err("DUE_REVIEWS_FAILED", str(exc))


# ---------------------------------------------------------------------------
# study_record_review
# ---------------------------------------------------------------------------

def handle_study_record_review(args: dict[str, Any], **_kwargs) -> str:
    """Record the result of a spaced-repetition review and update intervals."""
    try:
        vault = resolve_vault_path(args.get("vault_path"))
        note_ref = str(args.get("note") or "").strip()
        if not note_ref:
            return _err("MISSING_NOTE", "note is required")

        path, matches = _find_note(vault, note_ref)
        if matches:
            return _err(
                "NOTE_AMBIGUOUS",
                f"More than one note matched {note_ref!r}",
                {"matches": [p.relative_to(vault).as_posix() for p in matches[:20]]},
            )
        if not path:
            return _err("NOTE_NOT_FOUND", f"Note not found: {note_ref}")

        note, warnings = parse_note(path, vault, include_body=False)
        fm = note.get("frontmatter", {})
        old_rl = int(fm.get("review_level", 0))
        old_count = int(fm.get("review_count", 0))
        passed = bool(args.get("passed", True))

        # Optional: user can override review_level
        new_rl = args.get("new_review_level")
        if new_rl is not None:
            try:
                new_rl = int(new_rl)
                new_rl = max(0, min(5, new_rl))
            except (ValueError, TypeError):
                new_rl = old_rl
        else:
            new_rl = old_rl

        # Calculate next review
        effective_rl = new_rl if new_rl is not None else old_rl
        new_count, next_date = _calculate_next_review(old_count, effective_rl, passed)

        # Update frontmatter
        _upsert_frontmatter_field(path, "last_reviewed_at", date.today())
        _upsert_frontmatter_field(path, "next_review_at", next_date)
        _upsert_frontmatter_field(path, "review_count", new_count)
        if new_rl != old_rl:
            _upsert_frontmatter_field(path, "review_level", new_rl)

        _invalidate_review_stats(vault)
        _graph_cache_path(vault).unlink(missing_ok=True)

        # Optionally log error for failed reviews
        error_result = None
        if not passed and args.get("log_error"):
            cause = str(args.get("cause") or "未分类").strip()
            detail = str(args.get("detail") or "").strip()
            concepts = [_strip_wikilink(v) for v in _as_list(args.get("concepts"))]
            occurred = _parse_date(args.get("occurred_on"), default=date.today())
            block = "\n".join([
                f"### {occurred.isoformat()} {note['title']} (复习错误)",
                _record_field("Source", path.relative_to(vault).as_posix()),
                _record_field("Concepts", _md_list(concepts)),
                _record_field("Cause", cause),
                _record_field("Severity", args.get("severity") or "medium"),
                _record_field("Next action", f"明日重做 (Ebbinghaus reset, next={next_date.isoformat()})"),
                "",
                detail or "（复习未通过，间隔重置为 1 天）",
                "",
            ])
            err_path = _study_dir(vault) / "errors" / f"{occurred:%Y-%m}.md"
            if not err_path.exists():
                _write_text(err_path, f"# Study OS Error Log {occurred:%Y-%m}\n\n")
            _append_text(err_path, block)
            error_result = {"path": err_path.relative_to(vault).as_posix()}

        return _ok({
            "path": note["path"],
            "title": note["title"],
            "passed": passed,
            "review_level": {"old": old_rl, "new": new_rl},
            "review_count": {"old": old_count, "new": new_count},
            "last_reviewed_at": date.today().isoformat(),
            "next_review_at": next_date.isoformat(),
            "error_logged": error_result,
        }, warnings)
    except Exception as exc:
        return _err("RECORD_REVIEW_FAILED", str(exc))


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

STUDY_DUE_REVIEWS_SCHEMA = {
    "description": "Build a review queue from example notes. Defaults to due items and priority order (lowest review_level, then oldest review). Select explicitly by note paths, subjects/tags/concepts, difficulty, level, review state, and order.",
    "parameters": {
        "type": "object",
        "properties": {
            "vault_path": _VAULT_PROP,
            "folder": {"type": "string", "description": "Folder to scan. Defaults to 'examples'."},
            "subject": {"type": "string", "description": "Backward-compatible single subject selector; matches subject, tag, or concept case-insensitively."},
            "subjects": {"type": "array", "items": {"type": "string"}, "description": "Subject selectors. Combined with other selector types using AND."},
            "notes": {"type": "array", "items": {"type": "string"}, "description": "Exact vault-relative example paths. Use for a user-selected question set."},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Tag selectors without #."},
            "concepts": {"type": "array", "items": {"type": "string"}, "description": "Concept selectors; matches a concept name case-insensitively."},
            "difficulties": {"type": "array", "items": {"type": "string"}, "description": "Difficulty values, such as easy, medium, or hard."},
            "review_levels": {"type": "array", "items": {"type": "integer"}, "description": "Exact review levels (0-5)."},
            "min_review_level": {"type": "integer", "description": "Inclusive minimum review level (0-5)."},
            "max_review_level": {"type": "integer", "description": "Inclusive maximum review level (0-5)."},
            "review_state": {"type": "string", "enum": ["due", "all", "new", "reviewed"], "description": "due is default; all permits targeted practice of non-due items."},
            "match": {"type": "string", "enum": ["any", "all"], "description": "How multiple values in subjects/tags/concepts match; default any."},
            "sort": {"type": "string", "enum": ["priority", "oldest", "newest", "difficulty_asc", "difficulty_desc", "title"], "description": "Queue order; default priority."},
            "limit": {"type": "integer", "description": "Maximum due notes to return (default 30)."},
        },
    },
}

STUDY_RECORD_REVIEW_SCHEMA = {
    "description": "Record the result of a spaced-repetition review. Updates review_count, last_reviewed_at, next_review_at in the note's YAML frontmatter. On pass: Ebbinghaus interval advances. On fail: interval resets to 1 day. Optionally logs an error record for failed reviews.",
    "parameters": {
        "type": "object",
        "properties": {
            "vault_path": _VAULT_PROP,
            "note": {"type": "string", "description": "Vault-relative path, basename, title, or wikilink target of the reviewed note."},
            "passed": {"type": "boolean", "description": "Whether the review was successful (default true)."},
            "new_review_level": {"type": "integer", "description": "Optional: update review_level (0-5). If omitted, level stays unchanged."},
            "log_error": {"type": "boolean", "description": "If review failed, also log an error record (default false)."},
            "cause": {"type": "string", "description": "Error cause (for log_error). See study_profile.md for taxonomy."},
            "concepts": {"type": "array", "items": {"type": "string"}, "description": "Related concepts (for error log)."},
            "severity": {"type": "string", "description": "Error severity (for log_error)."},
            "detail": {"type": "string", "description": "Free-text detail about the review or error."},
            "occurred_on": {"type": "string", "description": "YYYY-MM-DD. Defaults to today."},
        },
        "required": ["note"],
    },
}


# ---------------------------------------------------------------------------
# study_sync_memory
# ---------------------------------------------------------------------------

def _count_due(vault: Path) -> tuple[int, int]:
    """Return (due_count, total_examples) for today."""
    today = date.today()
    due = 0
    total = 0
    for path in _iter_markdown_notes(vault, folder="examples"):
        note, _warnings = parse_note(path, vault, include_body=False)
        if note.get("layer") != "example":
            continue
        total += 1
        if _is_due(note, today):
            due += 1
    return due, total


def _recent_weak_concepts(vault: Path, days: int = 30) -> list[dict[str, Any]]:
    """Return concepts with the most errors in recent period, sorted by count."""
    start = date.today() - timedelta(days=days)
    end = date.today()
    errors = _collect_error_records(vault, start, end)
    concept_errors: Counter[str] = Counter()
    for r in errors:
        raw = (r.get("concepts") or "").strip()
        names = [c.strip() for c in raw.replace("[[", "").replace("]]", "").split(",") if c.strip()]
        for c_name in names:
            concept_errors[c_name] += 1
    return [{"concept": name, "error_count": count} for name, count in concept_errors.most_common(10)]


def handle_study_sync_memory(args: dict[str, Any], **_kwargs) -> str:
    """Build structured memory entries from current study state.

    Returns entries the agent should pass to Hermes's ``memory`` tool
    (target="memory", action="add"/"replace").  Does NOT write memory
    directly — the agent decides what to keep and how to consolidate.
    """
    try:
        vault = resolve_vault_path(args.get("vault_path"))
        today = date.today()
        due, total = _count_due(vault)
        weak = _recent_weak_concepts(vault, days=30)

        entries: list[dict[str, str]] = []

        if weak:
            top5 = ", ".join(f"{w['concept']}({w['error_count']})" for w in weak[:5])
            entries.append({
                "action": "replace",
                "content": (
                    f"StudyOS Math: 近30天最薄弱的概念（按错误次数）：{top5}。"
                ),
                "old_text": "StudyOS Math: 近30天最薄弱的概念",
            })

        entries.append({
            "action": "replace",
            "content": (
                f"StudyOS Math: 当前 {due}/{total} 道例题待复习（艾宾浩斯间隔到期）。"
            ),
            "old_text": "StudyOS Math: 当前",
        })

        if weak:
            weakest = weak[0]
            entries.append({
                "action": "replace",
                "content": (
                    f"StudyOS Math: 最弱概念 [[{weakest['concept']}]] "
                    f"（{weakest['error_count']} 次错误）。优先复习相关例题。"
                ),
                "old_text": f"StudyOS Math: 最弱概念",
            })

        entries.append({
            "action": "replace",
            "content": f"StudyOS Math: 上次同步 {today.isoformat()}。",
            "old_text": "StudyOS Math: 上次同步",
        })

        return _ok({
            "vault_path": str(vault),
            "due_count": due,
            "total_examples": total,
            "weak_concepts": weak,
            "memory_entries": entries,
            "timestamp": today.isoformat(),
        })
    except Exception as exc:
        return _err("SYNC_MEMORY_FAILED", str(exc))


STUDY_SYNC_MEMORY_SCHEMA = {
    "description": "Build structured memory entries from current study state (due reviews, weak concepts). Returns entries ready for Hermes's memory tool. Use after daily/weekly review to persist study progress across sessions.",
    "parameters": {
        "type": "object",
        "properties": {
            "vault_path": _VAULT_PROP,
        },
    },
}


# ---------------------------------------------------------------------------
# Concept dependency graph
# ---------------------------------------------------------------------------

def _build_concept_graph(vault: Path) -> dict[str, Any]:
    """Build a directed dependency graph from all notes in the vault.

    Returns a dict with:
      - ``prerequisites``: {concept: [concepts it depends on]}
      - ``dependents``: {concept: [concepts that depend on it]}
      - ``exercised_by``: {concept: [example paths]}
      - ``review_levels``: {concept: {"min": int, "avg": float, "count": int}}
      - ``note_count``: {concept: total notes referencing it}
    """
    prerequisites: dict[str, set[str]] = {}
    exercised_by: dict[str, list[str]] = {}
    review_by_concept: dict[str, list[int]] = {}

    for path in _iter_markdown_notes(vault):
        note, _warnings = parse_note(path, vault, include_body=False)
        note_concepts = [_strip_wikilink(c) for c in note.get("concepts", [])]
        if not note_concepts:
            continue

        layer = note.get("layer", "note")
        note_path = note["path"]

        for c in note_concepts:
            exercised_by.setdefault(c, []).append(note_path)

            fm = note.get("frontmatter", {})
            rl = fm.get("review_level")
            if isinstance(rl, (int, float)):
                review_by_concept.setdefault(c, []).append(int(rl))

        if layer in ("concept", "pattern"):
            for c in note_concepts:
                prereqs = [d for d in note_concepts if d != c]
                if prereqs:
                    prerequisites.setdefault(c, set()).update(prereqs)

    dependents: dict[str, set[str]] = {}
    for concept, prereqs in prerequisites.items():
        for p in prereqs:
            dependents.setdefault(p, set()).add(concept)

    review_levels: dict[str, dict[str, Any]] = {}
    for c, levels in review_by_concept.items():
        review_levels[c] = {
            "min": min(levels),
            "avg": round(sum(levels) / len(levels), 1),
            "max": max(levels),
            "count": len(levels),
        }

    note_count = {c: len(paths) for c, paths in exercised_by.items()}

    return {
        "prerequisites": {k: sorted(v) for k, v in prerequisites.items()},
        "dependents": {k: sorted(v) for k, v in dependents.items()},
        "exercised_by": {k: v for k, v in exercised_by.items()},
        "review_levels": review_levels,
        "note_count": note_count,
    }


_GRAPH_CACHE_TTL_HOURS = 1


def _graph_cache_path(vault: Path) -> Path:
    return _study_dir(vault) / "concept_graph.json"


def _load_graph_cache(vault: Path) -> dict[str, Any] | None:
    cache_path = _graph_cache_path(vault)
    if not cache_path.exists():
        return None
    try:
        data = json.loads(_read_text(cache_path))
        built_at = data.get("built_at", "")
        if built_at:
            age = datetime.now() - datetime.fromisoformat(built_at)
            if age > timedelta(hours=_GRAPH_CACHE_TTL_HOURS):
                return None
        return data.get("graph")
    except Exception:
        return None


def _save_graph_cache(vault: Path, graph: dict[str, Any]) -> None:
    _write_text(
        _graph_cache_path(vault),
        _json({"built_at": datetime.now().isoformat(), "graph": graph}),
    )


def _get_concept_graph(vault: Path, rebuild: bool = False) -> dict[str, Any]:
    if not rebuild:
        cached = _load_graph_cache(vault)
        if cached is not None:
            return cached
    graph = _build_concept_graph(vault)
    _save_graph_cache(vault, graph)
    return graph


def _concept_ancestors(
    concept: str, graph: dict[str, Any], max_depth: int = 5
) -> list[list[str]]:
    """Return chains of prerequisites (ancestors) for a concept."""
    prereqs = graph["prerequisites"]
    chains: list[list[str]] = []

    def walk(current: str, path: list[str], depth: int):
        if depth > max_depth:
            return
        deps = prereqs.get(current, [])
        if not deps:
            chains.append(path + [current])
            return
        for d in deps:
            if d in path:
                chains.append(path + [current, f"(cycle→{d})"])
            else:
                walk(d, path + [current], depth + 1)

    walk(concept, [], 0)
    return chains


def _concept_descendants(
    concept: str, graph: dict[str, Any], max_depth: int = 5
) -> list[list[str]]:
    """Return chains of dependents (descendants) for a concept."""
    deps = graph["dependents"]
    chains: list[list[str]] = []

    def walk(current: str, path: list[str], depth: int):
        if depth > max_depth:
            return
        children = deps.get(current, [])
        if not children:
            chains.append(path + [current])
            return
        for c in children:
            if c in path:
                chains.append(path + [current, f"(cycle→{c})"])
            else:
                walk(c, path + [current], depth + 1)

    walk(concept, [], 0)
    return chains


def _topological_order(concepts: list[str], graph: dict[str, Any]) -> list[str]:
    """Return concepts in dependency order (prerequisites first).

    Only includes concepts that exist in the graph. Cycles are broken
    arbitrarily (the first encountered edge is skipped).
    """
    prereqs = graph["prerequisites"]
    relevant: set[str] = set()

    def collect(c: str):
        if c in relevant:
            return
        relevant.add(c)
        for p in prereqs.get(c, []):
            collect(p)

    for c in concepts:
        collect(c)

    in_degree: dict[str, int] = {c: 0 for c in relevant}
    adj: dict[str, list[str]] = {c: [] for c in relevant}
    for c in relevant:
        for p in prereqs.get(c, []):
            if p in relevant and c != p:
                adj.setdefault(p, []).append(c)
                in_degree[c] = in_degree.get(c, 0) + 1

    queue = [c for c in relevant if in_degree.get(c, 0) == 0]
    order: list[str] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for child in adj.get(node, []):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    return order


def handle_study_concept_graph(args: dict[str, Any], **_kwargs) -> str:
    """Query the concept dependency graph from cache (rebuilt hourly or on demand)."""
    try:
        vault = resolve_vault_path(args.get("vault_path"))
        rebuild = bool(args.get("rebuild", False))
        graph = _get_concept_graph(vault, rebuild=rebuild)
        target = str(args.get("concept") or "").strip()
        weak_only = bool(args.get("weak_only", False))

        if weak_only:
            recent = _recent_weak_concepts(vault, days=30)
            weak_names = {w["concept"] for w in recent}

        if target:
            target = _strip_wikilink(target)
            ancestors = _concept_ancestors(target, graph)
            descendants = _concept_descendants(target, graph)
            prereqs = graph["prerequisites"].get(target, [])
            deps = graph["dependents"].get(target, [])
            examples = graph["exercised_by"].get(target, [])
            rl = graph["review_levels"].get(target)
            review_order = _topological_order([target], graph)

            affected_examples = []
            for d in deps:
                affected_examples.extend(graph["exercised_by"].get(d, []))

            return _ok({
                "concept": target,
                "direct_prerequisites": prereqs,
                "direct_dependents": deps,
                "ancestor_chains": ancestors,
                "descendant_chains": descendants,
                "exercised_in": examples[:20],
                "affected_examples": list(set(affected_examples))[:20],
                "review_level": rl,
                "note_count": graph["note_count"].get(target, 0),
                "recommended_review_order": review_order,
            })

        all_concepts = sorted(
            set(graph["prerequisites"]) | set(graph["dependents"]) | set(graph["exercised_by"])
        )

        bottleneck = sorted(
            [(c, len(graph["dependents"].get(c, []))) for c in all_concepts],
            key=lambda x: -x[1],
        )[:15]

        isolated = [
            c for c in all_concepts
            if not graph["prerequisites"].get(c) and not graph["dependents"].get(c)
        ]

        result: dict[str, Any] = {
            "total_concepts": len(all_concepts),
            "all_concepts": all_concepts,
            "concepts_with_dependencies": len(graph["prerequisites"]),
            "top_bottlenecks": [{"concept": c, "dependents": n} for c, n in bottleneck if n > 0],
            "isolated_concepts": len(isolated),
            "isolated_concept_names": isolated,
            "review_levels": graph["review_levels"],
        }

        if weak_only:
            result["weak_concepts"] = [
                {
                    "concept": c,
                    "prerequisites": graph["prerequisites"].get(c, []),
                    "dependents": graph["dependents"].get(c, []),
                    "review_level": graph["review_levels"].get(c),
                    "error_count": next(
                        (w["error_count"] for w in _recent_weak_concepts(vault, days=30) if w["concept"] == c), 0
                    ),
                    "recommended_review_order": _topological_order([c], graph),
                }
                for c in weak_names
                if c in all_concepts
            ]

        return _ok(result)
    except Exception as exc:
        return _err("CONCEPT_GRAPH_FAILED", str(exc))


STUDY_CONCEPT_GRAPH_SCHEMA = {
    "description": "Query the concept dependency graph (cached in .StudyOS/concept_graph.json, auto-refreshed hourly). Use rebuild=true after adding or editing notes. Without a target concept, returns all_concepts (full name list), top_bottlenecks, and isolated_concept_names. With a target concept, returns prerequisite chains (direct_prerequisites, direct_dependents, ancestor_chains, descendant_chains), exercised_in examples, and recommended_review_order.",
    "parameters": {
        "type": "object",
        "properties": {
            "vault_path": _VAULT_PROP,
            "concept": {"type": "string", "description": "Focus on a specific concept."},
            "weak_only": {"type": "boolean", "description": "Only return concepts with recent errors."},
            "rebuild": {"type": "boolean", "description": "Force rebuild the cache (default false, auto-refreshes hourly)."},
        },
    },
}


# ---------------------------------------------------------------------------
# Review statistics (cached JSON)
# ---------------------------------------------------------------------------

def _stats_cache_path(vault: Path) -> Path:
    return _study_dir(vault) / "review_stats.json"


def _invalidate_review_stats(vault: Path) -> None:
    cache = _stats_cache_path(vault)
    if cache.exists():
        cache.unlink()


def _build_review_stats(vault: Path) -> dict[str, Any]:
    today = date.today()
    total = 0
    by_level: Counter[int] = Counter()
    concept_levels: dict[str, list[int]] = {}
    concept_due: dict[str, int] = {}
    last_reviewed_dates: list[date] = []

    for path in _iter_markdown_notes(vault, folder="examples"):
        note, _warnings = parse_note(path, vault, include_body=False)
        if note.get("layer") != "example":
            continue
        total += 1
        fm = note.get("frontmatter", {})
        rl = int(fm.get("review_level", 0))
        by_level[rl] += 1

        lr = fm.get("last_reviewed_at")
        if lr:
            try:
                last_reviewed_dates.append(_parse_date(str(lr)))
            except Exception:
                pass

        for c in note.get("concepts", []):
            c_name = _strip_wikilink(c)
            concept_levels.setdefault(c_name, []).append(rl)

        if _is_due(note, today):
            for c in note.get("concepts", []):
                c_name = _strip_wikilink(c)
                concept_due[c_name] = concept_due.get(c_name, 0) + 1

    mastered = by_level.get(5, 0)
    progress = round(mastered / total * 100, 1) if total > 0 else 0.0

    concept_stats = {}
    for c, levels in concept_levels.items():
        concept_stats[c] = {
            "avg": round(sum(levels) / len(levels), 1),
            "min": min(levels),
            "max": max(levels),
            "count": len(levels),
            "due": concept_due.get(c, 0),
        }

    review_streak = 0
    if last_reviewed_dates:
        sorted_dates = sorted(set(last_reviewed_dates), reverse=True)
        check = today
        for d in sorted_dates:
            if d == check or d == check - timedelta(days=1):
                if d == check - timedelta(days=1):
                    check = d
                review_streak += 1
            elif d < check - timedelta(days=1):
                break

    return {
        "built_at": datetime.now().isoformat(),
        "total_examples": total,
        "by_review_level": {str(k): v for k, v in sorted(by_level.items())},
        "mastered": mastered,
        "progress_pct": progress,
        "due_today": sum(concept_due.values()),
        "review_streak_days": review_streak,
        "concepts": concept_stats,
    }


def _load_review_stats(vault: Path) -> dict[str, Any] | None:
    cache = _stats_cache_path(vault)
    if not cache.exists():
        return None
    try:
        return json.loads(_read_text(cache))
    except Exception:
        return None


def _save_review_stats(vault: Path, stats: dict[str, Any]) -> None:
    _write_text(_stats_cache_path(vault), _json(stats))


def handle_study_review_stats(args: dict[str, Any], **_kwargs) -> str:
    try:
        vault = resolve_vault_path(args.get("vault_path"))
        rebuild = bool(args.get("rebuild", False))
        if not rebuild:
            cached = _load_review_stats(vault)
            if cached is not None:
                return _ok({"cached": True, **cached})
        stats = _build_review_stats(vault)
        _save_review_stats(vault, stats)
        return _ok({"cached": False, **stats})
    except Exception as exc:
        return _err("REVIEW_STATS_FAILED", str(exc))


STUDY_REVIEW_STATS_SCHEMA = {
    "description": "Aggregated review statistics cached in .StudyOS/review_stats.json. Returns progress, review_level distribution, due count, review streak, and per-concept averages. Cache auto-invalidates after study_record_review.",
    "parameters": {
        "type": "object",
        "properties": {
            "vault_path": _VAULT_PROP,
            "rebuild": {"type": "boolean", "description": "Force rebuild instead of using cache."},
        },
    },
}


# ---------------------------------------------------------------------------
# Learning queue — new material that hasn't entered review yet
# ---------------------------------------------------------------------------

_LEARNING_STATES = ("未开始", "学习中", "已理解", "已掌握")


def _concept_learning_state(note: dict[str, Any]) -> str:
    fm = note.get("frontmatter", {})
    state = str(fm.get("learning_state", "未开始")).strip()
    return state if state in _LEARNING_STATES else "未开始"


def handle_study_learning_queue(args: dict[str, Any], **_kwargs) -> str:
    """Show new material that needs first-pass attention.

    Two sections:
      - New concepts: ordered by dependency (prerequisites first), filtered by learning_state.
      - New examples: never reviewed (review_count=0), ordered by difficulty then subject.
    """
    try:
        vault = resolve_vault_path(args.get("vault_path"))
        graph = _get_concept_graph(vault)
        state_filter = str(args.get("state") or "").strip()
        limit = _limit_from(args, default=30)

        new_concepts: list[dict[str, Any]] = []
        new_examples: list[dict[str, Any]] = []

        for path in _iter_markdown_notes(vault):
            note, _warnings = parse_note(path, vault, include_body=False)
            layer = note.get("layer", "note")
            fm = note.get("frontmatter", {})

            if layer in ("concept", "pattern"):
                ls = _concept_learning_state(note)
                if state_filter and ls != state_filter:
                    continue
                if ls in ("已掌握",):
                    continue
                deps = graph.get("prerequisites", {}).get(
                    _strip_wikilink(note.get("title", "")), []
                )
                new_concepts.append({
                    "path": note["path"],
                    "title": note["title"],
                    "learning_state": ls,
                    "prerequisites": deps or [],
                    "tags": note.get("tags", []),
                })

            elif layer == "example":
                rc = int(fm.get("review_count", 0))
                if rc > 0:
                    continue
                if state_filter:
                    rl = int(fm.get("review_level", 0))
                    if state_filter == "学习中" and rl != 0:
                        continue
                    if state_filter == "已理解" and rl == 0:
                        continue
                new_examples.append({
                    "path": note["path"],
                    "title": note["title"],
                    "review_level": int(fm.get("review_level", 0)),
                    "difficulty": fm.get("difficulty"),
                    "concepts": note.get("concepts", []),
                    "tags": note.get("tags", []),
                    "source": fm.get("source"),
                })

        new_examples.sort(key=lambda e: (
            {"easy": 1, "medium": 2, "hard": 3}.get(str(e.get("difficulty", "")).lower(), 2),
            e["title"],
        ))

        def _concept_order_key(c: dict[str, Any]) -> tuple[int, str]:
            deps_depth = max(
                (len(chain) for chain in _concept_ancestors(c["title"], graph)),
                default=0,
            )
            return (deps_depth, c["title"])

        new_concepts.sort(key=_concept_order_key)

        return _ok({
            "vault_path": str(vault),
            "new_concepts": new_concepts[:limit],
            "new_concepts_total": len(new_concepts),
            "new_examples": new_examples[:limit],
            "new_examples_total": len(new_examples),
        })
    except Exception as exc:
        return _err("LEARNING_QUEUE_FAILED", str(exc))


# ---------------------------------------------------------------------------
# Study session log
# ---------------------------------------------------------------------------

def handle_study_log_session(args: dict[str, Any], **_kwargs) -> str:
    """Log a study session to .StudyOS/sessions/YYYY-MM-DD.md."""
    try:
        vault = resolve_vault_path(args.get("vault_path"))
        occurred = _parse_date(args.get("occurred_on"), default=date.today())
        duration = args.get("duration_minutes")
        topics = _as_list(args.get("topics"))
        notes_created = _as_list(args.get("notes_created"))
        examples_attempted = _as_list(args.get("examples_attempted"))
        examples_passed = _as_list(args.get("examples_passed"))
        examples_failed = _as_list(args.get("examples_failed"))
        note_text = str(args.get("note") or "").strip()

        ses_dir = _study_dir(vault) / "sessions"
        ses_dir.mkdir(parents=True, exist_ok=True)
        ses_path = ses_dir / f"{occurred.isoformat()}.md"

        lines = [
            f"# Study Session {occurred.isoformat()}",
            "",
        ]
        if duration is not None:
            lines.append(f"- Duration: {duration} min")
        if topics:
            lines.append(f"- Topics: {', '.join(str(t) for t in topics)}")
        if notes_created:
            lines.append(f"- Notes created: {', '.join(str(n) for n in notes_created)}")
        if examples_attempted:
            lines.append(f"- Examples attempted: {len(examples_attempted)}")
            lines.append(f"  - Attempted: {', '.join(str(e) for e in examples_attempted)}")
        if examples_passed:
            lines.append(f"  - Passed: {', '.join(str(e) for e in examples_passed)}")
        if examples_failed:
            lines.append(f"  - Failed: {', '.join(str(e) for e in examples_failed)}")
        if note_text:
            lines.extend(["", note_text, ""])
        lines.append("")

        if ses_path.exists():
            _append_text(ses_path, "\n".join(lines))
        else:
            _write_text(ses_path, "\n".join(lines))

        return _ok({
            "vault_path": str(vault),
            "path": ses_path.relative_to(vault).as_posix(),
            "date": occurred.isoformat(),
        })
    except Exception as exc:
        return _err("LOG_SESSION_FAILED", str(exc))


def handle_study_update_concept_state(args: dict[str, Any], **_kwargs) -> str:
    """Update the learning_state of a concept/pattern note."""
    try:
        vault = resolve_vault_path(args.get("vault_path"))
        note_ref = str(args.get("note") or "").strip()
        new_state = str(args.get("learning_state") or "").strip()
        if new_state not in _LEARNING_STATES:
            return _err("INVALID_STATE", f"learning_state must be one of: {', '.join(_LEARNING_STATES)}")

        path, matches = _find_note(vault, note_ref)
        if matches:
            return _err("NOTE_AMBIGUOUS", f"Multiple matches for {note_ref!r}")
        if not path:
            return _err("NOTE_NOT_FOUND", f"Note not found: {note_ref}")

        note, warnings = parse_note(path, vault, include_body=False)
        if note.get("layer") not in ("concept", "pattern"):
            return _err("NOT_CONCEPT", "learning_state only applies to concept/pattern notes")

        old_state = _concept_learning_state(note)
        _upsert_frontmatter_field(path, "learning_state", new_state)
        if new_state == "已掌握":
            _upsert_frontmatter_field(path, "mastered_at", date.today().isoformat())

        return _ok({
            "path": note["path"],
            "title": note["title"],
            "learning_state": {"old": old_state, "new": new_state},
        }, warnings)
    except Exception as exc:
        return _err("UPDATE_CONCEPT_STATE_FAILED", str(exc))


STUDY_LEARNING_QUEUE_SCHEMA = {
    "description": "List new material that hasn't entered review yet: concepts with learning_state ≠ 已掌握, and examples with review_count=0 (never attempted). Concepts ordered by dependency (prerequisites first), examples ordered by difficulty.",
    "parameters": {
        "type": "object",
        "properties": {
            "vault_path": _VAULT_PROP,
            "state": {"type": "string", "description": "Filter by learning_state: 未开始, 学习中, 已理解."},
            "limit": {"type": "integer", "description": "Maximum items per section (default 30)."},
        },
    },
}

STUDY_LOG_SESSION_SCHEMA = {
    "description": "Log a study session to .StudyOS/sessions/YYYY-MM-DD.md. Use after every study session to track time, topics, and progress.",
    "parameters": {
        "type": "object",
        "properties": {
            "vault_path": _VAULT_PROP,
            "occurred_on": {"type": "string", "description": "YYYY-MM-DD. Defaults to today."},
            "duration_minutes": {"type": "integer", "description": "Session duration in minutes."},
            "topics": {"type": "array", "items": {"type": "string"}, "description": "Topics or subjects studied."},
            "notes_created": {"type": "array", "items": {"type": "string"}, "description": "New notes created this session."},
            "examples_attempted": {"type": "array", "items": {"type": "string"}, "description": "Examples attempted (E-#### or paths)."},
            "examples_passed": {"type": "array", "items": {"type": "string"}, "description": "Examples passed."},
            "examples_failed": {"type": "array", "items": {"type": "string"}, "description": "Examples failed."},
            "note": {"type": "string", "description": "Free-text notes about the session."},
        },
    },
}

STUDY_UPDATE_CONCEPT_STATE_SCHEMA = {
    "description": "Update the learning_state of a concept or pattern note. States: 未开始 → 学习中 → 已理解 → 已掌握.",
    "parameters": {
        "type": "object",
        "properties": {
            "vault_path": _VAULT_PROP,
            "note": {"type": "string", "description": "Concept/pattern note path, title, or alias."},
            "learning_state": {"type": "string", "description": "New state: 未开始, 学习中, 已理解, 已掌握."},
        },
        "required": ["note", "learning_state"],
    },
}


# ---------------------------------------------------------------------------
# Learning plan — import 题单 and track progress
# ---------------------------------------------------------------------------

def _plans_dir(vault: Path) -> Path:
    d = _study_dir(vault) / "learning_plans"
    d.mkdir(parents=True, exist_ok=True)
    return d


_KOD_RE = re.compile(r"考点\s*(\d+)[：:]\s*(.+?)(?:（(\d+)\s*题）)?$")
_SECTION_RE = re.compile(r"^[│\s├└─]*([一二三四五六七八九十]+)、(.+?)(?:（考点\s*[\d\-,]+)?(?:[，,]\s*(\d+)\s*题)?(?:）)?$")
_TREE_KAODIAN_RE = re.compile(r"考点\s*(\d+)[：:]\s*(.+?)(?:（(\d+)\s*题）)")
_TABLE_ID_RE = re.compile(r"\*\*(\d+)\*\*")
_H1_RE = re.compile(r"^#\s+(.+)")


def _parse_tidan(vault: Path, tidan_path: Path) -> dict[str, Any]:
    raw = _read_text(tidan_path)
    lines = raw.splitlines()
    topic = ""
    kaodian: list[dict[str, Any]] = []
    sections: list[dict[str, Any]] = []
    problem_ids: set[int] = set()
    rounds: list[dict[str, Any]] = []
    checklist: list[str] = []
    in_tree = False
    in_practice = False
    in_checklist = False
    current_section: dict[str, Any] | None = None
    waiting_for_tree = False

    for line in lines:
        m = _H1_RE.match(line)
        if m:
            topic = m.group(1).strip().lstrip("#").strip()
            continue

        stripped = line.strip()

        if waiting_for_tree and stripped == "```":
            in_tree = True
            waiting_for_tree = False
            continue
        if in_tree and stripped == "```":
            in_tree = False
            continue

        if "考点树状图" in stripped:
            waiting_for_tree = True
            continue

        if in_tree:
            sm = _SECTION_RE.match(line.strip())
            if sm:
                current_section = {
                    "name": f"{sm.group(1)}、{sm.group(2)}",
                    "kaodian": [],
                }
                sections.append(current_section)
                continue
            km = _TREE_KAODIAN_RE.search(line)
            if km and current_section is not None:
                kd = {
                    "id": int(km.group(1)),
                    "name": km.group(2).strip(),
                    "problem_count": int(km.group(3)) if km.group(3) else 0,
                }
                current_section["kaodian"].append(kd)
                kaodian.append(kd)
                continue

        if "## 练习计划" in line:
            in_practice = True
            in_checklist = False
            continue
        if "## 模块完成标志" in line:
            in_checklist = True
            in_practice = False
            continue
        if line.startswith("## ") and (in_practice or in_checklist):
            in_practice = False
            in_checklist = False
            continue

        if in_checklist and line.startswith("- ["):
            checklist.append(line.strip())
            continue

        for pid in _TABLE_ID_RE.findall(line):
            problem_ids.add(int(pid))

    return {
        "topic": topic,
        "source": tidan_path.relative_to(vault).as_posix(),
        "sections": sections,
        "kaodian": kaodian,
        "total_kaodian": len(kaodian),
        "total_problems": len(problem_ids),
        "problem_ids": sorted(problem_ids),
        "checklist": checklist,
        "imported_at": datetime.now().isoformat(),
        "completed_kaodian": [],
    }


def handle_study_import_plan(args: dict[str, Any], **_kwargs) -> str:
    try:
        vault = resolve_vault_path(args.get("vault_path"))
        tidan_ref = str(args.get("tidan") or "").strip()
        if not tidan_ref:
            review_dir = vault / "review"
            available = (
                sorted(
                    [p.relative_to(vault).as_posix() for p in review_dir.glob("*.md")]
                )
                if review_dir.exists()
                else []
            )
            return _ok({"available_tidan": available})
        tidan_path = vault / tidan_ref
        if not tidan_path.exists():
            return _err("TIDAN_NOT_FOUND", str(tidan_path))
        plan = _parse_tidan(vault, tidan_path)
        topic = plan["topic"]
        if not topic:
            topic = tidan_path.stem
            plan["topic"] = topic
        out_path = _plans_dir(vault) / f"{topic}.json"
        _write_text(out_path, _json(plan))
        return _ok(
            {"path": out_path.relative_to(vault).as_posix(), "plan": plan}
        )
    except Exception as exc:
        return _err("IMPORT_PLAN_FAILED", str(exc))


def handle_study_plan_progress(args: dict[str, Any], **_kwargs) -> str:
    try:
        vault = resolve_vault_path(args.get("vault_path"))
        topic = str(args.get("topic") or "").strip()
        all_plans: list[dict[str, Any]] = []
        for plan_path in sorted(_plans_dir(vault).glob("*.json")):
            try:
                plan = json.loads(_read_text(plan_path))
                plan["_file"] = plan_path.relative_to(vault).as_posix()
                all_plans.append(plan)
            except Exception:
                continue
        if topic:
            all_plans = [p for p in all_plans if p.get("topic") == topic]
        summaries = []
        for p in all_plans:
            completed = len(p.get("completed_kaodian", []))
            total = p.get("total_kaodian", 0)
            summaries.append(
                {
                    "topic": p["topic"],
                    "source": p.get("source", ""),
                    "progress_pct": round(completed / total * 100, 1) if total else 0,
                    "completed_kaodian": completed,
                    "total_kaodian": total,
                    "total_problems": p.get("total_problems", 0),
                    "checklist_total": len(p.get("checklist", [])),
                    "checklist_done": sum(
                        1 for c in p.get("checklist", []) if c.startswith("- [x]")
                    ),
                    "file": p.get("_file", ""),
                }
            )
        return _ok({"plans": summaries})
    except Exception as exc:
        return _err("PLAN_PROGRESS_FAILED", str(exc))


STUDY_IMPORT_PLAN_SCHEMA = {
    "description": "Import a 题单 (problem checklist) from the vault's review/ directory into a structured learning plan. Parses 考点 tree, problem IDs, practice plan, and completion checklist. If no tidan specified, lists available 题单 files.",
    "parameters": {
        "type": "object",
        "properties": {
            "vault_path": _VAULT_PROP,
            "tidan": {"type": "string", "description": "Vault-relative path to a 题单 .md file under review/."},
        },
    },
}

STUDY_PLAN_PROGRESS_SCHEMA = {
    "description": "Show progress on all imported learning plans from .StudyOS/learning_plans/. Returns per-plan completion stats (考点 done/total, checklist progress).",
    "parameters": {
        "type": "object",
        "properties": {
            "vault_path": _VAULT_PROP,
            "topic": {"type": "string", "description": "Filter to a specific topic name."},
        },
    },
}


# ---------------------------------------------------------------------------
# Standardized curriculum — single source of truth for "what to learn"
# ---------------------------------------------------------------------------

_CURRICULUM_VERSION = "1"

_CURRICULUM_TEMPLATE = {
    "version": _CURRICULUM_VERSION,
    "meta": {
        "topic": "",
        "textbook": "",
        "exercise_book": "",
        "created_at": "",
    },
    "sections": [],
}


def _curricula_dir(vault: Path) -> Path:
    d = _study_dir(vault) / "curricula"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _validate_curriculum(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(data.get("meta"), dict):
        errors.append("meta is required")
    else:
        meta = data["meta"]
        if not meta.get("topic"):
            errors.append("meta.topic is required")
        if not meta.get("textbook"):
            errors.append("meta.textbook is required")
    if not isinstance(data.get("sections"), list):
        errors.append("sections must be a list")
    else:
        for i, sec in enumerate(data["sections"]):
            if not isinstance(sec.get("title"), str) or not sec["title"].strip():
                errors.append(f"sections[{i}].title is required")
            for j, kd in enumerate(sec.get("kaodian", [])):
                if not isinstance(kd.get("name"), str) or not kd["name"].strip():
                    errors.append(f"sections[{i}].kaodian[{j}].name is required")
    return errors


def handle_study_create_curriculum(args: dict[str, Any], **_kwargs) -> str:
    try:
        vault = resolve_vault_path(args.get("vault_path"))
        data = args.get("data")
        if data is None:
            template = json.loads(json.dumps(_CURRICULUM_TEMPLATE))
            template["meta"]["created_at"] = datetime.now().isoformat()
            return _ok({"template": template})
        if not isinstance(data, dict):
            return _err("INVALID_DATA", "data must be a JSON object")
        errors = _validate_curriculum(data)
        if errors:
            return _err("VALIDATION_FAILED", "; ".join(errors))
        topic = data["meta"]["topic"].strip()
        data.setdefault("version", _CURRICULUM_VERSION)
        data["meta"]["created_at"] = data["meta"].get("created_at") or datetime.now().isoformat()
        out_path = _curricula_dir(vault) / f"{topic}.json"
        _write_text(out_path, _json(data))
        return _ok({
            "path": out_path.relative_to(vault).as_posix(),
            "topic": topic,
            "sections": len(data.get("sections", [])),
            "kaodian": sum(len(s.get("kaodian", [])) for s in data.get("sections", [])),
        })
    except Exception as exc:
        return _err("CREATE_CURRICULUM_FAILED", str(exc))


def handle_study_list_curricula(args: dict[str, Any], **_kwargs) -> str:
    try:
        vault = resolve_vault_path(args.get("vault_path"))
        topic = str(args.get("topic") or "").strip()
        curricula: list[dict[str, Any]] = []
        for p in sorted(_curricula_dir(vault).glob("*.json")):
            try:
                c = json.loads(_read_text(p))
                if topic and c.get("meta", {}).get("topic") != topic:
                    continue
                curricula.append({
                    "topic": c["meta"]["topic"],
                    "textbook": c["meta"].get("textbook", ""),
                    "exercise_book": c["meta"].get("exercise_book", ""),
                    "sections": len(c.get("sections", [])),
                    "kaodian": sum(len(s.get("kaodian", [])) for s in c.get("sections", [])),
                    "file": p.relative_to(vault).as_posix(),
                })
            except Exception:
                continue
        return _ok({"curricula": curricula})
    except Exception as exc:
        return _err("LIST_CURRICULA_FAILED", str(exc))


STUDY_CREATE_CURRICULUM_SCHEMA = {
    "description": "Create or update a standardized learning curriculum (JSON). Call without data to get an empty template. Curriculum is the single source of truth for 'what to learn' — generated from textbook/exercise book, not from inconsistently formatted review/ 题单.",
    "parameters": {
        "type": "object",
        "properties": {
            "vault_path": _VAULT_PROP,
            "data": {"type": "object", "description": "Full curriculum JSON matching the schema. Omit to get a template."},
        },
    },
}

STUDY_LIST_CURRICULA_SCHEMA = {
    "description": "List all standardized curricula in .StudyOS/curricula/. Each curriculum maps a textbook chapter to its 考点 tree with problem references.",
    "parameters": {
        "type": "object",
        "properties": {
            "vault_path": _VAULT_PROP,
            "topic": {"type": "string", "description": "Filter by topic name."},
        },
    },
}


# ---------------------------------------------------------------------------
# StudyOS projects, schedules, and prompt context
# ---------------------------------------------------------------------------

_PROJECT_DEFAULT_SUBJECTS = [
    {"id": "math", "label": "数学", "target_score": 120},
    {"id": "english", "label": "英语一", "target_score": 75},
    {"id": "politics", "label": "政治", "target_score": 75},
]

_VALID_PROMPT_INTENTS = {
    "planning",
    "organizing",
    "reviewing",
    "teaching",
    "assessment",
    "error_analysis",
    "schedule_adjustment",
}

_INTENT_SKILL = {
    "planning": "study-plan",
    "schedule_adjustment": "study-plan",
    "organizing": "study-organize",
    "reviewing": "study-review",
    "teaching": "study-teach",
    "assessment": "study-assessment",
    "error_analysis": "study-assessment",
}


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _default_project_manifest(args: dict[str, Any]) -> dict[str, Any]:
    now = _now_iso()
    requested_domain_pack = str(args.get("domain_pack") or "").strip()
    requested_domain = str(args.get("domain") or "").strip()
    requested_exam_type = str(args.get("exam_type") or "").strip()
    is_kaoyan = (
        requested_domain_pack == "kaoyan.v1"
        or requested_domain == "kaoyan"
        or requested_exam_type == "考研"
    )
    if is_kaoyan:
        defaults = {
            "project_id": "kaoyan-2027",
            "title": "2027 考研学习计划",
            "domain": "kaoyan",
            "exam_type": "考研",
            "exam_date": "2027-12-20",
            "phase": "foundation",
            "domain_pack": "kaoyan.v1",
            "workspace_type": "exam-vault",
            "subjects": list(_PROJECT_DEFAULT_SUBJECTS),
        }
    else:
        defaults = {
            "project_id": "general-learning",
            "title": "General Learning Project",
            "domain": "general",
            "exam_type": "none",
            "exam_date": "2099-12-31",
            "phase": "discovery",
            "domain_pack": requested_domain_pack or "general.v1",
            "workspace_type": "skill-vault",
            "subjects": [{"id": "learning", "label": "Learning"}],
        }
    return {
        "schema_version": "study_project.v1",
        "project_id": str(args.get("project_id") or defaults["project_id"]).strip(),
        "title": str(args.get("title") or defaults["title"]).strip(),
        "domain": str(args.get("domain") or defaults["domain"]).strip(),
        "exam_type": str(args.get("exam_type") or defaults["exam_type"]).strip(),
        "exam_date": str(args.get("exam_date") or defaults["exam_date"]).strip(),
        "timezone": str(args.get("timezone") or "Asia/Shanghai").strip(),
        "phase": str(args.get("phase") or defaults["phase"]).strip(),
        "domain_pack": str(args.get("domain_pack") or defaults["domain_pack"]).strip(),
        "workspace_type": str(args.get("workspace_type") or defaults["workspace_type"]).strip(),
        "artifact_policy": str(args.get("artifact_policy") or "lightweight").strip(),
        "subjects": args.get("subjects") if isinstance(args.get("subjects"), list) else defaults["subjects"],
        "prompt_policy": dict(DEFAULT_PROMPT_POLICY),
        "created_at": now,
        "updated_at": now,
    }


def handle_study_project(args: dict[str, Any], **_kwargs) -> str:
    try:
        vault = resolve_vault_path(args.get("vault_path"))
        action = str(args.get("action") or "status").strip()
        if action == "init":
            manifest = _default_project_manifest(args)
            ok, validated = validate_study_project(manifest)
            if not ok:
                return _err("VALIDATION_FAILED", "; ".join(validated))
            project_id = validated["project_id"]
            manifest_path = _project_manifest_path(vault, project_id)
            _write_text(manifest_path, _json(validated))
            active_path = _active_project_path(vault)
            _write_text(active_path, _json({"project_id": project_id}))
            return _ok(
                {
                    "project": validated,
                    "path": manifest_path.relative_to(vault).as_posix(),
                    "active_path": active_path.relative_to(vault).as_posix(),
                }
            )
        if action == "select":
            project_id = _validate_project_id(args.get("project_id"))
            manifest = _read_project_manifest(vault, project_id)
            active_path = _active_project_path(vault)
            _write_text(active_path, _json({"project_id": project_id}))
            return _ok({"project": manifest, "active_path": active_path.relative_to(vault).as_posix()})
        if action == "status":
            manifest = _read_project_manifest(vault, args.get("project_id"))
            project_id = manifest["project_id"]
            prompt_summary = _project_dir(vault, project_id) / "prompt_summary.md"
            schedules = sorted(_schedule_dir(vault, project_id).glob("*.json"))
            return _ok(
                {
                    "project": manifest,
                    "active": _resolve_project_id(vault, None) == project_id if _active_project_path(vault).exists() else False,
                    "prompt_summary_exists": prompt_summary.exists(),
                    "schedule_count": len(schedules),
                }
            )
        if action == "update_prompt_summary":
            manifest = _read_project_manifest(vault, args.get("project_id"))
            summary = str(args.get("summary") or "")
            max_chars = int(manifest.get("prompt_policy", {}).get("project_summary_max_chars", 1200))
            warnings: list[str] = []
            if len(summary) > max_chars:
                summary = summary[:max_chars]
                warnings.append(f"summary truncated to {max_chars} characters")
            path = _project_dir(vault, manifest["project_id"]) / "prompt_summary.md"
            _write_text(path, summary)
            return _ok({"project_id": manifest["project_id"], "path": path.relative_to(vault).as_posix(), "char_count": len(summary)}, warnings)
        return _err("INVALID_ACTION", f"Unsupported study_project action: {action}")
    except ValueError as exc:
        return _err("VALIDATION_FAILED", str(exc))
    except FileNotFoundError as exc:
        return _err("PROJECT_NOT_FOUND", str(exc))
    except Exception as exc:
        return _err("STUDY_PROJECT_FAILED", str(exc))


def _schedule_template(project: dict[str, Any]) -> dict[str, Any]:
    project_id = project["project_id"]
    if project.get("domain_pack") == "kaoyan.v1":
        phase_title = "基础阶段"
        phase_goal = "完成核心考点覆盖"
        event = {
            "id": "evt-20260701-math-derivative",
            "title": "数学：导数定义整理",
            "subject_id": "math",
            "type": "learning",
            "start": "2026-07-01T19:00:00+08:00",
            "end": "2026-07-01T21:00:00+08:00",
            "duration_minutes": 120,
            "goals": ["整理导数定义例题"],
            "source_curriculum": "一元函数微分学",
            "status": "planned",
        }
    else:
        subject_id = project.get("subjects", [{}])[0].get("id", "learning")
        phase_title = "Discovery"
        phase_goal = "Map one concrete learning objective to lightweight notes and source anchors"
        event = {
            "id": "evt-20260701-learning-scout",
            "title": "Scout one concept and source anchor",
            "subject_id": subject_id,
            "type": "learning",
            "start": "2026-07-01T19:00:00+08:00",
            "end": "2026-07-01T20:00:00+08:00",
            "duration_minutes": 60,
            "goals": ["Create or update one lightweight concept note only if it will be reused"],
            "source_curriculum": "project-roadmap",
            "status": "planned",
        }
    return {
        "schema_version": "study_schedule.v1",
        "schedule_id": f"{project_id}-master-plan",
        "project_id": project_id,
        "title": f"{project.get('title', project_id)} 学习计划",
        "timezone": project.get("timezone", "Asia/Shanghai"),
        "range": {"start": "2026-07-01", "end": "2026-07-31"},
        "phases": [
            {
                "id": project.get("phase", "foundation"),
                "title": phase_title,
                "start": "2026-07-01",
                "end": "2026-09-30",
                "goal": phase_goal,
            }
        ],
        "events": [event],
    }


def handle_study_schedule(args: dict[str, Any], **_kwargs) -> str:
    try:
        vault = resolve_vault_path(args.get("vault_path"))
        action = str(args.get("action") or "list").strip()
        if action == "template":
            project = _read_project_manifest(vault, args.get("project_id"))
            return _ok({"schedule": _schedule_template(project)})
        if action == "validate":
            project = _read_project_manifest(vault, args.get("project_id") or (args.get("data") or {}).get("project_id"))
            data = args.get("data")
            ok, validated = validate_study_schedule(data, project=project)
            if not ok:
                return _err("VALIDATION_FAILED", "; ".join(validated), {"errors": validated})
            return _ok({"schedule": validated})
        if action == "save":
            data = args.get("data")
            if not isinstance(data, dict):
                return _err("VALIDATION_FAILED", "data must be a JSON object")
            project = _read_project_manifest(vault, args.get("project_id") or data.get("project_id"))
            ok, validated = validate_study_schedule(data, project=project)
            if not ok:
                return _err("VALIDATION_FAILED", "; ".join(validated), {"errors": validated})
            path = _schedule_path(vault, project["project_id"], validated["schedule_id"])
            _write_text(path, _json(validated))
            return _ok({"schedule": validated, "path": path.relative_to(vault).as_posix()})
        if action == "list":
            project = _read_project_manifest(vault, args.get("project_id"))
            schedules = []
            for path in sorted(_schedule_dir(vault, project["project_id"]).glob("*.json")):
                try:
                    data = _read_json_file(path)
                    ok, validated = validate_study_schedule(data, project=project)
                    if not ok:
                        continue
                    schedules.append(
                        {
                            "schedule_id": validated["schedule_id"],
                            "project_id": validated["project_id"],
                            "title": validated["title"],
                            "timezone": validated["timezone"],
                            "range": validated["range"],
                            "event_count": len(validated.get("events", [])),
                            "path": path.relative_to(vault).as_posix(),
                        }
                    )
                except Exception:
                    continue
            return _ok({"project_id": project["project_id"], "schedules": schedules})
        if action == "read":
            project = _read_project_manifest(vault, args.get("project_id"))
            schedule_id = _validate_schedule_id(args.get("schedule_id"))
            path = _schedule_path(vault, project["project_id"], schedule_id)
            if not path.exists():
                return _err("SCHEDULE_NOT_FOUND", f"StudyOS schedule not found: {schedule_id}")
            data = _read_json_file(path)
            ok, validated = validate_study_schedule(data, project=project)
            if not ok:
                return _err("VALIDATION_FAILED", "; ".join(validated), {"errors": validated})
            return _ok({"schedule": validated, "path": path.relative_to(vault).as_posix()})
        return _err("INVALID_ACTION", f"Unsupported study_schedule action: {action}")
    except ValueError as exc:
        return _err("VALIDATION_FAILED", str(exc))
    except FileNotFoundError as exc:
        return _err("PROJECT_NOT_FOUND", str(exc))
    except Exception as exc:
        return _err("STUDY_SCHEDULE_FAILED", str(exc))


def _skill_path(skill_name: str) -> Path:
    return Path(__file__).resolve().parent / "skills" / skill_name / "SKILL.md"


def _read_prompt_fragment(kind: str, path: Path, max_chars: int) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"{kind} prompt source missing: {path.relative_to(Path(__file__).resolve().parent)}"
    content = _read_text(path)
    if len(content) > max_chars:
        return None, f"{kind} prompt source exceeds {max_chars} characters"
    return {"kind": kind, "source": path.as_posix(), "char_count": len(content), "content": content}, None


def handle_study_prompt_context(args: dict[str, Any], **_kwargs) -> str:
    try:
        vault = resolve_vault_path(args.get("vault_path"))
        intent = str(args.get("intent") or "").strip()
        if intent not in _VALID_PROMPT_INTENTS:
            return _err("INVALID_INTENT", f"Unsupported StudyOS intent: {intent}")
        project = _read_project_manifest(vault, args.get("project_id"))
        policy = {**DEFAULT_PROMPT_POLICY, **project.get("prompt_policy", {})}
        domain_pack = str(args.get("domain_pack") or project.get("domain_pack") or "").strip()
        fragments: list[dict[str, Any]] = []
        warnings: list[str] = []
        for kind, path, cap in (
            ("base", _skill_path("study-os"), int(policy["base_max_chars"])),
            ("intent", _skill_path(_INTENT_SKILL[intent]), int(policy["intent_max_chars"])),
        ):
            fragment, warning = _read_prompt_fragment(kind, path, cap)
            if warning:
                return _err("PROMPT_CONTEXT_TOO_LARGE" if "exceeds" in warning else "PROMPT_CONTEXT_SOURCE_MISSING", warning)
            if fragment:
                fragments.append(fragment)
        domain_skill = {
            "kaoyan.v1": "study-kaoyan",
            "engineering.v1": "study-engineering",
        }.get(domain_pack)
        if domain_skill:
            fragment, warning = _read_prompt_fragment("domain", _skill_path(domain_skill), int(policy["domain_max_chars"]))
            if warning:
                return _err("PROMPT_CONTEXT_TOO_LARGE" if "exceeds" in warning else "PROMPT_CONTEXT_SOURCE_MISSING", warning)
            if fragment:
                fragments.append(fragment)
        summary_path = _project_dir(vault, project["project_id"]) / "prompt_summary.md"
        if summary_path.exists():
            content = _read_text(summary_path)
            max_chars = int(policy["project_summary_max_chars"])
            if len(content) > max_chars:
                content = content[:max_chars]
                warnings.append(f"project_summary truncated to {max_chars} characters")
            fragments.append(
                {
                    "kind": "project_summary",
                    "source": summary_path.relative_to(vault).as_posix(),
                    "char_count": len(content),
                    "content": content,
                }
            )
        total = sum(fragment["char_count"] for fragment in fragments)
        if total > int(policy["total_max_chars"]):
            return _err("PROMPT_CONTEXT_TOO_LARGE", f"prompt context exceeds {policy['total_max_chars']} total characters")
        return _ok(
            {
                "intent": intent,
                "project_id": project["project_id"],
                "domain_pack": domain_pack,
                "fragments": fragments,
                "total_char_count": total,
            },
            warnings,
        )
    except ValueError as exc:
        return _err("VALIDATION_FAILED", str(exc))
    except FileNotFoundError as exc:
        return _err("PROJECT_NOT_FOUND", str(exc))
    except Exception as exc:
        return _err("STUDY_PROMPT_CONTEXT_FAILED", str(exc))


def _decision_path(vault: Path, project_id: str, decision_id: str) -> Path:
    return _decisions_dir(vault, project_id) / f"{_validate_schedule_id(decision_id)}.md"


def _md_bullets(items: Any) -> list[str]:
    values = [str(item).strip() for item in _as_list(items) if str(item).strip()]
    return [f"- {value}" for value in values] or ["- None"]


def _decision_markdown(decision: dict[str, Any]) -> str:
    lines = [
        "---",
        f"schema_version: learning_decision_record.v1",
        f"decision_id: {decision['decision_id']}",
        f"project_id: {decision['project_id']}",
        f"status: {decision['status']}",
        f"created_at: {decision['created_at']}",
        "---",
        "",
        f"# {decision['title']}",
        "",
        "## Decision",
        "",
        decision["decision"],
        "",
        "## Context",
        "",
        decision["context"] or "None",
        "",
        "## Options Considered",
        "",
        *_md_bullets(decision.get("options_considered")),
        "",
        "## Consequences",
        "",
        decision["consequences"] or "None",
        "",
        "## Linked Concepts",
        "",
        *_md_bullets(decision.get("linked_concepts")),
        "",
        "## Linked Sources",
        "",
        *_md_bullets(decision.get("linked_sources")),
        "",
        "## Linked Sessions",
        "",
        *_md_bullets(decision.get("linked_sessions")),
        "",
    ]
    return "\n".join(lines)


def _decision_summary(path: Path, vault: Path) -> dict[str, Any]:
    text = _read_text(path)
    fm, body, warning = _parse_frontmatter(text)
    headings = _extract_headings(body)
    return {
        "path": path.relative_to(vault).as_posix(),
        "decision_id": fm.get("decision_id") or path.stem,
        "project_id": fm.get("project_id"),
        "status": fm.get("status"),
        "title": headings[0]["text"] if headings else path.stem,
        "warning": warning,
    }


def handle_study_decision(args: dict[str, Any], **_kwargs) -> str:
    try:
        vault = resolve_vault_path(args.get("vault_path"))
        action = str(args.get("action") or "list").strip()
        project = _read_project_manifest(vault, args.get("project_id"))
        project_id = project["project_id"]
        if action == "create":
            title = str(args.get("title") or "").strip()
            decision_text = str(args.get("decision") or "").strip()
            if not title or not decision_text:
                return _err("VALIDATION_FAILED", "title and decision are required")
            created_at = _now_iso()
            requested_id = str(args.get("decision_id") or "").strip()
            base_id = requested_id or f"{len(list(_decisions_dir(vault, project_id).glob('*.md'))) + 1:04d}-{_slugify(title)}"
            decision_id = _validate_schedule_id(base_id)
            record = {
                "decision_id": decision_id,
                "project_id": project_id,
                "title": title,
                "status": str(args.get("status") or "accepted").strip(),
                "created_at": created_at,
                "decision": decision_text,
                "context": str(args.get("context") or "").strip(),
                "options_considered": _as_list(args.get("options_considered")),
                "consequences": str(args.get("consequences") or "").strip(),
                "linked_concepts": _as_list(args.get("linked_concepts")),
                "linked_sources": _as_list(args.get("linked_sources")),
                "linked_sessions": _as_list(args.get("linked_sessions")),
            }
            path = _decision_path(vault, project_id, decision_id)
            if path.exists():
                return _err("DECISION_EXISTS", f"LearningDecisionRecord already exists: {decision_id}")
            _write_text(path, _decision_markdown(record))
            return _ok({"decision": record, "path": path.relative_to(vault).as_posix()})
        if action == "list":
            decisions = [_decision_summary(path, vault) for path in sorted(_decisions_dir(vault, project_id).glob("*.md"))]
            return _ok({"project_id": project_id, "decisions": decisions})
        if action == "read":
            decision_id = _validate_schedule_id(args.get("decision_id"))
            path = _decision_path(vault, project_id, decision_id)
            if not path.exists():
                return _err("DECISION_NOT_FOUND", f"LearningDecisionRecord not found: {decision_id}")
            return _ok({"project_id": project_id, "decision_id": decision_id, "path": path.relative_to(vault).as_posix(), "content": _read_text(path)})
        return _err("INVALID_ACTION", f"Unsupported study_decision action: {action}")
    except ValueError as exc:
        return _err("VALIDATION_FAILED", str(exc))
    except FileNotFoundError as exc:
        return _err("PROJECT_NOT_FOUND", str(exc))
    except Exception as exc:
        return _err("STUDY_DECISION_FAILED", str(exc))


def _learning_record_path(vault: Path, project_id: str, record_id: str) -> Path:
    return _learning_records_dir(vault, project_id) / f"{_validate_schedule_id(record_id)}.md"


def _learning_record_markdown(record: dict[str, Any]) -> str:
    lines = [
        "---",
        "schema_version: learning_record.v1",
        f"record_id: {record['record_id']}",
        f"project_id: {record['project_id']}",
        f"status: {record['status']}",
        f"created_at: {record['created_at']}",
        "---",
        "",
        f"# {record['title']}",
        "",
        record["summary"],
        "",
        "## Evidence",
        "",
        record["evidence"] or "None",
        "",
        "## Implications",
        "",
        record["implications"] or "None",
        "",
        "## Linked Concepts",
        "",
        *_md_bullets(record.get("linked_concepts")),
        "",
        "## Linked Sources",
        "",
        *_md_bullets(record.get("linked_sources")),
        "",
    ]
    return "\n".join(lines)


def _learning_record_summary(path: Path, vault: Path) -> dict[str, Any]:
    text = _read_text(path)
    fm, body, warning = _parse_frontmatter(text)
    headings = _extract_headings(body)
    return {
        "path": path.relative_to(vault).as_posix(),
        "record_id": fm.get("record_id") or path.stem,
        "project_id": fm.get("project_id"),
        "status": fm.get("status"),
        "title": headings[0]["text"] if headings else path.stem,
        "warning": warning,
    }


def handle_study_learning_record(args: dict[str, Any], **_kwargs) -> str:
    try:
        vault = resolve_vault_path(args.get("vault_path"))
        action = str(args.get("action") or "list").strip()
        project = _read_project_manifest(vault, args.get("project_id"))
        project_id = project["project_id"]
        if action == "create":
            title = str(args.get("title") or "").strip()
            summary = str(args.get("summary") or "").strip()
            evidence = str(args.get("evidence") or "").strip()
            if not title or not summary or not evidence:
                return _err("VALIDATION_FAILED", "title, summary, and evidence are required")
            requested_id = str(args.get("record_id") or "").strip()
            base_id = requested_id or f"{len(list(_learning_records_dir(vault, project_id).glob('*.md'))) + 1:04d}-{_slugify(title, 'learning-record')}"
            record_id = _validate_schedule_id(base_id)
            record = {
                "record_id": record_id,
                "project_id": project_id,
                "title": title,
                "status": str(args.get("status") or "active").strip(),
                "created_at": _now_iso(),
                "summary": summary,
                "evidence": evidence,
                "implications": str(args.get("implications") or "").strip(),
                "linked_concepts": _as_list(args.get("linked_concepts")),
                "linked_sources": _as_list(args.get("linked_sources")),
            }
            path = _learning_record_path(vault, project_id, record_id)
            if path.exists():
                return _err("LEARNING_RECORD_EXISTS", f"LearningRecord already exists: {record_id}")
            _write_text(path, _learning_record_markdown(record))
            return _ok({"record": record, "path": path.relative_to(vault).as_posix()})
        if action == "list":
            records = [_learning_record_summary(path, vault) for path in sorted(_learning_records_dir(vault, project_id).glob("*.md"))]
            return _ok({"project_id": project_id, "records": records})
        if action == "read":
            record_id = _validate_schedule_id(args.get("record_id"))
            path = _learning_record_path(vault, project_id, record_id)
            if not path.exists():
                return _err("LEARNING_RECORD_NOT_FOUND", f"LearningRecord not found: {record_id}")
            return _ok({"project_id": project_id, "record_id": record_id, "path": path.relative_to(vault).as_posix(), "content": _read_text(path)})
        return _err("INVALID_ACTION", f"Unsupported study_learning_record action: {action}")
    except ValueError as exc:
        return _err("VALIDATION_FAILED", str(exc))
    except FileNotFoundError as exc:
        return _err("PROJECT_NOT_FOUND", str(exc))
    except Exception as exc:
        return _err("STUDY_LEARNING_RECORD_FAILED", str(exc))


def _lesson_path(vault: Path, project_id: str, lesson_id: str) -> Path:
    return _lessons_dir(vault, project_id) / f"{_validate_schedule_id(lesson_id)}.html"


def _lesson_summary(path: Path, vault: Path) -> dict[str, Any]:
    content = _read_text(path)
    title_match = re.search(r"<title>(.*?)</title>", content, flags=re.IGNORECASE | re.DOTALL)
    h1_match = re.search(r"<h1[^>]*>(.*?)</h1>", content, flags=re.IGNORECASE | re.DOTALL)
    title = title_match.group(1).strip() if title_match else (h1_match.group(1).strip() if h1_match else path.stem)
    title = re.sub(r"<[^>]+>", "", title)
    return {
        "path": path.relative_to(vault).as_posix(),
        "lesson_id": path.stem,
        "title": title,
        "size_bytes": path.stat().st_size,
    }


def handle_study_lesson(args: dict[str, Any], **_kwargs) -> str:
    try:
        vault = resolve_vault_path(args.get("vault_path"))
        action = str(args.get("action") or "list").strip()
        project = _read_project_manifest(vault, args.get("project_id"))
        project_id = project["project_id"]
        if action == "create":
            title = str(args.get("title") or "").strip()
            html = str(args.get("html") or "").strip()
            rationale = str(args.get("rationale") or "").strip()
            if not title or not html or not rationale:
                return _err("VALIDATION_FAILED", "title, html, and rationale are required")
            lowered = html.lower()
            if "<html" not in lowered or "</html>" not in lowered:
                return _err("VALIDATION_FAILED", "html must be a complete HTML document")
            requested_id = str(args.get("lesson_id") or "").strip()
            base_id = requested_id or f"{len(list(_lessons_dir(vault, project_id).glob('*.html'))) + 1:04d}-{_slugify(title, 'visual-lesson')}"
            lesson_id = _validate_schedule_id(base_id)
            path = _lesson_path(vault, project_id, lesson_id)
            if path.exists():
                return _err("LESSON_EXISTS", f"VisualLesson already exists: {lesson_id}")
            _write_text(path, html)
            meta = {
                "schema_version": "visual_lesson.v1",
                "lesson_id": lesson_id,
                "project_id": project_id,
                "title": title,
                "rationale": rationale,
                "created_at": _now_iso(),
                "linked_concepts": _as_list(args.get("linked_concepts")),
                "linked_sources": _as_list(args.get("linked_sources")),
                "html_path": path.relative_to(vault).as_posix(),
            }
            meta_path = _lessons_dir(vault, project_id) / f"{lesson_id}.json"
            _write_text(meta_path, _json(meta))
            return _ok({"lesson": meta, "path": path.relative_to(vault).as_posix(), "metadata_path": meta_path.relative_to(vault).as_posix()})
        if action == "list":
            lessons = [_lesson_summary(path, vault) for path in sorted(_lessons_dir(vault, project_id).glob("*.html"))]
            return _ok({"project_id": project_id, "lessons": lessons})
        if action == "read":
            lesson_id = _validate_schedule_id(args.get("lesson_id"))
            path = _lesson_path(vault, project_id, lesson_id)
            if not path.exists():
                return _err("LESSON_NOT_FOUND", f"VisualLesson not found: {lesson_id}")
            metadata_path = _lessons_dir(vault, project_id) / f"{lesson_id}.json"
            metadata = _read_json_file(metadata_path) if metadata_path.exists() else {}
            return _ok({"project_id": project_id, "lesson_id": lesson_id, "path": path.relative_to(vault).as_posix(), "metadata": metadata, "html": _read_text(path)})
        return _err("INVALID_ACTION", f"Unsupported study_lesson action: {action}")
    except ValueError as exc:
        return _err("VALIDATION_FAILED", str(exc))
    except FileNotFoundError as exc:
        return _err("PROJECT_NOT_FOUND", str(exc))
    except Exception as exc:
        return _err("STUDY_LESSON_FAILED", str(exc))


STUDY_PROJECT_SCHEMA = {
    "description": "Manage versioned StudyOS learning projects under the current vault.",
    "parameters": {
        "type": "object",
        "properties": {
            "vault_path": _VAULT_PROP,
            "action": {"type": "string", "enum": ["init", "select", "status", "update_prompt_summary"]},
            "project_id": {"type": "string", "description": "StudyOS project id, e.g. kaoyan-2027."},
            "title": {"type": "string"},
            "domain": {"type": "string"},
            "exam_type": {"type": "string"},
            "exam_date": {"type": "string"},
            "timezone": {"type": "string"},
            "phase": {"type": "string"},
            "domain_pack": {"type": "string"},
            "workspace_type": {
                "type": "string",
                "description": "Learning workspace shape, e.g. exam-vault, engineering-repo, skill-vault, or hybrid.",
            },
            "artifact_policy": {"type": "string", "description": "Persistence style such as lightweight or full."},
            "subjects": {"type": "array", "items": {"type": "object"}},
            "summary": {"type": "string", "description": "Prompt summary markdown for update_prompt_summary."},
        },
        "required": ["action"],
    },
}

STUDY_DECISION_SCHEMA = {
    "description": "Create, list, or read StudyOS LearningDecisionRecord markdown under the active project.",
    "parameters": {
        "type": "object",
        "properties": {
            "vault_path": _VAULT_PROP,
            "action": {"type": "string", "enum": ["create", "list", "read"]},
            "project_id": {"type": "string"},
            "decision_id": {"type": "string"},
            "title": {"type": "string"},
            "status": {"type": "string"},
            "decision": {"type": "string"},
            "context": {"type": "string"},
            "options_considered": {"type": "array", "items": {"type": "string"}},
            "consequences": {"type": "string"},
            "linked_concepts": {"type": "array", "items": {"type": "string"}},
            "linked_sources": {"type": "array", "items": {"type": "string"}},
            "linked_sessions": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["action"],
    },
}

STUDY_LEARNING_RECORD_SCHEMA = {
    "description": "Create, list, or read StudyOS LearningRecord markdown under the active project.",
    "parameters": {
        "type": "object",
        "properties": {
            "vault_path": _VAULT_PROP,
            "action": {"type": "string", "enum": ["create", "list", "read"]},
            "project_id": {"type": "string"},
            "record_id": {"type": "string"},
            "title": {"type": "string"},
            "status": {"type": "string"},
            "summary": {"type": "string"},
            "evidence": {"type": "string"},
            "implications": {"type": "string"},
            "linked_concepts": {"type": "array", "items": {"type": "string"}},
            "linked_sources": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["action"],
    },
}

STUDY_LESSON_SCHEMA = {
    "description": "Create, list, or read on-demand StudyOS VisualLesson HTML artifacts under the active project.",
    "parameters": {
        "type": "object",
        "properties": {
            "vault_path": _VAULT_PROP,
            "action": {"type": "string", "enum": ["create", "list", "read"]},
            "project_id": {"type": "string"},
            "lesson_id": {"type": "string"},
            "title": {"type": "string"},
            "rationale": {"type": "string", "description": "Why this topic needs a visual/interactive lesson."},
            "html": {"type": "string", "description": "Complete HTML document for create."},
            "linked_concepts": {"type": "array", "items": {"type": "string"}},
            "linked_sources": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["action"],
    },
}

STUDY_SCHEDULE_SCHEMA = {
    "description": "Template, validate, save, list, or read validated StudyOS schedule artifacts.",
    "parameters": {
        "type": "object",
        "properties": {
            "vault_path": _VAULT_PROP,
            "action": {"type": "string", "enum": ["template", "validate", "save", "list", "read"]},
            "project_id": {"type": "string"},
            "schedule_id": {"type": "string"},
            "data": {"type": "object", "description": "study_schedule.v1 artifact for validate/save."},
        },
        "required": ["action"],
    },
}

STUDY_PROMPT_CONTEXT_SCHEMA = {
    "description": "Return capped StudyOS prompt fragments for one project intent without mutating prompts.",
    "parameters": {
        "type": "object",
        "properties": {
            "vault_path": _VAULT_PROP,
            "intent": {
                "type": "string",
                "enum": ["planning", "organizing", "reviewing", "teaching", "assessment", "error_analysis", "schedule_adjustment"],
            },
            "project_id": {"type": "string"},
            "domain_pack": {"type": "string"},
        },
        "required": ["intent"],
    },
}
