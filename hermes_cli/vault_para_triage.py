"""PARA vault inbox triage helpers shared by the skill script and Slack plugin.

The workflow is intentionally edge-scoped:

* notes are read from an Obsidian-style vault inbox
* missing YAML frontmatter is added deterministically
* routing decisions are logged to an immutable audit trail
* user corrections are stored as examples for future runs
* Slack-facing commands call these helpers rather than touching core tools
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import yaml

from agent.skill_utils import parse_frontmatter
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

DEFAULT_REVIEW_DIR = "Resources/_Inbox Review"
DEFAULT_AUDIT_ROOT = ".hermes/para-triage"
DEFAULT_CAPTURE_ROOT = ".hermes/note-capture"
DEFAULT_CONFIG_REL_PATH = ".hermes/para-triage.yaml"

_CANONICAL_ROUTING_FIELDS = (
    "target_id",
    "target_class",
    "logical_path",
    "display_path",
    "resolved_target_path",
    "target_status",
    "trust_boundary",
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _iso_now() -> str:
    return _utc_now().isoformat().replace("+00:00", "Z")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _default_config() -> dict[str, Any]:
    return {
        "inbox_dir": "Inbox",
        "para_roots": {
            "projects": "Projects",
            "areas": "Areas",
            "resources": "Resources",
            "archives": "Archives",
        },
        "review_dir": DEFAULT_REVIEW_DIR,
        "audit": {
            "root": DEFAULT_AUDIT_ROOT,
        },
        "capture": {
            "root": DEFAULT_CAPTURE_ROOT,
            "source_archive_dir": "source-archive",
        },
        "frontmatter": {
            "title_field": "title",
            "created_field": "created",
            "updated_field": "updated",
            "tags_field": "tags",
            "triage_status_field": "para_triage_status",
            "triage_target_field": "para_target",
            "triage_confidence_field": "para_confidence",
            "triage_reason_field": "para_reason",
            "triaged_at_field": "para_triaged_at",
        },
        "routing": {
            "min_confidence": 0.7,
            "low_confidence_action": "review",
            "example_limit": 8,
            "rules": [],
            "target_resolver": {
                "enabled": False,
                "api_url": "",
                "timeout_seconds": 5.0,
            },
        },
        "projection": {
            "stores": {
                "vault": {
                    "enabled": True,
                    "path_prefix": "",
                },
                "second_brain": {
                    "enabled": True,
                    "path_prefix": "",
                },
            }
        },
    }


def resolve_vault_path(explicit: str | Path | None = None) -> Path:
    """Resolve the Obsidian vault path from an explicit arg, env, or fallback."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())

    env_path = os.getenv("OBSIDIAN_VAULT_PATH", "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates.append(Path("~/Documents/Obsidian Vault").expanduser())

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    first = candidates[0]
    raise FileNotFoundError(
        f"Could not resolve Obsidian vault path. Checked: {', '.join(str(p) for p in candidates)}. "
        f"Set OBSIDIAN_VAULT_PATH or pass --vault. First missing candidate: {first}"
    )


def _resolve_config_path(vault_path: Path, explicit: str | Path | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    return (vault_path / DEFAULT_CONFIG_REL_PATH).resolve()


def load_config(
    vault_path: Path,
    *,
    config_path: str | Path | None = None,
    config_override: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Path]:
    """Load triage config from YAML and merge it with defaults."""
    resolved_config_path = _resolve_config_path(vault_path, config_path)
    loaded: dict[str, Any] = {}
    if resolved_config_path.exists():
        data = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{resolved_config_path} must contain a YAML mapping")
        loaded = data
    merged = _deep_merge(_default_config(), loaded)
    if config_override:
        merged = _deep_merge(merged, config_override)
    return merged, resolved_config_path


def _state_root(vault_path: Path, config: dict[str, Any]) -> Path:
    configured = str(((config.get("audit") or {}).get("root")) or DEFAULT_AUDIT_ROOT).strip()
    root = Path(configured).expanduser()
    if not root.is_absolute():
        root = vault_path / root
    return root.resolve()


def _capture_root(vault_path: Path, config: dict[str, Any]) -> Path:
    configured = str(((config.get("capture") or {}).get("root")) or DEFAULT_CAPTURE_ROOT).strip()
    root = Path(configured).expanduser()
    if not root.is_absolute():
        root = vault_path / root
    return root.resolve()


def _source_archive_root(vault_path: Path, config: dict[str, Any]) -> Path:
    configured = str(((config.get("capture") or {}).get("source_archive_dir")) or "source-archive").strip()
    root = Path(configured).expanduser()
    if root.is_absolute():
        return root.resolve()
    return (_capture_root(vault_path, config) / configured).resolve()


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_load(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Could not parse %s; returning default", path)
        return default


def _json_dump(path: Path, payload: Any) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _jsonl_append(path: Path, payload: dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL row in %s", path)
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-") or "note"


def _extract_tags(frontmatter: dict[str, Any], field_name: str) -> list[str]:
    raw = frontmatter.get(field_name, [])
    if isinstance(raw, str):
        pieces = [raw]
    elif isinstance(raw, list):
        pieces = raw
    else:
        pieces = []
    out: list[str] = []
    for piece in pieces:
        text = str(piece or "").strip().lstrip("#")
        if text and text not in out:
            out.append(text)
    return out


def _extract_title(frontmatter: dict[str, Any], body: str, path: Path, *, title_field: str) -> str:
    title = str(frontmatter.get(title_field) or "").strip()
    if title:
        return title
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return path.stem.replace("-", " ").replace("_", " ").strip() or path.stem


def _excerpt(text: str, limit: int = 280) -> str:
    compact = re.sub(r"\s+", " ", text or "").strip()
    return compact[:limit]


def _render_note(frontmatter: dict[str, Any], body: str) -> str:
    yaml_text = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).strip()
    rendered_body = body.lstrip("\n")
    if rendered_body:
        return f"---\n{yaml_text}\n---\n\n{rendered_body}\n"
    return f"---\n{yaml_text}\n---\n"


def _unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for idx in range(2, 1000):
        candidate = path.with_name(f"{stem} {idx}{suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not find a free destination for {path}")


def _normalize_relative(path: Path, vault_path: Path) -> str:
    return str(path.resolve().relative_to(vault_path.resolve())).replace("\\", "/")


def _normalize_low_confidence_action(config: dict[str, Any]) -> str:
    action = str(((config.get("routing") or {}).get("low_confidence_action")) or "review").strip().lower()
    if action in {"review", "inbox", "auto-file"}:
        return action
    return "review"


def _enabled_projection_stores(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stores = (config.get("projection") or {}).get("stores") or {}
    enabled: dict[str, dict[str, Any]] = {}
    for store_name, store_cfg in stores.items():
        if not isinstance(store_cfg, dict):
            continue
        if not store_cfg.get("enabled", True):
            continue
        enabled[str(store_name)] = dict(store_cfg)
    if enabled:
        return enabled
    return {"vault": {"enabled": True, "path_prefix": ""}}


def _projection_relative_path(target: str, filename: str, store_cfg: dict[str, Any]) -> str:
    prefix = str(store_cfg.get("path_prefix") or "").strip().strip("/\\")
    rel = "/".join(part for part in (target.strip("/\\"), filename) if part)
    if prefix:
        rel = f"{prefix}/{rel}"
    return rel.replace("\\", "/")


def _inbox_path(vault_path: Path, config: dict[str, Any]) -> Path:
    inbox_dir = str(config.get("inbox_dir") or "Inbox").strip().strip("/\\")
    return (vault_path / inbox_dir).resolve()


def _available_targets(vault_path: Path, config: dict[str, Any]) -> list[str]:
    targets: list[str] = []
    para_roots = config.get("para_roots") or {}
    for rel_root in para_roots.values():
        root = (vault_path / str(rel_root)).resolve()
        if not root.exists():
            continue
        targets.append(_normalize_relative(root, vault_path))
        for candidate in sorted(p for p in root.rglob("*") if p.is_dir()):
            targets.append(_normalize_relative(candidate, vault_path))

    review_dir = str(config.get("review_dir") or DEFAULT_REVIEW_DIR).strip()
    if review_dir and review_dir not in targets:
        targets.append(review_dir)
    # Preserve order but dedupe.
    seen: set[str] = set()
    ordered: list[str] = []
    for item in targets:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def capture_structure_memory(vault_path: Path, config: dict[str, Any], *, state_root: Path | None = None) -> dict[str, Any]:
    """Snapshot PARA folder structure so later runs have stable context."""
    targets = _available_targets(vault_path, config)
    snapshot = {
        "captured_at": _iso_now(),
        "vault_path": str(vault_path),
        "inbox_dir": str(config.get("inbox_dir") or "Inbox"),
        "targets": targets,
        "para_roots": dict(config.get("para_roots") or {}),
    }
    if state_root is not None:
        _json_dump(state_root / "structure.json", snapshot)
    return snapshot


def _load_examples(state_root: Path) -> list[dict[str, Any]]:
    data = _json_load(state_root / "routing_examples.json", {"examples": []})
    examples = data.get("examples", []) if isinstance(data, dict) else []
    return [x for x in examples if isinstance(x, dict)]


def _save_examples(state_root: Path, examples: list[dict[str, Any]]) -> None:
    _json_dump(state_root / "routing_examples.json", {"examples": examples})


def _keyword_overlap(needles: Iterable[str], haystack: str) -> int:
    lowered = haystack.lower()
    total = 0
    for needle in needles:
        token = str(needle or "").strip().lower()
        if token and token in lowered:
            total += 1
    return total


def _best_examples(
    state_root: Path,
    *,
    title: str,
    body: str,
    tags: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    haystack = " ".join([title, body, " ".join(tags)]).lower()
    scored: list[tuple[int, dict[str, Any]]] = []
    for example in _load_examples(state_root):
        signals = example.get("signals") or []
        if isinstance(signals, str):
            signals = [signals]
        score = _keyword_overlap(signals, haystack)
        if score > 0:
            scored.append((score, example))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [example for _score, example in scored[:limit]]


def _rules_match(
    rule: dict[str, Any],
    *,
    title: str,
    body: str,
    tags: list[str],
) -> bool:
    haystack = " ".join([title, body]).lower()
    match_any = [str(x).strip().lower() for x in rule.get("match_any", []) if str(x).strip()]
    match_all = [str(x).strip().lower() for x in rule.get("match_all", []) if str(x).strip()]
    tags_any = [str(x).strip().lower().lstrip("#") for x in rule.get("tags_any", []) if str(x).strip()]
    if match_any and not any(token in haystack for token in match_any):
        return False
    if match_all and not all(token in haystack for token in match_all):
        return False
    if tags_any and not any(tag in {t.lower() for t in tags} for tag in tags_any):
        return False
    return bool(match_any or match_all or tags_any)


def _rule_decision(
    config: dict[str, Any],
    *,
    title: str,
    body: str,
    tags: list[str],
) -> dict[str, Any] | None:
    for idx, rule in enumerate((config.get("routing") or {}).get("rules", []), start=1):
        if not isinstance(rule, dict):
            continue
        target = str(rule.get("target") or "").strip()
        if not target:
            continue
        if _rules_match(rule, title=title, body=body, tags=tags):
            reason = str(rule.get("reason") or f"matched routing rule {idx}").strip()
            decision = {
                "target": target,
                "confidence": float(rule.get("confidence", 0.95)),
                "reason": reason,
                "source": "rule",
                "needs_feedback": bool(rule.get("needs_feedback", False)),
            }
            for key in _CANONICAL_ROUTING_FIELDS:
                value = rule.get(key)
                if value not in (None, ""):
                    decision[key] = value
            return decision
    return None


def _target_resolver_config(config: dict[str, Any]) -> dict[str, Any]:
    routing = config.get("routing") or {}
    resolver = routing.get("target_resolver") or {}
    return resolver if isinstance(resolver, dict) else {}


def _resolver_api_url(config: dict[str, Any]) -> str:
    resolver = _target_resolver_config(config)
    configured = str(resolver.get("api_url") or "").strip()
    if configured:
        return configured
    for env_name in ("HERMES_NOTE_TARGET_RESOLVER_URL", "NOTE_CAPTURE_TARGET_RESOLVER_URL"):
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    return ""


def _resolver_enabled(config: dict[str, Any]) -> bool:
    resolver = _target_resolver_config(config)
    if bool(resolver.get("enabled", False)):
        return True
    return bool(_resolver_api_url(config))


def _resolver_timeout_seconds(config: dict[str, Any]) -> float:
    resolver = _target_resolver_config(config)
    try:
        timeout = float(resolver.get("timeout_seconds", 5.0) or 5.0)
    except (TypeError, ValueError):
        timeout = 5.0
    return max(timeout, 0.1)


def _normalize_target(target: str, *, vault_path: Path, config: dict[str, Any]) -> str:
    target = (target or "").strip().strip("/\\")
    if not target:
        return str(config.get("review_dir") or DEFAULT_REVIEW_DIR)
    first = target.split("/", 1)[0].lower()
    allowed_roots = {str(v).split("/", 1)[0].lower() for v in (config.get("para_roots") or {}).values()}
    allowed_roots.add(str(config.get("inbox_dir") or "Inbox").split("/", 1)[0].lower())
    allowed_roots.add(str(config.get("review_dir") or DEFAULT_REVIEW_DIR).split("/", 1)[0].lower())
    if first not in allowed_roots:
        return str(config.get("review_dir") or DEFAULT_REVIEW_DIR)
    resolved = (vault_path / target).resolve()
    vault_resolved = vault_path.resolve()
    if not str(resolved).startswith(str(vault_resolved) + os.sep) and resolved != vault_resolved:
        raise ValueError(f"Target escapes the vault: {target}")
    return target.replace("\\", "/")


def _parse_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if "\n" in raw:
            raw = raw.split("\n", 1)[1]
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(raw[start : end + 1])
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
    return {}


def _copy_canonical_routing_fields(source: dict[str, Any], dest: dict[str, Any]) -> dict[str, Any]:
    for key in _CANONICAL_ROUTING_FIELDS:
        value = source.get(key)
        if value not in (None, ""):
            dest[key] = str(value).strip()
    return dest


def _normalize_resolver_decision(
    decision: dict[str, Any],
    *,
    vault_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    canonical_target = decision.get("canonical_target") or {}
    if not isinstance(canonical_target, dict):
        canonical_target = {}
    merged = dict(canonical_target)
    for key in (
        "target",
        "target_id",
        "target_class",
        "logical_path",
        "display_path",
        "resolved_target_path",
        "target_status",
        "trust_boundary",
    ):
        value = decision.get(key)
        if value not in (None, "") and key not in merged:
            merged[key] = value

    raw_target = ""
    for key in ("resolved_target_path", "target", "display_path"):
        candidate = str(merged.get(key) or "").strip()
        if candidate:
            raw_target = candidate
            break

    normalized = {
        "target": _normalize_target(raw_target, vault_path=vault_path, config=config),
        "confidence": float(decision.get("confidence", 0.0) or 0.0),
        "reason": str(decision.get("reason") or "resolved by target resolver").strip() or "resolved by target resolver",
        "source": str(decision.get("source") or "target_resolver"),
        "needs_feedback": bool(decision.get("needs_feedback", False)),
    }
    return _copy_canonical_routing_fields(merged, normalized)


def _resolver_decision(
    *,
    title: str,
    body: str,
    tags: list[str],
    targets: list[str],
    examples: list[dict[str, Any]],
    config: dict[str, Any],
    vault_path: Path,
    state_root: Path,
    structure: dict[str, Any],
) -> dict[str, Any] | None:
    if not _resolver_enabled(config):
        return None

    api_url = _resolver_api_url(config)
    if not api_url:
        return None

    payload = {
        "note": {
            "title": title,
            "tags": tags,
            "body": body,
            "excerpt": _excerpt(body, 1600),
        },
        "allowed_targets": targets,
        "review_target": str(config.get("review_dir") or DEFAULT_REVIEW_DIR),
        "examples": examples,
        "vault_path": str(vault_path),
        "state_root": str(state_root),
        "structure": structure,
    }
    body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        api_url,
        method="POST",
        data=body_bytes,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json; charset=utf-8",
        },
    )
    timeout = _resolver_timeout_seconds(config)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")

    parsed: Any
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = _parse_json_object(raw)

    if not isinstance(parsed, dict):
        raise ValueError("target resolver returned a non-object")

    decision = parsed
    for wrapper_key in ("decision", "result", "data"):
        wrapped = decision.get(wrapper_key)
        if isinstance(wrapped, dict):
            decision = wrapped
            break

    if not isinstance(decision, dict):
        raise ValueError("target resolver returned a non-object decision")
    return _normalize_resolver_decision(decision, vault_path=vault_path, config=config)


def _llm_decision(
    *,
    title: str,
    body: str,
    tags: list[str],
    targets: list[str],
    examples: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    from agent.auxiliary_client import call_llm

    prompt = {
        "task": "Route this note into the user's PARA vault.",
        "allowed_targets": targets,
        "review_target": str(config.get("review_dir") or DEFAULT_REVIEW_DIR),
        "examples": examples,
        "note": {
            "title": title,
            "tags": tags,
            "excerpt": _excerpt(body, 1600),
        },
        "rules": [
            "Choose exactly one target from allowed_targets.",
            "Prefer the narrowest existing folder that fits the note.",
            "Use the review_target when uncertain.",
            "Return JSON only.",
        ],
        "response_schema": {
            "target": "string",
            "confidence": "number 0..1",
            "reason": "short string",
            "needs_feedback": "boolean",
        },
    }
    response = call_llm(
        task="monitor",
        messages=[
            {
                "role": "system",
                "content": (
                    "You classify personal notes into an existing PARA vault. "
                    "Return ONLY one JSON object with keys target, confidence, reason, needs_feedback."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        max_tokens=500,
        temperature=0,
    )
    content = getattr(response.choices[0].message, "content", "") or ""
    parsed = _parse_json_object(str(content))
    if not parsed:
        raise ValueError("classifier returned no JSON object")
    return parsed


def _fallback_decision(config: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "target": str(config.get("review_dir") or DEFAULT_REVIEW_DIR),
        "confidence": 0.0,
        "reason": reason,
        "source": "fallback",
        "needs_feedback": True,
    }


def _classify_note(
    *,
    title: str,
    body: str,
    tags: list[str],
    vault_path: Path,
    config: dict[str, Any],
    state_root: Path,
    structure: dict[str, Any],
    classifier: Callable[..., dict[str, Any]] | None = None,
    target_resolver: Callable[..., dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    rule_hit = _rule_decision(config, title=title, body=body, tags=tags)
    if rule_hit:
        return rule_hit

    examples = _best_examples(
        state_root,
        title=title,
        body=body,
        tags=tags,
        limit=int(((config.get("routing") or {}).get("example_limit")) or 8),
    )

    try:
        resolver_decision = target_resolver(
            title=title,
            body=body,
            tags=tags,
            targets=list(structure.get("targets") or []),
            examples=examples,
            config=config,
            vault_path=vault_path,
            state_root=state_root,
            structure=structure,
        ) if target_resolver else _resolver_decision(
            title=title,
            body=body,
            tags=tags,
            targets=list(structure.get("targets") or []),
            examples=examples,
            config=config,
            vault_path=vault_path,
            state_root=state_root,
            structure=structure,
        )
    except Exception as exc:
        logger.warning("Vault target resolver failed: %s", exc)
        resolver_decision = None

    if isinstance(resolver_decision, dict):
        return _normalize_resolver_decision(resolver_decision, vault_path=vault_path, config=config)

    try:
        decision = classifier(
            title=title,
            body=body,
            tags=tags,
            targets=list(structure.get("targets") or []),
            examples=examples,
            config=config,
            vault_path=vault_path,
            state_root=state_root,
            structure=structure,
        ) if classifier else _llm_decision(
            title=title,
            body=body,
            tags=tags,
            targets=list(structure.get("targets") or []),
            examples=examples,
            config=config,
        )
    except Exception as exc:
        logger.warning("Vault note classification failed: %s", exc)
        return _fallback_decision(config, f"classifier failed: {exc}")

    if not isinstance(decision, dict):
        return _fallback_decision(config, "classifier returned a non-object")

    normalized = {
        "target": _normalize_target(str(decision.get("target") or ""), vault_path=vault_path, config=config),
        "confidence": float(decision.get("confidence", 0.0) or 0.0),
        "reason": str(decision.get("reason") or "classified").strip() or "classified",
        "source": str(decision.get("source") or "llm"),
        "needs_feedback": bool(decision.get("needs_feedback", False)),
    }
    return _copy_canonical_routing_fields(decision, normalized)


def _move_note(
    source_path: Path,
    dest_rel: str,
    *,
    vault_path: Path,
    content: str,
) -> Path:
    dest_dir = (vault_path / dest_rel).resolve()
    _ensure_dir(dest_dir)
    proposed = dest_dir / source_path.name
    if proposed.resolve() == source_path.resolve():
        dest_path = proposed
    else:
        dest_path = _unique_destination(proposed)
    dest_path.write_text(content, encoding="utf-8")
    if source_path.resolve() != dest_path.resolve() and source_path.exists():
        source_path.unlink()
    return dest_path


def _capture_event_path(capture_root: Path, entry_id: str) -> Path:
    return capture_root / "events" / f"{entry_id}.json"


def _projection_status_path(capture_root: Path) -> Path:
    return capture_root / "status" / "latest.json"


def _stage_projection_file(
    *,
    capture_root: Path,
    entry_id: str,
    store_name: str,
    relative_path: str,
    content: str,
    dry_run: bool,
) -> tuple[str, Path]:
    staged_rel = f"staging/{store_name}/{entry_id}/{relative_path}".replace("\\", "/")
    staged_path = capture_root / staged_rel
    if not dry_run:
        _ensure_dir(staged_path.parent)
        staged_path.write_text(content, encoding="utf-8")
    return staged_rel, staged_path


def _archive_source_note(
    *,
    source_path: Path,
    vault_path: Path,
    capture_root: Path,
    source_archive_root: Path,
    entry_id: str,
    dry_run: bool,
) -> tuple[str, Path]:
    archived_rel = f"source-archive/{entry_id}/{_normalize_relative(source_path, vault_path)}".replace("\\", "/")
    archived_path = capture_root / archived_rel
    if dry_run:
        return archived_rel, archived_path
    _ensure_dir(source_archive_root / entry_id / Path(_normalize_relative(source_path, vault_path)).parent)
    _ensure_dir(archived_path.parent)
    archived_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
    source_path.unlink()
    return archived_rel, archived_path


def _sync_contract(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "capture_model": "canonical_event_log",
        "write_policy": "trusted_staging_only",
        "projection_status_source": "structured_status",
        "stores": sorted(_enabled_projection_stores(config)),
    }


def _summarize_projection_status(capture_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    events_dir = capture_root / "events"
    stores: dict[str, dict[str, int]] = {}
    total_events = 0
    pending_events = 0
    needs_feedback = 0
    if events_dir.exists():
        for event_path in sorted(events_dir.glob("*.json")):
            event = _json_load(event_path, {})
            if not isinstance(event, dict):
                continue
            total_events += 1
            if event.get("needs_feedback"):
                needs_feedback += 1
            has_pending = False
            projection = ((event.get("projection") or {}).get("stores")) or {}
            for store_name, store_state in projection.items():
                state = str((store_state or {}).get("state") or "unknown")
                bucket = stores.setdefault(store_name, {})
                bucket[state] = bucket.get(state, 0) + 1
                if state == "pending":
                    has_pending = True
            if has_pending:
                pending_events += 1
    summary = {
        "captured_at": _iso_now(),
        "sync_contract": _sync_contract(config),
        "total_events": total_events,
        "pending_events": pending_events,
        "needs_feedback": needs_feedback,
        "stores": stores,
        "memory_hint": (
            "Hermes captures notes into a canonical internal event log first. "
            "Vault and second-brain outputs are downstream staged projections; "
            "check structured status for live sync health."
        ),
    }
    _json_dump(_projection_status_path(capture_root), summary)
    return summary


def _make_feedback_example(event: dict[str, Any], target: str, action: str) -> dict[str, Any]:
    tokens: list[str] = []
    for field in ("title", "excerpt"):
        for piece in re.findall(r"[A-Za-z0-9][A-Za-z0-9/_-]+", str(event.get(field) or "")):
            lowered = piece.lower()
            if lowered not in tokens:
                tokens.append(lowered)
    for tag in event.get("tags") or []:
        lowered = str(tag or "").strip().lower()
        if lowered and lowered not in tokens:
            tokens.append(lowered)
    return {
        "entry_id": event["entry_id"],
        "target": target,
        "action": action,
        "title": event.get("title", ""),
        "signals": tokens[:16],
        "excerpt": event.get("excerpt", ""),
        "tags": event.get("tags") or [],
        "learned_at": _iso_now(),
    }


def _feedback_files(state_root: Path) -> tuple[Path, Path]:
    return state_root / "feedback.jsonl", state_root / "feedback_index.json"


def _load_feedback_index(state_root: Path) -> dict[str, Any]:
    data = _json_load(state_root / "feedback_index.json", {})
    return data if isinstance(data, dict) else {}


def _save_feedback_index(state_root: Path, payload: dict[str, Any]) -> None:
    _json_dump(state_root / "feedback_index.json", payload)


def _audit_events_path(state_root: Path) -> Path:
    return state_root / "audit.jsonl"


def load_audit_events(
    *,
    vault_path: Path | None = None,
    config_path: str | Path | None = None,
    config_override: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], Path, dict[str, Any], Path]:
    vault = resolve_vault_path(vault_path)
    config, _resolved_config_path = load_config(vault, config_path=config_path, config_override=config_override)
    state_root = _ensure_dir(_state_root(vault, config))
    return _read_jsonl(_audit_events_path(state_root)), vault, config, state_root


def run_triage(
    *,
    vault_path: str | Path | None = None,
    config_path: str | Path | None = None,
    config_override: dict[str, Any] | None = None,
    classifier: Callable[..., dict[str, Any]] | None = None,
    target_resolver: Callable[..., dict[str, Any] | None] | None = None,
    dry_run: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Capture inbox notes into a canonical event log and stage projections."""
    vault = resolve_vault_path(vault_path)
    config, resolved_config_path = load_config(vault, config_path=config_path, config_override=config_override)
    state_root = _ensure_dir(_state_root(vault, config))
    capture_root = _ensure_dir(_capture_root(vault, config))
    source_archive_root = _ensure_dir(_source_archive_root(vault, config))
    structure = capture_structure_memory(vault, config, state_root=state_root)
    inbox = _inbox_path(vault, config)
    _ensure_dir(inbox)
    projection_stores = _enabled_projection_stores(config)

    run_id = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    min_conf = float(((config.get("routing") or {}).get("min_confidence")) or 0.7)
    low_conf_action = _normalize_low_confidence_action(config)

    processed: list[dict[str, Any]] = []
    note_paths = sorted(inbox.rglob("*.md"))
    if limit is not None:
        note_paths = note_paths[: max(limit, 0)]

    for note_path in note_paths:
        raw = note_path.read_text(encoding="utf-8")
        frontmatter, body = parse_frontmatter(raw)
        title_field = str(((config.get("frontmatter") or {}).get("title_field")) or "title")
        tags_field = str(((config.get("frontmatter") or {}).get("tags_field")) or "tags")
        title = _extract_title(frontmatter, body, note_path, title_field=title_field)
        tags = _extract_tags(frontmatter, tags_field)
        decision = _classify_note(
            title=title,
            body=body,
            tags=tags,
            vault_path=vault,
            config=config,
            state_root=state_root,
            structure=structure,
            classifier=classifier,
            target_resolver=target_resolver,
        )

        target = str(decision.get("target") or config.get("review_dir") or DEFAULT_REVIEW_DIR)
        confidence = float(decision.get("confidence", 0.0) or 0.0)
        needs_feedback = bool(decision.get("needs_feedback", False))
        if confidence < min_conf:
            needs_feedback = True
            if low_conf_action == "review":
                target = str(config.get("review_dir") or DEFAULT_REVIEW_DIR)
            elif low_conf_action == "inbox":
                target = _normalize_relative(note_path.parent, vault)

        frontmatter = dict(frontmatter or {})
        fields = config.get("frontmatter") or {}
        created_field = str(fields.get("created_field") or "created")
        updated_field = str(fields.get("updated_field") or "updated")
        status_field = str(fields.get("triage_status_field") or "para_triage_status")
        target_field = str(fields.get("triage_target_field") or "para_target")
        conf_field = str(fields.get("triage_confidence_field") or "para_confidence")
        reason_field = str(fields.get("triage_reason_field") or "para_reason")
        triaged_at_field = str(fields.get("triaged_at_field") or "para_triaged_at")

        frontmatter.setdefault(title_field, title)
        frontmatter.setdefault(created_field, _iso_now())
        frontmatter[updated_field] = _iso_now()
        frontmatter[status_field] = "needs-feedback" if needs_feedback else "captured"
        frontmatter[target_field] = target
        frontmatter[conf_field] = round(confidence, 3)
        frontmatter[reason_field] = str(decision.get("reason") or "").strip()
        frontmatter[triaged_at_field] = _iso_now()

        rendered = _render_note(frontmatter, body)
        entry_id = f"{run_id}-{uuid.uuid4().hex[:8]}"
        path_before = _normalize_relative(note_path, vault)
        default_store = next(iter(projection_stores))
        staged_projection_paths: dict[str, dict[str, Any]] = {}
        default_projected_rel = _projection_relative_path(target, note_path.name, projection_stores[default_store])
        for store_name, store_cfg in projection_stores.items():
            projected_rel = _projection_relative_path(target, note_path.name, store_cfg)
            staged_rel, _staged_path = _stage_projection_file(
                capture_root=capture_root,
                entry_id=entry_id,
                store_name=store_name,
                relative_path=projected_rel,
                content=rendered,
                dry_run=dry_run,
            )
            staged_projection_paths[store_name] = {
                "state": "pending",
                "target_relative_path": projected_rel,
                "staged_relative_path": staged_rel,
            }
        archived_rel, _archived_path = _archive_source_note(
            source_path=note_path,
            vault_path=vault,
            capture_root=capture_root,
            source_archive_root=source_archive_root,
            entry_id=entry_id,
            dry_run=dry_run,
        )

        capture_event = {
            "event_id": entry_id,
            "run_id": run_id,
            "captured_at": _iso_now(),
            "capture_model": "canonical_event_log",
            "status": "needs_feedback" if needs_feedback else "projection_pending",
            "needs_feedback": needs_feedback,
            "content_class": "vault_note",
            "vault_path": str(vault),
            "config_path": str(resolved_config_path),
            "title": title,
            "tags": tags,
            "source": {
                "original_relative_path": path_before,
                "archived_relative_path": archived_rel,
                "filename": note_path.name,
            },
            "routing": {
                "target": target,
                "confidence": confidence,
                "reason": str(decision.get("reason") or "").strip(),
                "source": str(decision.get("source") or "llm"),
            },
            "projection": {
                "stores": staged_projection_paths,
            },
            "canonical_content": rendered,
            "sync_contract": _sync_contract(config),
        }
        for key in _CANONICAL_ROUTING_FIELDS:
            value = decision.get(key)
            if value not in (None, ""):
                capture_event["routing"][key] = value
        if not dry_run:
            _json_dump(_capture_event_path(capture_root, entry_id), capture_event)

        event = {
            "entry_id": entry_id,
            "run_id": run_id,
            "timestamp": _iso_now(),
            "title": title,
            "tags": tags,
            "source": str(decision.get("source") or "llm"),
            "reason": str(decision.get("reason") or "").strip(),
            "confidence": confidence,
            "needs_feedback": needs_feedback,
            "path_before": path_before,
            "path_after": default_projected_rel,
            "target": target,
            "status": "dry-run" if dry_run else "captured",
            "excerpt": _excerpt(body, 300),
            "config_path": str(resolved_config_path),
            "archived_path": archived_rel,
            "stores": sorted(staged_projection_paths),
        }
        for key in _CANONICAL_ROUTING_FIELDS:
            value = decision.get(key)
            if value not in (None, ""):
                event[key] = value
        processed.append(event)
        if not dry_run:
            _jsonl_append(_audit_events_path(state_root), event)

    latest = {
        "run_id": run_id,
        "timestamp": _iso_now(),
        "processed": len(processed),
        "needs_feedback": sum(1 for item in processed if item["needs_feedback"]),
        "config_path": str(resolved_config_path),
        "vault_path": str(vault),
    }
    if not dry_run:
        _json_dump(state_root / "latest_run.json", latest)
        projection_summary = _summarize_projection_status(capture_root, config)
    else:
        projection_summary = {
            "sync_contract": _sync_contract(config),
            "total_events": len(processed),
            "pending_events": len(processed),
            "needs_feedback": sum(1 for item in processed if item["needs_feedback"]),
            "stores": {store_name: {"pending": len(processed)} for store_name in projection_stores},
        }

    return {
        "run_id": run_id,
        "vault_path": str(vault),
        "config_path": str(resolved_config_path),
        "state_root": str(state_root),
        "capture_root": str(capture_root),
        "processed": processed,
        "projection_status": projection_summary,
        "counts": {
            "processed": len(processed),
            "needs_feedback": sum(1 for item in processed if item["needs_feedback"]),
            "captured": sum(1 for item in processed if item["status"] == "captured"),
        },
        "dry_run": dry_run,
    }


def list_pending_feedback(
    *,
    vault_path: str | Path | None = None,
    config_path: str | Path | None = None,
    config_override: dict[str, Any] | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    events, _vault, _config, state_root = load_audit_events(
        vault_path=vault_path,
        config_path=config_path,
        config_override=config_override,
    )
    feedback_index = _load_feedback_index(state_root)
    pending: list[dict[str, Any]] = []
    for event in reversed(events):
        if not event.get("needs_feedback"):
            continue
        status = (feedback_index.get(event["entry_id"]) or {}).get("action")
        if status in {"approve", "correct", "ignore"}:
            continue
        pending.append(event)
        if len(pending) >= max(limit, 0):
            break
    return pending


def feedback_status(
    *,
    vault_path: str | Path | None = None,
    config_path: str | Path | None = None,
    config_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    events, vault, config, state_root = load_audit_events(
        vault_path=vault_path,
        config_path=config_path,
        config_override=config_override,
    )
    latest = _json_load(state_root / "latest_run.json", {})
    feedback_log = _read_jsonl(state_root / "feedback.jsonl")
    capture_root = _capture_root(vault, config)
    projection_status = _json_load(_projection_status_path(capture_root), {})
    return {
        "vault_path": str(vault),
        "pending": len(list_pending_feedback(vault_path=vault, config_path=config_path, config_override=config_override, limit=999999)),
        "audited": len(events),
        "feedback_events": len(feedback_log),
        "latest_run": latest or None,
        "projection_status": projection_status or None,
    }


def _find_event(events: list[dict[str, Any]], entry_id: str) -> dict[str, Any]:
    for event in events:
        if event.get("entry_id") == entry_id:
            return event
    raise KeyError(f"Audit entry not found: {entry_id}")


def _load_capture_event(capture_root: Path, entry_id: str) -> dict[str, Any]:
    payload = _json_load(_capture_event_path(capture_root, entry_id), {})
    return payload if isinstance(payload, dict) else {}


def _rewrite_projection_staging(
    *,
    vault: Path,
    config: dict[str, Any],
    capture_root: Path,
    capture_event: dict[str, Any],
    target: str,
) -> str:
    fields = config.get("frontmatter") or {}
    raw = str(capture_event.get("canonical_content") or "")
    frontmatter, body = parse_frontmatter(raw)
    frontmatter[str(fields.get("triage_status_field") or "para_triage_status")] = "corrected"
    frontmatter[str(fields.get("triage_target_field") or "para_target")] = target
    frontmatter[str(fields.get("updated_field") or "updated")] = _iso_now()
    rendered = _render_note(frontmatter, body)

    projection = ((capture_event.get("projection") or {}).get("stores")) or {}
    filename = str(((capture_event.get("source") or {}).get("filename")) or Path(str(capture_event.get("path_after") or "note.md")).name)
    stores = _enabled_projection_stores(config)
    relocated_path = ""
    for store_name, store_cfg in stores.items():
        projected_rel = _projection_relative_path(target, filename, store_cfg)
        staged_rel, staged_path = _stage_projection_file(
            capture_root=capture_root,
            entry_id=str(capture_event.get("event_id") or ""),
            store_name=store_name,
            relative_path=projected_rel,
            content=rendered,
            dry_run=False,
        )
        old_rel = str((projection.get(store_name) or {}).get("staged_relative_path") or "")
        if old_rel and old_rel != staged_rel:
            old_path = capture_root / old_rel
            if old_path.exists():
                old_path.unlink()
        projection[store_name] = {
            "state": "pending",
            "target_relative_path": projected_rel,
            "staged_relative_path": staged_rel,
        }
        if not relocated_path:
            relocated_path = projected_rel
    capture_event["canonical_content"] = rendered
    capture_event["status"] = "projection_pending"
    capture_event["needs_feedback"] = False
    capture_event["routing"] = dict(capture_event.get("routing") or {})
    capture_event["routing"]["target"] = target
    capture_event["projection"] = {"stores": projection}
    _json_dump(_capture_event_path(capture_root, str(capture_event.get("event_id") or "")), capture_event)
    _summarize_projection_status(capture_root, config)
    return relocated_path


def apply_feedback(
    *,
    action: str,
    entry_id: str,
    target: str | None = None,
    reviewer: str = "user",
    vault_path: str | Path | None = None,
    config_path: str | Path | None = None,
    config_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist reviewer feedback and update staged projection targets."""
    normalized_action = str(action or "").strip().lower()
    if normalized_action not in {"approve", "correct", "ignore"}:
        raise ValueError("action must be one of: approve, correct, ignore")

    events, vault, config, state_root = load_audit_events(
        vault_path=vault_path,
        config_path=config_path,
        config_override=config_override,
    )
    event = _find_event(events, entry_id)
    capture_root = _capture_root(vault, config)
    capture_event = _load_capture_event(capture_root, entry_id)
    resolved_target = None
    relocated_path = None
    if normalized_action == "correct":
        if not target:
            raise ValueError("correct requires a target path")
        resolved_target = _normalize_target(target, vault_path=vault, config=config)
        if not capture_event:
            raise KeyError(f"Capture event not found: {entry_id}")
        relocated_path = _rewrite_projection_staging(
            vault=vault,
            config=config,
            capture_root=capture_root,
            capture_event=capture_event,
            target=resolved_target,
        )
    elif normalized_action == "approve":
        resolved_target = str(event.get("target") or "")
        if capture_event:
            capture_event["needs_feedback"] = False
            capture_event["status"] = "projection_pending"
            _json_dump(_capture_event_path(capture_root, entry_id), capture_event)
            _summarize_projection_status(capture_root, config)

    feedback_record = {
        "entry_id": entry_id,
        "action": normalized_action,
        "reviewer": reviewer,
        "timestamp": _iso_now(),
        "target": resolved_target,
        "relocated_path": relocated_path,
    }
    _jsonl_append(state_root / "feedback.jsonl", feedback_record)
    feedback_index = _load_feedback_index(state_root)
    feedback_index[entry_id] = feedback_record
    _save_feedback_index(state_root, feedback_index)

    if normalized_action in {"approve", "correct"} and resolved_target:
        examples = _load_examples(state_root)
        new_example = _make_feedback_example(event, resolved_target, normalized_action)
        examples = [item for item in examples if item.get("entry_id") != entry_id]
        examples.append(new_example)
        _save_examples(state_root, examples[-200:])

    return feedback_record


def format_pending_feedback(items: list[dict[str, Any]]) -> str:
    if not items:
        return "No pending PARA feedback items."
    lines = ["Pending PARA feedback:"]
    for item in items:
        lines.append(
            f"- {item['entry_id']} | {item['path_after']} | conf={item['confidence']:.2f} | {item['reason']}"
        )
    return "\n".join(lines)


def format_run_report(summary: dict[str, Any]) -> str:
    counts = summary.get("counts") or {}
    processed = summary.get("processed") or []
    if not processed:
        return "[SILENT]"

    lines = [
        (
            f"Vault PARA triage run `{summary['run_id']}`: "
            f"{counts.get('captured', 0)} captured, {counts.get('needs_feedback', 0)} need feedback."
        ),
        f"Vault: {summary['vault_path']}",
    ]
    for item in processed[:12]:
        lines.append(
            f"- {item['entry_id']} | {item['path_before']} -> staged:{item['path_after']} "
            f"(conf={item['confidence']:.2f})"
        )
        if item.get("needs_feedback"):
            lines.append(
                f"  /para-feedback approve {item['entry_id']}  |  "
                f"/para-feedback correct {item['entry_id']} {item['target']}"
            )
    remaining = len(processed) - 12
    if remaining > 0:
        lines.append(f"...and {remaining} more item(s).")
    lines.append("On-demand audit: /para-feedback status | /para-feedback list")
    return "\n".join(lines)


def _format_status(status: dict[str, Any]) -> str:
    lines = [
        f"Vault: {status['vault_path']}",
        f"Audited entries: {status['audited']}",
        f"Pending feedback: {status['pending']}",
        f"Feedback events: {status['feedback_events']}",
    ]
    latest = status.get("latest_run")
    if latest:
        lines.append(
            f"Latest run: {latest.get('run_id')} at {latest.get('timestamp')} "
            f"({latest.get('processed', 0)} processed)"
        )
    projection = status.get("projection_status") or {}
    if projection:
        lines.append(
            f"Projection status: {projection.get('pending_events', 0)} pending event(s) "
            f"across {len(projection.get('stores') or {})} store(s)"
        )
    return "\n".join(lines)


def handle_feedback_command(raw_args: str) -> str:
    """Slash-command entrypoint used by the Slack/plugin feedback surface."""
    argv = shlex.split(raw_args or "")
    if not argv or argv[0] in {"help", "-h", "--help"}:
        return (
            "/para-feedback list [limit]\n"
            "/para-feedback approve <entry_id>\n"
            "/para-feedback correct <entry_id> <target>\n"
            "/para-feedback ignore <entry_id>\n"
            "/para-feedback status"
        )

    action = argv[0].lower()
    try:
        if action == "list":
            limit = int(argv[1]) if len(argv) > 1 else 20
            return format_pending_feedback(list_pending_feedback(limit=limit))
        if action == "status":
            return _format_status(feedback_status())
        if action == "approve":
            if len(argv) < 2:
                return "Usage: /para-feedback approve <entry_id>"
            record = apply_feedback(action="approve", entry_id=argv[1], reviewer="slack")
            return f"Approved {record['entry_id']}."
        if action == "correct":
            if len(argv) < 3:
                return "Usage: /para-feedback correct <entry_id> <target>"
            record = apply_feedback(
                action="correct",
                entry_id=argv[1],
                target=" ".join(argv[2:]),
                reviewer="slack",
            )
            return (
                f"Corrected {record['entry_id']} to {record.get('target')}."
                + (f" Relocated to {record['relocated_path']}." if record.get("relocated_path") else "")
            )
        if action == "ignore":
            if len(argv) < 2:
                return "Usage: /para-feedback ignore <entry_id>"
            record = apply_feedback(action="ignore", entry_id=argv[1], reviewer="slack")
            return f"Ignored {record['entry_id']}."
    except Exception as exc:
        return f"para-feedback error: {exc}"

    return "Unknown para-feedback subcommand. Use: list, approve, correct, ignore, status."


def install_cron_wrapper(
    *,
    vault_path: str | Path | None = None,
    config_path: str | Path | None = None,
    script_name: str = "vault-para-triage-nightly.py",
    output_format: str = "slack",
) -> Path:
    """Write a cron-safe launcher under ``HERMES_HOME/scripts``.

    Cron script jobs are constrained to that directory, so this helper bridges
    the optional-skill install location and the scheduler's trusted script root.
    """
    vault = resolve_vault_path(vault_path)
    resolved_config = _resolve_config_path(vault, config_path)
    scripts_dir = get_hermes_home() / "scripts"
    _ensure_dir(scripts_dir)

    clean_name = Path(script_name).name
    if not clean_name.endswith(".py"):
        clean_name = f"{clean_name}.py"
    target = (scripts_dir / clean_name).resolve()
    try:
        target.relative_to(scripts_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"Wrapper path escapes scripts dir: {script_name}") from exc

    content = (
        "#!/usr/bin/env python3\n"
        "\"\"\"Cron-safe PARA vault triage launcher.\"\"\"\n\n"
        "import sys\n\n"
        "from hermes_cli.vault_para_triage import main\n\n\n"
        "if __name__ == \"__main__\":\n"
        "    raise SystemExit(main([\n"
        f"        \"--vault\", {str(vault)!r},\n"
        f"        \"--config\", {str(resolved_config)!r},\n"
        "        \"run\",\n"
        f"        \"--format\", {output_format!r},\n"
        "    ]))\n"
    )
    target.write_text(content, encoding="utf-8")
    try:
        target.chmod(0o755)
    except OSError:
        pass
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PARA vault inbox triage helper")
    parser.add_argument("--vault", default=None, help="Absolute path to the Obsidian vault")
    parser.add_argument("--config", default=None, help="Path to para-triage.yaml")

    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser(
        "run",
        help="Process inbox notes into canonical capture events and staged PARA projections",
    )
    run_parser.add_argument("--dry-run", action="store_true", help="Do not write capture or staging files")
    run_parser.add_argument("--limit", type=int, default=None, help="Maximum inbox notes to process")
    run_parser.add_argument(
        "--format",
        choices=("text", "json", "slack"),
        default="text",
        help="Output format",
    )

    feedback_parser = sub.add_parser("feedback", help="Review or apply feedback to prior audit entries")
    feedback_sub = feedback_parser.add_subparsers(dest="feedback_action", required=True)

    feedback_list = feedback_sub.add_parser("list", help="List pending feedback items")
    feedback_list.add_argument("--limit", type=int, default=20)

    feedback_approve = feedback_sub.add_parser("approve", help="Approve an audit entry")
    feedback_approve.add_argument("entry_id")

    feedback_correct = feedback_sub.add_parser("correct", help="Correct an audit entry target")
    feedback_correct.add_argument("entry_id")
    feedback_correct.add_argument("target")

    feedback_ignore = feedback_sub.add_parser("ignore", help="Ignore an audit entry")
    feedback_ignore.add_argument("entry_id")

    sub.add_parser("status", help="Summarize audit and feedback state")

    install_wrapper = sub.add_parser(
        "install-wrapper",
        help="Write a cron-safe launcher into HERMES_HOME/scripts",
    )
    install_wrapper.add_argument(
        "--name",
        default="vault-para-triage-nightly.py",
        help="Filename to create under HERMES_HOME/scripts",
    )
    install_wrapper.add_argument(
        "--format",
        choices=("text", "json", "slack"),
        default="slack",
        help="Output format used by the generated nightly wrapper",
    )

    args = parser.parse_args(argv)

    if args.command == "run":
        summary = run_triage(
            vault_path=args.vault,
            config_path=args.config,
            dry_run=args.dry_run,
            limit=args.limit,
        )
        if args.format == "json":
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            print(format_run_report(summary))
        return 0

    if args.command == "status":
        print(_format_status(feedback_status(vault_path=args.vault, config_path=args.config)))
        return 0

    if args.command == "install-wrapper":
        target = install_cron_wrapper(
            vault_path=args.vault,
            config_path=args.config,
            script_name=args.name,
            output_format=args.format,
        )
        print(str(target))
        return 0

    if args.command == "feedback":
        if args.feedback_action == "list":
            print(
                format_pending_feedback(
                    list_pending_feedback(
                        vault_path=args.vault,
                        config_path=args.config,
                        limit=args.limit,
                    )
                )
            )
            return 0
        if args.feedback_action == "approve":
            result = apply_feedback(
                action="approve",
                entry_id=args.entry_id,
                reviewer="cli",
                vault_path=args.vault,
                config_path=args.config,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.feedback_action == "correct":
            result = apply_feedback(
                action="correct",
                entry_id=args.entry_id,
                target=args.target,
                reviewer="cli",
                vault_path=args.vault,
                config_path=args.config,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.feedback_action == "ignore":
            result = apply_feedback(
                action="ignore",
                entry_id=args.entry_id,
                reviewer="cli",
                vault_path=args.vault,
                config_path=args.config,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
