from __future__ import annotations

import csv
import hashlib
import ipaddress
import json
import os
import re
import secrets
import shutil
import socket
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    from hermes_constants import get_hermes_home
except Exception:
    def get_hermes_home() -> Path:
        return Path.home() / ".hermes"


PLUGIN_ID = "research-desk"
TOOLSET = "research-desk"
CLASSIFICATION = "public_research"
MAX_TOPIC_CHARS = 240
MAX_TARGETS = 20
MAX_DOMAINS = 12
MAX_WORKERS = 4
MAX_SOURCES = 12
MAX_EXTRACT_CHARS = 16_000
MAX_PROMPT_CHARS = 24_000
PLAN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{7,63}$", re.IGNORECASE)
RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{7,63}$", re.IGNORECASE)
SECRET_RE = re.compile(
    r"(?i)(api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret)\s*[:=]\s*[^\s,;]+"
)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)")
WINDOWS_USER_PATH_RE = re.compile(r"(?i)([A-Z]:[\\/]+Users[\\/])[^\\/\s]+")
POSIX_HOME_PATH_RE = re.compile(r"(?i)(/(?:home|Users)/)[^/\s]+")


STATUS_SCHEMA = {
    "name": "research_desk_status",
    "description": "Report the active profile, configured workspace, pinned worker revision, evidence policy, and readiness.",
    "parameters": {"type": "object", "properties": {}},
}

PLAN_SCHEMA = {
    "name": "research_desk_plan",
    "description": (
        "Validate a public-research report plan without network access or worker execution. "
        "Return a plan id and stable plan hash for later approval."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "Research topic."},
            "targets": {"type": "array", "items": {"type": "string"}, "maxItems": MAX_TARGETS},
            "source_domains": {"type": "array", "items": {"type": "string"}, "maxItems": MAX_DOMAINS},
            "frequency": {"type": "string", "enum": ["ad_hoc", "weekly", "monthly"]},
            "worker_count": {"type": "integer", "minimum": 1, "maximum": MAX_WORKERS},
            "output_format": {"type": "string", "enum": ["markdown", "json", "csv"]},
        },
        "required": ["topic"],
    },
}

RUN_SCHEMA = {
    "name": "research_desk_run",
    "description": (
        "Run an approved public-research plan through the Hermes evidence boundary and "
        "the LLM-only-network OpenManus worker mode. Hermes LLM synthesis is primary and explicit approval is required."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "plan_id": {"type": "string", "description": "Previously returned plan id."},
            "approved": {"type": "boolean", "description": "Human approval of the plan and run."},
            "acknowledge_side_effects": {"type": "boolean", "description": "Acknowledge public evidence retrieval and report writes."},
        },
        "required": ["plan_id", "approved", "acknowledge_side_effects"],
    },
}

EXPORT_SCHEMA = {
    "name": "research_desk_export",
    "description": (
        "Export a completed report only after explicit human approval. The export remains "
        "inside the configured Private Runner workspace."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "run_id": {"type": "string", "description": "Completed Research Desk run id."},
            "format": {"type": "string", "enum": ["markdown", "json", "csv"]},
            "approved": {"type": "boolean", "description": "Human approval to create the export."},
        },
        "required": ["run_id", "format", "approved"],
    },
}


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _hash(value: Any) -> str:
    if isinstance(value, bytes):
        data = value
    elif isinstance(value, str):
        data = value.encode("utf-8")
    else:
        data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _safe_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "Z"
    return f"{prefix}-{stamp}-{secrets.token_hex(4)}"


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        return SECRET_RE.sub(lambda match: f"{match.group(1)}=[REDACTED]", value)
    if isinstance(value, dict):
        return {str(key): _redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _redact_sensitive_text(value: str) -> str:
    value = SECRET_RE.sub(lambda match: f"{match.group(1)}=[REDACTED]", value)
    value = EMAIL_RE.sub("[REDACTED_EMAIL]", value)
    value = PHONE_RE.sub("[REDACTED_PHONE]", value)
    value = WINDOWS_USER_PATH_RE.sub(r"\1[REDACTED_USER]", value)
    return POSIX_HOME_PATH_RE.sub(r"\1[REDACTED_USER]", value)


def _redact_sensitive(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_sensitive_text(value)
    if isinstance(value, dict):
        return {str(key): _redact_sensitive(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_sensitive(item) for item in value]
    return value


def _public_source_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path or "/", "", ""))


def _entry() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly() or {}
    except Exception:
        cfg = {}
    plugins = cfg.get("plugins") if isinstance(cfg, dict) else {}
    entries = plugins.get("entries") if isinstance(plugins, dict) else {}
    value = entries.get(PLUGIN_ID) if isinstance(entries, dict) else {}
    return value if isinstance(value, dict) else {}


def _all_entries() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly() or {}
    except Exception:
        return {}
    plugins = cfg.get("plugins") if isinstance(cfg, dict) else {}
    entries = plugins.get("entries") if isinstance(plugins, dict) else {}
    return entries if isinstance(entries, dict) else {}


def _openmanus_entry() -> dict[str, Any]:
    value = _all_entries().get("openmanus")
    return value if isinstance(value, dict) else {}


def _worker_policy() -> tuple[bool, bool]:
    return bool(_openmanus_entry().get("allow_llm_network")), bool(_entry().get("pass_worker_model_secret"))


def _require_primary_llm(ctx) -> None:
    llm = getattr(ctx, "llm", None)
    if llm is None or not callable(getattr(llm, "complete_structured", None)):
        raise RuntimeError("Hermes LLM is required as the primary Research Desk engine")


def _source_revision() -> str:
    try:
        from plugins.openmanus.core import _source_revision as openmanus_revision

        return str(openmanus_revision())
    except Exception:
        return "unknown"


def _profile(ctx) -> str:
    return str(getattr(ctx, "profile_name", "default") or "default")


def _reject_identity(args: dict[str, Any]) -> None:
    for key in ("customer_id", "tenant_id", "organisation_id", "organization_id", "profile", "profile_name"):
        if key in args:
            raise ValueError(f"{key} is not accepted; the active Hermes profile is the execution principal")


def _configured_path(raw: Any, label: str) -> Path:
    value = str(raw or "").strip()
    if not value:
        raise ValueError(f"configure plugins.entries.{PLUGIN_ID}.{label}")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = get_hermes_home() / candidate
    resolved = candidate.resolve(strict=False)
    if candidate.is_symlink():
        raise ValueError(f"configured {label} must not be a symlink")
    if not resolved.is_dir():
        raise ValueError(f"configured {label} does not exist: {resolved}")
    return resolved


def _workspace(ctx, *, require_openmanus: bool = True) -> Path:
    entry = _entry()
    root = _configured_path(entry.get("workspace_root"), "workspace_root")
    profile_entry = entry.get("profile_name") or entry.get("profile")
    if profile_entry and str(profile_entry) != _profile(ctx):
        raise ValueError("configured Research Desk profile does not match the active Hermes profile")
    if require_openmanus:
        openmanus = _all_entries().get("openmanus") or {}
        open_root = _configured_path(openmanus.get("workspace_root"), "openmanus.workspace_root")
        try:
            root.relative_to(open_root)
        except ValueError as exc:
            raise ValueError("Research Desk workspace must be inside the configured OpenManus workspace") from exc
    return root


def _confined(root: Path, candidate: Path, *, must_exist: bool = False) -> Path:
    if candidate.is_symlink():
        raise ValueError("symlink paths are not accepted for Research Desk artifacts")
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("path escapes the configured Research Desk workspace") from exc
    if must_exist and not resolved.exists():
        raise ValueError(f"path does not exist: {resolved}")
    return resolved


def _domains() -> list[str]:
    raw = _entry().get("allowed_domains") or []
    if not isinstance(raw, list):
        raise ValueError("allowed_domains must be a list")
    values: list[str] = []
    for value in raw:
        domain = str(value).strip().lower().rstrip(".")
        if not domain or domain.startswith(".") or "*" in domain or "/" in domain:
            raise ValueError("allowed_domains must contain plain hostnames without wildcards")
        try:
            ipaddress.ip_address(domain)
        except ValueError:
            if "." not in domain:
                raise ValueError("allowed_domains must contain public hostnames")
        if domain not in values:
            values.append(domain)
    return values[:MAX_DOMAINS]


def _domain_allowed(url: str, domains: list[str]) -> bool:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    if parsed.username or parsed.password:
        return False
    if parsed.port not in {None, 80, 443}:
        return False
    host = parsed.hostname.lower().rstrip(".")
    try:
        address = ipaddress.ip_address(host)
        if address.is_private or address.is_loopback or address.is_link_local or address.is_reserved or address.is_multicast:
            return False
    except ValueError:
        pass
    try:
        from tools.url_safety import is_safe_url

        if not is_safe_url(url):
            return False
    except Exception:
        # The web-tool boundary remains the final network gate. A missing
        # optional safety helper must not make an allowlist less restrictive.
        pass
    return any(host == domain or host.endswith("." + domain) for domain in domains)


def _normalise_domains(values: Any, configured: list[str]) -> list[str]:
    if values is None or values == []:
        result = list(configured)
    elif not isinstance(values, list):
        raise ValueError("source_domains must be a list")
    else:
        result = []
        for value in values:
            domain = str(value).strip().lower().rstrip(".")
            if domain not in configured:
                raise ValueError(f"source domain is not configured: {domain}")
            if domain not in result:
                result.append(domain)
    if not result:
        raise ValueError("configure at least one allowed public research domain")
    return result[:MAX_DOMAINS]


def _normalise_plan(ctx, args: dict[str, Any]) -> dict[str, Any]:
    _reject_identity(args)
    topic = _redact_sensitive_text(str(args.get("topic") or "").strip())
    if not topic:
        raise ValueError("topic is required")
    if len(topic) > MAX_TOPIC_CHARS:
        raise ValueError(f"topic exceeds {MAX_TOPIC_CHARS} characters")
    targets = args.get("targets") or []
    if not isinstance(targets, list) or len(targets) > MAX_TARGETS:
        raise ValueError(f"targets must contain at most {MAX_TARGETS} items")
    clean_targets = [_redact_sensitive_text(str(value).strip()) for value in targets if str(value).strip()]
    workers = int(args.get("worker_count") or 2)
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"worker_count must be between 1 and {MAX_WORKERS}")
    frequency = str(args.get("frequency") or "weekly")
    if frequency not in {"ad_hoc", "weekly", "monthly"}:
        raise ValueError("frequency must be ad_hoc, weekly, or monthly")
    output_format = str(args.get("output_format") or "markdown")
    if output_format not in {"markdown", "json", "csv"}:
        raise ValueError("output_format must be markdown, json, or csv")
    workspace = _workspace(ctx, require_openmanus=True)
    domains = _normalise_domains(args.get("source_domains"), _domains())
    return {
        "schema_version": "research-desk.plan.v1",
        "classification": CLASSIFICATION,
        "profile": _profile(ctx),
        "topic": topic,
        "targets": clean_targets,
        "source_domains": domains,
        "frequency": frequency,
        "worker_count": workers,
        "output_format": output_format,
        "workspace": str(workspace),
        "created_at": _now(),
    }


def _plan_root() -> Path:
    root = get_hermes_home() / "research-desk" / "plans"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _receipt_root() -> Path:
    root = get_hermes_home() / "research-desk" / "receipts"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _save_plan(plan: dict[str, Any]) -> dict[str, Any]:
    plan_id = _safe_id("plan")
    digest = _hash(plan)
    path = _plan_root() / f"{plan_id}.json"
    path.write_text(_json({"plan_id": plan_id, "plan_hash": digest, "plan": plan}) + "\n", encoding="utf-8")
    return {
        "ok": True,
        "status": "approval_required",
        "plan_id": plan_id,
        "plan_hash": digest,
        "profile": plan["profile"],
        "classification": CLASSIFICATION,
        "topic": plan["topic"],
        "source_domains": plan["source_domains"],
        "external_communication": False,
        "openmanus_started": False,
        "plan_path": str(path),
    }


def _load_plan(ctx, plan_id: str) -> tuple[dict[str, Any], Path]:
    if not PLAN_ID_RE.fullmatch(plan_id):
        raise ValueError("invalid plan_id")
    path = _plan_root() / f"{plan_id}.json"
    if not path.is_file() or path.is_symlink():
        raise ValueError("plan_id was not found in the active Hermes home")
    payload = json.loads(path.read_text(encoding="utf-8"))
    plan = payload.get("plan") if isinstance(payload, dict) else None
    if not isinstance(plan, dict) or plan.get("profile") != _profile(ctx):
        raise ValueError("plan profile does not match the active Hermes profile")
    if payload.get("plan_hash") != _hash(plan):
        raise ValueError("plan hash mismatch")
    return plan, path


def _parse_json(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", value, flags=re.DOTALL | re.IGNORECASE)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                return {}
    return {}


def _dispatch(ctx, name: str, args: dict[str, Any], task_id: str) -> Any:
    return _parse_json(ctx.dispatch_tool(name, args, task_id=task_id))


def _search_rows(value: Any) -> list[dict[str, str]]:
    data = value.get("data") if isinstance(value, dict) else {}
    rows = data.get("web") if isinstance(data, dict) else None
    if not isinstance(rows, list) and isinstance(value, dict):
        rows = value.get("results") or value.get("items")
    if not isinstance(rows, list):
        return []
    result: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        url = row.get("url") or row.get("href") or ""
        if not isinstance(url, str) or not url.strip():
            continue
        result.append({
            "url": url.strip(),
            "title": str(row.get("title") or "").strip(),
            "description": str(row.get("description") or row.get("snippet") or "").strip(),
        })
    return result


def _extract_rows(value: Any) -> list[dict[str, str]]:
    rows = value.get("results") if isinstance(value, dict) else None
    if not isinstance(rows, list) and isinstance(value, dict):
        rows = value.get("data")
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list):
        return []
    result: list[dict[str, str]] = []
    for row in rows:
        if isinstance(row, str):
            result.append({"url": "", "title": "", "content": row[:MAX_EXTRACT_CHARS]})
        elif isinstance(row, dict):
            result.append({
                "url": str(row.get("url") or "").strip(),
                "title": str(row.get("title") or "").strip(),
                "content": str(row.get("content") or row.get("text") or row.get("markdown") or "")[:MAX_EXTRACT_CHARS],
            })
    return result


def _queries(plan: dict[str, Any]) -> list[str]:
    suffix = " ".join(plan.get("targets") or [])
    base = f"{plan['topic']} {suffix}".strip()
    return [f"{base} competitor pricing", f"{base} hiring jobs", f"{base} industry news"]


def _collect_evidence(ctx, plan: dict[str, Any], task_id: str) -> tuple[list[dict[str, Any]], list[str]]:
    domains = plan["source_domains"]
    candidates: list[dict[str, str]] = []
    errors: list[str] = []
    for query in _queries(plan):
        try:
            candidates.extend(_search_rows(_dispatch(ctx, "web_search", {"query": query, "limit": MAX_SOURCES}, task_id)))
        except Exception as exc:
            errors.append(f"search:{type(exc).__name__}")
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in candidates:
        source_url = row["url"]
        url = _public_source_url(source_url)
        if not _domain_allowed(source_url, domains) or url in seen:
            continue
        seen.add(url)
        selected.append({**row, "url": url, "title": _redact_sensitive_text(row.get("title") or "")})
        if len(selected) >= MAX_SOURCES:
            break
    if not selected:
        return [], errors + ["no allowlisted public sources returned"]
    evidence: list[dict[str, Any]] = []
    for start in range(0, len(selected), 5):
        batch = selected[start : start + 5]
        try:
            extracted = _extract_rows(_dispatch(ctx, "web_extract", {"urls": [row["url"] for row in batch], "char_limit": MAX_EXTRACT_CHARS}, task_id))
        except Exception as exc:
            errors.append(f"extract:{type(exc).__name__}")
            extracted = []
        by_url = {
            _public_source_url(str(row.get("url") or "")): row
            for row in extracted
            if row.get("url")
        }
        for row in batch:
            content_row = by_url.get(row["url"], {})
            content = _redact_sensitive_text(str(content_row.get("content") or "").strip())
            if not content:
                continue
            evidence.append({
                "source_id": f"source-{len(evidence) + 1:02d}",
                "url": row["url"],
                "title": _redact_sensitive_text(content_row.get("title") or row.get("title") or row["url"]),
                "accessed_at": _now(),
                "source_class": CLASSIFICATION,
                "content": content,
                "content_sha256": _hash(content),
            })
    if not evidence:
        errors.append("allowlisted sources returned no extractable content")
    return evidence, errors


def _source_metadata(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "source_id": row["source_id"],
            "url": row["url"],
            "title": _redact_sensitive_text(row["title"]),
            "accessed_at": row["accessed_at"],
            "source_class": row["source_class"],
            "content_sha256": row["content_sha256"],
        }
        for row in evidence
    ]


def _worker_prompt(plan: dict[str, Any], evidence: list[dict[str, Any]], index: int) -> str:
    compact = [
        {"source_id": row["source_id"], "url": row["url"], "title": row["title"], "text": row["content"][:3000]}
        for row in evidence
    ]
    roles = ["competitor and pricing analyst", "jobs and hiring analyst", "industry-news analyst", "evidence quality analyst"]
    packet = json.dumps(compact, ensure_ascii=False, sort_keys=True)
    return (
        "You are a Hermes-managed Research Desk worker. Use only the redacted evidence packet below. "
        "Do not browse, call web or MCP tools, inspect the environment, or invent sources. "
        "Return JSON with findings, where every finding has claim, source_refs, confidence, and caveat. "
        f"Your role is {roles[index % len(roles)]}. Topic: {plan['topic']}. "
        f"Evidence packet: {packet}"
    )[:MAX_PROMPT_CHARS]


def _worker_items(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, dict):
        return []
    rows = raw.get("items")
    return rows if isinstance(rows, list) else []


def _worker_payload(item: dict[str, Any]) -> dict[str, Any]:
    for key in ("result", "stdout", "text", "output"):
        value = item.get(key)
        parsed = _parse_json(value)
        if isinstance(parsed, dict) and parsed:
            return parsed
    return {}


def _valid_findings(payload: Any, source_ids: set[str]) -> list[dict[str, Any]]:
    rows = payload.get("findings") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []
    result: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        claim = _redact_sensitive_text(str(row.get("claim") or "").strip())
        refs = row.get("source_refs") or []
        if not claim or not isinstance(refs, list):
            continue
        clean_refs = [str(ref) for ref in refs if str(ref) in source_ids]
        if not clean_refs:
            continue
        result.append({
            "claim": claim[:2000],
            "source_refs": clean_refs,
            "confidence": str(row.get("confidence") or "unknown")[:40],
            "caveat": _redact_sensitive_text(str(row.get("caveat") or "").strip())[:1000],
        })
    return result


def _synthesise(ctx, plan: dict[str, Any], evidence: list[dict[str, Any]], workers: list[dict[str, Any]], task_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_ids = {row["source_id"] for row in evidence}
    worker_findings: list[dict[str, Any]] = []
    worker_summaries: list[str] = []
    for item in workers:
        payload = _worker_payload(item)
        worker_findings.extend(_valid_findings(payload, source_ids))
        if payload.get("summary"):
            worker_summaries.append(_redact_sensitive_text(str(payload["summary"])[:1000]))
    synthesis_meta: dict[str, Any] = {
        "mode": "worker_fallback",
        "primary_engine": "hermes_llm",
        "worker_findings": len(worker_findings),
    }
    if not hasattr(ctx, "llm"):
        return worker_findings, synthesis_meta
    source_index = [{"source_id": row["source_id"], "url": row["url"], "title": row["title"]} for row in evidence]
    input_text = json.dumps({"sources": source_index, "worker_findings": worker_findings, "worker_summaries": worker_summaries}, ensure_ascii=False)
    schema = {
        "type": "object",
        "properties": {
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string"},
                        "source_refs": {"type": "array", "items": {"type": "string"}},
                        "confidence": {"type": "string"},
                        "caveat": {"type": "string"},
                    },
                    "required": ["claim", "source_refs", "confidence", "caveat"],
                },
            }
        },
        "required": ["findings"],
    }
    try:
        response = ctx.llm.complete_structured(
            instructions=(
                "Create a public-research report finding list from the supplied evidence index and worker observations. "
                "Every claim must cite one or more exact source_id values. Drop unsupported claims. Do not provide legal, tax, financing, or incorporation advice."
            ),
            input=[{"type": "text", "text": input_text}],
            json_schema=schema,
            json_mode=True,
            schema_name="research_desk_findings",
            max_tokens=3000,
            purpose="research-desk-synthesis",
        )
        parsed = response.parsed if getattr(response, "parsed", None) is not None else _parse_json(getattr(response, "text", ""))
        findings = _valid_findings(parsed, source_ids)
        if findings:
            synthesis_meta.update({"mode": "host_llm_structured", "provider": getattr(response, "provider", ""), "model": getattr(response, "model", ""), "finding_count": len(findings)})
            return findings, synthesis_meta
    except Exception as exc:
        synthesis_meta["error_type"] = type(exc).__name__
    return worker_findings, synthesis_meta


def _write_receipt(run_id: str, payload: dict[str, Any]) -> Path:
    path = _receipt_root() / f"{run_id}.json"
    path.write_text(_json(_redact_sensitive(_redact(payload))) + "\n", encoding="utf-8")
    return path


def _receipt_payload(ctx, *, run_id: str, plan: dict[str, Any], status: str, plan_hash: str, source_metadata: list[dict[str, Any]], outputs: dict[str, str] | None = None, workers: list[dict[str, Any]] | None = None, synthesis: dict[str, Any] | None = None, errors: list[str] | None = None, approval: dict[str, Any] | None = None) -> dict[str, Any]:
    openmanus_revision = "unknown"
    openmanus_revision = _source_revision()
    return {
        "schema_version": "research-desk.receipt.v1",
        "run_id": run_id,
        "status": status,
        "profile": _profile(ctx),
        "classification": CLASSIFICATION,
        "plan_hash": plan_hash,
        "topic_sha256": _hash(plan.get("topic", "")),
        "openmanus_revision": openmanus_revision,
        "sources": source_metadata,
        "inputs": {"plan_sha256": plan_hash, "source_metadata_sha256": _hash(source_metadata)},
        "outputs": outputs or {},
        "workers": workers or [],
        "synthesis": synthesis or {},
        "approval": approval or {},
        "errors": errors or [],
        "created_at": _now(),
        "raw_customer_content_included": False,
    }


def status(ctx) -> dict[str, Any]:
    llm_network_ready, worker_secret_opt_in = _worker_policy()
    payload: dict[str, Any] = {
        "ok": True,
        "plugin": PLUGIN_ID,
        "profile": _profile(ctx),
        "classification": CLASSIFICATION,
        "evidence_policy": {"public_only": True, "allowlist_required": True, "source_hashes": True},
        "approval_policy": {"plan": True, "run": True, "export": True},
        "worker_policy": {
            "network": True,
            "network_scope": "llm_only",
            "web_and_mcp_tools": False,
            "secret_environment": "single_model_key_opt_in" if worker_secret_opt_in else False,
            "engine": "openmanus",
            "primary_engine": "hermes_llm",
            "llm_network_ready": llm_network_ready,
        },
        "recurring_orchestrator": "hermes-cron",
    }
    try:
        workspace = _workspace(ctx, require_openmanus=True)
        payload["workspace"] = str(workspace)
        payload["workspace_ready"] = True
    except Exception as exc:
        payload["workspace"] = ""
        payload["workspace_ready"] = False
        payload["workspace_error"] = str(exc)
    try:
        from plugins.openmanus.core import check_available

        payload["openmanus_revision"] = _source_revision()
        payload["openmanus_available"] = bool(check_available())
    except Exception as exc:
        payload["openmanus_revision"] = "unknown"
        payload["openmanus_available"] = False
        payload["openmanus_error"] = str(exc)
    try:
        configured_domains = _domains()
    except Exception as exc:
        configured_domains = []
        payload["workspace_error"] = payload.get("workspace_error") or str(exc)
    payload["runnable"] = bool(
        payload["workspace_ready"]
        and payload["openmanus_available"]
        and configured_domains
        and llm_network_ready
    )
    if not configured_domains:
        payload["readiness_note"] = "configure plugins.entries.research-desk.allowed_domains before planning a run"
    elif not llm_network_ready:
        payload["readiness_note"] = "enable plugins.entries.openmanus.allow_llm_network for the OpenManus LLM endpoint"
    return payload


def plan(ctx, args: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        normalized = _normalise_plan(ctx, args or {})
        return _save_plan(normalized)
    except Exception as exc:
        return {"ok": False, "status": "blocked", "external_communication": False, "openmanus_started": False, "error": str(exc)}


def run(ctx, args: dict[str, Any] | None = None) -> dict[str, Any]:
    args = args or {}
    run_id = _safe_id("run")
    plan_hash = ""
    evidence: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        _reject_identity(args)
        if not bool(args.get("approved")):
            raise PermissionError("approved=true is required for a Research Desk run")
        if not bool(args.get("acknowledge_side_effects")):
            raise PermissionError("acknowledge_side_effects=true is required for public retrieval and report writes")
        plan_value, _ = _load_plan(ctx, str(args.get("plan_id") or ""))
        plan_hash = _hash(plan_value)
        workspace = _workspace(ctx, require_openmanus=True)
        _require_primary_llm(ctx)
        llm_network_ready, worker_secret_opt_in = _worker_policy()
        if not llm_network_ready:
            raise PermissionError("OpenManus worker LLM network access is not enabled")
        run_root = _confined(workspace, workspace / "research-desk" / "runs" / run_id)
        run_root.mkdir(parents=True, exist_ok=True)
        task_id = f"research-desk-{run_id}"
        evidence, collect_errors = _collect_evidence(ctx, plan_value, task_id)
        errors.extend(collect_errors)
        if not evidence:
            receipt = _receipt_payload(ctx, run_id=run_id, plan=plan_value, status="failed", plan_hash=plan_hash, source_metadata=[], errors=errors, approval={"approved": True, "acknowledged": True})
            receipt_path = _write_receipt(run_id, receipt)
            return {"ok": False, "status": "failed", "run_id": run_id, "receipt_path": str(receipt_path), "errors": errors}
        packet = {
            "schema_version": "research-desk.evidence-packet.v1",
            "classification": CLASSIFICATION,
            "profile": _profile(ctx),
            "plan_hash": plan_hash,
            "sources": evidence,
            "instructions": "Use only redacted evidence, do not browse or call web/MCP tools, and cite source_id values.",
        }
        packet_path = run_root / "evidence_packet.json"
        packet_path.write_text(_json(packet) + "\n", encoding="utf-8")
        items = [_worker_prompt(plan_value, evidence, index) for index in range(plan_value["worker_count"])]
        worker_result = _dispatch(
            ctx,
            "openmanus_wide_research",
            {
                "items": items,
                "workspace": str(run_root),
                "dry_run": False,
                "allow_side_effects": True,
                "acknowledge_side_effects": True,
                "allow_network": True,
                "network_scope": "llm_only",
                "max_parallel": plan_value["worker_count"],
                "synthesize": False,
                "no_secret_env": not worker_secret_opt_in,
            },
            task_id,
        )
        if not isinstance(worker_result, dict) or not bool(worker_result.get("ok")):
            worker_status = worker_result.get("status") if isinstance(worker_result, dict) else "invalid"
            errors.append(f"openmanus_worker:{str(worker_status)[:80]}")
        workers = _worker_items(worker_result)
        findings, synthesis = _synthesise(ctx, plan_value, evidence, workers, task_id)
        report = {
            "schema_version": "research-desk.report.v1",
            "run_id": run_id,
            "profile": _profile(ctx),
            "classification": CLASSIFICATION,
            "topic": plan_value["topic"],
            "generated_at": _now(),
            "findings": findings,
            "sources": _source_metadata(evidence),
            "worker_count": len(workers),
            "synthesis": {
                "mode": synthesis.get("mode", "worker_fallback"),
                "primary_engine": "hermes_llm",
            },
            "limitations": ["Public sources only.", "Claims without valid source references were omitted."],
        }
        report_json_path = run_root / "report.json"
        report_md_path = run_root / "report.md"
        report_json_path.write_text(_json(report) + "\n", encoding="utf-8")
        lines = [f"# Research Desk: {plan_value['topic']}", "", f"Profile: `{_profile(ctx)}`", f"Classification: `{CLASSIFICATION}`", "", "## Findings", ""]
        if findings:
            for finding in findings:
                refs = ", ".join(finding["source_refs"])
                caveat = f" Caveat: {finding['caveat']}" if finding.get("caveat") else ""
                lines.append(f"- {finding['claim']} (sources: {refs}; confidence: {finding['confidence']}).{caveat}")
        else:
            lines.append("No supported findings were produced.")
        lines.extend(["", "## Sources", ""])
        for source in _source_metadata(evidence):
            lines.append(f"- {source['source_id']}: [{source['title']}]({source['url']}) — accessed {source['accessed_at']}")
        report_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        outputs = {"report_json_sha256": _hash(report_json_path.read_bytes()), "report_markdown_sha256": _hash(report_md_path.read_bytes())}
        worker_receipts = [{"index": index, "status": item.get("status", "unknown"), "output_sha256": _hash(_redact(item))} for index, item in enumerate(workers)]
        receipt = _receipt_payload(ctx, run_id=run_id, plan=plan_value, status="completed", plan_hash=plan_hash, source_metadata=_source_metadata(evidence), outputs=outputs, workers=worker_receipts, synthesis={"mode": synthesis.get("mode", "worker_fallback"), "history_sha256": _hash(synthesis)}, errors=errors, approval={"approved": True, "acknowledged": True})
        receipt_path = _write_receipt(run_id, receipt)
        return {"ok": True, "status": "completed", "run_id": run_id, "report_path": str(report_json_path), "evidence_packet_path": str(packet_path), "receipt_path": str(receipt_path), "finding_count": len(findings), "source_count": len(evidence)}
    except Exception as exc:
        receipt = _receipt_payload(ctx, run_id=run_id, plan={"topic": "", "profile": _profile(ctx)}, status="blocked", plan_hash=plan_hash, source_metadata=_source_metadata(evidence), errors=errors + [f"{type(exc).__name__}: {exc}"], approval={"approved": bool(args.get("approved")), "acknowledged": bool(args.get("acknowledge_side_effects"))})
        receipt_path = _write_receipt(run_id, receipt)
        return {"ok": False, "status": "blocked", "run_id": run_id, "receipt_path": str(receipt_path), "error": str(exc)}


def _load_run(ctx, run_id: str) -> tuple[Path, dict[str, Any]]:
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError("invalid run_id")
    root = _workspace(ctx, require_openmanus=True)
    run_root = _confined(root, root / "research-desk" / "runs" / run_id, must_exist=True)
    report_path = _confined(run_root, run_root / "report.json", must_exist=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("profile") != _profile(ctx):
        raise ValueError("run profile does not match the active Hermes profile")
    receipt_path = _receipt_root() / f"{run_id}.json"
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise ValueError("run receipt was not found in the active Hermes home")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "completed" or not bool((receipt.get("approval") or {}).get("approved")):
        raise ValueError("run is not a completed, approved Research Desk run")
    return run_root, report


def export(ctx, args: dict[str, Any] | None = None) -> dict[str, Any]:
    args = args or {}
    try:
        _reject_identity(args)
        if not bool(args.get("approved")):
            raise PermissionError("approved=true is required for export")
        fmt = str(args.get("format") or "markdown")
        if fmt not in {"markdown", "json", "csv"}:
            raise ValueError("format must be markdown, json, or csv")
        run_root, report = _load_run(ctx, str(args.get("run_id") or ""))
        export_root = _confined(run_root, run_root / "exports")
        export_root.mkdir(parents=True, exist_ok=True)
        run_id = str(args["run_id"])
        if fmt == "json":
            output = export_root / f"{run_id}.json"
            output.write_text(_json(report) + "\n", encoding="utf-8")
        elif fmt == "csv":
            output = export_root / f"{run_id}.csv"
            with output.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["claim", "source_refs", "confidence", "caveat"])
                writer.writeheader()
                for row in report.get("findings", []):
                    writer.writerow({"claim": row.get("claim", ""), "source_refs": ",".join(row.get("source_refs", [])), "confidence": row.get("confidence", ""), "caveat": row.get("caveat", "")})
        else:
            output = export_root / f"{run_id}.md"
            source_lines = "\n".join(f"- {row['source_id']}: [{row['title']}]({row['url']})" for row in report.get("sources", []))
            finding_lines = "\n".join(f"- {row['claim']} (sources: {', '.join(row.get('source_refs', []))})" for row in report.get("findings", [])) or "- No supported findings were produced."
            output.write_text(f"# {report.get('topic', 'Research Desk')}\n\n## Findings\n\n{finding_lines}\n\n## Sources\n\n{source_lines}\n", encoding="utf-8")
        return {"ok": True, "status": "exported", "run_id": run_id, "format": fmt, "path": str(output), "sha256": _hash(output.read_bytes()), "approved": True}
    except Exception as exc:
        return {"ok": False, "status": "blocked", "error": str(exc)}


def make_handlers(ctx) -> dict[str, Callable]:
    return {
        "status": lambda args=None, **_: _json(status(ctx)),
        "plan": lambda args=None, **_: _json(plan(ctx, args)),
        "run": lambda args=None, **_: _json(run(ctx, args)),
        "export": lambda args=None, **_: _json(export(ctx, args)),
        "slash": lambda command="", **_: _json(status(ctx)) if (command or "status").strip().lower() in {"", "status"} else "Use /research-desk status, or the hermes research-desk CLI for plan, run, and export.",
    }
