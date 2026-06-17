"""Read-only typed-agent spec parser, validator, and effective preview.

This module deliberately returns DTOs/reports only. It does not mutate profile
state, start MCP servers, alter prompt construction, or apply tool filtering.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tomllib
from pathlib import Path
from typing import Any

import yaml

from agent.agent_spec_models import (
    MCP_VALIDATION_STATES,
    SANDBOX_ENFORCEMENT_STATUSES,
    SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    AgentSpecDocument,
    McpServerPreview,
    SpecSource,
    ValidationIssue,
    ValidationReport,
    is_valid_reasoning_effort,
    to_plain,
    validation_status,
)
from hermes_cli.profiles import get_profile_dir, list_profiles, normalize_profile_name, profile_exists, validate_profile_name
from toolsets import validate_toolset

_ALLOWED_FIELDS = {
    "schema_version", "id", "profile_id", "display_name", "role_category", "description",
    "model", "reasoning_effort", "runtime", "toolsets", "mcp", "sandbox", "skills",
    "memory", "artifacts", "gates",
}
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_GATE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_SENSITIVE_RE = re.compile(r"token|secret|password|passwd|api[_ -]?key|authorization|headers|env|private[_ -]?key|ssh[_ -]?key", re.I)
_PREVIEW_SANDBOX_NOTE = "Preview declaration only; this slice does not change terminal/file access or runtime sandboxing."


def _issue(severity: str, code: str, message: str, field: str | None = None, source: str | None = None) -> ValidationIssue:
    return ValidationIssue(severity=severity, code=code, message=message, field=field, source=source)


def parse_agent_markdown(text: str, source: str = "<string>") -> AgentSpecDocument:
    if text.startswith("\ufeff"):
        text = text[1:]
    src = SpecSource("agent_spec", source, "loaded", 10)
    errors: list[ValidationIssue] = []
    raw: dict[str, Any] = {}
    body = ""
    if not text.startswith("---\n"):
        errors.append(_issue("error", "missing_frontmatter", "Agent markdown must start with YAML frontmatter", source=source))
        return AgentSpecDocument(src, raw, None, "agent_markdown", errors)
    try:
        end = text.index("\n---", 4)
        frontmatter = text[4:end]
        body = text[end + len("\n---"):].lstrip("\n")
        parsed = yaml.safe_load(frontmatter) or {}
        if not isinstance(parsed, dict):
            raise ValueError("frontmatter must be a mapping")
        raw = parsed
    except Exception as exc:
        errors.append(_issue("error", "invalid_frontmatter", f"Invalid YAML frontmatter: {exc}", source=source))
    return AgentSpecDocument(src, raw, body, "agent_markdown", errors)


def parse_agent_yaml(text: str, source: str = "<string>") -> AgentSpecDocument:
    src = SpecSource("agent_spec", source, "loaded", 10)
    try:
        parsed = yaml.safe_load(text) or {}
        if not isinstance(parsed, dict):
            raise ValueError("YAML spec must be a mapping")
        return AgentSpecDocument(src, parsed, None, "yaml", [])
    except Exception as exc:
        return AgentSpecDocument(src, {}, None, "yaml", [_issue("error", "invalid_yaml", str(exc), source=source)])


def parse_agent_toml(text: str, source: str = "<string>") -> AgentSpecDocument:
    src = SpecSource("agent_spec", source, "loaded", 10)
    try:
        return AgentSpecDocument(src, tomllib.loads(text), None, "toml", [])
    except Exception as exc:
        return AgentSpecDocument(src, {}, None, "toml", [_issue("error", "invalid_toml", str(exc), source=source)])


def load_agent_spec(path: str | Path) -> AgentSpecDocument:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    suffixes = [s.lower() for s in p.suffixes]
    if suffixes[-2:] == [".agent", ".md"] or p.name.endswith(".agent.md"):
        return parse_agent_markdown(text, str(p))
    if p.suffix.lower() in {".yaml", ".yml"}:
        return parse_agent_yaml(text, str(p))
    if p.suffix.lower() == ".toml":
        return parse_agent_toml(text, str(p))
    return AgentSpecDocument(SpecSource("agent_spec", str(p), "loaded", 10), {}, None, "unknown", [_issue("error", "unsupported_format", f"Unsupported spec format: {p}", source=str(p))])


def _safe_load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _repo_catalog_path() -> Path:
    return Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "agent_specs" / "mcp_catalog.yaml"


def load_mcp_catalog(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    raw = _safe_load_yaml_file(Path(path) if path else _repo_catalog_path())
    servers = raw.get("servers", raw)
    return servers if isinstance(servers, dict) else {}


def _configured_mcp_ids(profile_id: str | None = None) -> set[str]:
    cfg = _safe_load_yaml_file(get_profile_dir(profile_id or "default") / "config.yaml")
    servers = cfg.get("mcp_servers")
    return {str(k) for k in servers.keys()} if isinstance(servers, dict) else set()


def _clean_mcp_refs(refs: Any) -> list[dict[str, Any]]:
    if not isinstance(refs, list):
        return []
    return [ref for ref in refs if isinstance(ref, dict)]


def validate_mcp_references(refs: list[dict[str, Any]], *, catalog: dict[str, dict[str, Any]] | None = None, configured_ids: set[str] | None = None) -> list[McpServerPreview]:
    catalog = catalog if catalog is not None else load_mcp_catalog()
    configured_ids = configured_ids if configured_ids is not None else _configured_mcp_ids()
    previews: list[McpServerPreview] = []
    for ref in _clean_mcp_refs(refs):
        sid = str(ref.get("server_id", ""))
        tool = ref.get("tool")
        tool = str(tool) if tool is not None else None
        required = bool(ref.get("required", False))
        known = sid in catalog and sid != "mystery"
        configured = sid in configured_ids
        warnings: list[str] = []
        errors: list[str] = []
        if not known:
            state = "unknown_server_id"
            errors.append("server id is not in the static MCP catalog")
        elif configured:
            allowed = catalog.get(sid, {}).get("allowed_tools")
            if tool and isinstance(allowed, list) and tool not in allowed:
                state = "tool_not_in_catalog_or_discovery"
                errors.append("requested tool is not in static catalog")
            elif tool and allowed is None:
                state = "tool_discovery_unavailable"
                warnings.append("static tool list unavailable; live discovery was not performed")
            else:
                state = "known_in_catalog_and_configured"
        else:
            allowed = catalog.get(sid, {}).get("allowed_tools")
            if tool and allowed is None:
                state = "tool_discovery_unavailable"
                warnings.append("static tool list unavailable; live discovery was not performed")
            elif tool and isinstance(allowed, list) and tool not in allowed:
                state = "tool_not_in_catalog_or_discovery"
                errors.append("requested tool is not in static catalog")
            elif required:
                state = "known_in_catalog_but_required_missing"
                warnings.append("required MCP server is not configured")
            else:
                state = "known_in_catalog_but_not_configured_optional"
                warnings.append("optional MCP server is not configured")
        previews.append(McpServerPreview(sid, tool, required, state, configured, known, warnings, errors))
    return previews


def _skill_exists(profile_dir: Path, name: str) -> bool:
    for base in [profile_dir / "skills", Path.cwd() / "skills"]:
        if (base / name / "SKILL.md").is_file() or (base / name).exists():
            return True
        try:
            if any(p.parent.name == name for p in base.rglob("SKILL.md")):
                return True
        except Exception:
            pass
    return False


def _validate_artifact_path(path: str) -> bool:
    p = Path(path)
    return not (p.is_absolute() or ".." in p.parts)


def validate_agent_spec(doc: AgentSpecDocument, *, strict: bool = False, profile_id: str | None = None) -> ValidationReport:
    errors = list(doc.parse_errors)
    warnings: list[ValidationIssue] = []
    raw = doc.raw or {}
    source = doc.source.path
    schema = raw.get("schema_version")
    if schema not in SUPPORTED_SCHEMA_VERSIONS:
        errors.append(_issue("error", "unsupported_schema_version", "schema_version is missing or unsupported", "schema_version", source))
    if schema == "v1alpha1":
        warnings.append(_issue("warning", "schema_version_shorthand", "Use fully qualified schema version", "schema_version", source))
    spec_id = raw.get("id")
    if not isinstance(spec_id, str) or not _ID_RE.match(spec_id):
        errors.append(_issue("error", "invalid_id", "id must match [a-z0-9][a-z0-9_-]{0,63}", "id", source))
    prof = profile_id or raw.get("profile_id")
    profiles_to_check = []
    if raw.get("profile_id"):
        profiles_to_check.append(raw.get("profile_id"))
    if profile_id and profile_id not in profiles_to_check:
        profiles_to_check.append(profile_id)
    for candidate_profile in profiles_to_check or ([prof] if prof else []):
        try:
            canon = normalize_profile_name(str(candidate_profile)); validate_profile_name(canon)
            if not profile_exists(canon):
                errors.append(_issue("error", "unknown_profile", f"Profile does not exist: {canon}", "profile_id", source))
        except Exception as exc:
            errors.append(_issue("error", "invalid_profile_id", str(exc), "profile_id", source))
    for key in sorted(set(raw) - _ALLOWED_FIELDS):
        warnings.append(_issue("warning", "unknown_field", f"Unknown v1alpha1 field: {key}", key, source))
    effort = raw.get("reasoning_effort")
    if effort is not None and not is_valid_reasoning_effort(effort):
        errors.append(_issue("error", "invalid_reasoning_effort", f"Invalid reasoning_effort: {effort}", "reasoning_effort", source))
    toolsets = raw.get("toolsets") or {}
    if raw.get("toolsets") is not None and not isinstance(raw.get("toolsets"), dict):
        errors.append(_issue("error", "invalid_toolsets", "toolsets must be a mapping", "toolsets", source))
        toolsets = {}
    if isinstance(toolsets, dict):
        for field in ("enabled", "disabled"):
            values = toolsets.get(field) or []
            if not isinstance(values, list):
                errors.append(_issue("error", "invalid_toolsets", f"toolsets.{field} must be a list", f"toolsets.{field}", source))
                values = []
            for name in values:
                if not validate_toolset(str(name)):
                    errors.append(_issue("error", "unknown_toolset", f"Unknown toolset: {name}", f"toolsets.{field}", source))
    mcp_refs = raw.get("mcp") or []
    if raw.get("mcp") is not None and not isinstance(raw.get("mcp"), list):
        errors.append(_issue("error", "invalid_mcp", "mcp must be a list of server references", "mcp", source))
        mcp_refs = []
    clean_mcp_refs: list[dict[str, Any]] = []
    for index, ref in enumerate(mcp_refs):
        if not isinstance(ref, dict):
            errors.append(_issue("error", "invalid_mcp", "mcp entries must be mappings", f"mcp[{index}]", source))
            continue
        clean_mcp_refs.append(ref)
    for preview in validate_mcp_references(clean_mcp_refs, catalog=load_mcp_catalog(), configured_ids=_configured_mcp_ids(str(prof or "default"))):
        for msg in preview.errors:
            errors.append(_issue("error", preview.state, msg, "mcp", source))
        for msg in preview.warnings:
            warnings.append(_issue("warning", preview.state, msg, "mcp", source))
    if raw.get("runtime") is not None and not isinstance(raw.get("runtime"), dict):
        errors.append(_issue("error", "invalid_runtime", "runtime must be a mapping", "runtime", source))
    skills = raw.get("skills") or {}
    if raw.get("skills") is not None and not isinstance(raw.get("skills"), dict):
        errors.append(_issue("error", "invalid_skills", "skills must be a mapping", "skills", source))
        skills = {}
    if isinstance(skills, dict):
        pdir = get_profile_dir(str(prof or "default"))
        for name in skills.get("required") or []:
            if not _skill_exists(pdir, str(name)):
                errors.append(_issue("error", "missing_required_skill", f"Missing required skill: {name}", "skills.required", source))
    artifacts = raw.get("artifacts") or {}
    if raw.get("artifacts") is not None and not isinstance(raw.get("artifacts"), dict):
        errors.append(_issue("error", "invalid_artifacts", "artifacts must be a mapping", "artifacts", source))
        artifacts = {}
    if isinstance(artifacts, dict):
        roots = artifacts.get("allowed_roots") or []
        if not isinstance(roots, list):
            errors.append(_issue("error", "invalid_artifacts", "artifacts.allowed_roots must be a list", "artifacts.allowed_roots", source))
            roots = []
        for item in roots:
            if not _validate_artifact_path(str(item)):
                errors.append(_issue("error", "unsafe_artifact_path", f"Unsafe artifact path: {item}", "artifacts.allowed_roots", source))
    gates = raw.get("gates") or []
    if raw.get("gates") is not None and not isinstance(raw.get("gates"), list):
        errors.append(_issue("error", "invalid_gates", "gates must be a list", "gates", source))
        gates = []
    for index, gate in enumerate(gates):
        if not isinstance(gate, dict):
            errors.append(_issue("error", "invalid_gates", "gate entries must be mappings", f"gates[{index}]", source))
            continue
        gid = gate.get("id")
        if not isinstance(gid, str) or not _GATE_ID_RE.match(gid):
            errors.append(_issue("error", "unknown_gate_id", f"Invalid gate id: {gid}", "gates.id", source))
    sandbox = raw.get("sandbox") or {}
    if raw.get("sandbox") is not None and not isinstance(raw.get("sandbox"), dict):
        errors.append(_issue("error", "invalid_sandbox", "sandbox must be a mapping", "sandbox", source))
        sandbox = {}
    if isinstance(sandbox, dict):
        status = sandbox.get("enforcement_status", "declared_only")
        if status not in SANDBOX_ENFORCEMENT_STATUSES:
            errors.append(_issue("error", "invalid_sandbox_status", f"Invalid sandbox enforcement_status: {status}", "sandbox.enforcement_status", source))
        if sandbox.get("backend", "local") == "local" and status == "enforced":
            errors.append(_issue("error", "sandbox_overclaimed", "local backend cannot claim enforced sandboxing in preview", "sandbox.enforcement_status", source))
    report = ValidationReport("pass", errors, warnings, [], [doc.source])
    report.recompute_status(strict=strict)
    return report


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            key = str(k)
            out[key] = "[REDACTED]" if _SENSITIVE_RE.search(key) else _redact(v)
        return out
    if isinstance(value, list):
        return [_redact(v) for v in value]
    if isinstance(value, str) and (_SENSITIVE_RE.search(value) or "dummy-" in value):
        return "[REDACTED]"
    return value


def _model_from_config(profile_id: str) -> dict[str, Any]:
    cfg = _safe_load_yaml_file(get_profile_dir(profile_id) / "config.yaml")
    model_cfg = cfg.get("model")
    if isinstance(model_cfg, dict):
        return {"provider": _redact(model_cfg.get("provider")), "model": _redact(model_cfg.get("model")), "fallback": []}
    if isinstance(model_cfg, str):
        return {"provider": None, "model": _redact(model_cfg), "fallback": []}
    return {"provider": None, "model": None, "fallback": []}


def _build_preview(profile_id: str, doc: AgentSpecDocument | None, report: ValidationReport, *, strict: bool = False) -> dict[str, Any]:
    raw = doc.raw if doc else {}
    spec_status = "present" if doc and not doc.parse_errors else "missing" if doc is None else "invalid"
    sources = [SpecSource("legacy_profile", str(get_profile_dir(profile_id)), "read_only", 30)]
    if doc:
        sources.append(doc.source)
    model_raw = raw.get("model") if isinstance(raw.get("model"), dict) else {}
    model = {
        "provider": _redact(model_raw.get("provider")) if model_raw else _model_from_config(profile_id)["provider"],
        "model": _redact(model_raw.get("model")) if model_raw else _model_from_config(profile_id)["model"],
        "fallback": _redact(model_raw.get("fallback") or []),
    }
    effort = raw.get("reasoning_effort")
    reasoning = {"enabled": False if effort == "none" else (True if effort else None), "effort": None if effort == "none" else effort, "source": "agent_spec" if effort else "legacy"}
    ts = raw.get("toolsets") if isinstance(raw.get("toolsets"), dict) else {}
    enabled = [str(x) for x in ts.get("enabled") or []]
    disabled = [str(x) for x in ts.get("disabled") or []]
    mcp = validate_mcp_references(raw.get("mcp") or [], catalog=load_mcp_catalog(), configured_ids=_configured_mcp_ids(profile_id))
    sandbox_raw = raw.get("sandbox") if isinstance(raw.get("sandbox"), dict) else {}
    enforcement = sandbox_raw.get("enforcement_status") or "declared_only"
    if sandbox_raw.get("backend", "local") == "local" and enforcement == "enforced":
        enforcement = "declared_only"
    body = doc.body if doc else None
    skills = raw.get("skills") if isinstance(raw.get("skills"), dict) else {}
    artifacts = raw.get("artifacts") if isinstance(raw.get("artifacts"), dict) else {}
    memory = raw.get("memory") if isinstance(raw.get("memory"), dict) else {}
    gates_raw = raw.get("gates") or []
    warnings = report.warnings + [_issue("warning", "missing_agent_spec", "No typed agent spec found; legacy fallback remains active", source=str(get_profile_dir(profile_id)))] if doc is None else report.warnings
    status = validation_status(report.errors, warnings, strict=strict)
    return {
        "profile_id": profile_id,
        "schema_version": SCHEMA_VERSION,
        "spec_id": raw.get("id"),
        "status": status,
        "spec_status": spec_status,
        "sources": [to_plain(s) for s in sources],
        "model": model,
        "reasoning": reasoning,
        "runtime": {**(raw.get("runtime") if isinstance(raw.get("runtime"), dict) else {}), "preview_only": True},
        "toolsets": {"enabled": enabled, "disabled": disabled, "unknown": [x for x in enabled + disabled if not validate_toolset(x)], "applied_to_runtime": False},
        "mcp": [to_plain(x) for x in mcp],
        "sandbox": {"desired": sandbox_raw.get("desired"), "enforcement_status": enforcement, "backend": sandbox_raw.get("backend", "local"), "notes": [_PREVIEW_SANDBOX_NOTE]},
        "skills": {"required": skills.get("required", []), "recommended": skills.get("recommended", []), "missing_required": [i.message.rsplit(": ", 1)[-1] for i in report.errors if i.code == "missing_required_skill"], "checked_read_only": True},
        "memory": {"policy": memory.get("policy", "legacy"), "warnings": []},
        "instructions": {"body_present": bool(body), "body_sha256": hashlib.sha256(body.encode()).hexdigest() if body else None, "applied_to_prompt": False},
        "artifacts": {"allowed_roots": artifacts.get("allowed_roots", []), "requirements": artifacts.get("requirements", []), "unsafe_paths": [i.message.rsplit(": ", 1)[-1] for i in report.errors if i.code == "unsafe_artifact_path"]},
        "gates": [{"id": g.get("id"), "owner": g.get("owner"), "blocking": g.get("blocking"), "mode": "preview_only"} for g in gates_raw if isinstance(g, dict)],
        "warnings": [to_plain(w) for w in warnings],
        "errors": [to_plain(e) for e in report.errors],
        "read_only_guarantee": True,
        "enforcement_enabled": False,
    }


def preview_agent_spec(profile_id: str, *, spec_path: str | Path | None = None, strict: bool = False) -> dict[str, Any]:
    profile_id = normalize_profile_name(profile_id)
    validate_profile_name(profile_id)
    if not profile_exists(profile_id):
        report = ValidationReport("fail", [_issue("error", "unknown_profile", f"Profile does not exist: {profile_id}", "profile_id")])
        return _build_preview(profile_id, None, report, strict=strict)
    doc: AgentSpecDocument | None = None
    if spec_path:
        doc = load_agent_spec(spec_path)
    else:
        candidate = get_profile_dir(profile_id) / "agent-spec.md"
        if candidate.is_file():
            doc = load_agent_spec(candidate)
    report = validate_agent_spec(doc, strict=strict, profile_id=profile_id) if doc else ValidationReport("pass", [], [], [], [])
    if doc and profile_id:
        # In preview mode, --profile is the effective target. A fixture may carry
        # an example profile_id that is not present in a temp/test HERMES_HOME;
        # keep validation read-only and do not fail the requested profile preview
        # solely because the embedded example profile is absent.
        report.errors = [e for e in report.errors if e.code != "unknown_profile"]
        report.recompute_status(strict=strict)
    return _redact(_build_preview(profile_id, doc, report, strict=strict))


def _resolve_spec_path_or_id(path_or_id: str) -> Path:
    candidate = Path(path_or_id)
    if candidate.exists():
        return candidate
    # Read-only fixture/id convenience for the alpha CLI. This deliberately
    # searches repo-local test/example specs only; it never discovers or writes
    # live profile specs.
    fixtures_root = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "agent_specs"
    if fixtures_root.is_dir() and _ID_RE.match(path_or_id):
        for spec in sorted(fixtures_root.rglob("*")):
            if spec.suffix.lower() not in {".md", ".yaml", ".yml", ".toml"}:
                continue
            try:
                doc = load_agent_spec(spec)
            except Exception:
                continue
            if doc.raw.get("id") == path_or_id:
                return spec
    return candidate


def validate_spec_path(path_or_id: str, *, strict: bool = False, profile_id: str | None = None) -> dict[str, Any]:
    doc = load_agent_spec(_resolve_spec_path_or_id(path_or_id))
    report = validate_agent_spec(doc, strict=strict, profile_id=profile_id)
    return _redact({
        "status": report.status,
        "errors": [to_plain(e) for e in report.errors],
        "warnings": [to_plain(w) for w in report.warnings],
        "sources": [to_plain(s) for s in report.sources],
        "read_only_guarantee": True,
        "enforcement_enabled": False,
    })


def list_profile_specs() -> dict[str, Any]:
    profiles = []
    for info in list_profiles():
        spec = info.path / "agent-spec.md"
        status = "missing"
        if spec.is_file():
            status = "present" if validate_agent_spec(load_agent_spec(spec), profile_id=info.name).status != "fail" else "invalid"
        profiles.append({"name": info.name, "path": str(info.path), "spec_status": status, "legacy_fallback_active_when_missing_spec": status == "missing"})
    return {"profiles": profiles, "read_only_guarantee": True, "enforcement_enabled": False}


def render_json(payload: dict[str, Any]) -> str:
    return json.dumps(_redact(payload), indent=2, sort_keys=False) + "\n"


def render_text(payload: dict[str, Any]) -> str:
    if "profiles" in payload:
        lines = [
            "Agent spec profile coverage:",
            f"Profiles: {len(payload.get('profiles', []))}",
        ]
        for profile in payload.get("profiles", []):
            lines.extend(
                [
                    f"Profile: {profile.get('name')}",
                    f"  Path: {profile.get('path')}",
                    f"  Spec status: {profile.get('spec_status')}",
                    f"  Legacy fallback active when missing spec: {str(profile.get('legacy_fallback_active_when_missing_spec')).lower()}",
                ]
            )
        lines.append("Read-only guarantee: true")
        lines.append("Runtime enforcement: disabled")
        return "\n".join(lines) + "\n"

    lines = [
        f"Agent spec status: {payload.get('status')}",
        f"Profile: {payload.get('profile_id')}",
        f"Spec status: {payload.get('spec_status', payload.get('status'))}",
        "Sources:",
    ]
    for s in payload.get("sources", []):
        lines.append(f"  - {s.get('kind')} {s.get('status')} precedence={s.get('precedence')} path={s.get('path')}")
    lines += [
        f"Model: {payload.get('model')}",
        f"Reasoning: {payload.get('reasoning')}",
        f"Toolsets: {payload.get('toolsets')}",
        "MCP:",
    ]
    for m in payload.get("mcp", []):
        lines.append(f"  - {m.get('server_id')} tool={m.get('tool')} state={m.get('state')} required={m.get('required')}")
    lines.append(f"Sandbox: desired={payload.get('sandbox', {}).get('desired')} enforcement_status={payload.get('sandbox', {}).get('enforcement_status')} backend={payload.get('sandbox', {}).get('backend')}")
    lines.append(f"Memory: {payload.get('memory')}")
    lines.append(f"Artifacts: {payload.get('artifacts')}")
    lines.append(f"Gates: {payload.get('gates')}")
    for key in ("warnings", "errors"):
        lines.append(f"{key.title()}:")
        for issue in payload.get(key, []):
            lines.append(f"  - {issue.get('code')}: {issue.get('message')}")
    lines.append("Read-only guarantee: true")
    lines.append("Runtime enforcement: disabled")
    return "\n".join(lines) + "\n"
