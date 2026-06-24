"""Hermes Lungnis extension manifests, validation, and registry cache.

This module intentionally treats existing ``plugin.yaml`` and ``SKILL.md``
front matter as extension manifests. That keeps third-party authors on the
current Hermes surfaces while adding Lungnis-grade metadata, validation, and a
local cache for marketplace-style discovery.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home


RISK_LEVELS = {"low", "medium", "high", "critical"}
TRUST_LEVELS = {"official", "verified", "community", "local-only"}
SUPPORTED_EXTENSION_TYPES = {"plugin", "skill"}
DEFAULT_SUPPORTED_PLATFORMS = ["linux", "macos", "windows"]

SECURITY_PATTERNS: tuple[tuple[str, str, str], ...] = (
    ("python-eval", r"\beval\s*\(", "high"),
    ("python-exec", r"\bexec\s*\(", "high"),
    ("pickle-load", r"\bpickle\.loads?\s*\(", "high"),
    ("shell-true", r"shell\s*=\s*True", "medium"),
    ("os-system", r"\bos\.system\s*\(", "medium"),
    ("curl-pipe-shell", r"(curl|wget)\b.+\|\s*(sh|bash)\b", "critical"),
    ("world-writable", r"chmod\s+777\b", "medium"),
    ("secret-print", r"(print|console\.log)\s*\([^)]*(API_KEY|TOKEN|SECRET|PASSWORD)", "high"),
)


@dataclass
class ExtensionManifest:
    """Normalized extension metadata used by validation and registry cache."""

    name: str
    version: str = ""
    author: str = ""
    description: str = ""
    extension_type: str = "plugin"
    capabilities: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    required_secrets: list[str] = field(default_factory=list)
    risk_level: str = "medium"
    supported_platforms: list[str] = field(default_factory=lambda: list(DEFAULT_SUPPORTED_PLATFORMS))
    trust: str = "local-only"
    path: str = ""
    source: str = "local"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "type": self.extension_type,
            "capabilities": self.capabilities,
            "required_tools": self.required_tools,
            "required_secrets": self.required_secrets,
            "risk_level": self.risk_level,
            "supported_platforms": self.supported_platforms,
            "trust": self.trust,
            "path": self.path,
            "source": self.source,
        }


@dataclass
class ValidationFinding:
    check: str
    level: str
    message: str
    path: str = ""

    def to_dict(self) -> dict[str, str]:
        data = {"check": self.check, "level": self.level, "message": self.message}
        if self.path:
            data["path"] = self.path
        return data


def registry_cache_path() -> Path:
    path = get_hermes_home() / "extensions" / "registry_cache.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - PyYAML is a core dependency
        raise RuntimeError("PyYAML is required to read extension manifests") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def _skill_front_matter(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---", 4)
    if end == -1:
        return {}
    try:
        import yaml
        data = yaml.safe_load(text[4:end]) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict) and item.get("name"):
                out.append(str(item["name"]))
        return out
    return []


def _metadata_block(data: dict[str, Any]) -> dict[str, Any]:
    meta = data.get("metadata")
    if isinstance(meta, dict):
        lungnis = meta.get("lungnis")
        if isinstance(lungnis, dict):
            return lungnis
        hermes_lungnis = meta.get("hermes_lungnis")
        if isinstance(hermes_lungnis, dict):
            return hermes_lungnis
    return {}


def _infer_type(path: Path) -> str:
    if (path / "SKILL.md").exists() or path.name == "SKILL.md":
        return "skill"
    return "plugin"


def _manifest_path(path: Path) -> Path | None:
    if path.is_file():
        return path
    for name in ("plugin.yaml", "plugin.yml", "SKILL.md"):
        candidate = path / name
        if candidate.exists():
            return candidate
    return None


def load_extension_manifest(path: str | Path, *, source: str = "local") -> ExtensionManifest:
    root = Path(path)
    manifest_path = _manifest_path(root)
    if manifest_path is None:
        raise FileNotFoundError(f"No plugin.yaml, plugin.yml, or SKILL.md found in {root}")

    extension_type = _infer_type(root if root.is_dir() else manifest_path)
    if manifest_path.name == "SKILL.md":
        data = _skill_front_matter(manifest_path)
    else:
        data = _read_yaml(manifest_path)

    meta = _metadata_block(data)
    hermes_meta = {}
    if isinstance(data.get("metadata"), dict) and isinstance(data["metadata"].get("hermes"), dict):
        hermes_meta = data["metadata"]["hermes"]

    name = str(data.get("name") or (root.parent.name if root.name == "SKILL.md" else root.name))
    required_secrets = _as_str_list(
        data.get("required_secrets")
        or data.get("requires_env")
        or meta.get("required_secrets")
        or meta.get("requires_env")
    )
    capabilities = _as_str_list(
        data.get("capabilities")
        or data.get("provides_tools")
        or meta.get("capabilities")
        or hermes_meta.get("tags")
    )
    prereq_commands: Any = None
    if isinstance(data.get("prerequisites"), dict):
        prereq_commands = data["prerequisites"].get("commands")
    required_tools = _as_str_list(
        data.get("required_tools")
        or meta.get("required_tools")
        or hermes_meta.get("requires_toolsets")
        or data.get("dependencies")
        or prereq_commands
    )
    risk_level = str(data.get("risk_level") or meta.get("risk_level") or "medium").lower()
    trust = str(data.get("trust") or meta.get("trust") or ("official" if source == "bundled" else "local-only")).lower()
    platforms = _as_str_list(data.get("supported_platforms") or data.get("platforms") or meta.get("supported_platforms"))

    return ExtensionManifest(
        name=name,
        version=str(data.get("version", "")),
        author=str(data.get("author", "")),
        description=str(data.get("description", "")),
        extension_type=extension_type,
        capabilities=capabilities,
        required_tools=required_tools,
        required_secrets=required_secrets,
        risk_level=risk_level,
        supported_platforms=platforms or list(DEFAULT_SUPPORTED_PLATFORMS),
        trust=trust,
        path=str(root if root.is_dir() else root.parent),
        source=source,
    )


def _schema_findings(manifest: ExtensionManifest) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    required = {
        "name": manifest.name,
        "version": manifest.version,
        "author": manifest.author,
        "capabilities": manifest.capabilities,
        "risk_level": manifest.risk_level,
        "supported_platforms": manifest.supported_platforms,
    }
    for field_name, value in required.items():
        if value in ("", [], None):
            findings.append(ValidationFinding("schema", "error", f"Missing required field: {field_name}"))
    if manifest.extension_type not in SUPPORTED_EXTENSION_TYPES:
        findings.append(ValidationFinding("schema", "error", f"Unsupported type: {manifest.extension_type}"))
    if manifest.risk_level and manifest.risk_level not in RISK_LEVELS:
        findings.append(ValidationFinding("schema", "error", f"Invalid risk_level: {manifest.risk_level}"))
    if manifest.trust and manifest.trust not in TRUST_LEVELS:
        findings.append(ValidationFinding("schema", "warning", f"Unknown trust level: {manifest.trust}"))
    return findings


def _security_findings(root: Path) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    if root.is_file():
        roots = [root]
    else:
        roots = [
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in {".py", ".sh", ".js", ".ts", ".md"}
        ]
    for file_path in roots:
        rel = str(file_path.relative_to(root)) if root.is_dir() else file_path.name
        text = file_path.read_text(encoding="utf-8", errors="replace")[:200_000]
        for name, pattern, level in SECURITY_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL):
                findings.append(
                    ValidationFinding("security", level, f"Security lint matched {name}", rel)
                )
    return findings


def _dependency_findings(root: Path, manifest: ExtensionManifest) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    deps = list(manifest.required_tools)
    for req_file_name in ("requirements.txt", "pyproject.toml", "package.json"):
        req_file = root / req_file_name
        if req_file.exists():
            findings.append(ValidationFinding("dependencies", "info", f"Found dependency file: {req_file_name}", req_file_name))
    for dep in deps:
        if dep and re.match(r"^[A-Za-z0-9_.-]+$", dep) and not any(op in dep for op in ("==", ">=", "<", "~=", "@")):
            findings.append(
                ValidationFinding("dependencies", "warning", f"Dependency/tool '{dep}' has no version or availability check")
            )
    return findings


def _doc_findings(root: Path) -> list[ValidationFinding]:
    if (root / "README.md").exists() or (root / "SKILL.md").exists():
        return []
    return [ValidationFinding("docs", "error", "Missing README.md or SKILL.md")]


def _run_test_command(root: Path, command: str | None) -> list[ValidationFinding]:
    if not command:
        return []
    try:
        result = subprocess.run(
            shlex.split(command),
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception as exc:
        return [ValidationFinding("test", "error", f"Test command failed to run: {exc}")]
    if result.returncode != 0:
        output = (result.stderr or result.stdout or "").strip()
        return [ValidationFinding("test", "error", f"Test command exited {result.returncode}: {output[:500]}")]
    return [ValidationFinding("test", "info", "Test command passed")]


def validate_extension(path: str | Path, *, test_command: str | None = None, source: str = "local") -> dict[str, Any]:
    root = Path(path)
    manifest = load_extension_manifest(root, source=source)
    findings: list[ValidationFinding] = []
    findings.extend(_schema_findings(manifest))
    scan_root = root.parent if root.is_file() else root
    findings.extend(_security_findings(scan_root))
    findings.extend(_dependency_findings(scan_root, manifest))
    findings.extend(_doc_findings(scan_root))
    findings.extend(_run_test_command(scan_root, test_command))
    ok = not any(f.level == "error" for f in findings)
    return {
        "ok": ok,
        "manifest": manifest.to_dict(),
        "findings": [f.to_dict() for f in findings],
    }


def _scan_extension_dirs(base: Path, *, source: str) -> list[ExtensionManifest]:
    manifests: list[ExtensionManifest] = []
    if not base.is_dir():
        return manifests
    for manifest_path in sorted(base.rglob("plugin.yaml")) + sorted(base.rglob("plugin.yml")):
        try:
            manifests.append(load_extension_manifest(manifest_path.parent, source=source))
        except Exception:
            continue
    for skill_path in sorted(base.rglob("SKILL.md")):
        try:
            manifests.append(load_extension_manifest(skill_path.parent, source=source))
        except Exception:
            continue
    return manifests


def build_registry_cache(*, include_bundled: bool = True, include_user: bool = True) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    if include_bundled:
        from hermes_cli.plugins import get_bundled_plugins_dir
        repo_root = Path(__file__).resolve().parent.parent
        for base in (get_bundled_plugins_dir(), repo_root / "skills", repo_root / "optional-skills"):
            entries.extend(m.to_dict() for m in _scan_extension_dirs(base, source="bundled"))
    if include_user:
        home = get_hermes_home()
        for base in (home / "plugins", home / "skills"):
            entries.extend(m.to_dict() for m in _scan_extension_dirs(base, source="user"))

    dedup: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in entries:
        dedup[(entry["type"], entry["name"])] = entry
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "entries": sorted(dedup.values(), key=lambda e: (e["type"], e["name"])),
    }
    registry_cache_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def read_registry_cache() -> dict[str, Any]:
    path = registry_cache_path()
    if not path.exists():
        return build_registry_cache()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return build_registry_cache()
