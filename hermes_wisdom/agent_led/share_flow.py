"""Agent-guided Share flow.

``Share`` never publishes local files directly. A deterministic pre-pass
extracts requirements and scans for credential-shaped strings; a structured
agent packaging task then produces a portable package; the owner sees the
portability problems and package summary and must approve before the
existing publish path (``WisdomService.suggest`` -> review -> approve) is
invoked. Every step is persisted so the flow is resumable.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .agent import ModelCall, package_for_share
from .evidence import dependency_hints, read_frontmatter
from .history import history_path
from .schemas import SchemaRejected, SharePackage, PackagedFile
from ..contract import canonical_json_bytes, sha256_address

logger = logging.getLogger(__name__)

CREDENTIAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("slack_token", re.compile(r"\bxox[abpr]-[A-Za-z0-9-]{10,}\b")),
    ("openai_style_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b")),
    (
        "assignment",
        re.compile(
            r"(?i)\b(api[_-]?key|secret|token|password|passwd)\b\s*[:=]\s*['\"]?[A-Za-z0-9_\-/+]{16,}"
        ),
    ),
)
ORG_SPECIFIC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("home_path", re.compile(r"/(?:home|Users)/[A-Za-z0-9._-]+")),
    ("internal_host", re.compile(r"\b[a-z0-9-]+\.(?:internal|corp|local|lan)\b")),
    (
        "private_ip",
        re.compile(
            r"\b(?:10|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}(?:\.\d{1,3})?\b"
        ),
    ),
)
TEXT_SUFFIXES = {
    ".md",
    ".txt",
    ".py",
    ".sh",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".cfg",
    ".ini",
    ".js",
    ".ts",
}
MAX_FILE_BYTES = 200_000
MAX_TREE_BYTES = 1_000_000
MAX_FILES = 64

STATES = (
    "prepass",
    "packaged",
    "awaiting_approval",
    "changes_requested",
    "approved",
    "submitted",
    "cancelled",
)


def _flows_dir() -> Path:
    return history_path().parent / "share_flows"


def _list_files(root: Path) -> list[dict[str, str]]:
    if root.is_symlink() or not root.is_dir():
        raise SchemaRejected("skill root must be a regular directory")
    files: list[dict[str, str]] = []
    total = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise SchemaRejected("symlinks cannot be included in packaging input")
        if path.is_dir():
            continue
        if not path.is_file() or path.stat().st_nlink != 1:
            raise SchemaRejected("packaging input must contain regular, unlinked files")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            raise SchemaRejected(
                "unsupported packaging file; review the source before sharing"
            )
        if path.stat().st_size > MAX_FILE_BYTES:
            raise SchemaRejected("packaging input file exceeds its size limit")
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            raise SchemaRejected(
                "packaging input must be readable UTF-8 text"
            ) from None
        total += len(content.encode("utf-8"))
        if total > MAX_TREE_BYTES or len(files) >= MAX_FILES:
            raise SchemaRejected("packaging input exceeds its total size limit")
        files.append({"path": path.relative_to(root).as_posix(), "content": content})
    return files


def scan_credentials(files: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Return credential-shaped findings; matches are never echoed back."""
    findings: list[dict[str, Any]] = []
    for item in files:
        for number, line in enumerate(item["content"].splitlines(), start=1):
            for kind, pattern in CREDENTIAL_PATTERNS:
                if pattern.search(line):
                    findings.append({
                        "kind": kind,
                        "file": item["path"],
                        "line": number,
                    })
    return findings


def scan_org_specific(files: list[dict[str, str]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for item in files:
        for number, line in enumerate(item["content"].splitlines(), start=1):
            for kind, pattern in ORG_SPECIFIC_PATTERNS:
                match = pattern.search(line)
                if match:
                    findings.append({
                        "kind": kind,
                        "file": item["path"],
                        "line": number,
                        "token": match.group(0)[:80],
                    })
    return findings


def _existing_security_scan(skill_path: Path) -> dict[str, Any] | None:
    """Reuse the repo's guard/evaluator scan when importable."""
    try:
        from ..service import _scan_summary

        return _scan_summary(skill_path)
    except Exception as exc:  # pragma: no cover - optional dependency path
        logger.debug("existing security scan unavailable: %s", type(exc).__name__)
        return None


def prepass(skill_path: Path) -> dict[str, Any]:
    from ..qualification import snapshot_tree

    files = _list_files(skill_path)
    frontmatter = read_frontmatter(skill_path)
    env, commands = dependency_hints(frontmatter)
    content_hash, _tree = snapshot_tree(skill_path)
    return {
        "skill_name": skill_path.name,
        "source_content_hash": content_hash,
        "frontmatter_requirements": {
            "required_environment_variables": env,
            "required_commands": commands,
        },
        "scripts": [f["path"] for f in files if f["path"].startswith("scripts/")],
        "references": [f["path"] for f in files if f["path"].startswith("references/")],
        "credential_findings": scan_credentials(files),
        "org_specific_findings": scan_org_specific(files),
        "existing_security_scan": _existing_security_scan(skill_path),
        "files": files,
    }


class ShareFlow:
    """Resumable state machine; one JSON file per flow under HERMES_HOME."""

    def __init__(self, flow_id: str | None = None, *, root: Path | None = None) -> None:
        self.root = root or _flows_dir()
        self.flow_id = flow_id or uuid.uuid4().hex
        if not re.fullmatch(r"[a-f0-9]{32}", self.flow_id):
            raise ValueError("invalid share flow identity")
        self.path = self.root / f"{self.flow_id}.json"
        self.state: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.state["updated_at"] = datetime.now(timezone.utc).isoformat()
        fd, temporary = tempfile.mkstemp(
            prefix=f"{self.flow_id}-", suffix=".pending", dir=self.root
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(self.state, stream, indent=2, sort_keys=True, default=str)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    @property
    def status(self) -> str:
        return str(self.state.get("status") or "new")

    # -- steps -------------------------------------------------------------
    def start(self, skill_path: Path) -> dict[str, Any]:
        result = prepass(skill_path)
        self.state = {
            "flow_id": self.flow_id,
            "skill_name": result["skill_name"],
            "skill_path": str(skill_path),
            "source_content_hash": result["source_content_hash"],
            "status": "prepass",
            "prepass": {k: v for k, v in result.items() if k != "files"},
            "published": False,
        }
        self._save()
        return self.summary()

    def package(
        self,
        *,
        model_call: ModelCall | None = None,
        organization: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.status not in {"prepass", "changes_requested"}:
            raise RuntimeError(f"cannot package from state {self.status}")
        skill_path = Path(self.state["skill_path"])
        pre = prepass(skill_path)
        if pre["source_content_hash"] != self.state["source_content_hash"]:
            self.state["status"] = "prepass"
            self.state["source_content_hash"] = pre["source_content_hash"]
            self.state["prepass"] = {k: v for k, v in pre.items() if k != "files"}
        payload = {
            "skill_name": pre["skill_name"],
            "source_content_hash": pre["source_content_hash"],
            "files": pre["files"],
            "prepass": {
                k: v
                for k, v in pre.items()
                if k not in {"files", "existing_security_scan"}
            },
            "organization": organization or {},
            "requested_changes": self.state.get("requested_changes"),
        }
        try:
            package = package_for_share(payload, model_call=model_call)
            package = normalize_generated_package(package)
        except SchemaRejected as exc:
            self.state["last_error"] = str(exc)
            self._save()
            raise
        if (
            package.source_content_hash != pre["source_content_hash"]
            or package.skill_name != pre["skill_name"]
        ):
            raise SchemaRejected(
                "package does not reference the current source content hash"
            )
        leaked = scan_credentials([
            {"path": f.path, "content": f.content} for f in package.files
        ])
        if leaked:
            raise SchemaRejected(
                "package still contains credential-shaped strings", errors=leaked
            )
        self.state["package"] = package.model_dump(mode="json")
        self.state["package_hash"] = package_hash(package)
        self.state.pop("approved_package_hash", None)
        self.state["status"] = "awaiting_approval"
        self.state.pop("last_error", None)
        self._save()
        return self.summary()

    def request_changes(self, note: str) -> dict[str, Any]:
        if self.status != "awaiting_approval":
            raise RuntimeError(f"cannot request changes from state {self.status}")
        self.state["requested_changes"] = str(note).strip()[:2000]
        self.state["status"] = "changes_requested"
        self._save()
        return self.summary()

    def cancel(self) -> dict[str, Any]:
        self.state["status"] = "cancelled"
        self._save()
        return self.summary()

    def approve(
        self, *, submit: Callable[[SharePackage, dict[str, Any]], dict[str, Any]]
    ) -> dict[str, Any]:
        """Explicit approval; only now is the existing publish path invoked."""
        if self.status == "submitted":
            return self.summary()
        if self.status not in {"awaiting_approval", "approved"}:
            raise RuntimeError("approval requires a packaged flow awaiting approval")
        package = SharePackage.model_validate(self.state["package"])
        if self.state.get("package_hash") != package_hash(package):
            raise SchemaRejected("package changed; regenerate and review it again")
        if (
            self.status == "approved"
            and self.state.get("approved_package_hash") != self.state["package_hash"]
        ):
            raise SchemaRejected("approval does not match this package")
        self.state["approved_package_hash"] = self.state["package_hash"]
        self.state["status"] = "approved"
        self._save()
        result = submit(package, dict(self.state))
        self.state["status"] = "submitted"
        self.state["submission"] = result
        self._save()
        return self.summary()

    # -- presentation ----------------------------------------------------
    def summary(self) -> dict[str, Any]:
        pre = self.state.get("prepass") or {}
        package = self.state.get("package") or {}
        problems: list[str] = []
        for finding in pre.get("credential_findings") or []:
            problems.append(
                f"Credential-shaped {finding['kind']} in {finding['file']}:{finding['line']}"
            )
        for finding in pre.get("org_specific_findings") or []:
            problems.append(
                f"Organization-specific {finding['kind']} in {finding['file']}:{finding['line']}"
            )
        return {
            "flow_id": self.flow_id,
            "skill_name": self.state.get("skill_name"),
            "status": self.status,
            "source_content_hash": self.state.get("source_content_hash"),
            "package_hash": self.state.get("package_hash"),
            "prepared_review": (self.state.get("submission") or {}).get("prepared"),
            "portability_problems": problems,
            "package_summary": {
                "editorial_name": package.get("editorial_name"),
                "plain_description": package.get("plain_description"),
                "files": [f["path"] for f in package.get("files", [])],
                "requirements": package.get("requirements", []),
                "setup_instructions": package.get("setup_instructions", []),
                "credential_handoff": package.get("credential_handoff", []),
                "compatibility_limits": package.get("compatibility_limits", []),
                "verification_step": package.get("verification_step"),
                "removed_or_generalized": package.get("removed_or_generalized", []),
                "related_skills": package.get("related_skills", []),
            }
            if package
            else None,
            "next_actions": (
                ["approve", "request_changes", "cancel"]
                if self.status == "awaiting_approval"
                else []
            ),
            "published": bool(self.state.get("published")),
        }


def normalize_generated_package(package: SharePackage) -> SharePackage:
    """Materialize the generated metadata as part of the reviewable package."""
    import hermes_yaml as yaml
    from .setup_document import SETUP_PATH, render_setup_document

    files = []
    setup_path = SETUP_PATH
    setup = render_setup_document(package)
    for item in package.files:
        if item.path == setup_path:
            if item.content != setup:
                raise SchemaRejected(
                    "refs/wisdom-setup.md is reserved for the reviewed setup metadata"
                )
            continue
        if item.path == "SKILL.md" and not item.content.startswith("---"):
            header = yaml.safe_dump(
                {"name": package.skill_name, "description": package.plain_description},
                sort_keys=False,
                allow_unicode=True,
            )
            item = item.model_copy(
                update={"content": "---\n" + header + "---\n" + item.content}
            )
        files.append(item)
    files.append(PackagedFile(path=setup_path, content=setup))
    return package.model_copy(update={"files": files})


def package_hash(package: SharePackage) -> str:
    package = normalize_generated_package(package)
    return sha256_address(canonical_json_bytes(package.model_dump(mode="json")))


def verify_staged_package(package: SharePackage, target: Path) -> None:
    package = normalize_generated_package(package)
    if target.is_symlink() or not target.is_dir():
        raise SchemaRejected("staged package must be a regular directory")
    expected = {item.path: item.content.encode("utf-8") for item in package.files}
    found = {}
    for path in target.rglob("*"):
        if path.is_symlink():
            raise SchemaRejected("staged package contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file() or path.stat().st_nlink != 1:
            raise SchemaRejected("staged package contains an unsafe file")
        relative = path.relative_to(target).as_posix()
        if relative not in expected or path.stat().st_size != len(expected[relative]):
            raise SchemaRejected("staged package contains changed or unexpected files")
        found[relative] = path.read_bytes()
    if found != expected:
        raise SchemaRejected("staged package changed; review the exact bytes again")


def write_package_to_staging(package: SharePackage, staging_root: Path) -> Path:
    """Create a content-addressed snapshot without modifying any prior review."""
    from ..package import verify_content_files

    package = normalize_generated_package(
        SharePackage.model_validate(package.model_dump(mode="json"))
    )
    verify_content_files(
        [(item.path, "file", item.content.encode("utf-8")) for item in package.files],
        require_manifest=False,
    )
    if scan_credentials([
        {"path": item.path, "content": item.content} for item in package.files
    ]):
        raise SchemaRejected("staged package contains credential-shaped strings")
    staging_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if staging_root.is_symlink():
        raise SchemaRejected("staging root cannot be a symlink")
    namespace = staging_root / package_hash(package).removeprefix("sha256:")
    target = namespace / package.skill_name
    if namespace.is_symlink():
        raise SchemaRejected("staging namespace cannot be a symlink")
    if namespace.exists():
        verify_staged_package(package, target)
        return target
    with tempfile.TemporaryDirectory(prefix="package-", dir=staging_root) as temporary:
        staged_namespace = Path(temporary) / "snapshot"
        staged = staged_namespace / package.skill_name
        staged.mkdir(parents=True, mode=0o700)
        for item in package.files:
            destination = staged / item.path
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            destination.write_bytes(item.content.encode("utf-8"))
            destination.chmod(0o600)
        try:
            os.rename(staged_namespace, namespace)
        except OSError:
            if not namespace.exists():
                raise
        if namespace.is_symlink():
            raise SchemaRejected("staging namespace cannot be a symlink")
        verify_staged_package(package, target)
    return target


def submit_via_service(
    service: Any, staging_root: Path
) -> Callable[[SharePackage, dict[str, Any]], dict[str, Any]]:
    """Prepare approved generated bytes locally; never upload or publish here."""

    def _submit(package: SharePackage, _state: dict[str, Any]) -> dict[str, Any]:
        if _state.get("approved_package_hash") != package_hash(package):
            raise SchemaRejected("package has not been approved for local preparation")
        prepared = service.prepare_share_package(
            package, source_path=Path(_state["skill_path"]), staging_root=staging_root
        )
        return {"prepared": prepared, "published": False, "network_submission": False}

    return _submit
