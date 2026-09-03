"""Application service shared by Wisdom CLI, dashboard, and desktop APIs."""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import sys
import tempfile
import uuid
import webbrowser
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

from hermes_constants import get_hermes_home, get_skills_dir
from agent.skill_utils import load_skill_editorial_metadata
from tools.skill_usage import is_bundled, is_hub_installed
from tools.skills_guard import scan_skill, should_allow_install
from tools.skillevaluator_scan import run_tier1_scan, tier1_advisory_enabled

from .client import (
    WisdomAuthError,
    WisdomClient,
    WisdomConflict,
    WisdomError,
    WisdomNotFound,
    WisdomValidationError,
)
from .compatibility import CompatibilityResult, detect_local_capabilities, evaluate
from .consumption import WisdomConsumption
from .contract import (
    CONTRACT_PIN,
    PackageManifest,
    SystemSpecification,
    author_description_hash,
    canonical_json_bytes,
    org_directory_name,
    parse_manifest_bytes,
    sha256_address,
)
from .package import (
    MAX_FILES,
    MAX_FILE_BYTES,
    MAX_TREE_BYTES,
    PackagePolicyError,
    PreparedPackage,
    prepare_package,
    verify_content_files,
)
from .professionalism import (
    enqueue_review,
    exact_utf8_package,
    process_pending_reviews,
)
from .store import WisdomStore, utc_now


SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$")
ORG_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:_-]{0,127}$")
WISDOM_DISCLOSURE = (
    "Candidate signals stay on this profile. Only owner-approved private draft bytes, "
    "author copy, manifest metadata, and managed-install state reach the Gateway."
)
ORGANIZATION_NAME_REFRESH = timedelta(days=1)
ORGANIZATION_NAME_FAILURE_RETRY = timedelta(minutes=15)


def _parse_package_manifest(raw: bytes) -> PackageManifest:
    try:
        return parse_manifest_bytes(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise PackagePolicyError("package manifest is invalid") from exc


def _config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        value = (load_config() or {}).get("wisdom") or {}
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def portal_base_url() -> str:
    value = _config().get("portal_url")
    return (
        str(value).rstrip("/")
        if isinstance(value, str) and value.strip()
        else "https://portal.nousresearch.com"
    )


def _slug(value: str) -> str:
    candidate = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:64]
    if not SLUG_RE.fullmatch(candidate):
        raise PackagePolicyError(
            "skill name cannot be converted to a valid Wisdom slug"
        )
    return candidate


def _source_fingerprint(source: Path) -> str:
    rows: list[str] = []
    for path in sorted(source.rglob("*")):
        if path.is_file() and not path.is_symlink():
            rows.append(
                f"{path.relative_to(source).as_posix()} {sha256_address(path.read_bytes())}\n"
            )
    return sha256_address("".join(rows).encode("utf-8"))


def _write_active_org_marker(managed: Path, org_id: str) -> str:
    if not ORG_ID_RE.fullmatch(org_id):
        raise WisdomValidationError("team organization identity is malformed")
    directory_name = org_directory_name(org_id)
    managed.mkdir(parents=True, exist_ok=True, mode=0o700)
    marker = managed / ".active_org"
    pending = managed / f".active_org.{uuid.uuid4().hex}.pending"
    try:
        pending.write_text(directory_name + "\n", encoding="utf-8")
        pending.chmod(0o600)
        os.replace(pending, marker)
    finally:
        pending.unlink(missing_ok=True)
    return directory_name


def _verified_tree(root: Path) -> tuple[dict[str, str], str]:
    if not root.is_dir() or root.is_symlink():
        raise WisdomValidationError("managed package tree is missing or unsafe")
    files: list[tuple[str, str, bytes]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise WisdomValidationError("managed package tree contains a symbolic link")
        if path.is_file():
            files.append((path.relative_to(root).as_posix(), "file", path.read_bytes()))
    records, content_hash = verify_content_files(files)
    return {record.path: record.hash for record in records}, content_hash


def _editable_package_files(root: Path) -> list[dict[str, str]]:
    """Return a complete, policy-validated UTF-8 package for local editing."""
    if not root.is_dir() or root.is_symlink():
        raise PackagePolicyError("prepared package is missing or unsafe")
    files: list[tuple[str, str, bytes]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise PackagePolicyError("prepared package contains a symbolic link")
        if path.is_file():
            files.append((path.relative_to(root).as_posix(), "file", path.read_bytes()))
    records, _ = verify_content_files(files)
    bodies = {path: body for path, _mode, body in files}
    return [
        {
            "path": record.path,
            "mode": record.mode,
            "hash": record.hash,
            "content_utf8": bodies[record.path].decode("utf-8"),
        }
        for record in records
    ]


def _scan_summary(path: Path) -> dict[str, Any]:
    guard = scan_skill(path, source="community")
    allowed, reason = should_allow_install(guard)
    tier1 = run_tier1_scan(path) if tier1_advisory_enabled() else None
    return {
        "guard": {
            "verdict": guard.verdict,
            "allowed": allowed,
            "reason": reason,
            "findings": [
                {
                    "severity": finding.severity,
                    "category": finding.category,
                    "file": finding.file,
                    "line": finding.line,
                    "match": finding.match,
                }
                for finding in guard.findings
            ],
        },
        "skill_evaluator": (
            {
                "status": "available",
                "passed": tier1.passed,
                "findings": [
                    {
                        "check": finding.check,
                        "severity": finding.severity,
                        "file": finding.file,
                        "line": finding.line,
                        "message": finding.message,
                        "secrets_class": finding.is_secrets_class,
                    }
                    for finding in tier1.findings
                ],
                "incomplete_checks": tier1.incomplete_checks,
            }
            if tier1 and tier1.available
            else {
                "status": "disabled" if tier1 is None else "unavailable",
                "passed": None,
                "findings": [],
            }
        ),
    }


def _has_high_confidence_secret(scan: dict[str, Any]) -> bool:
    return any(
        item.get("secrets_class") and item.get("severity") in {"critical", "high"}
        for item in scan["skill_evaluator"]["findings"]
    )


def _extract_model_text(response: Any) -> str:
    from agent.auxiliary_client import extract_content_or_reasoning

    return extract_content_or_reasoning(response).strip()


def draft_description(skill_md: str) -> str:
    """Draft author copy with the configured model under normal routing rules."""
    try:
        from agent.auxiliary_client import call_llm

        response = call_llm(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Write one concise, outcome-oriented plain-text description of this Hermes skill. "
                        "Do not claim quality, safety, usage, popularity, or platform verification. "
                        "Return only the description, at most 600 characters."
                    ),
                },
                {"role": "user", "content": skill_md[:24000]},
            ],
            temperature=0.2,
            max_tokens=220,
            timeout=60,
        )
        text = _extract_model_text(response)
    except Exception as exc:
        raise WisdomError(
            "Hermes could not draft a description with the configured model. "
            "Check the model setup and try again."
        ) from exc
    if not text:
        raise PackagePolicyError(
            "the configured model did not return an author description"
        )
    return text


def _existing_author_description(skill_md: str) -> str | None:
    """Reuse explicit skill copy before asking a model to rewrite it.

    Most agent-created skills already carry a concise frontmatter description.
    Treating that as the editable initial owner copy keeps local review available
    when inference is offline and avoids an unnecessary model round trip. Skills
    without usable copy still use the configured model path below.
    """
    try:
        from agent.skill_utils import (
            extract_skill_editorial_metadata,
            parse_frontmatter,
        )

        frontmatter, _body = parse_frontmatter(skill_md)
    except Exception:
        return None
    value = frontmatter.get("description")
    description = value.strip() if isinstance(value, str) else ""
    editorial = extract_skill_editorial_metadata(
        frontmatter,
        fallback_name="",
        fallback_description=description,
    )
    return editorial["editorial_description"] or None


class WisdomService:
    def __init__(
        self, *, store: WisdomStore | None = None, client: WisdomClient | None = None
    ) -> None:
        self.store = store or WisdomStore()
        self._client = client

    @property
    def client(self) -> WisdomClient:
        if self._client is None:
            self._client = WisdomClient(
                timeout=float(_config().get("request_timeout", 30))
            )
        return self._client

    @property
    def consumption(self) -> WisdomConsumption:
        return WisdomConsumption(
            store=self.store,
            client=self.client,
            scan=_scan_summary,
            config=_config(),
        )

    def _enqueue_professionalism_review(
        self,
        *,
        subject_id: str,
        content_hash: str,
        package_root: Path,
        author_description: str,
    ) -> dict[str, Any]:
        return enqueue_review(
            self.store,
            skill_id=subject_id,
            content_hash=content_hash,
            package=exact_utf8_package(package_root),
            author_description=author_description,
        )

    def process_professionalism_reviews(
        self, *, max_jobs: int = 1, review_id: str | None = None
    ) -> list[dict[str, Any]]:
        return process_pending_reviews(
            self.store, max_jobs=max_jobs, review_id=review_id
        )

    def organization_display_name(
        self, *, organization_id: str | None = None, force: bool = False
    ) -> str | None:
        """Return a verified, cached organization name for local presentation."""

        active_org_id = self.store.active_org_id()
        target_org_id = organization_id or active_org_id
        if target_org_id is None or target_org_id != active_org_id:
            return None
        cached = self.store.organization_display_name(target_org_id)
        cached_name = (
            str(cached["display_name"]) if cached and cached["display_name"] else None
        )
        cached_resolved = bool(cached and cached.get("resolved"))
        if cached and not force:
            try:
                checked_at = datetime.fromisoformat(str(cached["checked_at"]))
                if checked_at.tzinfo is None:
                    checked_at = checked_at.replace(tzinfo=timezone.utc)
                ttl = (
                    ORGANIZATION_NAME_REFRESH
                    if cached_resolved
                    else ORGANIZATION_NAME_FAILURE_RETRY
                )
                if datetime.now(timezone.utc) - checked_at < ttl:
                    return cached_name
            except (TypeError, ValueError):
                pass
        try:
            from hermes_cli.nous_account import get_nous_portal_account_info

            account = get_nous_portal_account_info(force_fresh=True)
            verified_name = (
                account.org_name.strip()
                if account.org_id == target_org_id
                and isinstance(account.org_name, str)
                and account.org_name.strip()
                else None
            )
        except Exception:
            verified_name = None
        self.store.record_organization_display_name_check(target_org_id, verified_name)
        return verified_name or cached_name

    def _project_candidate_event(
        self, event: dict[str, Any], *, organization_name: str | None
    ) -> dict[str, Any]:
        sequence = int(event.get("qualification_sequence") or 1)
        return {
            **event,
            "qualification_sequence": sequence,
            "notice_variant": "first" if sequence == 1 else "returning",
            "organization_name": organization_name,
        }

    def local_candidate_events(
        self, *, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        events = self.store.local_events(kind="wisdom.candidate", session_id=session_id)
        organization_name = self.organization_display_name()
        return [
            self._project_candidate_event(event, organization_name=organization_name)
            for event in events
        ]

    def pending_candidate_events(
        self, *, session_id: str, surface: str
    ) -> list[dict[str, Any]]:
        events = self.store.pending_surface_events(
            kind="wisdom.candidate", session_id=session_id, surface=surface
        )
        organization_name = self.organization_display_name()
        return [
            self._project_candidate_event(event, organization_name=organization_name)
            for event in events
        ]

    def candidate_professionalism_review(
        self, *, skill_id: str, content_hash: str
    ) -> dict[str, Any] | None:
        row = self.store.professionalism_review(
            skill_id=skill_id,
            content_hash=content_hash,
            author_description_hash=author_description_hash(""),
        )
        if row is None:
            local = self.store.local_skill(skill_id)
            source = Path(str(local["canonical_path"])) if local else None
            if (
                source is None
                or not source.is_dir()
                or _source_fingerprint(source) != content_hash
            ):
                return None
            row = self._enqueue_professionalism_review(
                subject_id=skill_id,
                content_hash=content_hash,
                package_root=source,
                author_description="",
            )
        return row.get("result") or {"status": row["state"]}

    def finish_candidate_professionalism_review(
        self, *, skill_id: str, content_hash: str
    ) -> dict[str, Any] | None:
        row = self.store.professionalism_review(
            skill_id=skill_id,
            content_hash=content_hash,
            author_description_hash=author_description_hash(""),
        )
        if row is None:
            self.candidate_professionalism_review(
                skill_id=skill_id, content_hash=content_hash
            )
            row = self.store.professionalism_review(
                skill_id=skill_id,
                content_hash=content_hash,
                author_description_hash=author_description_hash(""),
            )
        if row is None:
            return None
        if row.get("state") != "complete":
            self.store.expedite_professionalism_review(str(row["id"]))
            process_pending_reviews(
                self.store,
                max_jobs=2,
                review_id=str(row["id"]),
                retry_delay_seconds=0,
            )
        return self.candidate_professionalism_review(
            skill_id=skill_id, content_hash=content_hash
        )

    def _require_professionalism_review(
        self,
        *,
        subject_id: str,
        content_hash: str,
        package_root: Path,
        author_description: str,
    ) -> dict[str, Any]:
        job = self._enqueue_professionalism_review(
            subject_id=subject_id,
            content_hash=content_hash,
            package_root=package_root,
            author_description=author_description,
        )
        if job.get("state") != "complete":
            self.store.expedite_professionalism_review(str(job["id"]))
            process_pending_reviews(
                self.store,
                max_jobs=1,
                review_id=str(job["id"]),
                terminal_on_failure=True,
            )
            job = (
                self.store.professionalism_review(
                    skill_id=subject_id,
                    content_hash=content_hash,
                    author_description_hash=author_description_hash(author_description),
                )
                or job
            )
        result = job.get("result")
        if not isinstance(result, dict):
            raise WisdomConflict(
                "Professionalism check is still running; try submission again shortly",
                code="professionalism_review_pending",
            )
        return result

    def require_setup(self) -> None:
        wisdom = _config()
        active_org_id = self.store.active_org_id()
        if (
            wisdom.get("enabled") is not True
            or not wisdom.get("disclosure_acknowledged_at")
            or self.store.existing_installation_identity() is None
            or active_org_id is None
            or not ORG_ID_RE.fullmatch(active_org_id)
        ):
            raise PackagePolicyError(
                "Collective Wisdom is not set up for this profile; run `hermes wisdom setup` first"
            )
        try:
            token_org_id = self.client.display_org_id
        except Exception:
            # The last server-verified org remains usable offline. Gateway is
            # authoritative whenever a network operation is attempted.
            return
        if token_org_id and token_org_id != active_org_id:
            raise PackagePolicyError(
                "the authenticated organization changed; rerun `hermes wisdom setup` before using Collective Wisdom"
            )

    def setup(self, *, disclosure_accepted: bool = False) -> dict[str, Any]:
        if not disclosure_accepted:
            raise PackagePolicyError(
                "setup requires explicit acceptance of the local telemetry and private-draft disclosure"
            )
        capability = self.client.capability()
        org_id = self.client.display_org_id
        if not org_id:
            raise WisdomValidationError(
                "team organization identity is missing from the current token"
            )
        if not ORG_ID_RE.fullmatch(org_id):
            raise WisdomValidationError("team organization identity is malformed")
        installation_id = self.store.existing_installation_identity()
        if installation_id is None or self.store.active_org_id() not in {None, org_id}:
            installation_id = "hwi_" + uuid.uuid4().hex
        registered = self.client.register_identity(installation_id)
        managed = get_skills_dir() / "_wisdom"
        # Publish the server-verified org marker before switching the local
        # ledger. A crash may temporarily select the new verified org while
        # setup asks to resume, but can never keep loading the stale org after
        # Gateway accepted the change.
        managed_org = _write_active_org_marker(managed, org_id)
        self.store.activate_installation_identity(installation_id, org_id)
        recovered = self.reconcile_pending_install_records()
        recovered.extend(self.consumption.recover())
        candidates = self.scan_candidates()
        from hermes_cli.config import load_config, save_config

        config = load_config()
        wisdom = config.get("wisdom")
        wisdom = dict(wisdom) if isinstance(wisdom, dict) else {}
        acknowledged_at = str(wisdom.get("disclosure_acknowledged_at") or utc_now())
        wisdom["enabled"] = True
        wisdom["disclosure_acknowledged_at"] = acknowledged_at
        config["wisdom"] = wisdom
        save_config(config)
        return {
            "ok": True,
            "installation_id": installation_id,
            "organization_id": org_id,
            "registered": registered,
            "capabilities": capability.get("features", []),
            "display_scopes": list(self.client.display_scopes),
            "database": str(self.store.path),
            "managed_directory": str(managed / managed_org),
            "candidate_count": len(candidates),
            "recovered_gateway_records": recovered,
            "disclosure": WISDOM_DISCLOSURE,
            "disclosure_acknowledged_at": acknowledged_at,
        }

    def reconcile_pending_install_records(self) -> list[str]:
        """Resume interrupted installs from their last durable journal phase."""
        recovered: list[str] = []
        installation_id = self.store.installation_identity()
        for operation in self.store.pending_operations():
            if operation["kind"] != "install":
                continue
            plan = json.loads(operation["payload_json"])
            skill_id = str(plan.get("skill_id") or operation["entity_id"])
            lock = self.store.acquire_operation_lock(skill_id)
            if not lock:
                continue
            try:
                plan_org = str(plan.get("org_id") or "")
                if plan_org and plan_org != self.store.active_org_id():
                    raw_staging = plan.get("staging_path")
                    staging = (
                        Path(raw_staging)
                        if isinstance(raw_staging, str) and raw_staging
                        else None
                    )
                    if staging is not None and staging.is_dir():
                        abandoned = (
                            self.store.root
                            / "recovery"
                            / str(operation["id"])
                            / "stale-org-staging"
                        )
                        abandoned.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                        if not abandoned.exists():
                            os.replace(staging, abandoned)
                    self.store.deactivate_install(skill_id)
                    self.store.advance(str(operation["id"]), "stale_org", done=True)
                    continue
                if operation["phase"] in {"staged", "files_committed"}:
                    receipt = str(plan.get("receipt") or "")
                    if not receipt:
                        continue
                    self._resume_install(
                        str(operation["id"]),
                        plan,
                        plan_path=self.store.root / "plans" / f"{receipt}.json",
                    )
                    recovered.append(skill_id)
                    continue
                if operation["phase"] != "local_ledger_committed":
                    continue
                server = self.client.record_install(
                    skill_id=plan["skill_id"],
                    installation_id=installation_id,
                    version=int(plan["version"]),
                    takedown_generation=int(plan["takedown_generation"]),
                    update_mode=plan.get("update_mode"),
                )
            except (WisdomConflict, WisdomNotFound, WisdomValidationError):
                local = self.store.installation(str(plan.get("skill_id", "")))
                if local:
                    target = Path(str(local["target_path"]))
                    quarantine = (
                        self.store.root
                        / "recovery"
                        / str(operation["id"])
                        / "gateway-rejected"
                    )
                    quarantine.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    if target.exists() and not quarantine.exists():
                        os.replace(target, quarantine)
                    self.store.deactivate_install(str(plan["skill_id"]))
                self.store.advance(operation["id"], "gateway_rejected", done=True)
                continue
            except Exception:
                continue
            else:
                local = self.store.installation(str(plan["skill_id"]))
                if local:
                    self.store.record_install({
                        **local,
                        "baseline": local["baseline"],
                        "update_mode": server.effective_update_mode,
                    })
                self.store.advance(operation["id"], "gateway_recorded", done=True)
                recovered.append(str(operation["entity_id"]))
            finally:
                self.store.release_operation_lock(skill_id, lock)
        return recovered

    def status(self) -> dict[str, Any]:
        try:
            client = self.client
            live = True
            error_kind = None
            capability = client.capability()
            scopes = list(client.display_scopes)
            authenticated_org_id = client.display_org_id
            admin_gate = (
                client.identity.get("claims", {}).get("tool_gateway_admin") is True
            )
        except WisdomAuthError as exc:
            live = False
            error_kind = "authentication"
            capability = {}
            scopes = []
            authenticated_org_id = None
            admin_gate = False
            error = str(exc)
        except Exception as exc:
            live = False
            error_kind = "unavailable"
            capability = {}
            scopes = []
            authenticated_org_id = None
            admin_gate = False
            error = str(exc)
        wisdom = _config()
        installation_id = self.store.existing_installation_identity()
        verified_org_id = self.store.active_org_id()
        locally_configured = bool(
            wisdom.get("enabled") is True
            and wisdom.get("disclosure_acknowledged_at")
            and installation_id
            and verified_org_id
            and ORG_ID_RE.fullmatch(verified_org_id)
        )
        organization_changed = bool(
            authenticated_org_id
            and verified_org_id
            and authenticated_org_id != verified_org_id
        )
        return {
            "configured": locally_configured and not organization_changed,
            "setup_required_reason": (
                "organization_changed"
                if organization_changed
                else None
                if locally_configured
                else "not_configured"
            ),
            "gateway_available": live,
            "error": None if live else error,
            "error_kind": error_kind,
            "capability_advertised": "wisdom" in (capability.get("features") or []),
            "entitled": "wisdom:read" in scopes,
            "display_scopes": scopes,
            "dogfood_admin_claim": admin_gate,
            "installation_id": installation_id,
            "verified_org_id": verified_org_id,
            "authenticated_org_id": authenticated_org_id,
            "local_store": str(self.store.path),
            "pending_operations": self.store.pending_operations(),
            "contract": asdict(CONTRACT_PIN),
        }

    def _eligible_paths(self) -> list[Path]:
        root = get_skills_dir().resolve()
        if not root.exists():
            return []
        paths: list[Path] = []
        for skill_md in sorted(root.rglob("SKILL.md")):
            try:
                rel = skill_md.relative_to(root)
            except ValueError:
                continue
            if any(
                part in {".archive", "_org", "_wisdom", ".hub"} for part in rel.parts
            ):
                continue
            path = skill_md.parent
            name = path.name
            if is_bundled(name) or is_hub_installed(name):
                continue
            paths.append(path)
        return paths

    def _candidate_source(
        self, skill_name: str, local_skill_id: str | None = None
    ) -> Path:
        eligible = {path.resolve() for path in self._eligible_paths()}
        if local_skill_id:
            local = self.store.local_skill(local_skill_id)
            if local:
                source = Path(str(local["canonical_path"])).resolve()
                if source in eligible and source.name == skill_name:
                    return source
            raise PackagePolicyError(
                "this local skill changed or moved; scan local skills and try again"
            )
        matches = [path for path in eligible if path.name == skill_name]
        if not matches:
            raise PackagePolicyError("skill is not eligible for Collective Wisdom")
        if len(matches) > 1:
            raise PackagePolicyError(
                "more than one eligible local skill has this name; select it from the dashboard"
            )
        return matches[0]

    def _candidate_editorial_copy(
        self, *, skill_id: str, content_hash: str
    ) -> dict[str, str | None]:
        for event in self.store.local_events(kind="wisdom.candidate"):
            if (
                str(event.get("skill_id")) != skill_id
                or str(event.get("content_hash")) != content_hash
            ):
                continue
            payload = event.get("payload")
            payload = payload if isinstance(payload, dict) else {}
            return {
                "editorial_name": (
                    str(payload["editorial_name"])
                    if isinstance(payload.get("editorial_name"), str)
                    else None
                ),
                "editorial_description": (
                    str(payload["editorial_description"])
                    if isinstance(payload.get("editorial_description"), str)
                    else None
                ),
            }
        return {"editorial_name": None, "editorial_description": None}

    def scan_candidates(self) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        qualified = {
            (str(event["skill_id"]), str(event["content_hash"])): event
            for event in self.local_candidate_events()
        }
        eligible_paths = self._eligible_paths()
        self.store.mark_missing_skills({str(path.resolve()) for path in eligible_paths})
        for path in eligible_paths:
            source_hash = _source_fingerprint(path)
            skill_id = self.store.register_skill(
                path, content_hash=source_hash, source_kind="local"
            )
            event = qualified.get((skill_id, source_hash))
            editorial = load_skill_editorial_metadata(path)
            if event:
                payload = event.get("payload")
                payload = payload if isinstance(payload, dict) else {}
                for key in ("editorial_name", "editorial_description"):
                    if isinstance(payload.get(key), str):
                        editorial[key] = payload[key]
            contribution = self.store.latest_draft_for_source(skill_id, source_hash)
            if contribution and str(contribution["state"]) != "prepared":
                # The exact bytes have already entered the contribution flow.
                # A material source change gets a new hash and becomes eligible again.
                continue
            try:
                # Preparation remains manual; this dry structural pass rejects
                # scripts/templates/unsupported bytes without uploading anything.
                prepare_package(
                    path,
                    overlay_root=self.store.root / "scan-overlays",
                    author_description=f"Local skill {path.name}.",
                    owner="local-scan",
                    installation_id=self.store.installation_identity(),
                )
                eligibility = "eligible"
                reason = None
            except PackagePolicyError as exc:
                eligibility = "instruction_only_fork_required"
                reason = str(exc)
            candidate_review = (
                self.candidate_professionalism_review(
                    skill_id=skill_id, content_hash=source_hash
                )
                if event
                else None
            )
            candidates.append({
                "local_skill_id": skill_id,
                "name": path.name,
                **editorial,
                "path": str(path),
                "content_hash": source_hash,
                "eligibility": eligibility,
                "reason": (
                    reason
                    if reason
                    else json.dumps(event["payload"]["local_reasons"], sort_keys=True)
                    if event
                    else None
                ),
                "qualification": (
                    str(event["qualification"]) if event else "manual_selection"
                ),
                "qualification_sequence": (
                    int(event["qualification_sequence"]) if event else None
                ),
                "notice_variant": (str(event["notice_variant"]) if event else None),
                "organization_name": (
                    event.get("organization_name") if event else None
                ),
                "contribution_state": "prepared" if contribution else "new",
                "professionalism_check": candidate_review,
            })
        return candidates

    def scan(self, skill_name: str | None = None) -> dict[str, Any]:
        selected = self.scan_candidates()
        if skill_name:
            selected = [item for item in selected if item["name"] == skill_name]
            if not selected:
                raise PackagePolicyError(f"local skill not found: {skill_name}")
        return {
            "candidates": [
                {**item, "local_scan": _scan_summary(Path(item["path"]))}
                for item in selected
            ]
        }

    def _prepared_result(self, draft: dict[str, Any]) -> dict[str, Any]:
        overlay = Path(str(draft["overlay_path"]))
        manifest = _parse_package_manifest(
            (overlay / "skill.manifest.json").read_bytes()
        )
        review = self.store.professionalism_review(
            skill_id=str(draft["skill_id"]),
            content_hash=str(draft["content_hash"]),
            author_description_hash=str(draft["description_hash"]),
        )
        return {
            "network_submission": False,
            "local_draft_id": str(draft["id"]),
            "overlay_path": str(overlay),
            "drafted_description": str(draft["description"]),
            "system_specification": manifest.requirements.model_dump(mode="json"),
            "files": _editable_package_files(overlay),
            "local_scan": _scan_summary(overlay),
            "professionalism_check": (
                review.get("result")
                if review and isinstance(review.get("result"), dict)
                else {"status": str(review.get("state") if review else "pending")}
            ),
            "next_step": (
                "Review and save the complete local package, then send its exact bytes "
                "for owner-only server review."
            ),
        }

    def suggest(
        self,
        skill_name: str | None = None,
        *,
        description: str | None = None,
        system_specification: dict[str, Any] | None = None,
        allow_private_secret_review: bool = False,
        local_skill_id: str | None = None,
    ) -> dict[str, Any]:
        if not skill_name:
            return {"candidates": self.scan_candidates(), "network_submission": False}
        source = self._candidate_source(skill_name, local_skill_id)
        source_hash = _source_fingerprint(source)
        skill_id = self.store.register_skill(
            source, content_hash=source_hash, source_kind="local"
        )
        editorial = self._candidate_editorial_copy(
            skill_id=skill_id, content_hash=source_hash
        )
        prepared = self.store.prepared_draft(skill_id, source_hash)
        existing = self.store.latest_draft_for_source(skill_id, source_hash)
        if prepared is None and existing is not None:
            state = str(existing["state"])
            if state == "published":
                message = (
                    "this exact skill version is already shared; change the local content "
                    "before contributing a new version"
                )
            else:
                message = (
                    f"this exact skill version is already in the contribution flow ({state}); "
                    "open its existing draft instead"
                )
            raise WisdomConflict(message, code="wisdom_source_already_contributed")
        if prepared is None:
            skill_md = (source / "SKILL.md").read_text(encoding="utf-8")
            author_copy = (
                description
                or _existing_author_description(skill_md)
                or draft_description(skill_md)
            )
            local_package = prepare_package(
                source,
                overlay_root=self.store.root / "drafts",
                author_description=author_copy,
                owner=str(self.client.identity.get("owner")),
                installation_id=self.store.installation_identity(),
                editorial_name=editorial["editorial_name"],
                editorial_description=editorial["editorial_description"],
            )
            local_id = f"local:{skill_id}:{source_hash.removeprefix('sha256:')[:16]}"
            self.store.record_draft({
                "id": local_id,
                "skill_id": skill_id,
                "source_hash": source_hash,
                "overlay_path": str(local_package.overlay),
                "state": "prepared",
                "description": local_package.description,
                "content_hash": local_package.content_hash,
                "description_hash": local_package.description_hash,
                "manifest_hash": local_package.manifest_hash,
            })
            self._enqueue_professionalism_review(
                subject_id=skill_id,
                content_hash=local_package.content_hash,
                package_root=local_package.overlay,
                author_description=local_package.description,
            )
            local = self.store.draft(local_id)
            if local is None:  # pragma: no cover - SQLite write/read invariant
                raise WisdomValidationError("prepared local draft was not persisted")
            return self._prepared_result(local)
        if description is None:
            return self._prepared_result(prepared)
        if system_specification is None:
            raise PackagePolicyError(
                "submission requires explicit owner approval of the System Specification"
            )
        overlay = Path(prepared["overlay_path"])
        existing_manifest = _parse_package_manifest(
            (overlay / "skill.manifest.json").read_bytes()
        )
        approved_manifest = PackageManifest(
            schema_version=existing_manifest.schema_version,
            name=existing_manifest.name,
            requirements=SystemSpecification.model_validate(system_specification),
        )
        manifest_path = overlay / "skill.manifest.json"
        temporary_manifest = manifest_path.with_suffix(".json.pending")
        temporary_manifest.write_bytes(
            canonical_json_bytes(approved_manifest.model_dump(mode="json"))
        )
        temporary_manifest.chmod(0o600)
        os.replace(temporary_manifest, manifest_path)
        package = prepare_package(
            overlay,
            overlay_root=self.store.root / "submissions",
            author_description=description,
            owner=str(self.client.identity.get("owner")),
            installation_id=self.store.installation_identity(),
        )
        local_scan = _scan_summary(package.overlay)
        if local_scan["guard"]["allowed"] is False:
            raise PackagePolicyError(
                f"built-in guard blocked the exact staged package: {local_scan['guard']['reason']}"
            )
        if _has_high_confidence_secret(local_scan) and not allow_private_secret_review:
            raise PackagePolicyError(
                "high-confidence local secret finding paused upload; rerun with the explicit "
                "--send-for-owner-only-server-review confirmation"
            )
        if _source_fingerprint(source) != source_hash:
            raise PackagePolicyError(
                "source changed while the review overlay was being prepared"
            )
        professionalism_review = self._require_professionalism_review(
            subject_id=skill_id,
            content_hash=package.content_hash,
            package_root=package.overlay,
            author_description=package.description,
        )
        self.client.upload_private_objects(package.objects)
        server = self.client.submit_draft(
            slug=_slug(skill_name),
            commit=package.commit,
            content_hash=package.content_hash,
            description=package.description,
            professionalism_review=professionalism_review,
        )
        self.store.set_draft_state(str(prepared["id"]), "submitted")
        self.store.record_draft({
            "id": server.id,
            "skill_id": skill_id,
            "source_hash": source_hash,
            "overlay_path": str(package.overlay),
            "draft_commit": server.draftCommit,
            "server_revision": server.updatedAt,
            "state": server.state,
            "description": server.authorDescription or "",
            "content_hash": server.contentHash,
            "description_hash": server.authorDescriptionHash
            or package.description_hash,
            "manifest_hash": server.packageManifestHash or package.manifest_hash,
        })
        return {
            "draft": server.model_dump(mode="json"),
            "local_scan": local_scan,
            "professionalism_check": professionalism_review,
            "notice": "Draft bytes are owner-private; nothing is published until hash-bound approval.",
        }

    def save_prepared(
        self,
        draft_id: str,
        *,
        author_description: str,
        files: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Save owner edits locally without performing any network operation."""
        local = self.store.draft(draft_id)
        if local is None or not draft_id.startswith("local:"):
            raise WisdomNotFound("prepared draft not found")
        if str(local["state"]) != "prepared":
            raise WisdomConflict(
                f"this local draft cannot be edited while it is {str(local['state']).replace('_', ' ')}",
                code=f"state_is_{local['state']}",
            )

        overlay = Path(str(local["overlay_path"]))
        authoritative = {
            item["path"]: item for item in _editable_package_files(overlay)
        }
        if len(files) > MAX_FILES:
            raise PackagePolicyError(f"package exceeds {MAX_FILES} files")

        requested: dict[str, bytes] = {}
        for item in files:
            path = item.get("path")
            content = item.get("content_utf8")
            if not isinstance(path, str) or not isinstance(content, str):
                raise PackagePolicyError(
                    "every edited file requires a path and UTF-8 content"
                )
            if path in requested:
                raise PackagePolicyError(f"duplicate edited file: {path}")
            requested[path] = content.encode("utf-8")

        verify_content_files([(path, "file", body) for path, body in requested.items()])
        if set(requested) != set(authoritative):
            raise PackagePolicyError(
                "a saved local draft must include the complete existing package without adding, removing, or renaming files"
            )

        namespace = sha256_address(draft_id.encode("utf-8")).removeprefix("sha256:")
        local_inputs = self.store.root / "local-edit-inputs"
        local_inputs.mkdir(parents=True, exist_ok=True, mode=0o700)
        with tempfile.TemporaryDirectory(prefix="edit-", dir=local_inputs) as raw_stage:
            stage = Path(raw_stage)
            for path, body in requested.items():
                destination = stage.joinpath(*PurePosixPath(path).parts)
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                destination.write_bytes(body)
                destination.chmod(0o600)
            package = prepare_package(
                stage,
                overlay_root=self.store.root / "local-edits" / namespace,
                author_description=author_description,
                owner=str(self.client.identity.get("owner")),
                installation_id=self.store.installation_identity(),
            )

        self.store.record_draft({
            **local,
            "overlay_path": str(package.overlay),
            "state": "prepared",
            "description": package.description,
            "content_hash": package.content_hash,
            "description_hash": package.description_hash,
            "manifest_hash": package.manifest_hash,
        })
        self._enqueue_professionalism_review(
            subject_id=str(local["skill_id"]),
            content_hash=package.content_hash,
            package_root=package.overlay,
            author_description=package.description,
        )
        saved = self.store.draft(draft_id)
        if saved is None:  # pragma: no cover - SQLite write/read invariant
            raise WisdomValidationError("saved local draft was not persisted")
        return self._prepared_result(saved)

    def dismiss_local_candidate(
        self, local_skill_id: str, content_hash: str
    ) -> dict[str, Any]:
        local = self.store.local_skill(local_skill_id)
        if local is None or str(local.get("current_hash")) != content_hash:
            raise WisdomNotFound("local candidate not found")
        self.store.dismiss_candidate(local_skill_id, content_hash)
        prepared = self.store.prepared_draft(local_skill_id, content_hash)
        if prepared is not None:
            self.store.set_draft_state(str(prepared["id"]), "declined")
        return {"dismissed": True}

    def portal_review_url(self, draft_id: str) -> str:
        """Return the authenticated Portal review URL for this profile's team."""
        org_id = self.store.active_org_id()
        if not org_id:
            raise PackagePolicyError("Collective Wisdom is not set up for this profile")
        org_slug = org_id.split(":", 1)[-1]
        return (
            f"{portal_base_url()}/orgs/{quote(org_slug, safe='')}/wisdom/review/"
            f"{quote(draft_id, safe='')}"
        )

    def portal_skill_url(
        self, skill_id: str, *, version: int | None = None
    ) -> str | None:
        """Return the Portal detail URL when this profile has a verified team."""
        org_id = self.store.active_org_id()
        if not org_id:
            return None
        org_slug = org_id.split(":", 1)[-1]
        url = (
            f"{portal_base_url()}/orgs/{quote(org_slug, safe='')}/wisdom/skills/"
            f"{quote(skill_id, safe='')}"
        )
        return f"{url}?version={version}" if version is not None else url

    def _candidate_event_context(
        self, event_id: str
    ) -> tuple[dict[str, Any], str, str, str]:
        event = self._candidate_event(event_id)
        if event.get("state") != "unread":
            raise WisdomConflict(
                "this local candidate has already been handled",
                code="candidate_already_handled",
            )

        local_skill_id = str(event["skill_id"])
        content_hash = str(event["content_hash"])
        local = self.store.local_skill(local_skill_id)
        if local is None or str(local.get("current_hash")) != content_hash:
            raise WisdomConflict(
                "this local skill changed after qualification; use its current version instead",
                code="candidate_source_changed",
            )
        source = Path(str(local["canonical_path"]))
        if not source.is_dir() or _source_fingerprint(source) != content_hash:
            raise WisdomConflict(
                "this local skill changed after qualification; scan local skills and try again",
                code="candidate_source_changed",
            )
        skill_name = source.name
        # Re-run the normal eligibility boundary so a callback cannot turn a
        # stale or excluded local path into an upload authority.
        self._candidate_source(skill_name, local_skill_id)
        return event, local_skill_id, content_hash, skill_name

    def _candidate_event(self, event_id: str) -> dict[str, Any]:
        """Return a candidate event even after another surface resolved it.

        A Telegram button can outlive the local unread marker.  Reading the
        event is safe; mutation still goes through the authoritative draft
        state and the exact-source checks below.
        """
        event = self.store.local_event(event_id)
        if event is None or event.get("kind") != "wisdom.candidate":
            raise WisdomNotFound("local candidate not found")
        return event

    def _existing_candidate_draft(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """Reconcile a candidate's existing draft with Gateway authority."""
        local_skill_id = str(event["skill_id"])
        content_hash = str(event["content_hash"])
        existing = self.store.latest_draft_for_source(local_skill_id, content_hash)
        if existing is None or str(existing["id"]).startswith("local:"):
            return None

        draft_id = str(existing["id"])
        authoritative = self.client.draft(draft_id).draft
        state = authoritative.state
        if state == "declined":
            self.store.set_draft_state(draft_id, state)
            self.store.dismiss_candidate(local_skill_id, content_hash)
        elif state in {
            "owner_approved",
            "publishing",
            "pending_moderation",
            "changes_requested",
            "published",
            "invalidated",
        }:
            self.store.complete_contribution(draft_id, state)
        else:
            self.store.set_draft_state(draft_id, state)
        payload = event.get("payload")
        payload = payload if isinstance(payload, dict) else {}
        return {
            "draft_id": draft_id,
            "skill_name": authoritative.slug
            or str(payload.get("skill_name") or "Local skill"),
            "qualification": str(event.get("qualification") or ""),
            "state": state,
            "portal_url": self.portal_review_url(draft_id),
            "created": False,
        }

    def draft_candidate(self, event_id: str) -> dict[str, Any]:
        """Create or resume an owner-private draft for one exact candidate."""
        self.require_setup()
        event = self._candidate_event(event_id)
        existing = self._existing_candidate_draft(event)
        if existing is not None:
            return existing
        event, local_skill_id, content_hash, skill_name = self._candidate_event_context(
            event_id
        )
        qualification = str(event.get("qualification") or "")

        prepared = self.store.prepared_draft(local_skill_id, content_hash)
        prepared_result = (
            self._prepared_result(prepared)
            if prepared is not None
            else self.suggest(skill_name, local_skill_id=local_skill_id)
        )
        submitted = self.suggest(
            skill_name,
            description=str(prepared_result["drafted_description"]),
            system_specification=dict(prepared_result["system_specification"]),
            local_skill_id=local_skill_id,
        )
        draft = submitted.get("draft")
        if not isinstance(draft, dict) or not isinstance(draft.get("id"), str):
            raise WisdomValidationError("Gateway did not return an owner-private draft")
        draft_id = str(draft["id"])
        return {
            "draft_id": draft_id,
            "skill_name": skill_name,
            "qualification": qualification,
            "state": str(draft.get("state") or "ready"),
            "portal_url": self.portal_review_url(draft_id),
            "created": True,
        }

    def approve_candidate(self, event_id: str) -> dict[str, Any]:
        """Use an explicit qualification action as exact-package owner consent."""
        drafted = self.draft_candidate(event_id)
        state = str(drafted["state"])
        if state in {"pending_moderation", "published"}:
            return {
                **drafted,
                "publication_state": state,
                "already_advanced": True,
            }
        if state in {"changes_requested", "declined", "invalidated"}:
            return {
                **drafted,
                "publication_state": state,
                "already_advanced": True,
            }

        if state in {"owner_approved", "publishing"}:
            return self._resume_candidate_publication(drafted)
        if state != "ready":
            raise WisdomConflict(
                f"this private draft is currently {state.replace('_', ' ')}",
                code=f"state_is_{state}",
            )

        draft_id = str(drafted["draft_id"])
        # Approval still passes through the ordinary authoritative re-fetch and
        # three-hash receipt. The button is consent, not a bypass around review.
        try:
            self.review(draft_id, acknowledge=True)
            result = self.approve(draft_id)
        except WisdomConflict:
            # Portal approval and Telegram approval may race. Re-read Gateway
            # and accept the committed winner instead of surfacing a stale
            # button error or creating a second publication proposal.
            event = self._candidate_event(event_id)
            refreshed = self._existing_candidate_draft(event)
            if refreshed is None:
                raise
            refreshed_state = str(refreshed["state"])
            if refreshed_state in {"owner_approved", "publishing"}:
                return self._resume_candidate_publication(refreshed)
            if refreshed_state in {
                "pending_moderation",
                "published",
                "changes_requested",
                "declined",
                "invalidated",
            }:
                return {
                    **refreshed,
                    "publication_state": refreshed_state,
                    "already_advanced": True,
                }
            raise
        publication = result.get("publication")
        publication = publication if isinstance(publication, dict) else {}
        publication_state = str(publication.get("state") or "")
        return {
            **drafted,
            "state": publication_state,
            "publication_state": publication_state,
            "approval": result,
        }

    def defer_candidate_prompt(self, event_id: str, *, surface: str) -> dict[str, Any]:
        """Hide one surface prompt without declining the qualified skill."""
        if surface not in {"desktop", "slack", "telegram"}:
            raise WisdomValidationError("unsupported candidate notification surface")
        event = self._candidate_event(event_id)
        if event.get("organization_id") != self.store.active_org_id():
            raise WisdomNotFound("local candidate not found")
        self.store.mark_surface_delivered([event_id], surface=surface)
        payload = event.get("payload")
        payload = payload if isinstance(payload, dict) else {}
        return {
            "event_id": event_id,
            "skill_name": str(payload.get("skill_name") or "Local skill"),
            "qualification": str(event.get("qualification") or ""),
            "state": "deferred",
        }

    def _resume_candidate_publication(self, drafted: dict[str, Any]) -> dict[str, Any]:
        """Resume the Gateway's idempotent publication coordinator."""
        draft_id = str(drafted["draft_id"])
        receipt, _authoritative = self._validated_review_receipt(draft_id)
        published = self.client.publish(
            draft_id, content_hash=str(receipt["content_hash"])
        )
        publication_state = str(published.get("state") or "")
        if publication_state not in {"pending_moderation", "published"}:
            raise WisdomValidationError("Gateway returned an invalid publication state")
        self.store.complete_contribution(draft_id, publication_state)
        self.store.consume_receipt(draft_id)
        return {
            **drafted,
            "state": publication_state,
            "publication_state": publication_state,
            "approval": {"publication": published},
        }

    def decline_candidate(self, event_id: str) -> dict[str, Any]:
        """Decline local bytes or withdraw their existing private contribution."""
        event = self._candidate_event(event_id)
        existing = self._existing_candidate_draft(event)
        if existing is not None:
            state = str(existing["state"])
            if state in {"published", "declined"}:
                return {
                    **existing,
                    "already_advanced": True,
                }
            draft_id = str(existing["draft_id"])
            try:
                self.decline(draft_id)
            except WisdomConflict:
                refreshed = self._existing_candidate_draft(event)
                if refreshed is not None and str(refreshed["state"]) in {
                    "published",
                    "declined",
                }:
                    return {
                        **refreshed,
                        "already_advanced": True,
                    }
                raise
            return {
                **existing,
                "state": "declined",
                "withdrawn": state == "pending_moderation",
            }

        event, local_skill_id, content_hash, skill_name = self._candidate_event_context(
            event_id
        )
        result = self.dismiss_local_candidate(local_skill_id, content_hash)
        return {
            **result,
            "skill_name": skill_name,
            "qualification": str(event.get("qualification") or ""),
            "state": "declined",
        }

    def review(
        self, draft_id: str, *, acknowledge: bool, portal: bool = False
    ) -> dict[str, Any]:
        reconstructed = self.client.reconstruct_draft(draft_id)
        draft = reconstructed.detail.draft
        manifest_body = next(
            body
            for path, _, body in reconstructed.files
            if path == "skill.manifest.json"
        )
        manifest_hash = sha256_address(manifest_body)
        if manifest_hash != draft.packageManifestHash:
            raise WisdomValidationError(
                "server draft package manifest hash does not match exact bytes"
            )
        description_hash = author_description_hash(draft.authorDescription or "")
        if description_hash != draft.authorDescriptionHash:
            raise WisdomValidationError(
                "server draft author description hash does not match displayed copy"
            )
        result = {
            "draft": draft.model_dump(mode="json"),
            "effective_policy": reconstructed.detail.effective_policy,
            "files": [
                {
                    "path": path,
                    "mode": mode,
                    "hash": sha256_address(body),
                    "content_utf8": body.decode("utf-8", errors="replace"),
                }
                for path, mode, body in reconstructed.files
            ],
            "hashes": {
                "content": reconstructed.content_hash,
                "author_description": description_hash,
                "package_manifest": manifest_hash,
            },
            "receipt": None,
        }
        if portal:
            url = self.portal_review_url(draft_id)
            webbrowser.open(url)
            result["portal_url"] = url
        if acknowledge:
            result["receipt"] = self.store.save_receipt(
                draft_id=draft_id,
                server_revision=draft.updatedAt,
                content_hash=reconstructed.content_hash,
                description_hash=description_hash,
                manifest_hash=manifest_hash,
            )
        return result

    def revise(
        self,
        draft_id: str,
        *,
        author_description: str,
        files: list[dict[str, str]],
        expected_content_hash: str,
        expected_description_hash: str,
        expected_manifest_hash: str,
        allow_private_secret_review: bool = False,
    ) -> dict[str, Any]:
        """Create and upload an edited successor without mutating reviewed bytes."""
        current = self.client.reconstruct_draft(draft_id)
        draft = current.detail.draft
        manifest_body = next(
            body for path, _, body in current.files if path == "skill.manifest.json"
        )
        authoritative = (
            current.content_hash,
            author_description_hash(draft.authorDescription or ""),
            sha256_address(manifest_body),
        )
        expected = (
            expected_content_hash,
            expected_description_hash,
            expected_manifest_hash,
        )
        if authoritative != expected:
            raise WisdomConflict(
                "this draft changed after it was opened; reload it before saving",
                code="stale_revision",
            )
        if draft.state not in {"ready", "changes_requested", "invalidated"}:
            raise WisdomConflict(
                f"this draft cannot be edited while it is {draft.state.replace('_', ' ')}",
                code=f"state_is_{draft.state}",
            )
        if len(files) > MAX_FILES:
            raise PackagePolicyError(f"package exceeds {MAX_FILES} files")

        requested: dict[str, bytes] = {}
        total = 0
        for item in files:
            path = item.get("path")
            content = item.get("content_utf8")
            if not isinstance(path, str) or not isinstance(content, str):
                raise PackagePolicyError(
                    "every edited file requires a path and UTF-8 content"
                )
            if path in requested:
                raise PackagePolicyError(f"duplicate edited file: {path}")
            body = content.encode("utf-8")
            if len(body) > MAX_FILE_BYTES:
                raise PackagePolicyError(f"file exceeds {MAX_FILE_BYTES} bytes: {path}")
            total += len(body)
            if total > MAX_TREE_BYTES:
                raise PackagePolicyError(
                    f"package exceeds {MAX_TREE_BYTES} total bytes"
                )
            requested[path] = body

        authoritative_files = {path: (mode, body) for path, mode, body in current.files}
        if set(requested) != set(authoritative_files):
            raise PackagePolicyError(
                "an edited revision must include the complete existing package without adding, removing, or renaming files"
            )
        if any(mode != "file" for mode, _ in authoritative_files.values()):
            raise PackagePolicyError(
                "executable Wisdom content cannot be edited or published"
            )

        revision_inputs = self.store.root / "revision-inputs"
        revision_inputs.mkdir(parents=True, exist_ok=True, mode=0o700)
        revision_namespace = sha256_address(draft_id.encode("utf-8")).removeprefix(
            "sha256:"
        )
        with tempfile.TemporaryDirectory(
            prefix="edit-", dir=revision_inputs
        ) as raw_stage:
            stage = Path(raw_stage)
            for path, body in requested.items():
                destination = stage.joinpath(*PurePosixPath(path).parts)
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                destination.write_bytes(body)
                destination.chmod(0o600)
            package = prepare_package(
                stage,
                overlay_root=self.store.root / "revisions" / revision_namespace,
                author_description=author_description,
                owner=str(self.client.identity.get("owner")),
                installation_id=self.store.installation_identity(),
            )

        local_scan = _scan_summary(package.overlay)
        if local_scan["guard"]["allowed"] is False:
            raise PackagePolicyError(
                f"built-in guard blocked the exact edited package: {local_scan['guard']['reason']}"
            )
        if _has_high_confidence_secret(local_scan) and not allow_private_secret_review:
            raise PackagePolicyError(
                "high-confidence local secret finding paused upload; explicitly confirm owner-only server review to continue"
            )

        local = self.store.draft(draft_id)
        review_subject = str(local["skill_id"]) if local else f"draft:{draft_id}"
        professionalism_review = self._require_professionalism_review(
            subject_id=review_subject,
            content_hash=package.content_hash,
            package_root=package.overlay,
            author_description=package.description,
        )

        self.client.upload_private_objects(package.objects)
        revised = self.client.revise_draft(
            draft_id,
            commit=package.commit,
            content_hash=package.content_hash,
            description=package.description,
            expected_content_hash=expected_content_hash,
            expected_description_hash=expected_description_hash,
            expected_manifest_hash=expected_manifest_hash,
            professionalism_review=professionalism_review,
        )
        if local:
            self.store.set_draft_state(draft_id, "invalidated")
            self.store.record_draft({
                "id": revised.id,
                "skill_id": str(local["skill_id"]),
                "source_hash": str(local["source_hash"]),
                "overlay_path": str(package.overlay),
                "draft_commit": revised.draftCommit,
                "server_revision": revised.updatedAt,
                "state": revised.state,
                "description": revised.authorDescription or "",
                "content_hash": revised.contentHash,
                "description_hash": revised.authorDescriptionHash
                or package.description_hash,
                "manifest_hash": revised.packageManifestHash or package.manifest_hash,
            })
        self.store.consume_receipt(draft_id)
        return {
            "draft": revised.model_dump(mode="json"),
            "local_scan": local_scan,
            "professionalism_check": professionalism_review,
            "notice": "Changes were saved as a new owner-private revision and rescanned.",
        }

    def _validated_review_receipt(
        self, draft_id: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        receipt = self.store.receipt(draft_id)
        if not receipt:
            raise PackagePolicyError(
                "approval requires a fresh complete-package review receipt"
            )
        current = self.client.reconstruct_draft(draft_id)
        draft = current.detail.draft
        manifest_body = next(
            body for path, _, body in current.files if path == "skill.manifest.json"
        )
        expected = (
            current.content_hash,
            author_description_hash(draft.authorDescription or ""),
            sha256_address(manifest_body),
            draft.updatedAt,
        )
        received = (
            receipt["content_hash"],
            receipt["description_hash"],
            receipt["manifest_hash"],
            receipt["server_revision"],
        )
        if expected != received:
            raise PackagePolicyError(
                "review receipt is stale; review the complete server draft again"
            )
        return receipt, draft.model_dump(mode="json")

    def approve(self, draft_id: str) -> dict[str, Any]:
        receipt, _draft = self._validated_review_receipt(draft_id)
        approved = self.client.approve(
            draft_id,
            content_hash=receipt["content_hash"],
            description_hash=receipt["description_hash"],
            manifest_hash=receipt["manifest_hash"],
        )
        published = self.client.publish(draft_id, content_hash=receipt["content_hash"])
        publication_state = published.get("state")
        if publication_state not in {"pending_moderation", "published"}:
            raise WisdomValidationError("Gateway returned an invalid publication state")
        self.store.complete_contribution(draft_id, str(publication_state))
        self.store.consume_receipt(draft_id)
        return {"approved": approved.model_dump(mode="json"), "publication": published}

    def decline(self, draft_id: str) -> dict[str, Any]:
        result = self.client.decline(draft_id)
        local = self.store.draft(draft_id)
        if local:
            self.store.dismiss_candidate(
                str(local["skill_id"]), str(local["source_hash"])
            )
            self.store.set_draft_state(draft_id, "declined")
        self.store.consume_receipt(draft_id)
        return result

    def _record_owner_draft_state(self, draft_id: str, state: str) -> None:
        """Mirror an authoritative owner-draft state into the local ledger."""
        local = self.store.draft(draft_id)
        if not local:
            return
        if state == "declined":
            self.store.dismiss_candidate(
                str(local["skill_id"]), str(local["source_hash"])
            )
            self.store.set_draft_state(draft_id, state)
        elif state in {
            "owner_approved",
            "publishing",
            "pending_moderation",
            "changes_requested",
            "published",
            "invalidated",
        }:
            self.store.complete_contribution(draft_id, state)
        else:
            self.store.set_draft_state(draft_id, state)

    def _owner_draft_state(self, draft_id: str) -> dict[str, Any]:
        draft = self.client.draft(draft_id).draft
        self._record_owner_draft_state(draft_id, draft.state)
        return draft.model_dump(mode="json")

    def _resume_owner_publication(
        self, draft_id: str, draft: dict[str, Any]
    ) -> dict[str, Any]:
        receipt, authoritative = self._validated_review_receipt(draft_id)
        published = self.client.publish(
            draft_id, content_hash=str(receipt["content_hash"])
        )
        state = str(published.get("state") or "")
        if state not in {"pending_moderation", "published"}:
            raise WisdomValidationError("Gateway returned an invalid publication state")
        self._record_owner_draft_state(draft_id, state)
        self.store.consume_receipt(draft_id)
        return {
            "draft": authoritative,
            "publication": published,
            "publication_state": state,
        }

    def approve_owner_draft(self, draft_id: str) -> dict[str, Any]:
        """Approve or reconcile an owner draft across Portal/Telegram races."""
        self.require_setup()
        draft = self._owner_draft_state(draft_id)
        state = str(draft["state"])
        if state in {
            "pending_moderation",
            "changes_requested",
            "published",
            "declined",
            "invalidated",
        }:
            return {
                "draft": draft,
                "publication_state": state,
                "already_advanced": True,
            }
        if state in {"owner_approved", "publishing"}:
            return self._resume_owner_publication(draft_id, draft)
        if state != "ready":
            raise WisdomConflict(
                f"this private draft is currently {state.replace('_', ' ')}",
                code=f"state_is_{state}",
            )

        try:
            self.review(draft_id, acknowledge=True)
            result = self.approve(draft_id)
        except (WisdomConflict, PackagePolicyError):
            # A Portal action may win after the first read or after the review
            # receipt is created. Resolve the committed state instead of
            # treating a repeated explicit confirmation as a generic failure.
            refreshed = self._owner_draft_state(draft_id)
            refreshed_state = str(refreshed["state"])
            if refreshed_state in {"owner_approved", "publishing"}:
                return self._resume_owner_publication(draft_id, refreshed)
            if refreshed_state in {
                "pending_moderation",
                "changes_requested",
                "published",
                "declined",
                "invalidated",
            }:
                return {
                    "draft": refreshed,
                    "publication_state": refreshed_state,
                    "already_advanced": True,
                }
            raise
        publication = result.get("publication")
        publication = publication if isinstance(publication, dict) else {}
        return {
            "draft": draft,
            **result,
            "publication_state": str(publication.get("state") or "published"),
        }

    def decline_owner_draft(self, draft_id: str) -> dict[str, Any]:
        """Decline or reconcile an owner draft without stale-button errors."""
        self.require_setup()
        draft = self._owner_draft_state(draft_id)
        state = str(draft["state"])
        if state in {"published", "declined"}:
            return {"draft": draft, "state": state, "already_advanced": True}
        try:
            self.decline(draft_id)
        except WisdomConflict:
            refreshed = self._owner_draft_state(draft_id)
            refreshed_state = str(refreshed["state"])
            if refreshed_state in {"published", "declined"}:
                return {
                    "draft": refreshed,
                    "state": refreshed_state,
                    "already_advanced": True,
                }
            raise
        return {
            "draft": {**draft, "state": "declined"},
            "state": "declined",
            "withdrawn": state == "pending_moderation",
        }

    def list_skills(self) -> dict[str, Any]:
        response = self.client.list_skills()
        return response.model_dump(mode="json")

    def command_home(self) -> dict[str, Any]:
        """Return the compact, profile-scoped state used by messaging UIs."""
        status = self.status()
        if (
            not status["configured"]
            or not status["gateway_available"]
            or not status["capability_advertised"]
            or not status["entitled"]
            or not status["dogfood_admin_claim"]
        ):
            return {"status": status}
        self.require_setup()
        skills = self.search_skills()
        drafts = self.list_owner_drafts()
        candidates = self.list_candidates(qualified_only=True)
        installations = self.list_installations()
        notifications = self.notifications(mark_seen=False).get("events", [])
        return {
            "status": status,
            "organization_id": self.store.active_org_id(),
            "counts": {
                "published": len(skills),
                "suggested": len(candidates),
                "drafts": len(drafts),
                "installed": len([
                    item for item in installations if item.get("state") == "active"
                ]),
                "notifications": len(notifications),
            },
        }

    def list_owner_drafts(self) -> list[dict[str, Any]]:
        """List the current user's authoritative owner-private drafts."""
        self.require_setup()
        return [item.model_dump(mode="json") for item in self.client.list_drafts()]

    def list_installations(self) -> list[dict[str, Any]]:
        """List local managed installs joined to authoritative update state."""
        self.require_setup()
        installation_id = self.store.installation_identity()
        remote = {
            str(item["skill_id"]): item
            for item in self.client.installations(installation_id)
        }
        results: list[dict[str, Any]] = []
        for item in self.store.installations():
            authoritative = remote.get(str(item["skill_id"])) or {}
            results.append({
                **item,
                "latest_version": authoritative.get("latest_version"),
                "effective_update_mode": authoritative.get("update_mode")
                or item.get("update_mode"),
                "skill_state": authoritative.get("skill_state"),
            })
        return results

    def list_candidates(
        self,
        *,
        qualified_only: bool = True,
        query: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return qualified suggestions, or all manually eligible local skills."""
        self.require_setup()
        candidates = self.scan_candidates()
        if qualified_only:
            candidates = [
                item
                for item in candidates
                if item.get("qualification") in {"high_usage", "refinement"}
            ]
        needle = str(query or "").strip().casefold()
        if needle:
            candidates = [
                item
                for item in candidates
                if needle in str(item.get("name") or "").casefold()
            ]
        return candidates

    def search_skills(self, query: str | None = None) -> list[dict[str, Any]]:
        """Search authoritative discovery pages without inventing client authority."""
        self.require_setup()
        skills: list[dict[str, Any]] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        for _page in range(100):
            response = self.client.list_skills(cursor=cursor)
            skills.extend(item.model_dump(mode="json") for item in response.skills)
            cursor = response.next_cursor
            if not cursor:
                break
            if cursor in seen_cursors:
                raise WisdomValidationError(
                    "Gateway returned a repeated Wisdom discovery cursor"
                )
            seen_cursors.add(cursor)
        else:
            raise WisdomValidationError("Wisdom discovery exceeded the page limit")
        needle = str(query or "").strip().casefold()
        if not needle:
            return skills
        return [
            item
            for item in skills
            if any(
                needle in str(item.get(field) or "").casefold()
                for field in (
                    "slug",
                    "author_description",
                    "created_by_user_id",
                    "created_by_display_name",
                    "author_name",
                    "author",
                )
            )
        ]

    def resolve_skill(
        self,
        reference: str,
        *,
        include_compatibility: bool = True,
    ) -> dict[str, Any]:
        """Resolve an opaque ID or exact slug, keeping misses opaque."""
        raw, _version = self._resolve_install_ref(reference)
        try:
            return self.show(raw, include_compatibility=include_compatibility)
        except WisdomNotFound:
            matches = [
                item for item in self.search_skills(raw) if item.get("slug") == raw
            ]
            if len(matches) != 1:
                raise WisdomNotFound("Wisdom skill not found")
            return self.show(
                str(matches[0]["id"]),
                include_compatibility=include_compatibility,
            )

    def prepare_local_submission(self, skill_name: str) -> dict[str, Any]:
        """Prepare and submit one manually selected local skill as owner-private."""
        self.require_setup()
        candidates = [
            item
            for item in self.list_candidates(qualified_only=False, query=skill_name)
            if item.get("name") == skill_name
        ]
        if len(candidates) != 1:
            raise WisdomNotFound("local skill not found")
        candidate = candidates[0]
        if candidate.get("eligibility") != "eligible":
            raise PackagePolicyError(
                str(candidate.get("reason") or "skill needs an instruction-only fork")
            )
        prepared = self.suggest(
            skill_name,
            local_skill_id=str(candidate["local_skill_id"]),
        )
        submitted = self.suggest(
            skill_name,
            description=str(prepared["drafted_description"]),
            system_specification=dict(prepared["system_specification"]),
            local_skill_id=str(candidate["local_skill_id"]),
        )
        draft = submitted.get("draft")
        if not isinstance(draft, dict) or not isinstance(draft.get("id"), str):
            raise WisdomValidationError("Gateway did not return an owner-private draft")
        return {
            "draft": draft,
            "portal_url": self.portal_review_url(str(draft["id"])),
            "notice": submitted.get("notice"),
        }

    def show(
        self,
        skill_id: str,
        *,
        include_compatibility: bool = True,
    ) -> dict[str, Any]:
        detail = self.client.skill(skill_id)
        result = detail.model_dump(mode="json")
        versions = [
            int(item["version"])
            for item in detail.versions
            if isinstance(item.get("version"), int) and int(item["version"]) > 0
        ]
        if versions:
            latest = self.client.version(skill_id, max(versions))
            result["latest_version_detail"] = latest.model_dump(mode="json")
            specification = latest.version.get("system_spec")
            if include_compatibility and isinstance(specification, dict):
                parsed_specification = SystemSpecification.model_validate(specification)
                result["local_compatibility"] = asdict(
                    evaluate(
                        parsed_specification,
                        detect_local_capabilities(parsed_specification),
                    )
                )
        if include_compatibility:
            result["local_installation"] = self.store.installation(skill_id)
        portal_url = self.portal_skill_url(skill_id)
        if portal_url:
            result["portal_url"] = portal_url
        return result

    def versions(self, skill_id: str) -> list[dict[str, Any]]:
        return self.client.skill(skill_id).versions

    def version_detail(
        self,
        reference: str,
        version: int,
        *,
        include_compatibility: bool = True,
    ) -> dict[str, Any]:
        """Resolve a skill reference and return one immutable version's metadata."""
        if version < 1:
            raise WisdomValidationError("Wisdom version must be a positive integer")
        raw, _selected_version = self._resolve_install_ref(reference)
        try:
            detail = self.client.version(raw, version)
        except WisdomNotFound:
            matches = [
                item for item in self.search_skills(raw) if item.get("slug") == raw
            ]
            if len(matches) != 1:
                raise WisdomNotFound("Wisdom skill not found")
            detail = self.client.version(str(matches[0]["id"]), version)
        result = detail.model_dump(mode="json")
        skill = result.get("skill") or {}
        skill_id = str(skill.get("id") or raw)
        specification = (result.get("version") or {}).get("system_spec")
        if include_compatibility and isinstance(specification, dict):
            parsed_specification = SystemSpecification.model_validate(specification)
            result["local_compatibility"] = asdict(
                evaluate(
                    parsed_specification,
                    detect_local_capabilities(parsed_specification),
                )
            )
        if include_compatibility:
            result["local_installation"] = self.store.installation(skill_id)
        portal_url = self.portal_skill_url(skill_id, version=version)
        if portal_url:
            result["portal_url"] = portal_url
        return result

    def _content_authority(self) -> str:
        installation_id = self.store.existing_installation_identity()
        if not installation_id:
            raise PackagePolicyError(
                "run `hermes wisdom setup` before downloading managed content"
            )
        return installation_id

    def version_content(self, skill_id: str, version: int) -> dict[str, Any]:
        detail = self.client.skill(skill_id)
        generation = int(detail.skill.get("takedown_generation", -1))
        if generation < 0:
            raise WisdomValidationError(
                "Gateway omitted the content authorization generation"
            )
        response, files = self.client.content(
            skill_id,
            version,
            installation_id=self._content_authority(),
            takedown_generation=generation,
        )
        return {
            "commit": response.commit,
            "content_hash": response.content_hash,
            "files": [
                {
                    "path": path,
                    "mode": mode,
                    "hash": sha256_address(body),
                    "content_utf8": body.decode("utf-8", errors="replace"),
                }
                for path, mode, body in files
            ],
        }

    def _resolve_install_ref(self, reference: str) -> tuple[str, int | None]:
        parsed = urlparse(reference)
        raw = (
            parsed.path.rstrip("/").split("/")[-1]
            if parsed.scheme in {"http", "https"}
            else reference
        )
        if parsed.scheme in {"http", "https"}:
            selected = parse_qs(parsed.query).get("version", [])
            if selected:
                raw_version = selected[-1]
                if not raw_version.isdigit() or int(raw_version) < 1:
                    raise PackagePolicyError("invalid Wisdom version selector")
                return raw, int(raw_version)
        if "@v" in raw:
            skill_id, raw_version = raw.rsplit("@v", 1)
            if not raw_version.isdigit() or int(raw_version) < 1:
                raise PackagePolicyError("invalid Wisdom version selector")
            return skill_id, int(raw_version)
        return raw, None

    def install_plan(
        self, reference: str, *, update_mode: str | None = None
    ) -> dict[str, Any]:
        skill_id, selected_version = self._resolve_install_ref(reference)
        existing = self.store.installation(skill_id)
        if existing and existing["state"] == "active":
            raise PackagePolicyError(
                "skill is already managed; use `hermes wisdom update`"
            )
        detail = self.client.skill(skill_id)
        if detail.skill.get("state") != "active":
            raise PackagePolicyError("only active Wisdom skills can be installed")
        versions = detail.versions
        if not versions:
            raise PackagePolicyError("Wisdom skill has no published versions")
        version_number = selected_version or max(
            int(item["version"]) for item in versions
        )
        version_detail = self.client.version(skill_id, version_number)
        takedown_generation = int(detail.skill.get("takedown_generation", -1))
        if takedown_generation < 0:
            raise WisdomValidationError(
                "Gateway omitted the install authorization generation"
            )
        content, files = self.client.content(
            skill_id,
            version_number,
            installation_id=self._content_authority(),
            takedown_generation=takedown_generation,
        )
        manifest_body = next(
            (body for path, _mode, body in files if path == "skill.manifest.json"),
            None,
        )
        if manifest_body is None:
            raise WisdomValidationError("version content has no package manifest")
        manifest = _parse_package_manifest(manifest_body)
        declared_specification = version_detail.version.get("system_spec")
        if (
            not isinstance(declared_specification, dict)
            or SystemSpecification.model_validate(declared_specification)
            != manifest.requirements
        ):
            raise WisdomValidationError(
                "version metadata does not match the exact package manifest"
            )
        spec = manifest.requirements
        compatibility = evaluate(spec, detect_local_capabilities(spec))
        if compatibility.outcome == "blocked_pending_action":
            allowed = False
        else:
            allowed = True
        receipt = "wip_" + uuid.uuid4().hex
        plan = {
            "receipt": receipt,
            "skill_id": skill_id,
            "slug": str(detail.skill.get("slug") or skill_id),
            "version": version_number,
            "content_hash": content.content_hash,
            "manifest_hash": sha256_address(manifest_body),
            "takedown_generation": takedown_generation,
            "update_mode": update_mode,
            "compatibility": asdict(compatibility),
            "allowed": allowed,
        }
        plan_dir = self.store.root / "plans"
        plan_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        plan_path = plan_dir / f"{receipt}.json"
        plan_path.write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")
        plan_path.chmod(0o600)
        return plan

    def install_apply(
        self, receipt: str, *, accept_partial: bool = False
    ) -> dict[str, Any]:
        plan_path = self.store.root / "plans" / f"{receipt}.json"
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            skill_id = str(plan["skill_id"])
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            raise PackagePolicyError(
                "install plan receipt is missing or invalid"
            ) from exc
        lock = self.store.acquire_operation_lock(skill_id)
        if not lock:
            raise PackagePolicyError(
                "another managed operation is already active for this skill"
            )
        try:
            return self._install_apply_unlocked(
                plan, plan_path=plan_path, accept_partial=accept_partial
            )
        finally:
            self.store.release_operation_lock(skill_id, lock)

    def _install_apply_unlocked(
        self,
        plan: dict[str, Any],
        *,
        plan_path: Path,
        accept_partial: bool,
    ) -> dict[str, Any]:
        pending = next(
            (
                item
                for item in self.store.pending_operations()
                if item["kind"] == "install" and item["entity_id"] == plan["skill_id"]
            ),
            None,
        )
        if pending:
            return self._resume_install(
                str(pending["id"]),
                json.loads(pending["payload_json"]),
                plan_path=plan_path,
            )
        current_skill = self.client.skill(str(plan["skill_id"]))
        if current_skill.skill.get("state") != "active":
            raise PackagePolicyError("only active Wisdom skills can be installed")
        if int(current_skill.skill.get("takedown_generation", -1)) != int(
            plan["takedown_generation"]
        ):
            raise PackagePolicyError(
                "install authorization changed after planning; create a new plan"
            )
        response, files = self.client.content(
            plan["skill_id"],
            int(plan["version"]),
            installation_id=self._content_authority(),
            takedown_generation=int(plan["takedown_generation"]),
        )
        exact_records, exact_hash = verify_content_files(files)
        if exact_hash != response.content_hash or exact_hash != str(
            plan.get("content_hash")
        ):
            raise WisdomValidationError("download changed after install planning")
        manifest_body = next(
            (body for path, _mode, body in files if path == "skill.manifest.json"),
            None,
        )
        if manifest_body is None or sha256_address(manifest_body) != plan.get(
            "manifest_hash"
        ):
            raise WisdomValidationError(
                "package manifest changed after install planning"
            )
        manifest = _parse_package_manifest(manifest_body)
        compatibility = evaluate(
            manifest.requirements,
            detect_local_capabilities(manifest.requirements),
        )
        plan["compatibility"] = asdict(compatibility)
        plan["allowed"] = compatibility.outcome != "blocked_pending_action"
        if compatibility.outcome == "blocked_pending_action":
            raise PackagePolicyError(
                "blocked compatibility requirements prevent activation"
            )
        if (
            compatibility.outcome in {"partial", "compatible_after_setup"}
            and not accept_partial
        ):
            raise PackagePolicyError(
                "compatibility changed after planning; explicit acceptance is required"
            )
        org_id = self.store.active_org_id()
        if not org_id:
            raise PackagePolicyError("run `hermes wisdom setup` before installing")
        managed_root = (
            get_skills_dir() / "_wisdom" / org_directory_name(org_id)
        ).resolve()
        managed_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        target = (managed_root / plan["slug"]).resolve()
        try:
            target.relative_to(managed_root)
        except ValueError as exc:
            raise PackagePolicyError(
                "managed install target escaped the Wisdom root"
            ) from exc
        if target.exists():
            raise PackagePolicyError(
                "managed target already exists without an active ledger entry"
            )
        staging = Path(tempfile.mkdtemp(prefix=f".{plan['slug']}-", dir=managed_root))
        try:
            for raw_path, mode, body in files:
                destination = staging.joinpath(*PurePosixPath(raw_path).parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(body)
                destination.chmod(0o600)
            local_scan = _scan_summary(staging)
            if local_scan["guard"]["allowed"] is False:
                raise PackagePolicyError(
                    f"built-in guard blocked installation: {local_scan['guard']['reason']}"
                )
            baseline = {record.path: record.hash for record in exact_records}
            plan.update({
                "org_id": org_id,
                "staging_path": str(staging),
                "target_path": str(target),
                "content_hash": exact_hash,
                "baseline": baseline,
                "local_scan": local_scan,
            })
            operation = self.store.journal("install", plan["skill_id"], "staged", plan)
            return self._resume_install(operation, plan, plan_path=plan_path)
        except BaseException:
            if staging.exists() and not any(
                item["kind"] == "install" and item["entity_id"] == plan["skill_id"]
                for item in self.store.pending_operations()
            ):
                shutil.rmtree(staging)
            raise

    def _resume_install(
        self, operation_id: str, plan: dict[str, Any], *, plan_path: Path
    ) -> dict[str, Any]:
        operation = self.store.operation(operation_id)
        if not operation:
            raise PackagePolicyError("install recovery journal is missing")
        phase = str(operation["phase"])
        staging = Path(str(plan["staging_path"]))
        target = Path(str(plan["target_path"]))
        if phase == "staged":
            if staging.exists():
                staged_baseline, staged_hash = _verified_tree(staging)
                if (
                    staged_hash != plan["content_hash"]
                    or staged_baseline != plan["baseline"]
                ):
                    raise WisdomValidationError(
                        "staged install bytes changed after validation"
                    )
                os.replace(staging, target)
            elif not target.exists():
                raise PackagePolicyError("staged install bytes are unavailable")
            else:
                target_baseline, target_hash = _verified_tree(target)
                if (
                    target_hash != plan["content_hash"]
                    or target_baseline != plan["baseline"]
                ):
                    raise WisdomValidationError(
                        "committed install bytes do not match the recovery journal"
                    )
            self.store.advance(operation_id, "files_committed")
            phase = "files_committed"
        if phase == "files_committed":
            self.store.record_install({
                "skill_id": plan["skill_id"],
                "org_id": plan["org_id"],
                "slug": plan["slug"],
                "version": plan["version"],
                "content_hash": plan["content_hash"],
                "baseline": plan["baseline"],
                "target_path": str(target),
                "update_mode": plan.get("update_mode") or "MANUAL",
            })
            self.store.advance(operation_id, "local_ledger_committed")
            phase = "local_ledger_committed"
        if phase == "local_ledger_committed":
            try:
                server = self.client.record_install(
                    skill_id=plan["skill_id"],
                    installation_id=self.store.installation_identity(),
                    version=int(plan["version"]),
                    takedown_generation=int(plan["takedown_generation"]),
                    update_mode=plan.get("update_mode"),
                )
            except (WisdomConflict, WisdomNotFound, WisdomValidationError):
                quarantine = (
                    self.store.root / "recovery" / operation_id / "gateway-rejected"
                )
                quarantine.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                if target.exists() and not quarantine.exists():
                    os.replace(target, quarantine)
                self.store.deactivate_install(str(plan["skill_id"]))
                self.store.advance(operation_id, "gateway_rejected", done=True)
                raise
            self.store.record_install({
                "skill_id": plan["skill_id"],
                "org_id": plan["org_id"],
                "slug": plan["slug"],
                "version": plan["version"],
                "content_hash": plan["content_hash"],
                "baseline": plan["baseline"],
                "target_path": str(target),
                "update_mode": server.effective_update_mode,
            })
            self.store.advance(operation_id, "gateway_recorded", done=True)
            plan_path.unlink(missing_ok=True)
            return {
                "installed": True,
                "skill_id": plan["skill_id"],
                "version": plan["version"],
                "path": str(target),
                "content_hash": plan["content_hash"],
                "effective_update_mode": server.effective_update_mode,
                "compatibility": plan["compatibility"],
                "local_scan": plan["local_scan"],
            }
        raise PackagePolicyError(f"unsupported install recovery phase: {phase}")

    def check(self, *, apply_automatic: bool = False) -> dict[str, Any]:
        return self.consumption.check(apply_automatic=apply_automatic)

    def update_plan(self, skill_id: str) -> dict[str, Any]:
        return self.consumption.update_plan(skill_id)

    def update_apply(
        self,
        receipt: str,
        *,
        accept_sensitive: bool = False,
        accept_partial: bool = False,
        preserve_modified: bool = False,
    ) -> dict[str, Any]:
        return self.consumption.update_apply(
            receipt,
            accept_sensitive=accept_sensitive,
            accept_partial=accept_partial,
            preserve_modified=preserve_modified,
        )

    def update_all(self, *, apply: bool = False) -> dict[str, Any]:
        checked = self.consumption.check(apply_automatic=False)
        if not apply:
            return checked
        applied: list[dict[str, Any]] = []
        pending: list[dict[str, Any]] = []
        for item in checked["installations"]:
            plan = item.get("plan")
            if item.get("state") != "update_available" or not isinstance(plan, dict):
                continue
            try:
                applied.append(
                    self.consumption.update_apply(str(plan["receipt"]), automatic=False)
                )
            except PackagePolicyError as exc:
                pending.append({"skill_id": item["skill_id"], "reason": str(exc)})
        checked["applied"] = applied
        checked["pending_action"] = pending
        return checked

    def uninstall(self, skill_id: str) -> dict[str, Any]:
        return self.consumption.uninstall(skill_id)

    def notifications(self, *, mark_seen: bool = False) -> dict[str, Any]:
        return self.consumption.notifications(mark_seen=mark_seen)
