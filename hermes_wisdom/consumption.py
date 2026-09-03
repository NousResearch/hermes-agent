"""Crash-safe managed updates, feed reconciliation, and local notifications."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import uuid
from html import escape
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable
from urllib.parse import quote, urlencode

from hermes_constants import get_skills_dir

from .client import (
    WisdomClient,
    WisdomConflict,
    WisdomNotFound,
    WisdomValidationError,
)
from .compatibility import detect_local_capabilities, evaluate
from .contract import (
    PackageManifest,
    SystemSpecification,
    org_directory_name,
    parse_manifest_bytes,
    sha256_address,
)
from .package import PackagePolicyError, verify_content_files
from .qualification import process_due_stability_jobs
from .store import WisdomStore
from .review_presentation import aggregate_review_text, full_review_text


UPDATE_MODES = {"MANUAL", "AUTO_WITH_NOTICE", "REQUIRED"}
logger = logging.getLogger(__name__)


def _tree_hashes(root: Path) -> dict[str, str]:
    if not root.is_dir() or root.is_symlink():
        raise PackagePolicyError("managed install target is missing or unsafe")
    tree: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise PackagePolicyError("managed install contains a symbolic link")
        if path.is_file():
            tree[path.relative_to(root).as_posix()] = sha256_address(path.read_bytes())
    return tree


def _manifest(root: Path) -> PackageManifest:
    try:
        return parse_manifest_bytes((root / "skill.manifest.json").read_bytes())
    except (OSError, ValueError) as exc:
        raise PackagePolicyError(
            "managed package manifest is missing or invalid"
        ) from exc


def _sensitive_expansion(
    previous: SystemSpecification, incoming: SystemSpecification
) -> list[str]:
    reasons: list[str] = []

    def added(label: str, old: list[str], new: list[str]) -> None:
        values = sorted(set(new) - set(old))
        if values:
            reasons.append(f"new {label}: {', '.join(values)}")

    added("credentials", previous.credentials, incoming.credentials)
    added("connections", previous.connections, incoming.connections)
    added("filesystem reads", previous.filesystem.read, incoming.filesystem.read)
    added("filesystem writes", previous.filesystem.write, incoming.filesystem.write)
    added(
        "network destinations",
        previous.network.destinations,
        incoming.network.destinations,
    )
    added("hardware requirements", previous.hardware, incoming.hardware)
    old_tools = {item.name: item for item in previous.tools}
    for tool in incoming.tools:
        before = old_tools.get(tool.name)
        if tool.requires_admin and (before is None or not before.requires_admin):
            reasons.append(f"new privileged tool requirement: {tool.name}")
    for field in ("shell", "browser", "code"):
        if getattr(incoming.runtime, field) and not getattr(previous.runtime, field):
            reasons.append(f"new {field} permission")
    if previous.runtime.sandbox and not incoming.runtime.sandbox:
        reasons.append("sandbox no longer required")
    return reasons


def _public_notification_safe(event: dict[str, Any]) -> bool:
    return event["category"] == "new_skill" or (
        event["category"] == "publication_decision"
        and event.get("state") in {"published", "approved"}
    )


def _safe_target(store: WisdomStore, installation: dict[str, Any]) -> Path:
    org_id = store.active_org_id()
    if not org_id or installation["org_id"] != org_id:
        raise PackagePolicyError("managed installation belongs to another organization")
    managed_root = (get_skills_dir() / "_wisdom" / org_directory_name(org_id)).resolve()
    target = Path(str(installation["target_path"])).resolve()
    try:
        target.relative_to(managed_root)
    except ValueError as exc:
        raise PackagePolicyError(
            "managed target escaped the active Wisdom root"
        ) from exc
    if target.name != installation["slug"]:
        raise PackagePolicyError("managed target no longer matches its ledger identity")
    return target


def _unique_fork_path(slug: str) -> Path:
    root = get_skills_dir().resolve()
    if Path(slug).name != slug:
        raise PackagePolicyError("managed skill slug cannot escape the skills root")
    for suffix in ["local-fork", *[f"local-fork-{index}" for index in range(2, 1000)]]:
        candidate = (root / f"{slug}-{suffix}").resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise PackagePolicyError(
                "managed skill fork escaped the skills root"
            ) from exc
        if not candidate.exists():
            return candidate
    raise PackagePolicyError("could not allocate a unique unmanaged fork name")


class WisdomConsumption:
    def __init__(
        self,
        *,
        store: WisdomStore,
        client: WisdomClient,
        scan: Callable[[Path], dict[str, Any]],
        config: dict[str, Any],
    ) -> None:
        self.store = store
        self.client = client
        self.scan = scan
        self.config = config

    def _portal_url(
        self,
        *,
        skill_id: str | None = None,
        version: int | None = None,
        draft_id: str | None = None,
    ) -> str | None:
        org_id = self.store.active_org_id()
        if not org_id:
            return None
        org_slug = org_id.split(":", 1)[-1]
        base = str(
            self.config.get("portal_url") or "https://portal.nousresearch.com"
        ).rstrip("/")
        if draft_id:
            return (
                f"{base}/orgs/{quote(org_slug, safe='')}/wisdom/review/"
                f"{quote(draft_id, safe='')}"
            )
        if not skill_id:
            return None
        url = (
            f"{base}/orgs/{quote(org_slug, safe='')}/wisdom/skills/"
            f"{quote(skill_id, safe='')}"
        )
        return f"{url}?{urlencode({'version': version})}" if version else url

    def _notification_catalog(
        self, events: list[dict[str, Any]]
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        by_id: dict[str, dict[str, Any]] = {}
        by_slug: dict[str, dict[str, Any]] = {}
        for installation in self.store.installations():
            item = dict(installation)
            by_id[str(item["skill_id"])] = item
            slug = item.get("slug")
            if isinstance(slug, str) and slug:
                by_slug[slug] = item

        unresolved = {
            str(event["skill_id"])
            for event in events
            if str(event["skill_id"]) not in by_id
        }
        if not unresolved:
            return by_id, by_slug

        cursor: str | None = None
        try:
            for _page in range(20):
                response = self.client.list_skills(cursor=cursor)
                for skill in response.skills:
                    item = skill.model_dump(mode="json")
                    by_id[skill.id] = item
                    by_slug[skill.slug] = item
                cursor = response.next_cursor
                if not cursor or not (unresolved - by_id.keys()):
                    break
        except Exception:
            # Notification rendering is an enhancement. A temporary discovery
            # failure must not break update checks or expose opaque identifiers.
            logger.debug("Wisdom notification enrichment failed", exc_info=True)
        return by_id, by_slug

    def _notification_projection(
        self, events: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[str]]:
        by_id, by_slug = self._notification_catalog(events)
        installed_ids = {
            str(item["skill_id"])
            for item in self.store.installations()
            if str(item.get("state") or "active") == "active"
        }
        local_installation_id = self.store.existing_installation_identity()
        installation_updates = {
            (str(item["skill_id"]), item.get("version"))
            for item in events
            if str(item.get("kind")) == "installation_updated"
            and local_installation_id
            and str((item.get("payload") or {}).get("installation_id") or "")
            == local_installation_id
        }
        grouped: dict[tuple[str, str, int | None, str], dict[str, Any]] = {}
        ignored: list[str] = []

        for event in events:
            event_id = str(event["event_id"])
            payload = event.get("payload")
            payload = payload if isinstance(payload, dict) else {}
            kind = str(event.get("kind") or "")
            skill_id = str(event.get("skill_id") or payload.get("skill_id") or "")
            state = str(payload.get("state") or "")
            version_value = event.get("version") or payload.get("version")
            version = (
                int(version_value)
                if isinstance(version_value, int) and version_value > 0
                else None
            )

            category: str | None = None
            if kind == "owner_decision":
                category = "publication_decision"
            elif kind == "new":
                category = "new_skill"
            elif (
                kind == "installed"
                and local_installation_id
                and str(payload.get("installation_id") or "") == local_installation_id
            ):
                category = "installed"
            elif (
                kind == "installation_updated"
                and local_installation_id
                and str(payload.get("installation_id") or "") == local_installation_id
            ):
                category = "updated"
            elif kind == "updated" and skill_id in installed_ids:
                category = (
                    "updated"
                    if (skill_id, version) in installation_updates
                    else "update_available"
                )
            elif kind in {"archived", "taken_down"} and skill_id in installed_ids:
                category = "unavailable"

            if category is None:
                ignored.append(event_id)
                continue

            slug = str(payload.get("slug") or "")
            catalog_item = by_id.get(skill_id)
            if not catalog_item and slug:
                catalog_item = by_slug.get(slug)
            if catalog_item:
                authoritative_id = str(catalog_item.get("id") or skill_id)
                slug = str(catalog_item.get("slug") or slug)
                if version is None:
                    latest = catalog_item.get("latest_version")
                    version = latest if isinstance(latest, int) and latest > 0 else None
                skill_id = authoritative_id
            elif not slug:
                ignored.append(event_id)
                continue

            draft_id = str(payload.get("draft_id") or "") or None
            use_draft_link = category == "publication_decision" and (
                state != "published" or catalog_item is None
            )
            portal_url = self._portal_url(
                skill_id=skill_id,
                version=version,
                draft_id=draft_id if use_draft_link else None,
            )
            key = (category, skill_id, version, state)
            existing = grouped.get(key)
            if existing:
                existing["source_event_ids"].append(event_id)
                continue
            grouped[key] = {
                "event_id": event_id,
                "source_event_ids": [event_id],
                "category": category,
                "kind": kind,
                "skill_id": skill_id,
                "skill_name": slug,
                "version": version,
                "state": state or None,
                "moderation_note": payload.get("moderation_note"),
                "portal_url": portal_url,
                "occurred_at": event.get("created_at") or payload.get("occurred_at"),
                "security_check": (
                    catalog_item.get("security_check") if catalog_item else None
                ),
                "professionalism_check": (
                    catalog_item.get("professionalism_check") if catalog_item else None
                ),
            }

        return list(grouped.values()), ignored

    @staticmethod
    def _telegram_notification_text(event: dict[str, Any]) -> tuple[str, str]:
        name = str(event["skill_name"])
        version = f" · v{event['version']}" if event.get("version") else ""
        category = str(event["category"])
        state = str(event.get("state") or "")
        if category == "publication_decision":
            if state in {"published", "approved"}:
                return "✅ Your skill was published", f"{name}{version}"
            if state == "changes_requested":
                return "✏️ Your skill needs changes", f"{name}{version}"
            if state in {"declined", "rejected"}:
                return "↩️ Your skill was not published", f"{name}{version}"
            return "📣 Your contribution changed", f"{name}{version}"
        if category == "new_skill":
            return "🆕 New skill from your team", f"{name}{version}"
        if category == "installed":
            return "⬇️ Installed on this device", f"{name}{version}"
        if category == "updated":
            return "✅ Updated on this device", f"{name}{version}"
        if category == "update_available":
            return "⬆️ Update available", f"{name}{version}"
        return "⚠️ Installed skill is unavailable", f"{name}{version}"

    def _remote_installations(self) -> dict[str, dict[str, Any]]:
        identity = self.store.installation_identity()
        return {
            str(item["skill_id"]): item for item in self.client.installations(identity)
        }

    def _content(
        self, skill_id: str, version: int, takedown_generation: int
    ) -> tuple[Any, list[tuple[str, str, bytes]]]:
        installation_id = self.store.existing_installation_identity()
        if not installation_id:
            raise PackagePolicyError(
                "run `hermes wisdom setup` before downloading managed content"
            )
        return self.client.content(
            skill_id,
            version,
            installation_id=installation_id,
            takedown_generation=takedown_generation,
        )

    def check(self, *, apply_automatic: bool = False) -> dict[str, Any]:
        qualification_events = process_due_stability_jobs(store=self.store)
        feed = self.poll_feed()
        decisions = self.poll_owner_decisions()
        remote = self._remote_installations()
        results: list[dict[str, Any]] = []
        for local in self.store.installations():
            if local["state"] != "active":
                continue
            authoritative = remote.get(str(local["skill_id"]))
            if not authoritative:
                results.append({"skill_id": local["skill_id"], "state": "not_recorded"})
                continue
            skill_state = str(authoritative.get("skill_state") or "active")
            if skill_state == "taken_down":
                results.append({
                    "skill_id": local["skill_id"],
                    "state": "taken_down",
                    "local_installation_preserved": True,
                })
                continue
            if skill_state == "archived":
                results.append({"skill_id": local["skill_id"], "state": "archived"})
                continue
            latest = authoritative.get("latest_version")
            if not isinstance(latest, int) or latest <= int(local["version"]):
                results.append({"skill_id": local["skill_id"], "state": "current"})
                continue
            plan = self.update_plan(str(local["skill_id"]), remote=authoritative)
            mode = str(authoritative.get("update_mode") or local["update_mode"])
            should_apply = apply_automatic and mode in {"AUTO_WITH_NOTICE", "REQUIRED"}
            if should_apply and plan["auto_allowed"]:
                result = self.update_apply(plan["receipt"], automatic=True)
                results.append({
                    "skill_id": local["skill_id"],
                    "state": "updated",
                    "result": result,
                })
            else:
                results.append({
                    "skill_id": local["skill_id"],
                    "state": "update_available",
                    "plan": plan,
                })
        telegram = self.dispatch_telegram()
        slack = self.dispatch_slack()
        return {
            "installations": results,
            "qualification_events": qualification_events,
            "feed": feed,
            "owner_decisions": decisions,
            "telegram": telegram,
            "slack": slack,
        }

    def update_plan(
        self, skill_id: str, *, remote: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        lock = self.store.acquire_operation_lock(skill_id)
        if not lock:
            raise PackagePolicyError(
                "another managed operation is already active for this skill"
            )
        try:
            return self._update_plan_unlocked(skill_id, remote=remote)
        finally:
            self.store.release_operation_lock(skill_id, lock)

    def _update_plan_unlocked(
        self, skill_id: str, *, remote: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        installation = self.store.installation(skill_id)
        if not installation or installation["state"] != "active":
            raise PackagePolicyError("managed installation is not active")
        target = _safe_target(self.store, installation)
        authoritative = remote or self._remote_installations().get(skill_id)
        if not authoritative:
            raise PackagePolicyError(
                "Gateway has no active record for this installation"
            )
        if authoritative.get("skill_state") != "active":
            raise PackagePolicyError("inactive Wisdom skills cannot be updated")
        version = authoritative.get("latest_version")
        if not isinstance(version, int) or version <= int(installation["version"]):
            return {
                "skill_id": skill_id,
                "state": "current",
                "installed_version": installation["version"],
            }
        try:
            generation = int(authoritative["takedown_generation"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PackagePolicyError(
                "Gateway returned incomplete update authorization"
            ) from exc
        response, files = self._content(skill_id, version, generation)
        records, content_hash = verify_content_files(files)
        if content_hash != response.content_hash:
            raise WisdomValidationError("update content failed integrity validation")
        baseline = {record.path: record.hash for record in records}
        staging_parent = self.store.root / "update-plans"
        staging_parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        reusable = self._reusable_update_plan(
            staging_parent,
            skill_id=skill_id,
            from_version=int(installation["version"]),
            version=version,
            content_hash=content_hash,
            baseline=baseline,
        )
        if reusable:
            receipt, staging, plan_path = reusable
        else:
            receipt = "wup_" + uuid.uuid4().hex
            plan_path = staging_parent / f"{receipt}.json"
            staging = Path(tempfile.mkdtemp(prefix=f".{receipt}-", dir=staging_parent))
            for raw_path, _mode, body in files:
                destination = staging.joinpath(*PurePosixPath(raw_path).parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(body)
                destination.chmod(0o600)
        scan = self.scan(staging)
        if scan["guard"]["allowed"] is False:
            shutil.rmtree(staging)
            plan_path.unlink(missing_ok=True)
            raise PackagePolicyError(
                f"built-in guard blocked update: {scan['guard']['reason']}"
            )
        incoming = _manifest(staging)
        compatibility = evaluate(
            incoming.requirements,
            detect_local_capabilities(incoming.requirements),
        )
        try:
            previous = _manifest(target)
            sensitive = _sensitive_expansion(
                previous.requirements, incoming.requirements
            )
        except PackagePolicyError:
            sensitive = [
                "existing managed manifest is invalid; requirement expansion cannot be proven safe"
            ]
        if any(
            item.get("secrets_class") and item.get("severity") in {"critical", "high"}
            for item in scan.get("skill_evaluator", {}).get("findings", [])
        ):
            sensitive.append("high-confidence local secret finding")
        modified = _tree_hashes(target) != installation["baseline"]
        mode = str(authoritative.get("update_mode") or installation["update_mode"])
        if mode not in UPDATE_MODES:
            shutil.rmtree(staging)
            raise PackagePolicyError("Gateway returned an unsupported update mode")
        plan = {
            "receipt": receipt,
            "kind": "update",
            "skill_id": skill_id,
            "slug": installation["slug"],
            "from_version": int(installation["version"]),
            "version": version,
            "content_hash": content_hash,
            "baseline": baseline,
            "previous_baseline": installation["baseline"],
            "previous_installation": installation,
            "target_path": str(target),
            "staging_path": str(staging),
            "takedown_generation": generation,
            "update_mode": mode,
            "modified": modified,
            "sensitive_expansion": sensitive,
            "compatibility": asdict(compatibility),
            "local_scan": scan,
            "auto_allowed": (
                not sensitive
                and (not modified or mode == "REQUIRED")
                and compatibility.outcome == "compatible"
                and scan["guard"]["allowed"] is True
                and not scan["guard"].get("findings")
                and not scan.get("skill_evaluator", {}).get("findings")
            ),
        }
        plan_path.write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")
        plan_path.chmod(0o600)
        return plan

    def _reusable_update_plan(
        self,
        root: Path,
        *,
        skill_id: str,
        from_version: int,
        version: int,
        content_hash: str,
        baseline: dict[str, str],
    ) -> tuple[str, Path, Path] | None:
        """Reuse one exact, unapplied plan and remove superseded idle plans."""

        root = root.resolve()
        pending_receipts: set[str] = set()
        for operation in self.store.pending_operations():
            if operation["kind"] != "update":
                continue
            try:
                payload = json.loads(operation["payload_json"])
            except (TypeError, json.JSONDecodeError):
                continue
            receipt = payload.get("receipt")
            if isinstance(receipt, str):
                pending_receipts.add(receipt)

        reusable: tuple[str, Path, Path] | None = None
        for plan_path in root.glob("wup_*.json"):
            try:
                plan = json.loads(plan_path.read_text(encoding="utf-8"))
                receipt = str(plan["receipt"])
                if plan.get("skill_id") != skill_id:
                    continue
                staging = Path(str(plan["staging_path"])).resolve()
                staging.relative_to(root)
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            try:
                exact = (
                    receipt not in pending_receipts
                    and int(plan.get("from_version", 0)) == from_version
                    and int(plan.get("version", 0)) == version
                    and plan.get("content_hash") == content_hash
                    and plan.get("baseline") == baseline
                )
            except (TypeError, ValueError):
                exact = False
            try:
                exact = exact and _tree_hashes(staging) == baseline
            except PackagePolicyError:
                exact = False
            if exact and reusable is None:
                reusable = (receipt, staging, plan_path)
                continue
            if receipt not in pending_receipts:
                if staging.is_dir() and not staging.is_symlink():
                    shutil.rmtree(staging)
                plan_path.unlink(missing_ok=True)
        return reusable

    def update_apply(
        self,
        receipt: str,
        *,
        accept_sensitive: bool = False,
        accept_partial: bool = False,
        preserve_modified: bool = False,
        automatic: bool = False,
    ) -> dict[str, Any]:
        plan_path = self.store.root / "update-plans" / f"{receipt}.json"
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            skill_id = str(plan["skill_id"])
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            raise PackagePolicyError(
                "update plan receipt is missing or invalid"
            ) from exc
        lock = self.store.acquire_operation_lock(skill_id)
        if not lock:
            raise PackagePolicyError(
                "another managed operation is already active for this skill"
            )
        try:
            return self._update_apply_unlocked(
                receipt,
                accept_sensitive=accept_sensitive,
                accept_partial=accept_partial,
                preserve_modified=preserve_modified,
                automatic=automatic,
            )
        finally:
            self.store.release_operation_lock(skill_id, lock)

    def _update_apply_unlocked(
        self,
        receipt: str,
        *,
        accept_sensitive: bool = False,
        accept_partial: bool = False,
        preserve_modified: bool = False,
        automatic: bool = False,
    ) -> dict[str, Any]:
        plan_path = self.store.root / "update-plans" / f"{receipt}.json"
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PackagePolicyError(
                "update plan receipt is missing or invalid"
            ) from exc
        pending = next(
            (
                item
                for item in self.store.pending_operations()
                if item["kind"] == "update"
                and item["entity_id"] == str(plan.get("skill_id"))
            ),
            None,
        )
        if pending:
            persisted = json.loads(pending["payload_json"])
            return self._resume_update(
                str(pending["id"]),
                persisted,
                preserve=bool(persisted.get("preserve_required")),
            )
        installation = self.store.installation(str(plan["skill_id"]))
        if not installation or installation["state"] != "active":
            raise PackagePolicyError("managed installation is not active")
        target = _safe_target(self.store, installation)
        if int(installation["version"]) != int(plan["from_version"]):
            raise PackagePolicyError("update plan is stale; create a new plan")
        authoritative = self._remote_installations().get(str(plan["skill_id"]))
        if not authoritative or authoritative.get("skill_state") != "active":
            raise PackagePolicyError(
                "Gateway no longer authorizes this managed update; create a new plan"
            )
        try:
            remote_generation = int(authoritative["takedown_generation"])
            remote_installed = int(authoritative["installed_version"])
            remote_latest = int(authoritative["latest_version"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PackagePolicyError(
                "Gateway returned incomplete installation authority"
            ) from exc
        remote_mode = str(authoritative.get("update_mode") or "")
        if (
            remote_generation != int(plan["takedown_generation"])
            or remote_installed != int(plan["from_version"])
            or remote_latest != int(plan["version"])
            or remote_mode != str(plan["update_mode"])
        ):
            raise PackagePolicyError(
                "update authority or policy changed after planning; create a new plan"
            )
        response, remote_files = self._content(
            str(plan["skill_id"]),
            int(plan["version"]),
            remote_generation,
        )
        records, remote_hash = verify_content_files(remote_files)
        remote_baseline = {record.path: record.hash for record in records}
        if (
            remote_hash != response.content_hash
            or remote_hash != str(plan["content_hash"])
            or remote_baseline != plan["baseline"]
        ):
            raise WisdomValidationError(
                "authoritative update bytes changed after planning"
            )
        staging = Path(str(plan["staging_path"]))
        if _tree_hashes(staging) != plan["baseline"]:
            raise WisdomValidationError("staged update bytes changed after planning")
        current_scan = self.scan(staging)
        if current_scan["guard"]["allowed"] is False:
            raise PackagePolicyError(
                f"built-in guard blocked update: {current_scan['guard']['reason']}"
            )
        incoming = _manifest(staging)
        compatibility = evaluate(
            incoming.requirements,
            detect_local_capabilities(incoming.requirements),
        )
        try:
            previous = _manifest(target)
            sensitive = _sensitive_expansion(
                previous.requirements, incoming.requirements
            )
        except PackagePolicyError:
            sensitive = [
                "existing managed manifest is invalid; requirement expansion cannot be proven safe"
            ]
        if any(
            item.get("secrets_class") and item.get("severity") in {"critical", "high"}
            for item in current_scan.get("skill_evaluator", {}).get("findings", [])
        ):
            sensitive.append("high-confidence local secret finding")
        plan["local_scan"] = current_scan
        plan["compatibility"] = asdict(compatibility)
        plan["sensitive_expansion"] = sensitive
        modified_now = _tree_hashes(target) != installation["baseline"]
        plan["modified"] = modified_now
        plan["auto_allowed"] = (
            not sensitive
            and (not modified_now or remote_mode == "REQUIRED")
            and compatibility.outcome == "compatible"
            and current_scan["guard"]["allowed"] is True
            and not current_scan["guard"].get("findings")
            and not current_scan.get("skill_evaluator", {}).get("findings")
        )
        if sensitive and not accept_sensitive:
            raise PackagePolicyError(
                "update adds sensitive requirements; explicit confirmation is required"
            )
        outcome = compatibility.outcome
        if outcome == "blocked_pending_action":
            raise PackagePolicyError(
                "blocked compatibility requirements prevent activation"
            )
        if outcome in {"partial", "compatible_after_setup"} and not accept_partial:
            raise PackagePolicyError(
                "compatibility action is required; explicitly accept the plan"
            )
        mode = str(plan["update_mode"])
        if modified_now and mode != "REQUIRED" and not preserve_modified:
            raise PackagePolicyError(
                "managed files changed locally; preserve an unmanaged fork before updating"
            )
        if automatic and not plan["auto_allowed"]:
            raise PackagePolicyError(
                "automatic update is not safe without explicit action"
            )
        preserve = modified_now and (mode == "REQUIRED" or preserve_modified)
        plan["preserve_required"] = preserve
        if preserve and not plan.get("fork_path"):
            plan["fork_path"] = str(_unique_fork_path(str(plan["slug"])))
        operation = self.store.journal("update", str(plan["skill_id"]), "planned", plan)
        operation_row = self.store.operation(operation)
        if operation_row:
            plan = operation_row["payload"]
        return self._resume_update(
            operation,
            plan,
            preserve=preserve,
        )

    def _resume_update(
        self, operation_id: str, plan: dict[str, Any], *, preserve: bool
    ) -> dict[str, Any]:
        operation = self.store.operation(operation_id)
        if not operation:
            raise PackagePolicyError("update recovery journal is missing")
        phase = str(operation["phase"])
        target = Path(str(plan["target_path"]))
        staging = Path(str(plan["staging_path"]))
        recovery = self.store.root / "recovery" / operation_id
        recovery.mkdir(parents=True, exist_ok=True, mode=0o700)
        if phase == "planned" and preserve:
            exact = recovery / "managed-original"
            if not exact.exists():
                os.replace(target, exact)
            fork = Path(str(plan["fork_path"]))
            if fork.exists():
                if _tree_hashes(fork) != _tree_hashes(exact):
                    raise WisdomValidationError(
                        "preserved unmanaged fork changed during update recovery"
                    )
            else:
                pending_fork = fork.with_name(f".{fork.name}.{operation_id}.pending")
                if pending_fork.exists():
                    shutil.rmtree(pending_fork)
                shutil.copytree(exact, pending_fork, copy_function=shutil.copy2)
                os.replace(pending_fork, fork)
            plan["recovery_path"] = str(exact)
            self.store.replace_operation_payload(operation_id, plan)
            self.store.advance(operation_id, "fork_preserved")
            phase = "fork_preserved"
        if phase == "planned":
            self.store.advance(operation_id, "fork_preserved")
            phase = "fork_preserved"
        if phase == "fork_preserved":
            expected = dict(plan["baseline"])
            already_swapped = not staging.is_dir()
            if staging.is_dir():
                if _tree_hashes(staging) != expected:
                    raise WisdomValidationError(
                        "staged update bytes changed after planning"
                    )
            elif not target.is_dir() or _tree_hashes(target) != expected:
                raise PackagePolicyError(
                    "staged update bytes are unavailable for recovery"
                )
            backup = recovery / "replaced-managed"
            if (
                not preserve
                and not already_swapped
                and target.exists()
                and _tree_hashes(target) != dict(plan["previous_baseline"])
            ):
                raise PackagePolicyError(
                    "managed files changed during update planning; create a fresh plan"
                )
            if (
                not preserve
                and not already_swapped
                and not target.exists()
                and not backup.exists()
            ):
                raise PackagePolicyError(
                    "managed update source disappeared before the atomic swap"
                )
            if not already_swapped and target.exists() and not backup.exists():
                os.replace(target, backup)
            if not already_swapped and not target.exists():
                os.replace(staging, target)
            self.store.advance(operation_id, "files_committed")
            phase = "files_committed"
        if phase == "files_committed":
            installation = self.store.installation(str(plan["skill_id"]))
            if not installation:
                raise PackagePolicyError(
                    "managed installation ledger entry disappeared"
                )
            self.store.record_install({
                "skill_id": plan["skill_id"],
                "org_id": installation["org_id"],
                "slug": plan["slug"],
                "version": plan["version"],
                "content_hash": plan["content_hash"],
                "baseline": plan["baseline"],
                "target_path": str(target),
                "update_mode": plan["update_mode"],
            })
            self.store.advance(operation_id, "local_ledger_committed")
            phase = "local_ledger_committed"
        if phase == "local_ledger_committed":
            try:
                server = self.client.record_install(
                    skill_id=str(plan["skill_id"]),
                    installation_id=self.store.installation_identity(),
                    version=int(plan["version"]),
                    takedown_generation=int(plan["takedown_generation"]),
                    update_mode=str(plan["update_mode"]),
                )
            except (WisdomConflict, WisdomNotFound, WisdomValidationError) as exc:
                rejected = recovery / "gateway-rejected"
                if target.exists() and not rejected.exists():
                    os.replace(target, rejected)
                terminal_codes = {
                    "skill_archived",
                    "skill_taken_down",
                    "takedown_generation_changed",
                    "skill_not_found",
                    "version_not_found",
                }
                if isinstance(exc, WisdomNotFound) or exc.code in terminal_codes:
                    self.store.deactivate_install(str(plan["skill_id"]))
                    self.store.advance(operation_id, "gateway_rejected", done=True)
                    raise
                previous_installation = plan.get("previous_installation")
                previous_tree = (
                    recovery / "managed-original"
                    if preserve
                    else recovery / "replaced-managed"
                )
                if not isinstance(previous_installation, dict):
                    raise PackagePolicyError(
                        "update recovery is missing the previous installation ledger"
                    ) from exc
                if not previous_tree.is_dir() or _tree_hashes(previous_tree) != dict(
                    previous_installation["baseline"]
                ):
                    raise PackagePolicyError(
                        "previous managed bytes are unavailable for rollback"
                    ) from exc
                os.replace(previous_tree, target)
                self.store.record_install(previous_installation)
                self.store.advance(
                    operation_id, "gateway_rejected_rolled_back", done=True
                )
                plan_path = self.store.root / "update-plans" / f"{plan['receipt']}.json"
                plan_path.unlink(missing_ok=True)
                raise
            current = self.store.installation(str(plan["skill_id"]))
            if current:
                self.store.record_install({
                    **current,
                    "baseline": current["baseline"],
                    "update_mode": server.effective_update_mode,
                })
            self.store.advance(operation_id, "gateway_recorded", done=True)
            plan_path = self.store.root / "update-plans" / f"{plan['receipt']}.json"
            plan_path.unlink(missing_ok=True)
            return {
                "updated": True,
                "skill_id": plan["skill_id"],
                "version": plan["version"],
                "path": str(target),
                "preserved_fork": plan.get("fork_path"),
                "recovery_path": plan.get("recovery_path"),
                "effective_update_mode": server.effective_update_mode,
            }
        raise PackagePolicyError(f"unsupported update recovery phase: {phase}")

    def uninstall(self, skill_id: str) -> dict[str, Any]:
        lock = self.store.acquire_operation_lock(skill_id)
        if not lock:
            raise PackagePolicyError(
                "another managed operation is already active for this skill"
            )
        try:
            return self._uninstall_unlocked(skill_id)
        finally:
            self.store.release_operation_lock(skill_id, lock)

    def _uninstall_unlocked(self, skill_id: str) -> dict[str, Any]:
        installation = self.store.installation(skill_id)
        if not installation:
            raise PackagePolicyError("managed installation not found")
        target = _safe_target(self.store, installation)
        payload = {
            "skill_id": skill_id,
            "slug": installation["slug"],
            "target_path": str(target),
        }
        operation = self.store.journal("uninstall", skill_id, "validated", payload)
        return self._resume_uninstall(operation, payload)

    def _resume_uninstall(
        self, operation_id: str, plan: dict[str, Any]
    ) -> dict[str, Any]:
        operation = self.store.operation(operation_id)
        if not operation:
            raise PackagePolicyError("uninstall recovery journal is missing")
        phase = str(operation["phase"])
        target = Path(str(plan["target_path"]))
        slug = str(plan["slug"])
        if Path(operation_id).name != operation_id or Path(slug).name != slug:
            raise PackagePolicyError("uninstall recovery path is invalid")
        trash_root = (self.store.root / "trash").resolve()
        trash = (trash_root / operation_id / slug).resolve()
        try:
            trash.relative_to(trash_root)
        except ValueError as exc:
            raise PackagePolicyError(
                "uninstall recovery path escaped the Wisdom trash root"
            ) from exc
        if phase == "validated":
            trash.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if target.exists() and not trash.exists():
                os.replace(target, trash)
            self.store.advance(operation_id, "files_removed")
            phase = "files_removed"
        if phase == "files_removed":
            self.store.deactivate_install(str(plan["skill_id"]))
            self.store.advance(operation_id, "local_ledger_committed")
            phase = "local_ledger_committed"
        if phase == "local_ledger_committed":
            result = self.client.deactivate_install(
                self.store.installation_identity(), str(plan["skill_id"])
            )
            self.store.advance(operation_id, "gateway_deactivated", done=True)
            return {
                "uninstalled": True,
                "skill_id": plan["skill_id"],
                "state": result.state,
                "recoverable_path": str(trash),
            }
        raise PackagePolicyError(f"unsupported uninstall recovery phase: {phase}")

    def recover(self) -> list[str]:
        recovered: list[str] = []
        for operation in self.store.pending_operations():
            entity_id = str(operation["entity_id"])
            lock = self.store.acquire_operation_lock(entity_id)
            if not lock:
                continue
            try:
                payload = json.loads(operation["payload_json"])
                if operation["kind"] == "update":
                    self._resume_update(
                        operation["id"],
                        payload,
                        preserve=bool(payload.get("preserve_required")),
                    )
                elif operation["kind"] == "uninstall":
                    self._resume_uninstall(operation["id"], payload)
                else:
                    continue
                recovered.append(str(operation["entity_id"]))
            except Exception:
                continue
            finally:
                self.store.release_operation_lock(entity_id, lock)
        return recovered

    def poll_feed(self) -> dict[str, Any]:
        cursor = self.store.feed_cursor()
        inserted = 0
        pages = 0
        installation_id = self.store.installation_identity()
        installed = {str(item["skill_id"]) for item in self.store.installations()}
        while True:
            page = self.client.feed(cursor, installation_id=installation_id)
            events = [item.model_dump(mode="json") for item in page.events]
            notification = self.config.get("notifications") or {}
            cadences: dict[str, str] = {}
            for event in events:
                kind = str(event["kind"])
                if kind == "new":
                    cadence = str(notification.get("new_skills", "daily"))
                elif kind == "updated" and str(event["skill_id"]) not in installed:
                    cadence = str(notification.get("new_skills", "daily"))
                elif kind in {"updated", "installation_updated"}:
                    cadence = str(notification.get("installed_updates", "immediate"))
                else:
                    cadence = str(notification.get("decisions", "immediate"))
                cadences[str(event["event_id"])] = (
                    cadence
                    if cadence in {"immediate", "daily", "weekly", "off"}
                    else "immediate"
                )
            now = datetime.now(timezone.utc).isoformat()
            inserted += self.store.persist_feed_page(
                events, next_cursor=page.next_cursor, cadences=cadences, now=now
            )
            pages += 1
            cursor = page.next_cursor
            if not page.has_more:
                break
            if pages >= 100:
                raise WisdomValidationError(
                    "Wisdom feed exceeded the bounded page limit"
                )
        return {"inserted": inserted, "pages": pages, "cursor": cursor}

    def poll_owner_decisions(self) -> dict[str, Any]:
        inserted = 0
        for draft in self.client.list_drafts():
            local = self.store.draft(draft.id)
            if not local or str(local["state"]) == draft.state:
                continue
            previous = str(local["state"])
            if draft.state == "declined":
                self.store.set_draft_state(draft.id, draft.state)
                self.store.dismiss_candidate(
                    str(local["skill_id"]), str(local["source_hash"])
                )
            elif draft.state in {
                "owner_approved",
                "publishing",
                "pending_moderation",
                "changes_requested",
                "published",
                "invalidated",
            }:
                self.store.complete_contribution(draft.id, draft.state)
            else:
                self.store.set_draft_state(draft.id, draft.state)
            if draft.state not in {
                "approved",
                "rejected",
                "declined",
                "published",
                "changes_requested",
            }:
                continue
            if draft.state == "changes_requested":
                self.store.consume_receipt(draft.id)
            cadence = str(
                (self.config.get("notifications") or {}).get("decisions", "immediate")
            )
            if cadence not in {"immediate", "daily", "weekly", "off"}:
                cadence = "immediate"
            inserted += int(
                self.store.persist_local_notice(
                    event_id=f"draft-decision:{draft.id}:{draft.state}",
                    kind="owner_decision",
                    skill_id=str(local["skill_id"]),
                    payload={
                        "draft_id": draft.id,
                        "slug": draft.slug,
                        "previous_state": previous,
                        "state": draft.state,
                        "moderation_note": draft.moderationNote,
                        "moderation_decider_user_id": draft.moderationDeciderUserId,
                        "moderation_decided_at": draft.moderationDecidedAt,
                    },
                    cadence=cadence,
                )
            )
        return {"inserted": inserted}

    def notifications(self, *, mark_seen: bool = False) -> dict[str, Any]:
        events = self.store.feed_events(unseen_only=True)
        notifications, ignored = self._notification_projection(events)
        self.store.mark_feed_local_seen(ignored)
        if mark_seen:
            source_ids = [
                str(event_id)
                for item in notifications
                for event_id in item["source_event_ids"]
            ]
            self.store.mark_feed_local_seen(source_ids)
        return {"events": notifications}

    def dispatch_telegram(self) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        due = self.store.feed_events(telegram_due_at=now)
        if not due:
            return {"attempted": False, "delivered": 0}
        notifications, ignored = self._notification_projection(due)
        try:
            from gateway.config import Platform, load_gateway_config

            home = load_gateway_config().get_home_channel(Platform.TELEGRAM)
            private_home = bool(home and not str(home.chat_id).startswith("-"))
        except Exception:
            private_home = False
        excluded: list[str] = []
        if not private_home:
            public_notifications: list[dict[str, Any]] = []
            for event in notifications:
                if _public_notification_safe(event):
                    public_notifications.append(event)
                else:
                    excluded.extend(
                        str(event_id) for event_id in event["source_event_ids"]
                    )
            notifications = public_notifications
        selected = notifications[:8]
        if not selected:
            self.store.mark_feed_telegram_delivered([*ignored, *excluded])
            return {"attempted": False, "delivered": 0}
        lines = [
            "<b>Collective Wisdom</b>",
            f"{len(selected)} new {'update' if len(selected) == 1 else 'updates'}",
        ]
        button_rows: list[list[dict[str, str]]] = []
        rich_items: list[dict[str, object]] = []
        for event in selected:
            heading, detail = self._telegram_notification_text(event)
            checks = (
                full_review_text(
                    event.get("security_check"), event.get("professionalism_check")
                )
                if private_home
                else aggregate_review_text(
                    event.get("security_check"), event.get("professionalism_check")
                )
            )
            detail = f"{detail}\n{checks}"
            lines.extend(["", f"<b>{escape(heading)}</b>", escape(detail)])
            rich_items.append({"heading": heading, "detail": detail})
            row: list[dict[str, str]] = []
            portal_url = event.get("portal_url")
            if isinstance(portal_url, str) and portal_url:
                row.append({
                    "label": "View ↗",
                    "url": portal_url,
                })
            if private_home and event["category"] == "new_skill":
                row.append({
                    "label": "Install",
                    "callback_data": f"wi:plan:install:{event['skill_id']}",
                })
            elif private_home and event["category"] == "update_available":
                row.append({
                    "label": "Update",
                    "callback_data": f"wi:plan:update:{event['skill_id']}",
                })
            # Keep one row per notification even when an item has no actions,
            # so rich-message controls cannot drift onto the next skill.
            button_rows.append(row)
        try:
            from tools.send_message_tool import send_telegram_notification_pane

            raw = send_telegram_notification_pane(
                message="\n".join(lines),
                button_rows=button_rows,
                items=rich_items,
            )
            result = json.loads(raw) if isinstance(raw, str) else raw
        except Exception as exc:
            return {"attempted": True, "delivered": 0, "error": str(exc)}
        if not isinstance(result, dict) or not result.get("success"):
            return {
                "attempted": True,
                "delivered": 0,
                "error": str(
                    result.get("error") if isinstance(result, dict) else result
                ),
            }
        ids = [
            str(event_id) for item in selected for event_id in item["source_event_ids"]
        ]
        ids.extend(ignored)
        ids.extend(excluded)
        self.store.mark_feed_telegram_delivered(ids)
        return {"attempted": True, "delivered": len(selected)}

    def dispatch_slack(self) -> dict[str, Any]:
        """Deliver due Wisdom feed items to Slack's configured home chat.

        A public home channel receives only collective publication notices and
        Portal links. Device-local install/update state and mutation controls
        are emitted only when the configured home is a DM.
        """
        try:
            from gateway.config import Platform, load_gateway_config

            config = load_gateway_config()
            platform_config = config.platforms.get(Platform.SLACK)
            home = config.get_home_channel(Platform.SLACK)
            if not platform_config or not platform_config.enabled or not home:
                return {"attempted": False, "delivered": 0}
            private_home = str(home.chat_id).startswith("D")
        except Exception:
            return {"attempted": False, "delivered": 0}

        now = datetime.now(timezone.utc).isoformat()
        due = self.store.feed_events(surface="slack", surface_due_at=now)
        if not due:
            return {"attempted": False, "delivered": 0}
        notifications, ignored = self._notification_projection(due)
        excluded: list[str] = []
        if not private_home:
            public_notifications: list[dict[str, Any]] = []
            for event in notifications:
                if _public_notification_safe(event):
                    public_notifications.append(event)
                else:
                    excluded.extend(
                        str(event_id) for event_id in event["source_event_ids"]
                    )
            notifications = public_notifications
        selected = notifications[:8]
        if not selected:
            self.store.mark_feed_surface_delivered(
                [*ignored, *excluded], surface="slack"
            )
            return {"attempted": False, "delivered": 0}

        lines = [
            "Collective Wisdom",
            f"{len(selected)} new {'update' if len(selected) == 1 else 'updates'}",
        ]
        button_rows: list[list[dict[str, str]]] = []
        items: list[dict[str, object]] = []
        for event in selected:
            heading, detail = self._telegram_notification_text(event)
            checks = (
                full_review_text(
                    event.get("security_check"), event.get("professionalism_check")
                )
                if private_home
                else aggregate_review_text(
                    event.get("security_check"), event.get("professionalism_check")
                )
            )
            detail = f"{detail}\n{checks}"
            lines.extend(["", heading, detail])
            items.append({"heading": heading, "detail": detail})
            row: list[dict[str, str]] = []
            portal_url = event.get("portal_url")
            if isinstance(portal_url, str) and portal_url:
                row.append({"label": "View in Portal ↗", "url": portal_url})
            if private_home and event["category"] == "new_skill":
                row.append({
                    "label": "Install",
                    "callback_data": f"wi:plan:install:{event['skill_id']}",
                })
            elif private_home and event["category"] == "update_available":
                row.append({
                    "label": "Update",
                    "callback_data": f"wi:plan:update:{event['skill_id']}",
                })
            button_rows.append(row)
        try:
            from tools.send_message_tool import send_slack_wisdom_notification_pane

            result = send_slack_wisdom_notification_pane(
                message="\n".join(lines),
                button_rows=button_rows,
                items=items,
            )
        except Exception as exc:
            return {"attempted": True, "delivered": 0, "error": str(exc)}
        if not isinstance(result, dict) or not result.get("success"):
            return {
                "attempted": True,
                "delivered": 0,
                "error": str(
                    result.get("error") if isinstance(result, dict) else result
                ),
            }
        delivered_ids = [
            str(event_id) for item in selected for event_id in item["source_event_ids"]
        ]
        self.store.mark_feed_surface_delivered(
            [*delivered_ids, *ignored, *excluded], surface="slack"
        )
        return {"attempted": True, "delivered": len(selected)}
