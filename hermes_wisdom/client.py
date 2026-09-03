"""Bounded, schema-validated Gateway client for Collective Wisdom."""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal
from urllib.parse import quote

import requests
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from tools.skills_sync_client import (
    KIND_BLOB,
    KIND_COMMIT,
    KIND_TREE,
    ObjectSet,
    SyncClient,
)
from tools.skills_sync_client import (
    resolve_identity,
    resolve_sync_base_url,
    wire_address,
)

from .contract import ContentFile, SystemSpecification, parse_manifest_bytes
from .package import PackagePolicyError, verify_content_files

logger = logging.getLogger(__name__)


class WisdomError(RuntimeError):
    exit_code = 8

    def __init__(
        self, message: str, *, status: int | None = None, code: str | None = None
    ):
        super().__init__(message)
        self.status = status
        self.code = code


class WisdomAuthError(WisdomError):
    exit_code = 3


class WisdomNotFound(WisdomError):
    exit_code = 4


class WisdomConflict(WisdomError):
    exit_code = 5


class WisdomValidationError(WisdomError):
    exit_code = 6


class WireModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class Draft(WireModel):
    id: str
    orgId: str
    ownerUserId: str
    slug: str
    draftCommit: str
    contentHash: str
    authorDescription: str | None
    authorDescriptionHash: str | None
    state: Literal[
        "vetting",
        "ready",
        "owner_approved",
        "publishing",
        "pending_moderation",
        "changes_requested",
        "published",
        "declined",
        "invalidated",
    ]
    moderationNote: str | None = None
    moderationDeciderUserId: str | None = None
    moderationDecidedAt: str | None = None
    packageManifestHash: str | None
    packageManifestSchemaVersion: int | None
    systemSpec: SystemSpecification | None
    scan: dict[str, Any] | None
    scanVerdict: str | None
    security_check: dict[str, Any] | None = None
    professionalism_check: dict[str, Any] | None = None
    explanation: str | None
    updatedAt: str

    @model_validator(mode="after")
    def require_return_metadata(self) -> Draft:
        if self.state == "changes_requested" and (
            not self.moderationNote
            or not self.moderationDeciderUserId
            or not self.moderationDecidedAt
        ):
            raise ValueError(
                "changes_requested drafts require complete moderator return metadata"
            )
        return self


class DraftDetail(WireModel):
    draft: Draft
    effective_policy: dict[str, Any]


class DraftResponse(WireModel):
    draft: Draft


class DraftList(WireModel):
    drafts: list[Draft]


class SkillSummary(WireModel):
    id: str
    slug: str
    state: str
    created_by_user_id: str
    latest_version: int | None
    author_description: str | None
    verified_facts: dict[str, Any] | None
    explanation: str | None
    install_count: int
    takedown_generation: int
    system_spec: SystemSpecification | None
    security_check: dict[str, Any] | None = None
    professionalism_check: dict[str, Any] | None = None


class SkillList(WireModel):
    skills: list[SkillSummary]
    next_cursor: str | None


class SkillDetail(WireModel):
    skill: dict[str, Any]
    versions: list[dict[str, Any]]


class VersionDetail(WireModel):
    skill: dict[str, Any]
    version: dict[str, Any]


class VersionContent(WireModel):
    commit: str
    content_hash: str
    files: list[dict[str, Any]]


class InstallationRecord(WireModel):
    installed_version: int = Field(gt=0)
    effective_update_mode: Literal["MANUAL", "AUTO_WITH_NOTICE", "REQUIRED"]


class InstallationItem(WireModel):
    skill_id: str
    installed_version: int = Field(gt=0)
    latest_version: int | None = Field(default=None, gt=0)
    update_mode: Literal["MANUAL", "AUTO_WITH_NOTICE", "REQUIRED"]
    skill_state: str
    takedown_generation: int | None = Field(default=None, ge=0)


class InstallationList(WireModel):
    installations: list[InstallationItem]


class InstallationDeactivation(WireModel):
    skill_id: str
    installation_id: str
    state: Literal["inactive"]


class FeedEvent(WireModel):
    event_id: str
    kind: Literal[
        "new",
        "updated",
        "archived",
        "taken_down",
        "restored",
        "installed",
        "installation_updated",
        "uninstalled",
    ]
    skill_id: str
    version: int | None = Field(default=None, gt=0)
    takedown_generation: int | None = Field(default=None, ge=0)
    installation_id: str | None
    update_mode: Literal["MANUAL", "AUTO_WITH_NOTICE", "REQUIRED"] | None
    occurred_at: datetime


class Feed(WireModel):
    events: list[FeedEvent]
    next_cursor: str
    has_more: bool


@dataclass(frozen=True)
class ReconstructedDraft:
    detail: DraftDetail
    files: list[tuple[str, str, bytes]]
    content_files: list[ContentFile]
    content_hash: str


def _walk_private_commit(
    sync: SyncClient, commit_hash: str
) -> list[tuple[str, str, bytes]]:
    kind, body = sync.get_object(commit_hash)
    if wire_address(body) != commit_hash or kind != KIND_COMMIT:
        raise WisdomValidationError(
            "owner-private draft commit failed integrity validation"
        )
    try:
        commit = json.loads(body.decode("utf-8"))
        tree_hash = str(commit["tree"])
    except (
        KeyError,
        TypeError,
        ValueError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise WisdomValidationError("owner-private draft commit is malformed") from exc
    out: list[tuple[str, str, bytes]] = []
    seen_nodes: set[tuple[str, str]] = set()

    def walk(tree: str, prefix: str) -> None:
        node_key = (tree, prefix)
        if node_key in seen_nodes:
            raise WisdomValidationError("owner-private draft contains a tree cycle")
        seen_nodes.add(node_key)
        node_kind, node_body = sync.get_object(tree)
        if node_kind != KIND_TREE or wire_address(node_body) != tree:
            raise WisdomValidationError(
                "owner-private draft tree failed integrity validation"
            )
        try:
            parsed = json.loads(node_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WisdomValidationError(
                "owner-private draft tree is malformed"
            ) from exc
        entries = parsed.get("entries")
        if not isinstance(entries, list):
            raise WisdomValidationError("owner-private draft tree has no entries")
        for entry in entries:
            if not isinstance(entry, dict):
                raise WisdomValidationError(
                    "owner-private draft tree entry is malformed"
                )
            name = entry.get("name")
            address = entry.get("hash")
            entry_kind = entry.get("kind")
            if not isinstance(name, str) or not isinstance(address, str):
                raise WisdomValidationError(
                    "owner-private draft tree entry is malformed"
                )
            path = f"{prefix}{name}"
            if entry_kind == KIND_TREE:
                walk(address, path + "/")
            elif entry_kind == KIND_BLOB:
                blob_kind, blob = sync.get_object(address)
                if blob_kind != KIND_BLOB or wire_address(blob) != address:
                    raise WisdomValidationError(
                        f"draft blob failed integrity validation: {path}"
                    )
                out.append((
                    path,
                    "exec" if entry.get("mode") == "exec" else "file",
                    blob,
                ))
            else:
                raise WisdomValidationError(f"unsupported draft object kind at {path}")

    walk(tree_hash, "")
    return out


class WisdomClient:
    def __init__(self, *, timeout: float = 30.0) -> None:
        identity = resolve_identity()
        base = resolve_sync_base_url()
        if not base:
            raise WisdomAuthError("Collective Wisdom Gateway is not configured")
        self.identity = identity
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {identity['api_key']}",
            "Accept": "application/json",
        })
        self.sync = SyncClient(self.base, identity["api_key"], timeout=timeout)

    @property
    def display_scopes(self) -> tuple[str, ...]:
        scopes = self.identity.get("claims", {}).get("wisdom_scopes")
        return (
            tuple(item for item in scopes if isinstance(item, str))
            if isinstance(scopes, list)
            else ()
        )

    @property
    def display_org_id(self) -> str | None:
        claims = self.identity.get("claims", {})
        value = claims.get("org_id") or claims.get("orgId")
        return str(value) if value else None

    def _url(self, path: str) -> str:
        return f"{self.base}/v1/sync/wisdom/{path.lstrip('/')}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        model: type[BaseModel] | None = None,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        try:
            response = self.session.request(
                method,
                self._url(path),
                json=json_body,
                params=params,
                timeout=self.timeout,
            )
        except requests.Timeout as exc:
            raise WisdomError("Collective Wisdom Gateway timed out") from exc
        except requests.RequestException as exc:
            raise WisdomError("Collective Wisdom Gateway is unavailable") from exc
        body: Any = None
        if response.content:
            try:
                body = response.json()
            except ValueError as exc:
                raise WisdomError(
                    "Collective Wisdom Gateway returned a malformed response"
                ) from exc
        code = body.get("error") if isinstance(body, dict) else None
        if response.status_code in (401, 403):
            raise WisdomAuthError(
                "Collective Wisdom is unavailable for this account",
                status=response.status_code,
                code=code,
            )
        if response.status_code == 404:
            raise WisdomNotFound(
                "Collective Wisdom item not found", status=404, code=code
            )
        if response.status_code == 409:
            raise WisdomConflict(
                str(code or "Collective Wisdom state changed; refresh and retry"),
                status=409,
                code=code,
            )
        if response.status_code == 422:
            raise WisdomValidationError(
                str(code or "Collective Wisdom rejected invalid content"),
                status=422,
                code=code,
            )
        if response.status_code < 200 or response.status_code >= 300:
            raise WisdomError(
                f"Collective Wisdom Gateway failed ({response.status_code})",
                status=response.status_code,
                code=code,
            )
        if model is None:
            return body
        try:
            return model.model_validate(body)
        except ValidationError as exc:
            raise WisdomError(
                "Collective Wisdom Gateway response failed schema validation"
            ) from exc

    def capability(self) -> dict[str, Any]:
        try:
            response = self.session.get(
                f"{self.base}/v1/sync/capabilities", timeout=self.timeout
            )
            response.raise_for_status()
            body = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise WisdomError("Gateway capabilities are unavailable") from exc
        if "wisdom" not in (body.get("features") or []):
            raise WisdomAuthError("Gateway does not advertise Collective Wisdom")
        return body

    def register_identity(self, installation_id: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "installation-identities",
            json_body={"installation_id": installation_id},
        )

    def upload_private_objects(self, objects: ObjectSet) -> None:
        self.sync.put_objects(objects.objects)

    def submit_draft(
        self,
        *,
        slug: str,
        commit: str,
        content_hash: str,
        description: str,
        professionalism_review: dict[str, Any] | None = None,
    ) -> Draft:
        payload: dict[str, Any] = {
            "slug": slug,
            "draft_commit": commit,
            "content_hash": content_hash,
            "author_description": description,
        }
        if professionalism_review is not None:
            payload["professionalism_review"] = professionalism_review
        body = self._request(
            "POST",
            "drafts",
            model=DraftResponse,
            json_body=payload,
        )
        return body.draft

    def list_drafts(self) -> list[Draft]:
        return self._request("GET", "drafts", model=DraftList).drafts

    def draft(self, draft_id: str) -> DraftDetail:
        return self._request(
            "GET", f"drafts/{quote(draft_id, safe='')}", model=DraftDetail
        )

    def reconstruct_draft(self, draft_id: str) -> ReconstructedDraft:
        detail = self.draft(draft_id)
        files = _walk_private_commit(self.sync, detail.draft.draftCommit)
        records, content_hash = verify_content_files(files)
        if content_hash != detail.draft.contentHash:
            raise WisdomValidationError(
                "server draft content does not match its authoritative hash"
            )
        return ReconstructedDraft(
            detail=detail, files=files, content_files=records, content_hash=content_hash
        )

    def revise_draft(
        self,
        draft_id: str,
        *,
        commit: str,
        content_hash: str,
        description: str,
        expected_content_hash: str,
        expected_description_hash: str,
        expected_manifest_hash: str,
        professionalism_review: dict[str, Any] | None = None,
    ) -> Draft:
        """Create a newly vetted successor for an immutable owner draft."""
        payload: dict[str, Any] = {
            "draft_commit": commit,
            "content_hash": content_hash,
            "author_description": description,
            "expected_content_hash": expected_content_hash,
            "expected_author_description_hash": expected_description_hash,
            "expected_package_manifest_hash": expected_manifest_hash,
        }
        if professionalism_review is not None:
            payload["professionalism_review"] = professionalism_review
        body = self._request(
            "POST",
            f"drafts/{quote(draft_id, safe='')}/revise",
            json_body=payload,
        )
        return Draft.model_validate(body.get("draft", body))

    def approve(
        self,
        draft_id: str,
        *,
        content_hash: str,
        description_hash: str,
        manifest_hash: str,
    ) -> Draft:
        body = self._request(
            "POST",
            f"drafts/{quote(draft_id, safe='')}/approve",
            json_body={
                "content_hash": content_hash,
                "author_description_hash": description_hash,
                "package_manifest_hash": manifest_hash,
            },
        )
        return Draft.model_validate(body.get("draft", body))

    def publish(self, draft_id: str, *, content_hash: str) -> dict[str, Any]:
        return self._request(
            "POST",
            f"drafts/{quote(draft_id, safe='')}/publish",
            json_body={"content_hash": content_hash, "base_commit": None},
        )

    def decline(self, draft_id: str) -> dict[str, Any]:
        return self._request("POST", f"drafts/{quote(draft_id, safe='')}/decline")

    def list_skills(self, *, cursor: str | None = None) -> SkillList:
        return self._request(
            "GET",
            "skills",
            model=SkillList,
            params={"cursor": cursor} if cursor else None,
        )

    def skill(self, skill_id: str) -> SkillDetail:
        return self._request(
            "GET", f"skills/{quote(skill_id, safe='')}", model=SkillDetail
        )

    def version(self, skill_id: str, version: int) -> VersionDetail:
        return self._request(
            "GET",
            f"skills/{quote(skill_id, safe='')}/versions/{version}",
            model=VersionDetail,
        )

    def content(
        self,
        skill_id: str,
        version: int,
        *,
        installation_id: str,
        takedown_generation: int,
    ) -> tuple[VersionContent, list[tuple[str, str, bytes]]]:
        response = self._request(
            "GET",
            f"skills/{quote(skill_id, safe='')}/versions/{version}/content",
            model=VersionContent,
            params={
                "installation_id": installation_id,
                "takedown_generation": takedown_generation,
            },
        )
        decoded: list[tuple[str, str, bytes]] = []
        for item in response.files:
            try:
                raw = base64.b64decode(item["content_base64"], validate=True)
            except (KeyError, TypeError, ValueError) as exc:
                raise WisdomValidationError(
                    "version content contains malformed base64"
                ) from exc
            if wire_address(raw) != item.get("hash"):
                raise WisdomValidationError(
                    f"version blob failed integrity validation: {item.get('path', '?')}"
                )
            decoded.append((str(item["path"]), str(item["mode"]), raw))
        records, derived = verify_content_files(decoded)
        if derived != response.content_hash:
            raise WisdomValidationError(
                "version content hash failed integrity validation"
            )
        manifest_body = next(
            (body for path, _, body in decoded if path == "skill.manifest.json"), None
        )
        if manifest_body is None:
            raise WisdomValidationError("version content has no package manifest")
        try:
            parse_manifest_bytes(manifest_body)
        except (UnicodeDecodeError, ValueError) as exc:
            raise WisdomValidationError("version package manifest is invalid") from exc
        return response, decoded

    def record_install(
        self,
        *,
        skill_id: str,
        installation_id: str,
        version: int,
        takedown_generation: int,
        update_mode: str | None,
    ) -> InstallationRecord:
        payload: dict[str, Any] = {
            "skill_id": skill_id,
            "installation_id": installation_id,
            "version": version,
            "takedown_generation": takedown_generation,
        }
        if update_mode:
            payload["update_mode"] = update_mode
        return self._request(
            "POST", "installations", model=InstallationRecord, json_body=payload
        )

    def installations(self, installation_id: str) -> list[dict[str, Any]]:
        result = self._request(
            "GET",
            f"installations/{quote(installation_id, safe='')}",
            model=InstallationList,
        )
        return [item.model_dump(mode="json") for item in result.installations]

    def deactivate_install(
        self, installation_id: str, skill_id: str
    ) -> InstallationDeactivation:
        return self._request(
            "DELETE",
            f"installations/{quote(installation_id, safe='')}/skills/{quote(skill_id, safe='')}",
            model=InstallationDeactivation,
        )

    def feed(
        self, cursor: str | None = None, *, installation_id: str | None = None
    ) -> Feed:
        params: dict[str, str] = {}
        if cursor:
            params["cursor"] = cursor
        if installation_id:
            params["installation_id"] = installation_id
        return self._request("GET", "feed", model=Feed, params=params or None)
