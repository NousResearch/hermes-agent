"""
Administrative Control-Plane API module for Hermes Agent.

Provides safe, profile-scoped CRUD operations for runtime profiles, skills,
and context files under `/v1/admin`.
"""

import hashlib
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import unquote

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

from agent.skill_utils import parse_frontmatter
from gateway.request_profile import api_request_profile as _api_request_profile
from hermes_cli.profiles import (
    create_profile,
    get_profile_dir,
    list_profiles,
    normalize_profile_name,
    read_profile_meta,
    validate_profile_name,
    write_profile_meta,
)

MANIFEST_FILENAME = ".control_plane_manifest.json"
MAX_ADMIN_PAYLOAD_BYTES = 5_000_000  # 5 MB limit for admin payloads

# Field length bounds
MAX_OWNERSHIP_LEN = 256
MAX_NAME_LEN = 128
MAX_DISPLAY_NAME_LEN = 256
MAX_DESCRIPTION_LEN = 4096
MAX_SOUL_LEN = 100_000
MAX_USER_CONTEXT_LEN = 100_000
MAX_SKILL_CONTENT_LEN = 1_000_000
MAX_FILE_CONTENT_LEN = 2_000_000

# File security allowlists and forbidden sets
ALLOWED_EXACT_FILES = frozenset({"SOUL.md", "memories/USER.md", "memories/MEMORY.md"})
ALLOWED_SUBTREES = ("context/", "memories/")
FORBIDDEN_EXACT_FILES = frozenset({
    ".env",
    "config.yaml",
    "state.db",
    "state.db-wal",
    "state.db-shm",
    "hermes_state.db",
    "auth.json",
    "auth.lock",
    "gateway.pid",
    "gateway_state.json",
    "processes.json",
    MANIFEST_FILENAME,
})
FORBIDDEN_PREFIXES = (
    "response_store.db",
    ".",  # hidden files
)

SLUG_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def _admin_error(message: str, *, code: str = "invalid_request", param: Optional[str] = None, status: int = 400) -> Any:
    """Format an OpenAI-style JSON error response."""
    return web.json_response(
        {
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": param,
                "code": code,
            }
        },
        status=status,
    )


def _admin_json_response(data: dict, status: int = 200) -> Any:
    """Return web.json_response with ETag header if digest is present."""
    headers = {}
    digest = data.get("digest")
    if digest and isinstance(digest, str):
        headers["ETag"] = f'"{digest}"'
    return web.json_response(data, status=status, headers=headers)


def _sha256_digest(data: Union[str, bytes]) -> str:
    """Compute SHA-256 digest prefixed with sha256:."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _atomic_write_file(file_path: Path, content: Union[str, bytes], mode: int = 0o600) -> None:
    """Write content to file_path atomically with owner-only permissions."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = f".{file_path.name}.tmp.{uuid.uuid4().hex}"
    tmp_path = file_path.with_name(tmp_name)
    if isinstance(content, str):
        tmp_path.write_text(content, encoding="utf-8")
    else:
        tmp_path.write_bytes(content)
    try:
        os.chmod(str(tmp_path), mode)
    except OSError:
        pass
    os.replace(str(tmp_path), str(file_path))


def read_ownership_manifest(profile_dir: Path) -> Optional[dict]:
    """Read ownership manifest file inside profile root."""
    manifest_path = profile_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Malformed control plane manifest JSON")
        return data
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Malformed control plane manifest in {profile_dir}: {exc}") from exc
    except Exception:
        return None


def write_ownership_manifest(profile_dir: Path, manifest_data: dict) -> dict:
    """Write or update ownership manifest inside profile root atomically."""
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest_path = profile_dir / MANIFEST_FILENAME
    if manifest_path.is_file():
        existing = read_ownership_manifest(profile_dir) or {}
    else:
        existing = {}

    rev = manifest_data.get("revision")
    if rev is None:
        rev = existing.get("revision", 1)
    else:
        try:
            rev = int(rev)
        except (ValueError, TypeError):
            rev = existing.get("revision", 1)

    manifest = {
        "managed_by": str(manifest_data.get("managed_by") or existing.get("managed_by") or "").strip(),
        "tenant_id": str(manifest_data.get("tenant_id") or existing.get("tenant_id") or "").strip(),
        "resource_id": str(manifest_data.get("resource_id") or existing.get("resource_id") or "").strip(),
        "revision": rev,
        "spec_digest": str(manifest_data.get("spec_digest") if manifest_data.get("spec_digest") is not None else existing.get("spec_digest", "")),
        "created_at": existing.get("created_at") or now,
        "updated_at": now,
        "skills": manifest_data.get("skills") if manifest_data.get("skills") is not None else existing.get("skills", {}),
        "files": manifest_data.get("files") if manifest_data.get("files") is not None else existing.get("files", {}),
        "baseline_files": manifest_data.get("baseline_files") if manifest_data.get("baseline_files") is not None else existing.get("baseline_files", []),
    }
    _atomic_write_file(profile_dir / MANIFEST_FILENAME, json.dumps(manifest, indent=2))
    return manifest


def extract_caller_ownership(request: Any, body: Optional[dict] = None) -> dict:
    """Extract ownership identification (managed_by, tenant_id, resource_id) from headers or body."""
    headers = request.headers
    body = body or {}

    managed_by = (
        headers.get("X-Hermes-Owner-Managed-By")
        or headers.get("X-Control-Plane-Managed-By")
        or body.get("managed_by")
        or ""
    )
    if isinstance(managed_by, str):
        managed_by = managed_by.strip()
    else:
        managed_by = ""

    tenant_id = (
        headers.get("X-Hermes-Owner-Tenant-Id")
        or headers.get("X-Control-Plane-Tenant-Id")
        or body.get("tenant_id")
        or ""
    )
    if isinstance(tenant_id, str):
        tenant_id = tenant_id.strip()
    else:
        tenant_id = ""

    resource_id = (
        headers.get("X-Hermes-Owner-Resource-Id")
        or headers.get("X-Control-Plane-Resource-Id")
        or body.get("resource_id")
        or ""
    )
    if isinstance(resource_id, str):
        resource_id = resource_id.strip()
    else:
        resource_id = ""

    return {
        "managed_by": managed_by,
        "tenant_id": tenant_id,
        "resource_id": resource_id,
    }


def verify_profile_ownership(
    profile_name: str, profile_dir: Path, caller_ownership: dict
) -> Tuple[bool, Optional[str], Optional[dict]]:
    """Verify caller ownership for a named profile.

    Returns: (is_owned, error_code, manifest)
    """
    if not caller_ownership.get("managed_by") or not caller_ownership.get("tenant_id") or not caller_ownership.get("resource_id"):
        return False, "missing_ownership", None

    try:
        canon = normalize_profile_name(profile_name)
    except ValueError:
        return False, "invalid_profile_name", None

    if canon == "default":
        return False, "default_profile_protected", None

    if not profile_dir.is_dir():
        return False, "profile_not_found", None

    try:
        manifest = read_ownership_manifest(profile_dir)
    except ValueError:
        return False, "corrupt_manifest", None

    if not manifest:
        return False, "unmanaged_profile", None

    for field in ("managed_by", "tenant_id", "resource_id"):
        if caller_ownership.get(field) != manifest.get(field):
            return False, "ownership_mismatch", manifest

    return True, None, manifest


def validate_file_relative_path(profile_dir: Path, rel_path_str: str) -> Tuple[bool, Optional[str], Optional[Path], int]:
    """Validate relative file path against directory traversal and security allowlist.

    Returns: (is_valid, error_message, target_resolved_path, status_code)
    """
    if not rel_path_str or not isinstance(rel_path_str, str):
        return False, "Invalid path", None, 400

    # Reject absolute paths before stripping leading slash
    if rel_path_str.startswith("/") or rel_path_str.startswith("\\") or re.match(r"^[a-zA-Z]:", rel_path_str):
        return False, "Absolute paths are forbidden", None, 400

    unquoted = unquote(rel_path_str).strip()
    if unquoted.startswith("/") or unquoted.startswith("\\") or "\0" in unquoted:
        return False, "Absolute paths and null bytes are forbidden", None, 400

    parts = Path(unquoted).parts
    if ".." in parts or "." in parts or ".." in rel_path_str or "%2e%2e" in rel_path_str.lower():
        return False, "Path traversal forbidden", None, 403

    profile_resolved = profile_dir.resolve()
    target_path = (profile_dir / unquoted).resolve()

    # Out of root check
    try:
        rel_posix = target_path.relative_to(profile_resolved).as_posix()
    except ValueError:
        return False, "Path resolves outside profile root", None, 403

    # Forbidden files & prefixes check
    filename = target_path.name
    if rel_posix in FORBIDDEN_EXACT_FILES or filename in FORBIDDEN_EXACT_FILES:
        return False, f"File {rel_posix} is restricted", None, 403

    for prefix in FORBIDDEN_PREFIXES:
        if filename.startswith(prefix) and rel_posix not in ALLOWED_EXACT_FILES:
            return False, f"File {rel_posix} is restricted", None, 403

    # Symlink escape check
    curr = profile_resolved
    for part in Path(rel_posix).parts:
        curr = curr / part
        if curr.is_symlink():
            try:
                sym_target = curr.resolve()
                sym_target.relative_to(profile_resolved)
            except ValueError:
                return False, "Symlink points outside profile root", None, 403

    # Allowlist check
    if rel_posix in ALLOWED_EXACT_FILES:
        return True, None, target_path, 200

    for allowed_prefix in ALLOWED_SUBTREES:
        if rel_posix.startswith(allowed_prefix):
            return True, None, target_path, 200

    return False, f"File path {rel_posix} is not in managed allowlist", None, 403


def _check_admin_request(adapter: Any, request: Any) -> Optional[Any]:
    """Validate bearer auth, admin feature gate, and payload size bounds."""
    auth_err = adapter._check_auth(request)
    if auth_err:
        return auth_err

    if not getattr(adapter, "_admin_config_rw", False):
        return _admin_error(
            "Admin API is disabled. Opt in via gateway.api_server.admin_config_rw: true in config.yaml.",
            code="admin_api_disabled",
            status=403,
        )

    if request.content_length and request.content_length > MAX_ADMIN_PAYLOAD_BYTES:
        return _admin_error(
            f"Payload exceeds maximum allowed size of {MAX_ADMIN_PAYLOAD_BYTES} bytes.",
            code="payload_too_large",
            status=413,
        )

    return None


async def _parse_admin_body(request: Any) -> Tuple[Optional[dict], Optional[Any]]:
    """Read request body safely and enforce size bounds."""
    if not request.can_read_body:
        return {}, None
    try:
        raw_body = await request.read()
        if len(raw_body) > MAX_ADMIN_PAYLOAD_BYTES:
            return None, _admin_error(
                f"Payload exceeds maximum allowed size of {MAX_ADMIN_PAYLOAD_BYTES} bytes.",
                code="payload_too_large",
                status=413,
            )
        if not raw_body:
            return {}, None
        body = json.loads(raw_body.decode("utf-8"))
        if not isinstance(body, dict):
            return None, _admin_error("JSON body must be an object", status=400)
        return body, None
    except json.JSONDecodeError:
        return None, _admin_error("Invalid JSON body", status=400)
    except Exception:
        return None, _admin_error("Failed to read request body", status=400)


def _check_if_match(request: Any, current_digest: str) -> Optional[Any]:
    """Check If-Match header against current ETag/digest. Return error response if precondition fails."""
    if_match = request.headers.get("If-Match")
    if not if_match:
        return None
    cleaned_if_match = if_match.strip().strip('"').strip("'")
    cleaned_digest = current_digest.strip().strip('"').strip("'")
    if cleaned_if_match == "*":
        return None
    if cleaned_if_match != cleaned_digest:
        return _admin_error(
            f"If-Match precondition failed: expected digest {cleaned_digest}, got {cleaned_if_match}",
            status=412,
            code="precondition_failed",
        )
    return None


def _get_profile_param(request: Any, body: Optional[dict] = None) -> str:
    body = body or {}
    return request.match_info.get("target_profile") or request.match_info.get("profile") or body.get("name") or ""


def _canonical_profile_repr(profile_name: str, profile_dir: Path, manifest: dict) -> dict:
    """Build canonical dictionary representation for a managed profile."""
    meta = read_profile_meta(profile_dir) if profile_dir.is_dir() else {}
    display_name = profile_name
    meta_yaml_path = profile_dir / "profile.yaml"
    if meta_yaml_path.is_file():
        try:
            import yaml
            yaml_data = yaml.safe_load(meta_yaml_path.read_text(encoding="utf-8")) or {}
            if isinstance(yaml_data, dict) and yaml_data.get("display_name"):
                display_name = str(yaml_data["display_name"]).strip()
        except Exception:
            pass

    description = meta.get("description", "")

    soul_path = profile_dir / "SOUL.md"
    soul = soul_path.read_text(encoding="utf-8") if soul_path.is_file() else ""

    user_mem_path = profile_dir / "memories" / "USER.md"
    user_context = user_mem_path.read_text(encoding="utf-8") if user_mem_path.is_file() else ""

    digest_input = json.dumps(
        {
            "name": profile_name,
            "managed_by": manifest.get("managed_by", ""),
            "tenant_id": manifest.get("tenant_id", ""),
            "resource_id": manifest.get("resource_id", ""),
            "display_name": display_name,
            "description": description,
            "soul": soul,
            "user_context": user_context,
        },
        sort_keys=True,
    )
    actual_digest = _sha256_digest(digest_input)
    applied_digest = str(manifest.get("spec_digest") or "")
    drifted = (actual_digest != applied_digest) if applied_digest else False

    return {
        "object": "hermes.admin.profile",
        "name": profile_name,
        "display_name": display_name,
        "is_default": False,
        "ownership": {
            "managed_by": manifest.get("managed_by"),
            "tenant_id": manifest.get("tenant_id"),
            "resource_id": manifest.get("resource_id"),
        },
        "revision": manifest.get("revision", 1),
        "digest": actual_digest,
        "applied_digest": applied_digest,
        "drifted": drifted,
        "description": description,
        "soul": soul,
        "user_context": user_context,
        "created_at": manifest.get("created_at"),
        "updated_at": manifest.get("updated_at"),
    }


def _invalidate_skills_prompt_cache() -> None:
    """Invoke skills prompt cache invalidation path."""
    try:
        from agent.prompt_builder import clear_skills_system_prompt_cache
        clear_skills_system_prompt_cache(clear_snapshot=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Profile Route Handlers
# ---------------------------------------------------------------------------

async def handle_list_profiles(adapter: Any, request: Any) -> Any:
    """GET /v1/admin/profiles — list managed profiles."""
    guard = _check_admin_request(adapter, request)
    if guard:
        return guard

    caller_ownership = extract_caller_ownership(request)
    if not caller_ownership["managed_by"] or not caller_ownership["tenant_id"]:
        return _admin_error(
            "Ownership parameters (managed_by, tenant_id) are required",
            status=400,
            code="missing_ownership",
        )

    resource_filter = caller_ownership.get("resource_id") or ""
    all_profs = list_profiles()
    managed_list = []

    for p in all_profs:
        if p.is_default:
            continue
        try:
            manifest = read_ownership_manifest(p.path)
        except ValueError:
            continue
        if not manifest:
            continue

        owner_matches = (
            manifest.get("managed_by") == caller_ownership["managed_by"]
            and manifest.get("tenant_id") == caller_ownership["tenant_id"]
        )
        resource_matches = not resource_filter or manifest.get("resource_id") == resource_filter
        if owner_matches and resource_matches:
            managed_list.append(_canonical_profile_repr(p.name, p.path, manifest))

    managed_list.sort(key=lambda x: x["name"])
    return _admin_json_response({"object": "list", "data": managed_list})


async def handle_create_update_profile(adapter: Any, request: Any) -> Any:
    """PUT/POST /v1/admin/profiles — create or update a named managed profile."""
    guard = _check_admin_request(adapter, request)
    if guard:
        return guard

    body, body_err = await _parse_admin_body(request)
    if body_err:
        return body_err

    # Cloning parameters are strictly rejected
    if "clone_from" in body or "clone_config" in body or "clone_all" in body:
        return _admin_error(
            "Cloning profiles (clone_from, clone_config, clone_all) is forbidden in Admin API",
            status=400,
            code="cloning_forbidden",
        )

    profile_name = _get_profile_param(request, body)
    if not profile_name:
        return _admin_error("Profile name is required", param="profile", status=400)
    if len(profile_name) > MAX_NAME_LEN:
        return _admin_error(f"Profile name exceeds maximum length of {MAX_NAME_LEN}", status=400, code="payload_too_large")

    try:
        canon = normalize_profile_name(profile_name)
        validate_profile_name(canon)
    except ValueError as exc:
        return _admin_error(str(exc), param="profile", status=400)

    if canon == "default":
        return _admin_error("Default profile (~/.hermes) cannot be managed via admin API", status=403, code="default_profile_protected")

    caller_ownership = extract_caller_ownership(request, body)
    if not caller_ownership["managed_by"] or not caller_ownership["tenant_id"] or not caller_ownership["resource_id"]:
        return _admin_error(
            "Ownership parameters (managed_by, tenant_id, resource_id) are required",
            status=400,
            code="missing_ownership",
        )

    for k, v in caller_ownership.items():
        if len(v) > MAX_OWNERSHIP_LEN:
            return _admin_error(f"Ownership field {k} exceeds maximum length of {MAX_OWNERSHIP_LEN}", status=400, code="payload_too_large")

    display_name = body.get("display_name")
    if display_name is not None:
        if not isinstance(display_name, str):
            return _admin_error("display_name must be a string", status=400)
        display_name = display_name.strip()
        if len(display_name) > MAX_DISPLAY_NAME_LEN:
            return _admin_error(f"display_name exceeds maximum length of {MAX_DISPLAY_NAME_LEN}", status=400, code="payload_too_large")

    description = body.get("description")
    if description is not None:
        if not isinstance(description, str):
            return _admin_error("description must be a string", status=400)
        description = description.strip()
        if len(description) > MAX_DESCRIPTION_LEN:
            return _admin_error(f"description exceeds maximum length of {MAX_DESCRIPTION_LEN}", status=400, code="payload_too_large")

    soul = body.get("soul")
    if soul is not None:
        if not isinstance(soul, str):
            return _admin_error("soul must be a string", status=400)
        if len(soul) > MAX_SOUL_LEN:
            return _admin_error(f"soul exceeds maximum length of {MAX_SOUL_LEN}", status=400, code="payload_too_large")

    user_context = body.get("user_context")
    if user_context is not None:
        if not isinstance(user_context, str):
            return _admin_error("user_context must be a string", status=400)
        if len(user_context) > MAX_USER_CONTEXT_LEN:
            return _admin_error(f"user_context exceeds maximum length of {MAX_USER_CONTEXT_LEN}", status=400, code="payload_too_large")

    profile_dir = get_profile_dir(canon)

    if profile_dir.is_dir():
        is_owned, err_code, manifest = verify_profile_ownership(canon, profile_dir, caller_ownership)
        if not is_owned or not manifest:
            if err_code == "missing_ownership":
                return _admin_error("Ownership parameters are required", status=400, code="missing_ownership")
            if err_code == "corrupt_manifest":
                return _admin_error(f"Profile '{canon}' has corrupt manifest", status=409, code="corrupt_manifest")
            return _admin_error(
                f"Profile '{canon}' exists but is not owned by the caller ownership tuple",
                status=409,
                code="ownership_mismatch" if err_code == "ownership_mismatch" else "unmanaged_profile",
            )

        current_repr = _canonical_profile_repr(canon, profile_dir, manifest)
        if_match_err = _check_if_match(request, current_repr["digest"])
        if if_match_err:
            return if_match_err

        desired_disp = display_name if display_name is not None else current_repr["display_name"]
        desired_desc = description if description is not None else current_repr["description"]
        desired_soul = soul if soul is not None else current_repr["soul"]
        desired_user_ctx = user_context if user_context is not None else current_repr["user_context"]

        changed = (
            desired_disp != current_repr["display_name"]
            or desired_desc != current_repr["description"]
            or desired_soul != current_repr["soul"]
            or desired_user_ctx != current_repr["user_context"]
        )

        if not changed and not current_repr["drifted"]:
            return _admin_json_response(_canonical_profile_repr(canon, profile_dir, manifest), status=200)

        # Apply profile updates - do not swallow errors
        write_profile_meta(profile_dir, description=desired_desc, description_auto=False)

        meta_path = profile_dir / "profile.yaml"
        if meta_path.is_file() or desired_disp != canon:
            import yaml
            loaded = yaml.safe_load(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
            if not isinstance(loaded, dict):
                loaded = {}
            loaded["display_name"] = desired_disp
            from utils import atomic_yaml_write
            atomic_yaml_write(meta_path, loaded, sort_keys=False)

        files_map = dict(manifest.get("files", {}))
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        if soul is not None or desired_soul != current_repr["soul"]:
            _atomic_write_file(profile_dir / "SOUL.md", desired_soul, mode=0o600)
            files_map["SOUL.md"] = {
                "managed": True,
                "revision": files_map.get("SOUL.md", {}).get("revision", 0) + 1,
                "digest": _sha256_digest(desired_soul),
                "updated_at": now_iso,
            }

        if user_context is not None or desired_user_ctx != current_repr["user_context"]:
            _atomic_write_file(profile_dir / "memories" / "USER.md", desired_user_ctx, mode=0o600)
            files_map["memories/USER.md"] = {
                "managed": True,
                "revision": files_map.get("memories/USER.md", {}).get("revision", 0) + 1,
                "digest": _sha256_digest(desired_user_ctx),
                "updated_at": now_iso,
            }

        temp_manifest = {**manifest, "files": files_map}
        readback_repr = _canonical_profile_repr(canon, profile_dir, temp_manifest)
        new_applied_digest = readback_repr["digest"]
        next_rev = manifest.get("revision", 1) + 1

        updated_manifest_data = {
            **manifest,
            "revision": next_rev,
            "spec_digest": new_applied_digest,
            "files": files_map,
        }
        final_manifest = write_ownership_manifest(profile_dir, updated_manifest_data)
        return _admin_json_response(_canonical_profile_repr(canon, profile_dir, final_manifest), status=200)
    else:
        # Create fresh managed profile
        try:
            created_dir = create_profile(
                name=canon,
                clone_from=None,
                clone_config=False,
                clone_all=False,
                no_alias=True,
                no_skills=True,
                description=description if description else None,
            )
        except Exception as exc:
            return _admin_error(f"Failed to create profile '{canon}': {exc}", status=500)

        desired_disp = display_name if display_name is not None else canon
        desired_desc = description or ""
        desired_soul = soul or ""
        desired_user_ctx = user_context or ""

        baseline_files = [p.relative_to(created_dir).as_posix() for p in created_dir.rglob("*") if p.is_file()]

        meta_path = created_dir / "profile.yaml"
        if desired_disp != canon or meta_path.is_file():
            import yaml
            loaded = yaml.safe_load(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
            if not isinstance(loaded, dict):
                loaded = {}
            loaded["display_name"] = desired_disp
            from utils import atomic_yaml_write
            atomic_yaml_write(meta_path, loaded, sort_keys=False)

        files_map = {}
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        if soul is not None or desired_soul:
            _atomic_write_file(created_dir / "SOUL.md", desired_soul, mode=0o600)
            files_map["SOUL.md"] = {"managed": True, "revision": 1, "digest": _sha256_digest(desired_soul), "updated_at": now_iso}

        if user_context is not None or desired_user_ctx:
            _atomic_write_file(created_dir / "memories" / "USER.md", desired_user_ctx, mode=0o600)
            files_map["memories/USER.md"] = {"managed": True, "revision": 1, "digest": _sha256_digest(desired_user_ctx), "updated_at": now_iso}

        for auto_file in ("SOUL.md", "memories/USER.md"):
            auto_path = created_dir / auto_file
            if auto_path.is_file() and auto_file not in files_map:
                content_text = auto_path.read_text(encoding="utf-8")
                files_map[auto_file] = {
                    "managed": True,
                    "revision": 1,
                    "digest": _sha256_digest(content_text),
                    "updated_at": now_iso,
                }

        temp_manifest = {
            "managed_by": caller_ownership["managed_by"],
            "tenant_id": caller_ownership["tenant_id"],
            "resource_id": caller_ownership["resource_id"],
            "revision": 1,
            "skills": {},
            "files": files_map,
            "baseline_files": baseline_files,
        }
        readback_repr = _canonical_profile_repr(canon, created_dir, temp_manifest)
        spec_digest = readback_repr["digest"]

        manifest_data = {
            **temp_manifest,
            "spec_digest": spec_digest,
        }
        manifest = write_ownership_manifest(created_dir, manifest_data)
        return _admin_json_response(_canonical_profile_repr(canon, created_dir, manifest), status=201)


async def handle_get_profile(adapter: Any, request: Any) -> Any:
    """GET /v1/admin/profiles/{profile} — read managed profile details."""
    guard = _check_admin_request(adapter, request)
    if guard:
        return guard

    profile_name = _get_profile_param(request)
    profile_dir = get_profile_dir(profile_name)
    caller_ownership = extract_caller_ownership(request)

    is_owned, err_code, manifest = verify_profile_ownership(profile_name, profile_dir, caller_ownership)
    if not is_owned or not manifest:
        if err_code == "missing_ownership":
            return _admin_error("Ownership parameters are required", status=400, code="missing_ownership")
        if err_code == "default_profile_protected":
            return _admin_error("Default profile is protected", status=403, code="default_profile_protected")
        if err_code == "corrupt_manifest":
            return _admin_error(f"Profile '{profile_name}' has corrupt manifest", status=409, code="corrupt_manifest")
        if err_code == "ownership_mismatch":
            return _admin_error(f"Profile '{profile_name}' ownership mismatch", status=409, code="ownership_mismatch")
        return _admin_error(f"Profile '{profile_name}' not found or unowned", status=404, code="profile_not_found")

    return _admin_json_response(_canonical_profile_repr(profile_name, profile_dir, manifest))


async def handle_delete_profile(adapter: Any, request: Any) -> Any:
    """DELETE /v1/admin/profiles/{profile} — delete a named managed profile.

    Preflight-first deletion algorithm:
    1. Verify caller ownership tuple. Missing parameters return 400.
    2. Check active/busy status: request scope, running gateway process, or active runs return 409 profile_active_conflict / profile_busy_conflict.
    3. If profile directory does not exist, return 200 idempotent success if caller ownership tuple is complete.
    4. Preflight scan profile directory: if any unmanaged or unknown resource exists that is not in the ownership manifest (files, skills, baseline_files), return 409 profile_not_empty WITHOUT mutating any resource.
    5. Fully delete profile directory and verify absence before returning 200 deleted: true.
    """
    guard = _check_admin_request(adapter, request)
    if guard:
        return guard

    profile_name = _get_profile_param(request)
    try:
        canon = normalize_profile_name(profile_name)
    except ValueError as exc:
        return _admin_error(str(exc), status=400)

    if canon == "default":
        return _admin_error("Cannot delete default profile", status=403, code="default_profile_protected")

    profile_dir = get_profile_dir(canon)
    caller_ownership = extract_caller_ownership(request)

    if not profile_dir.is_dir():
        if not caller_ownership.get("managed_by") or not caller_ownership.get("tenant_id") or not caller_ownership.get("resource_id"):
            return _admin_error("Ownership parameters are required", status=400, code="missing_ownership")
        # Idempotent deletion for already-absent profile with complete ownership tuple
        return _admin_json_response({"object": "hermes.admin.profile.deleted", "name": canon, "deleted": True})

    is_owned, err_code, manifest = verify_profile_ownership(canon, profile_dir, caller_ownership)
    if not is_owned or not manifest:
        if err_code == "missing_ownership":
            return _admin_error("Ownership parameters are required", status=400, code="missing_ownership")
        if err_code == "corrupt_manifest":
            return _admin_error(f"Profile '{canon}' has corrupt manifest", status=409, code="corrupt_manifest")
        return _admin_error(
            f"Profile '{canon}' exists but is not owned by caller ownership tuple",
            status=409,
            code="ownership_mismatch" if err_code == "ownership_mismatch" else "unmanaged_profile",
        )

    current_repr = _canonical_profile_repr(canon, profile_dir, manifest)
    if_match_err = _check_if_match(request, current_repr["digest"])
    if if_match_err:
        return if_match_err

    # 1. Request scope active check
    current_scope = _api_request_profile.get()
    if current_scope and normalize_profile_name(current_scope) == canon:
        return _admin_error(f"Profile '{canon}' is currently active in request scope and cannot be deleted", status=409, code="profile_active_conflict")

    # 2. Running gateway process check
    for prof in list_profiles():
        if normalize_profile_name(prof.name) == canon and prof.gateway_running:
            return _admin_error(f"Profile '{canon}' has a running gateway process and cannot be deleted", status=409, code="profile_active_conflict")

    # 3. Active runs check on adapter
    run_statuses = getattr(adapter, "_run_statuses", {})
    if isinstance(run_statuses, dict):
        for run_id, info in run_statuses.items():
            if isinstance(info, dict):
                st = info.get("status")
                run_prof = info.get("profile")
                if st in ("queued", "running", "in_progress", "stopping"):
                    if run_prof and normalize_profile_name(run_prof) == canon:
                        return _admin_error(f"Profile '{canon}' has active runs and cannot be deleted", status=409, code="profile_busy_conflict")

    # Preflight scan for unmanaged resources
    skills_map = manifest.get("skills", {})
    files_map = manifest.get("files", {})
    baseline_files = set(manifest.get("baseline_files", []))
    known_baselines = {MANIFEST_FILENAME, "profile.yaml", "config.yaml", ".env"}

    managed_skill_slugs = set(skills_map.keys())
    managed_file_paths = set(files_map.keys())

    for item in profile_dir.rglob("*"):
        if item.is_dir():
            continue

        try:
            rel = item.relative_to(profile_dir).as_posix()
        except ValueError:
            rel = item.relative_to(profile_dir).as_posix()

        if rel in managed_file_paths or rel in baseline_files or rel in known_baselines:
            continue

        parts = Path(rel).parts
        if len(parts) >= 2 and parts[0] == "skills" and parts[1] in managed_skill_slugs:
            continue

        return _admin_error(
            f"Profile '{canon}' contains unmanaged resource '{rel}' and cannot be deleted",
            status=409,
            code="profile_not_empty",
        )

    # Safe deletion
    try:
        shutil.rmtree(profile_dir)
    except Exception as exc:
        return _admin_error(f"Failed to delete profile directory '{canon}': {exc}", status=500, code="profile_deletion_failed")

    if profile_dir.exists():
        return _admin_error(f"Failed to fully delete profile directory '{canon}'", status=500, code="profile_deletion_failed")

    return _admin_json_response({"object": "hermes.admin.profile.deleted", "name": canon, "deleted": True})


# ---------------------------------------------------------------------------
# Skill Route Handlers
# ---------------------------------------------------------------------------

def _canonical_skill_repr(profile_name: str, skill_slug: str, skill_path: Path, content: str, revision: int = 1) -> dict:
    """Build canonical representation for a skill."""
    frontmatter, body = parse_frontmatter(content)
    digest = _sha256_digest(content)
    mtime = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(skill_path.stat().st_mtime if skill_path.exists() else time.time()))

    return {
        "object": "hermes.admin.skill",
        "profile": profile_name,
        "skill_slug": skill_slug,
        "revision": revision,
        "digest": digest,
        "metadata": frontmatter,
        "path": f"skills/{skill_slug}/SKILL.md",
        "content": content,
        "updated_at": mtime,
    }


async def handle_list_skills(adapter: Any, request: Any) -> Any:
    """GET /v1/admin/profiles/{profile}/skills — list skills under a profile."""
    guard = _check_admin_request(adapter, request)
    if guard:
        return guard

    profile_name = _get_profile_param(request)
    profile_dir = get_profile_dir(profile_name)
    caller_ownership = extract_caller_ownership(request)

    is_owned, err_code, manifest = verify_profile_ownership(profile_name, profile_dir, caller_ownership)
    if not is_owned or not manifest:
        if err_code == "missing_ownership":
            return _admin_error("Ownership parameters are required", status=400, code="missing_ownership")
        if err_code == "corrupt_manifest":
            return _admin_error(f"Profile '{profile_name}' has corrupt manifest", status=409, code="corrupt_manifest")
        return _admin_error(f"Profile '{profile_name}' not found or unowned", status=409 if err_code == "ownership_mismatch" else 404)

    skills_meta = manifest.get("skills", {})
    skill_list = []

    for skill_slug, meta in sorted(skills_meta.items()):
        if not isinstance(meta, dict) or not meta.get("managed"):
            continue
        md = profile_dir / "skills" / skill_slug / "SKILL.md"
        if md.is_file():
            try:
                content = md.read_text(encoding="utf-8")
                rev = meta.get("revision", 1)
                skill_list.append(_canonical_skill_repr(profile_name, skill_slug, md, content, revision=rev))
            except Exception:
                continue

    return _admin_json_response({"object": "list", "data": skill_list})


async def handle_create_update_skill(adapter: Any, request: Any) -> Any:
    """PUT/POST /v1/admin/profiles/{profile}/skills/{skill_slug} — create or update a skill."""
    guard = _check_admin_request(adapter, request)
    if guard:
        return guard

    body, body_err = await _parse_admin_body(request)
    if body_err:
        return body_err

    profile_name = _get_profile_param(request, body)
    skill_slug = request.match_info.get("skill_slug", "") or body.get("name") or body.get("skill_slug") or ""
    if not skill_slug or not isinstance(skill_slug, str) or not SLUG_RE.match(skill_slug):
        return _admin_error("Invalid skill slug", param="skill_slug", status=400)
    if len(skill_slug) > MAX_NAME_LEN:
        return _admin_error(f"skill_slug exceeds maximum length of {MAX_NAME_LEN}", status=400, code="payload_too_large")

    profile_dir = get_profile_dir(profile_name)
    caller_ownership = extract_caller_ownership(request, body)
    is_owned, err_code, manifest = verify_profile_ownership(profile_name, profile_dir, caller_ownership)
    if not is_owned or not manifest:
        if err_code == "missing_ownership":
            return _admin_error("Ownership parameters are required", status=400, code="missing_ownership")
        if err_code == "corrupt_manifest":
            return _admin_error(f"Profile '{profile_name}' has corrupt manifest", status=409, code="corrupt_manifest")
        return _admin_error(f"Profile '{profile_name}' not found or unowned", status=409 if err_code == "ownership_mismatch" else 404)

    content = body.get("content") or body.get("skill_md")
    if content is None:
        name = body.get("name") or skill_slug
        desc = body.get("description") or ""
        content = f"---\nname: {name}\ndescription: {desc}\n---\n# {name}\n"
    elif not isinstance(content, str):
        return _admin_error("content must be a string", status=400)

    # Frontmatter validation
    frontmatter, body_text = parse_frontmatter(content)
    fm_name = str(frontmatter.get("name") or "").strip()
    fm_desc = str(frontmatter.get("description") or "").strip()
    if not fm_name or not fm_desc:
        return _admin_error(
            "Skill frontmatter must include a non-empty name and description",
            status=400,
            code="invalid_skill",
        )
    if fm_name != skill_slug:
        return _admin_error(
            f"Skill frontmatter name '{fm_name}' must match path slug '{skill_slug}'",
            status=400,
            code="invalid_skill",
        )

    if len(content) > MAX_SKILL_CONTENT_LEN:
        return _admin_error(f"content exceeds maximum length of {MAX_SKILL_CONTENT_LEN}", status=400, code="payload_too_large")

    skill_md_path = profile_dir / "skills" / skill_slug / "SKILL.md"
    skill_dir = profile_dir / "skills" / skill_slug
    skills_meta = manifest.get("skills", {})
    skill_meta = skills_meta.get(skill_slug) or {}

    # Check unmanaged adoption conflict
    if (skill_md_path.exists() or skill_dir.exists()) and not skill_meta.get("managed"):
        return _admin_error(
            f"Skill '{skill_slug}' exists on disk but is not managed by this profile",
            status=409,
            code="unmanaged_resource_conflict",
        )

    desired_digest = _sha256_digest(content)
    current_digest = skill_meta.get("digest") or (_sha256_digest(skill_md_path.read_text(encoding="utf-8")) if skill_md_path.is_file() else "")

    if skill_md_path.is_file() and skill_meta.get("managed"):
        if_match_err = _check_if_match(request, current_digest)
        if if_match_err:
            return if_match_err

        # Idempotent repeat: if content digest is identical
        if current_digest == desired_digest:
            rev = skill_meta.get("revision", 1)
            return _admin_json_response(_canonical_skill_repr(profile_name, skill_slug, skill_md_path, content, revision=rev), status=200)

        next_rev = skill_meta.get("revision", 1) + 1
        is_new = False
    else:
        next_rev = 1
        is_new = True

    _atomic_write_file(skill_md_path, content, mode=0o600)

    skills_meta[skill_slug] = {
        "managed": True,
        "revision": next_rev,
        "digest": desired_digest,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest["skills"] = skills_meta
    write_ownership_manifest(profile_dir, manifest)

    _invalidate_skills_prompt_cache()
    return _admin_json_response(_canonical_skill_repr(profile_name, skill_slug, skill_md_path, content, revision=next_rev), status=201 if is_new else 200)


async def handle_get_skill(adapter: Any, request: Any) -> Any:
    """GET /v1/admin/profiles/{profile}/skills/{skill_slug} — read skill details."""
    guard = _check_admin_request(adapter, request)
    if guard:
        return guard

    profile_name = _get_profile_param(request)
    skill_slug = request.match_info.get("skill_slug", "")
    if not skill_slug or not SLUG_RE.match(skill_slug):
        return _admin_error("Invalid skill slug", param="skill_slug", status=400)

    profile_dir = get_profile_dir(profile_name)
    caller_ownership = extract_caller_ownership(request)
    is_owned, err_code, manifest = verify_profile_ownership(profile_name, profile_dir, caller_ownership)
    if not is_owned or not manifest:
        if err_code == "missing_ownership":
            return _admin_error("Ownership parameters are required", status=400, code="missing_ownership")
        if err_code == "corrupt_manifest":
            return _admin_error(f"Profile '{profile_name}' has corrupt manifest", status=409, code="corrupt_manifest")
        return _admin_error(f"Profile '{profile_name}' not found or unowned", status=409 if err_code == "ownership_mismatch" else 404)

    skills_meta = manifest.get("skills", {})
    skill_meta = skills_meta.get(skill_slug)
    if not skill_meta or not skill_meta.get("managed"):
        return _admin_error(f"Skill '{skill_slug}' not found under profile '{profile_name}'", status=404, code="skill_not_found")

    skill_md_path = profile_dir / "skills" / skill_slug / "SKILL.md"
    if not skill_md_path.is_file():
        return _admin_error(f"Skill '{skill_slug}' not found under profile '{profile_name}'", status=404, code="skill_not_found")

    content = skill_md_path.read_text(encoding="utf-8")
    rev = skill_meta.get("revision", 1)
    return _admin_json_response(_canonical_skill_repr(profile_name, skill_slug, skill_md_path, content, revision=rev))


async def handle_delete_skill(adapter: Any, request: Any) -> Any:
    """DELETE /v1/admin/profiles/{profile}/skills/{skill_slug} — delete a skill under profile."""
    guard = _check_admin_request(adapter, request)
    if guard:
        return guard

    profile_name = _get_profile_param(request)
    skill_slug = request.match_info.get("skill_slug", "")
    if not skill_slug or not SLUG_RE.match(skill_slug):
        return _admin_error("Invalid skill slug", param="skill_slug", status=400)

    profile_dir = get_profile_dir(profile_name)
    caller_ownership = extract_caller_ownership(request)
    is_owned, err_code, manifest = verify_profile_ownership(profile_name, profile_dir, caller_ownership)
    if not is_owned or not manifest:
        if err_code == "missing_ownership":
            return _admin_error("Ownership parameters are required", status=400, code="missing_ownership")
        if err_code == "corrupt_manifest":
            return _admin_error(f"Profile '{profile_name}' has corrupt manifest", status=409, code="corrupt_manifest")
        return _admin_error(f"Profile '{profile_name}' not found or unowned", status=409 if err_code == "ownership_mismatch" else 404)

    skills_meta = manifest.get("skills", {})
    if skill_slug not in skills_meta or not skills_meta[skill_slug].get("managed"):
        return _admin_error(f"Skill '{skill_slug}' not found under profile '{profile_name}'", status=404, code="skill_not_found")

    skill_dir = profile_dir / "skills" / skill_slug
    skill_md_path = skill_dir / "SKILL.md"
    if not skill_md_path.is_file():
        return _admin_error(f"Skill '{skill_slug}' not found under profile '{profile_name}'", status=404, code="skill_not_found")
    current_digest = _sha256_digest(skill_md_path.read_text(encoding="utf-8"))
    if_match_err = _check_if_match(request, current_digest)
    if if_match_err:
        return if_match_err

    if skill_dir.is_dir():
        try:
            shutil.rmtree(skill_dir)
        except Exception as exc:
            return _admin_error(f"Failed to delete skill '{skill_slug}': {exc}", status=500)

    skills_meta.pop(skill_slug, None)
    manifest["skills"] = skills_meta
    write_ownership_manifest(profile_dir, manifest)

    _invalidate_skills_prompt_cache()
    return _admin_json_response({"object": "hermes.admin.skill.deleted", "skill_slug": skill_slug, "deleted": True})


# ---------------------------------------------------------------------------
# File Route Handlers
# ---------------------------------------------------------------------------

def _canonical_file_repr(profile_name: str, rel_path: str, target_path: Path, content: Union[str, bytes], revision: int = 1) -> dict:
    """Build canonical representation for a managed file."""
    digest = _sha256_digest(content)
    size = len(content) if isinstance(content, bytes) else len(content.encode("utf-8"))
    mtime = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(target_path.stat().st_mtime if target_path.exists() else time.time()))

    return {
        "object": "hermes.admin.file",
        "profile": profile_name,
        "path": rel_path,
        "revision": revision,
        "digest": digest,
        "size": size,
        "content": content if isinstance(content, str) else content.decode("utf-8", errors="replace"),
        "updated_at": mtime,
    }


async def handle_list_files(adapter: Any, request: Any) -> Any:
    """GET /v1/admin/profiles/{profile}/files — list managed files under a profile."""
    guard = _check_admin_request(adapter, request)
    if guard:
        return guard

    profile_name = _get_profile_param(request)
    profile_dir = get_profile_dir(profile_name)
    caller_ownership = extract_caller_ownership(request)

    is_owned, err_code, manifest = verify_profile_ownership(profile_name, profile_dir, caller_ownership)
    if not is_owned or not manifest:
        if err_code == "missing_ownership":
            return _admin_error("Ownership parameters are required", status=400, code="missing_ownership")
        if err_code == "corrupt_manifest":
            return _admin_error(f"Profile '{profile_name}' has corrupt manifest", status=409, code="corrupt_manifest")
        return _admin_error(f"Profile '{profile_name}' not found or unowned", status=409 if err_code == "ownership_mismatch" else 404)

    files_meta = manifest.get("files", {})
    file_list = []

    for rel_path, meta in sorted(files_meta.items()):
        if not isinstance(meta, dict) or not meta.get("managed"):
            continue
        p = profile_dir / rel_path
        if p.is_file():
            try:
                content = p.read_text(encoding="utf-8")
                rev = meta.get("revision", 1)
                file_list.append(_canonical_file_repr(profile_name, rel_path, p, content, revision=rev))
            except Exception:
                continue

    return _admin_json_response({"object": "list", "data": file_list})


async def handle_create_update_file(adapter: Any, request: Any) -> Any:
    """PUT /v1/admin/profiles/{profile}/files/{path:.*} — create or update a managed file."""
    guard = _check_admin_request(adapter, request)
    if guard:
        return guard

    profile_name = _get_profile_param(request)
    rel_path_str = request.match_info.get("path", "")

    profile_dir = get_profile_dir(profile_name)
    caller_ownership = extract_caller_ownership(request)
    is_owned, err_code, manifest = verify_profile_ownership(profile_name, profile_dir, caller_ownership)
    if not is_owned or not manifest:
        if err_code == "missing_ownership":
            return _admin_error("Ownership parameters are required", status=400, code="missing_ownership")
        if err_code == "corrupt_manifest":
            return _admin_error(f"Profile '{profile_name}' has corrupt manifest", status=409, code="corrupt_manifest")
        return _admin_error(f"Profile '{profile_name}' not found or unowned", status=409 if err_code == "ownership_mismatch" else 404)

    valid, err_msg, target_path, status_code = validate_file_relative_path(profile_dir, rel_path_str)
    if not valid or not target_path:
        return _admin_error(err_msg or "Forbidden file path", status=status_code, code="path_forbidden")

    body, body_err = await _parse_admin_body(request)
    if body_err:
        return body_err

    content = body.get("content")
    if content is None:
        return _admin_error("content field is required", param="content", status=400)
    if not isinstance(content, str):
        return _admin_error("content must be a string", status=400)

    if len(content) > MAX_FILE_CONTENT_LEN:
        return _admin_error(f"content exceeds maximum length of {MAX_FILE_CONTENT_LEN}", status=400, code="payload_too_large")

    rel_posix = target_path.relative_to(profile_dir.resolve()).as_posix()
    files_meta = manifest.get("files", {})
    file_meta = files_meta.get(rel_posix) or {}

    # Check unmanaged adoption conflict
    if target_path.exists() and not file_meta.get("managed"):
        return _admin_error(
            f"File '{rel_posix}' exists on disk but is not managed by this profile",
            status=409,
            code="unmanaged_resource_conflict",
        )

    desired_digest = _sha256_digest(content)
    current_digest = file_meta.get("digest") or (_sha256_digest(target_path.read_text(encoding="utf-8")) if target_path.is_file() else "")

    if target_path.is_file() and file_meta.get("managed"):
        if_match_err = _check_if_match(request, current_digest)
        if if_match_err:
            return if_match_err

        # Idempotent repeat check
        if current_digest == desired_digest:
            rev = file_meta.get("revision", 1)
            return _admin_json_response(_canonical_file_repr(profile_name, rel_posix, target_path, content, revision=rev), status=200)

        next_rev = file_meta.get("revision", 1) + 1
        is_new = False
    else:
        next_rev = 1
        is_new = True

    _atomic_write_file(target_path, content, mode=0o600)

    files_meta[rel_posix] = {
        "managed": True,
        "revision": next_rev,
        "digest": desired_digest,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest["files"] = files_meta

    # If updating SOUL.md or memories/USER.md, update profile spec_digest too
    if rel_posix in ("SOUL.md", "memories/USER.md"):
        readback_repr = _canonical_profile_repr(profile_name, profile_dir, manifest)
        manifest["spec_digest"] = readback_repr["digest"]

    write_ownership_manifest(profile_dir, manifest)
    return _admin_json_response(_canonical_file_repr(profile_name, rel_posix, target_path, content, revision=next_rev), status=201 if is_new else 200)


async def handle_get_file(adapter: Any, request: Any) -> Any:
    """GET /v1/admin/profiles/{profile}/files/{path:.*} — read a managed file."""
    guard = _check_admin_request(adapter, request)
    if guard:
        return guard

    profile_name = _get_profile_param(request)
    rel_path_str = request.match_info.get("path", "")

    profile_dir = get_profile_dir(profile_name)
    caller_ownership = extract_caller_ownership(request)
    is_owned, err_code, manifest = verify_profile_ownership(profile_name, profile_dir, caller_ownership)
    if not is_owned or not manifest:
        if err_code == "missing_ownership":
            return _admin_error("Ownership parameters are required", status=400, code="missing_ownership")
        if err_code == "corrupt_manifest":
            return _admin_error(f"Profile '{profile_name}' has corrupt manifest", status=409, code="corrupt_manifest")
        return _admin_error(f"Profile '{profile_name}' not found or unowned", status=409 if err_code == "ownership_mismatch" else 404)

    valid, err_msg, target_path, status_code = validate_file_relative_path(profile_dir, rel_path_str)
    if not valid or not target_path:
        return _admin_error(err_msg or "Forbidden file path", status=status_code, code="path_forbidden")

    rel_posix = target_path.relative_to(profile_dir.resolve()).as_posix()
    files_meta = manifest.get("files", {})
    file_meta = files_meta.get(rel_posix)

    if not file_meta or not file_meta.get("managed") or not target_path.is_file():
        return _admin_error(f"File '{rel_path_str}' not found", status=404, code="file_not_found")

    content = target_path.read_text(encoding="utf-8")
    rev = file_meta.get("revision", 1)
    return _admin_json_response(_canonical_file_repr(profile_name, rel_posix, target_path, content, revision=rev))


async def handle_delete_file(adapter: Any, request: Any) -> Any:
    """DELETE /v1/admin/profiles/{profile}/files/{path:.*} — delete a managed file."""
    guard = _check_admin_request(adapter, request)
    if guard:
        return guard

    profile_name = _get_profile_param(request)
    rel_path_str = request.match_info.get("path", "")

    profile_dir = get_profile_dir(profile_name)
    caller_ownership = extract_caller_ownership(request)
    is_owned, err_code, manifest = verify_profile_ownership(profile_name, profile_dir, caller_ownership)
    if not is_owned or not manifest:
        if err_code == "missing_ownership":
            return _admin_error("Ownership parameters are required", status=400, code="missing_ownership")
        if err_code == "corrupt_manifest":
            return _admin_error(f"Profile '{profile_name}' has corrupt manifest", status=409, code="corrupt_manifest")
        return _admin_error(f"Profile '{profile_name}' not found or unowned", status=409 if err_code == "ownership_mismatch" else 404)

    valid, err_msg, target_path, status_code = validate_file_relative_path(profile_dir, rel_path_str)
    if not valid or not target_path:
        return _admin_error(err_msg or "Forbidden file path", status=status_code, code="path_forbidden")

    rel_posix = target_path.relative_to(profile_dir.resolve()).as_posix()
    files_meta = manifest.get("files", {})

    if rel_posix not in files_meta or not files_meta[rel_posix].get("managed"):
        return _admin_error(f"File '{rel_path_str}' not found", status=404, code="file_not_found")

    if not target_path.is_file():
        return _admin_error(f"File '{rel_path_str}' not found", status=404, code="file_not_found")
    current_digest = _sha256_digest(target_path.read_text(encoding="utf-8"))
    if_match_err = _check_if_match(request, current_digest)
    if if_match_err:
        return if_match_err

    if target_path.is_file():
        try:
            target_path.unlink()
        except Exception as exc:
            return _admin_error(f"Failed to delete file '{rel_posix}': {exc}", status=500)

    files_meta.pop(rel_posix, None)
    manifest["files"] = files_meta

    if rel_posix in ("SOUL.md", "memories/USER.md"):
        readback_repr = _canonical_profile_repr(profile_name, profile_dir, manifest)
        manifest["spec_digest"] = readback_repr["digest"]

    write_ownership_manifest(profile_dir, manifest)

    return _admin_json_response({"object": "hermes.admin.file.deleted", "path": rel_posix, "deleted": True})
