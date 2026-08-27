"""Portable strict-persona bundle validation for Hermes.

Skill Foundry exports a self-contained directory containing ``persona.yaml``,
``PACK.json`` and every member skill. Hermes treats that directory as
untrusted until both content and activation approvals are bound to the exact
pack and tree digests in the profile-local approval registry.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from hermes_constants import get_hermes_home

MANIFEST_NAME = "persona.yaml"
DEFAULT_MAX_PROMPT_BYTES = 96 * 1024
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "strict",
        "name",
        "description",
        "pack_digest",
        "persona_tree_digest",
        "approvals",
        "members",
        "instruction",
        "instruction_digest",
    }
)


class StrictPersonaError(ValueError):
    """A strict persona failed its portable runtime contract."""


def approval_registry_path() -> Path:
    """Return the fixed profile-local persona approval authority path."""
    return get_hermes_home() / "approvals" / "persona-bundles.json"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or not _DIGEST_RE.fullmatch(value):
        raise StrictPersonaError(f"{label} is not a SHA-256 digest")
    return value


def _validate_containment(root: Path, bundles_root: Path) -> Path:
    if root.is_symlink():
        raise StrictPersonaError("strict persona root must not be a symlink")
    try:
        resolved_root = root.resolve(strict=True)
        resolved_bundles = bundles_root.resolve(strict=True)
        resolved_root.relative_to(resolved_bundles)
    except (OSError, ValueError) as exc:
        raise StrictPersonaError("strict persona root escapes the bundle directory") from exc
    return resolved_root


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    manifest_path = root / MANIFEST_NAME
    try:
        paths = sorted(root.rglob("*"), key=lambda item: item.as_posix())
        for path in paths:
            if path.is_symlink():
                raise StrictPersonaError(
                    f"symlink is forbidden in strict persona: {path}"
                )
            if path == manifest_path or not path.is_file():
                continue
            relative = path.relative_to(root).as_posix().encode()
            data = path.read_bytes()
            digest.update(len(relative).to_bytes(4, "big"))
            digest.update(relative)
            digest.update(len(data).to_bytes(8, "big"))
            digest.update(data)
    except OSError as exc:
        raise StrictPersonaError("strict persona tree cannot be read safely") from exc
    return digest.hexdigest()


def _validate_pack(pack: object) -> Dict[str, Any]:
    if not isinstance(pack, dict) or pack.get("kind") != "persona":
        raise StrictPersonaError("strict persona PACK.json is invalid")
    for key in ("name", "description", "persona_name", "mission", "voice"):
        if not isinstance(pack.get(key), str) or not pack[key].strip():
            raise StrictPersonaError(f"strict persona PACK.json field is invalid: {key}")
    if not _NAME_RE.fullmatch(pack["name"]):
        raise StrictPersonaError("strict persona PACK.json name is invalid")

    members = pack.get("member_skills")
    if not isinstance(members, list) or not members:
        raise StrictPersonaError("strict persona PACK.json has no members")
    names: List[str] = []
    for member in members:
        if not isinstance(member, dict) or set(member) != {"skill", "role"}:
            raise StrictPersonaError("strict persona member contract is invalid")
        name = member.get("skill")
        role = member.get("role")
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
            raise StrictPersonaError("strict persona member name is invalid")
        if not isinstance(role, str) or not role.strip():
            raise StrictPersonaError("strict persona member role is invalid")
        names.append(name)
    if len(names) != len(set(names)):
        raise StrictPersonaError("strict persona member names are duplicated")

    rules = pack.get("composition_rules")
    if not isinstance(rules, list) or not rules or not all(
        isinstance(rule, str) and rule.strip() for rule in rules
    ):
        raise StrictPersonaError("strict persona composition rules are invalid")
    handoffs = pack.get("handoffs")
    if not isinstance(handoffs, dict) or not all(
        isinstance(name, str)
        and name.strip()
        and isinstance(trigger, str)
        and trigger.strip()
        for name, trigger in handoffs.items()
    ):
        raise StrictPersonaError("strict persona handoffs are invalid")
    return pack


def build_persona_instruction(pack: Dict[str, Any]) -> str:
    """Render every behavioural PACK field deterministically."""
    pack = _validate_pack(pack)
    lines = [
        f"Persona: {pack['persona_name']}",
        f"Persona mission: {pack['mission']}",
        f"Voice and behaviour: {pack['voice']}",
        "",
        "Member roles:",
    ]
    for member in pack["member_skills"]:
        lines.append(f"- {member['skill']}: {member['role']}")
    lines.extend(["", "Composition rules:"])
    for index, rule in enumerate(pack["composition_rules"], start=1):
        lines.append(f"{index}. {rule}")
    lines.extend(["", "Handoffs:"])
    for name, trigger in pack["handoffs"].items():
        lines.append(f"- {name}: {trigger}")
    lines.extend(
        [
            "",
            "Runtime contract:",
            "- Route work to the smallest relevant member set.",
            "- Apply composition rules before conflicting member guidance.",
            "- Follow declared handoffs and stop rather than inventing an undeclared one.",
            "- Never continue if a required member is missing, disabled, or digest-mismatched.",
        ]
    )
    return "\n".join(lines).strip()


def _load_approval_registry(path: Path) -> Mapping[str, Mapping[str, str]]:
    if not path.is_file() or path.is_symlink():
        raise StrictPersonaError("persona approval registry is missing or unsafe")
    try:
        resolved_path = path.resolve(strict=True)
        resolved_home = get_hermes_home().resolve(strict=True)
        resolved_path.relative_to(resolved_home)
    except (OSError, ValueError) as exc:
        raise StrictPersonaError(
            "persona approval registry escapes the active Hermes home"
        ) from exc
    if os.name != "nt":
        stat = path.stat()
        if stat.st_uid != os.getuid() or stat.st_mode & 0o022:
            raise StrictPersonaError("persona approval registry ownership or mode is unsafe")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StrictPersonaError("persona approval registry is invalid") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise StrictPersonaError("persona approval registry schema is invalid")
    approved = data.get("approved")
    if not isinstance(approved, dict):
        raise StrictPersonaError("persona approval registry has no approved mapping")
    return approved


def _require_approval(
    manifest: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, str]],
    key: str,
    purpose: str,
) -> None:
    approvals = manifest.get("approvals")
    approval = approvals.get(key) if isinstance(approvals, Mapping) else None
    if not isinstance(approval, Mapping) or approval.get("purpose") != purpose:
        raise StrictPersonaError(f"strict persona {key} approval contract is invalid")
    digest = _validate_digest(approval.get("bundle_digest"), f"{key} approval digest")
    expected = {
        "purpose": purpose,
        "persona": manifest["name"],
        "pack_digest": manifest["pack_digest"],
        "persona_tree_digest": manifest["persona_tree_digest"],
    }
    authority = registry.get(digest)
    if not isinstance(authority, Mapping) or dict(authority) != expected:
        raise StrictPersonaError(f"strict persona {key} approval is not authorized")


def load_strict_persona_bundle(
    manifest_path: Path,
    *,
    bundles_root: Path,
    registry_path: Optional[Path] = None,
    include_member_content: bool = False,
) -> Dict[str, Any]:
    """Verify one canonical Skill Foundry export and return bundle metadata."""
    manifest_path = Path(manifest_path)
    if manifest_path.name != MANIFEST_NAME or manifest_path.is_symlink():
        raise StrictPersonaError("strict persona manifest path is invalid")
    root = _validate_containment(manifest_path.parent, Path(bundles_root))
    try:
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise StrictPersonaError("strict persona manifest is invalid") from exc
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_FIELDS:
        raise StrictPersonaError("strict persona manifest fields are invalid")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "persona"
        or manifest.get("strict") is not True
    ):
        raise StrictPersonaError("strict persona manifest contract is invalid")
    name = manifest.get("name")
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise StrictPersonaError("strict persona name is invalid")

    pack_path = root / "PACK.json"
    if not pack_path.is_file() or pack_path.is_symlink():
        raise StrictPersonaError("strict persona PACK.json is missing or unsafe")
    try:
        pack_bytes = pack_path.read_bytes()
    except OSError as exc:
        raise StrictPersonaError("strict persona PACK.json cannot be read") from exc
    pack_digest = _validate_digest(manifest.get("pack_digest"), "pack digest")
    if _sha256(pack_bytes) != pack_digest:
        raise StrictPersonaError("strict persona PACK.json digest mismatch")
    try:
        pack = _validate_pack(json.loads(pack_bytes))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StrictPersonaError("strict persona PACK.json is invalid") from exc
    if pack["name"] != name:
        raise StrictPersonaError("strict persona PACK.json name mismatch")

    instruction = build_persona_instruction(pack)
    if manifest.get("instruction") != instruction:
        raise StrictPersonaError("strict persona instruction does not match PACK.json")
    instruction_digest = _validate_digest(
        manifest.get("instruction_digest"), "instruction digest"
    )
    if _sha256(instruction.encode()) != instruction_digest:
        raise StrictPersonaError("strict persona instruction digest mismatch")

    expected_members = pack["member_skills"]
    members = manifest.get("members")
    if not isinstance(members, list) or len(members) != len(expected_members):
        raise StrictPersonaError("strict persona member set is invalid")
    loaded_members: List[str] = []
    roles: Dict[str, str] = {}
    member_contents: Dict[str, str] = {}
    for manifest_member, pack_member in zip(members, expected_members):
        if not isinstance(manifest_member, dict) or set(manifest_member) != {
            "skill",
            "role",
            "digest",
        }:
            raise StrictPersonaError("strict persona member manifest is invalid")
        name_value = manifest_member.get("skill")
        role = manifest_member.get("role")
        if not isinstance(name_value, str) or not _NAME_RE.fullmatch(name_value):
            raise StrictPersonaError("strict persona member name is invalid")
        if not isinstance(role, str) or not role.strip():
            raise StrictPersonaError("strict persona member role is invalid")
        if name_value != pack_member["skill"] or role != pack_member["role"]:
            raise StrictPersonaError("strict persona member order or role mismatch")
        digest = _validate_digest(manifest_member.get("digest"), "member digest")
        member_dir = root / "skills" / name_value
        skill_md = member_dir / "SKILL.md"
        if member_dir.is_symlink() or not skill_md.is_file() or skill_md.is_symlink():
            raise StrictPersonaError(f"strict persona member is missing or unsafe: {name_value}")
        try:
            member_bytes = skill_md.read_bytes()
        except OSError as exc:
            raise StrictPersonaError(
                f"strict persona member cannot be read: {name_value}"
            ) from exc
        if _sha256(member_bytes) != digest:
            raise StrictPersonaError(f"strict persona member digest mismatch: {name_value}")
        if include_member_content:
            try:
                member_contents[name_value] = member_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise StrictPersonaError(
                    f"strict persona member is not UTF-8: {name_value}"
                ) from exc
        loaded_members.append(name_value)
        roles[name_value] = role

    tree_digest = _validate_digest(manifest.get("persona_tree_digest"), "tree digest")
    if _tree_digest(root) != tree_digest:
        raise StrictPersonaError("strict persona tree digest mismatch")

    registry = _load_approval_registry(registry_path or approval_registry_path())
    _require_approval(manifest, registry, "content", "persona-content")
    _require_approval(manifest, registry, "activation", "persona-activation")
    return {
        "name": name,
        "slug": name,
        "description": str(manifest.get("description") or "").strip(),
        "skills": loaded_members,
        "roles": roles,
        "member_contents": member_contents,
        "instruction": instruction,
        "path": str(manifest_path),
        "persona_root": str(root),
        "strict": True,
        "kind": "persona",
    }


def build_strict_persona_invocation(
    info: Mapping[str, Any],
    *,
    bundles_root: Path,
    disabled_names: set[str],
    user_instruction: str = "",
    task_id: Optional[str] = None,
    max_prompt_bytes: int = DEFAULT_MAX_PROMPT_BYTES,
) -> Tuple[str, List[str], List[str]]:
    """Revalidate and render a complete strict persona invocation."""
    verified = load_strict_persona_bundle(
        Path(str(info["path"])),
        bundles_root=bundles_root,
        include_member_content=True,
    )
    members = list(verified["skills"])
    disabled = [name for name in members if name in disabled_names]
    if disabled:
        raise StrictPersonaError(
            "required strict persona members are disabled: " + ", ".join(disabled)
        )

    blocks = [
        f"## Active member: {name}\nRole: {verified['roles'][name]}\n\n"
        + verified["member_contents"][name]
        for name in members
    ]
    header = [
        "[STRICT PERSONA — content and activation approvals verified; all member and package digests verified]",
        "",
        verified["instruction"],
    ]
    if user_instruction:
        header.extend(["", f"User instruction: {user_instruction}"])
    message = "\n\n".join(["\n".join(header), *blocks])
    if isinstance(max_prompt_bytes, bool) or max_prompt_bytes <= 0:
        raise StrictPersonaError("persona prompt limit must be a positive integer")
    if len(message.encode()) > max_prompt_bytes:
        raise StrictPersonaError("strict persona prompt exceeds the runtime byte limit")

    try:
        from tools.skill_usage import bump_use

        for name in members:
            bump_use(name, task_id=task_id)
    except Exception:
        pass
    return message, members, []
