"""Read-only and safe mutation services for generated channel skills."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import logging
from pathlib import Path
import re
import shutil
from typing import Any

from agent.skill_utils import parse_frontmatter
from gateway.channel_skill_generation import (
    ChannelSkillRegistry,
    _create_skill_via_manager,
    _generate_with_auxiliary_llm,
    _validated_or_fallback,
)
from hermes_constants import get_hermes_home
from gateway.channel_skills import ChannelSkillResolver, ChannelSkillResolutionError

logger = logging.getLogger(__name__)
_SAFE_CHANNEL_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def _safe_channel_id(channel_id: str) -> bool:
    return bool(_SAFE_CHANNEL_ID.fullmatch(str(channel_id)))


def _find_skill(name: str):
    from tools.skill_manager_tool import _find_skill as find_skill
    return find_skill(name)


class ChannelSkillAdmin:
    """Administrative operations over the generated channel-skill registry."""

    def __init__(self, registry: ChannelSkillRegistry | None = None, *, config_extra: dict[str, Any] | None = None) -> None:
        self.registry = registry or ChannelSkillRegistry()
        self.config_extra = config_extra if isinstance(config_extra, dict) else {}

    def list(self, *, status: str | None = None, include_static: bool = False,
             source: str | None = None, effective: bool = False) -> list[dict[str, Any]]:
        records = self.registry.load().get("channels", {})
        result = []
        for channel_id, record in records.items():
            if not isinstance(record, dict):
                continue
            item = self._status_record(str(channel_id), record)
            item["source"] = "generated"
            item["binding_type"] = "generated"
            item["effective_skills"] = list(record.get("resolved_skills") or ([record.get("skill")] if record.get("skill") else []))
            if status and item["status"] != status:
                continue
            if source and source != "generated":
                continue
            result.append(item)
        if include_static or source == "static":
            for binding in self._static_bindings():
                item = self._static_status_record(binding)
                if status and item["status"] != status:
                    continue
                if source and source != "static":
                    continue
                duplicate_index = next((index for index, existing in enumerate(result)
                                        if existing["channel_id"] == binding.channel_id), None)
                if duplicate_index is None:
                    result.append(item)
                else:
                    item["overridden_generated"] = result[duplicate_index]
                    result[duplicate_index] = item
        if effective:
            for item in result:
                item["effective_skills"] = self.effective(item["channel_id"])["effective_skills"]
        return sorted(result, key=lambda item: item["channel_id"])

    def show(self, channel_id: str) -> dict[str, Any]:
        for binding in self._static_bindings():
            if binding.channel_id == str(channel_id):
                item = self._static_status_record(binding)
                generated = self.registry.get(channel_id)
                if generated is not None:
                    item["overridden_generated"] = self._status_record(str(channel_id), generated)
                return item
        record = self.registry.get(channel_id)
        if record is not None:
            result = self._status_record(str(channel_id), record)
            result.update({"source": "generated", "binding_type": "generated"})
            result["effective_skills"] = list(record.get("resolved_skills") or ([record.get("skill")] if record.get("skill") else []))
            return result
        for binding in self._static_bindings():
            if binding.channel_id == str(channel_id):
                return self._static_status_record(binding)
        raise KeyError(f"Unknown channel skill: {channel_id}")

    def effective(self, channel_id: str, parent_id: str | None = None) -> dict[str, Any]:
        resolver = ChannelSkillResolver(self.config_extra)
        resolved = resolver.resolve(str(channel_id), parent_id)
        if resolved:
            return {
                "channel_id": str(channel_id),
                "effective_skills": list(resolved.skills),
                "profile_chain": list(resolved.profile_chain),
                "matched_channel_id": resolved.matched_channel_id,
                "inherited_from_parent": resolved.inherited_from_parent,
                "source": "static",
            }
        record = self.registry.get(channel_id)
        if record:
            return {
                "channel_id": str(channel_id),
                "effective_skills": list(record.get("resolved_skills") or ([record.get("skill")] if record.get("skill") else [])),
                "profile_chain": [],
                "matched_channel_id": str(channel_id),
                "inherited_from_parent": False,
                "source": "generated",
            }
        return {
            "channel_id": str(channel_id), "effective_skills": [], "profile_chain": [],
            "matched_channel_id": None, "inherited_from_parent": False, "source": "none",
        }

    def validate(self) -> dict[str, Any]:
        issues: list[dict[str, Any]] = []
        resolver = ChannelSkillResolver(self.config_extra)
        for binding in self._static_bindings():
            try:
                skills = resolver.resolve_profile_skills(binding.profile) if binding.profile else list(binding.skills)
            except ChannelSkillResolutionError as exc:
                issues.append({"channel_id": binding.channel_id, "type": "invalid_profile", "message": str(exc)})
                continue
            for skill in skills:
                found = _find_skill(skill)
                try:
                    skill_path = Path(found["path"]) if found and found.get("path") else None
                    exists = bool(skill_path and (skill_path / "SKILL.md").is_file())
                except (TypeError, ValueError, OSError):
                    exists = False
                if not exists:
                    issues.append({"channel_id": binding.channel_id, "type": "missing_skill", "skill": skill})
        for item in self.list():
            if item["status"] in {"missing", "corrupt"}:
                issues.append({"channel_id": item["channel_id"], "type": item["status"]})
        return {"valid": not issues, "issues": issues}

    def inspect(self, channel_id: str) -> dict[str, Any]:
        info = self.show(channel_id)
        path = info["file"].get("path")
        if path and info["file"].get("exists"):
            info["content"] = Path(path).read_text(encoding="utf-8")
        else:
            info["content"] = None
        return info

    def remove(self, channel_id: str, *, delete_skill: bool = False) -> dict[str, Any]:
        record = self.registry.get(channel_id)
        if record is None:
            raise KeyError(f"Unknown channel skill: {channel_id}")
        skill_name = str(record.get("skill") or "")
        deleted = False
        with self.registry.lock(str(channel_id)):
            record = self.registry.get(channel_id)
            if record is None:
                raise KeyError(f"Unknown channel skill: {channel_id}")
            if delete_skill:
                path = self._safe_generated_skill_path(skill_name, str(channel_id))
                if path is None:
                    raise ValueError("Refusing to delete a skill that is not a generated channel skill")
                if path.exists():
                    path.unlink()
                    try:
                        path.parent.rmdir()
                    except OSError:
                        pass
                    deleted = True
            self.registry.delete(channel_id)
        return {"channel_id": str(channel_id), "detached": True, "skill_deleted": deleted}

    def regenerate(self, channel_id: str, *, force: bool = False, dry_run: bool = False,
                   backup: bool = False) -> dict[str, Any]:
        if not _safe_channel_id(str(channel_id)):
            raise ValueError("Invalid channel ID for channel-skill regeneration")
        record = self.registry.get(channel_id)
        if record is None:
            raise KeyError(f"Unknown channel skill: {channel_id}")
        skill_name = str(record.get("skill") or f"discord-channel-{channel_id}")
        expected_skill_name = f"discord-channel-{channel_id}"
        if skill_name != expected_skill_name:
            raise ValueError("Refusing to regenerate a non-generated channel skill")
        current = self.show(channel_id)
        if current["status"] == "manual_override" and not force:
            raise ValueError("Skill was manually edited; use --force to overwrite it")
        base_skills = [str(value) for value in (record.get("base_skills") or [])]
        content = _generate_with_auxiliary_llm(
            skill_name=skill_name,
            channel_id=str(channel_id),
            channel_name=str(record.get("channel_name") or ""),
            topic=str(record.get("topic") or ""),
            parent_id=record.get("parent_id"),
            initial_message="",
            base_skills=base_skills,
            timeout_seconds=30,
        )
        content = _validated_or_fallback(content, skill_name, str(channel_id), base_skills)
        if dry_run:
            return {"channel_id": str(channel_id), "skill": skill_name, "changed": True, "content": content}
        backup_path = None
        with self.registry.lock(str(channel_id)):
            found = _find_skill(skill_name)
            safe_skill_md = None
            if found and found.get("path"):
                safe_skill_md = self._safe_generated_skill_path(skill_name, str(channel_id))
                if safe_skill_md is None:
                    raise ValueError("Refusing to regenerate a skill outside the generated channel-skill path")
            if backup and safe_skill_md is not None:
                current_path = safe_skill_md
                if current_path.is_file():
                    backup_root = (self.registry.path.parent / "channel-skills" / "backups").resolve()
                    backup_dir = (backup_root / str(channel_id)).resolve()
                    if not backup_dir.is_relative_to(backup_root):
                        raise ValueError("Invalid backup path")
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
                    backup_target = backup_dir / f"{stamp}-SKILL.md"
                    shutil.copy2(current_path, backup_target)
                    backup_path = str(backup_target)
            if found and found.get("path"):
                from tools.skill_manager_tool import _edit_skill
                result = _edit_skill(skill_name, content)
            else:
                result = _create_skill_via_manager(skill_name, content)
            if not result.get("success"):
                raise OSError(str(result))
            updated = dict(record)
            updated.update({
                "status": "ready",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "manual_override": False,
                "content_sha256": "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "skill_path": result.get("skill_md"),
            })
            self.registry.put(str(channel_id), updated)
        return {"channel_id": str(channel_id), "skill": skill_name, "changed": True, "backup_path": backup_path}

    def repair(self, *, apply: bool = False) -> dict[str, Any]:
        issues = []
        for item in self.list():
            if item["status"] != "ready":
                issues.append({
                    "channel_id": item["channel_id"],
                    "status": item["status"],
                    "action": "inspect_or_regenerate",
                })
            elif item["file"].get("hash") and item["record"].get("content_sha256") \
                    and item["file"]["hash"] != item["record"]["content_sha256"]:
                issues.append({
                    "channel_id": item["channel_id"],
                    "status": "manual_override",
                    "action": "regenerate_with_force",
                })
        if apply:
            for issue in issues:
                if issue["status"] == "generating":
                    record = self.registry.get(issue["channel_id"])
                    if record:
                        record = dict(record)
                        record.update({"status": "failed", "updated_at": datetime.now(timezone.utc).isoformat()})
                        self.registry.put(issue["channel_id"], record)
            return {"applied": True, "issues": issues}
        return {"applied": False, "issues": issues}

    def _status_record(self, channel_id: str, record: dict[str, Any]) -> dict[str, Any]:
        skill_name = str(record.get("skill") or "")
        file_info = self._file_info(skill_name, channel_id, record)
        status = _derive_status(record, file_info)
        return {
            "channel_id": channel_id,
            "platform": record.get("platform"),
            "profile": record.get("profile"),
            "status": status,
            "record": record,
            "file": file_info,
        }

    def _static_bindings(self):
        return ChannelSkillResolver(self.config_extra).bindings

    def _static_status_record(self, binding) -> dict[str, Any]:
        skills = list(binding.skills)
        if binding.profile:
            try:
                skills = ChannelSkillResolver(self.config_extra).resolve_profile_skills(binding.profile)
            except ChannelSkillResolutionError:
                skills = []
        primary = skills[-1] if skills else ""
        file_info = self._file_info(primary, binding.channel_id, {"skill": primary}, require_channel_id=False) if primary else {
            "path": None, "exists": False, "frontmatter": "missing", "name": "missing",
            "channel_id": "missing", "hash": None,
        }
        return {
            "channel_id": binding.channel_id,
            "platform": "discord",
            "profile": binding.profile,
            "status": "configured" if skills else "invalid",
            "source": "static",
            "binding_type": "profile" if binding.profile else "exact",
            "skills": skills,
            "effective_skills": skills,
            "record": {"skills": skills, "profile": binding.profile, "generated": False},
            "file": file_info,
        }

    def _file_info(self, skill_name: str, channel_id: str, record: dict[str, Any], *, require_channel_id: bool = True) -> dict[str, Any]:
        try:
            found = _find_skill(skill_name) if skill_name else None
            raw_skill_dir = found.get("path") if found else None
            skill_dir = Path(raw_skill_dir) if raw_skill_dir else None
            skill_md = skill_dir / "SKILL.md" if skill_dir else None
        except (TypeError, ValueError, OSError):
            return {
                "path": None, "exists": False, "frontmatter": "invalid", "name": "invalid",
                "channel_id": "invalid", "hash": None,
            }
        info: dict[str, Any] = {
            "path": str(skill_md) if skill_md else None,
            "exists": bool(skill_md and skill_md.is_file()),
            "frontmatter": "missing",
            "name": "missing",
            "channel_id": "missing",
            "hash": None,
        }
        if not info["exists"] or skill_md is None:
            return info
        try:
            content = skill_md.read_text(encoding="utf-8")
            frontmatter, body = parse_frontmatter(content)
            info["hash"] = "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()
            info["frontmatter"] = "valid" if frontmatter and body.strip() else "invalid"
            info["name"] = "matched" if frontmatter.get("name") == skill_name else "mismatched"
            info["channel_id"] = (
                "not_applicable" if not require_channel_id else (
                    "matched" if str(frontmatter.get("metadata", {}).get("hermes", {}).get("channel_id")) == channel_id
                    else "mismatched"
                )
            )
        except (OSError, UnicodeError, ValueError, TypeError):
            logger.warning("Failed to inspect channel skill %s", skill_md, exc_info=True)
            info["frontmatter"] = "invalid"
        return info

    def _safe_generated_skill_path(self, skill_name: str, channel_id: str) -> Path | None:
        expected = f"discord-channel-{channel_id}"
        if skill_name != expected:
            return None
        found = _find_skill(skill_name)
        if not found or not found.get("path"):
            return None
        try:
            skill_dir = Path(found["path"]).resolve()
        except (TypeError, ValueError, OSError):
            return None
        skill_root = (get_hermes_home() / "skills").resolve()
        if not skill_dir.is_relative_to(skill_root):
            return None
        skill_md = skill_dir / "SKILL.md"
        if skill_md.is_symlink():
            return None
        info = self._file_info(skill_name, channel_id, {"skill": skill_name})
        if info["frontmatter"] != "valid" or info["name"] != "matched" or info["channel_id"] != "matched":
            return None
        return skill_md

    def _remove_registry_record(self, channel_id: str) -> None:
        data = self.registry.load()
        data.get("channels", {}).pop(str(channel_id), None)
        from utils import atomic_write_text
        import hermes_yaml as yaml
        atomic_write_text(
            self.registry.path,
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
            create_mode=0o600,
        )


def _derive_status(record: dict[str, Any], file_info: dict[str, Any]) -> str:
    if record.get("status") == "generating":
        return "generating"
    if not file_info["exists"]:
        return "missing"
    if file_info["frontmatter"] != "valid" or file_info["name"] != "matched" or file_info["channel_id"] != "matched":
        return "corrupt"
    expected_hash = record.get("content_sha256")
    if expected_hash and expected_hash != file_info.get("hash"):
        return "manual_override"
    return "ready"
