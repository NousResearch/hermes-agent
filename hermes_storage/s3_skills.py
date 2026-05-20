"""
hermes_storage/s3_skills.py — S3-backed skill artifact storage for SaaS-mode Hermes.

Provides S3SkillSource: reads/writes skill manifests (SKILL.md) and their
supporting files to S3 using tenant-prefixed keys for isolation.

Bucket scheme: single shared bucket with tenant-scoped key prefixes.
  Key format: hermes-skills/{tenant_slug}/{skill_name}/{file_path}

Rationale for prefix-per-tenant (vs per-tenant bucket):
  - S3 bucket creation requires AWS API calls (slow, has per-account limit).
  - Prefix-based isolation is just as secure with per-tenant IAM policy conditions
    (s3:prefix condition key).
  - Ops simpler: one bucket, one CloudFormation resource.
  - Downside: a single misconfigured IAM policy could expose all tenants.
    Mitigated by strict per-tenant IAM conditions and S3 Block Public Access.

Assumption surfaced:
  - HERMES_MODE=saas and a bucket name in env var S3_SKILLS_BUCKET (or default).
  - boto3 credentials available via instance role (Fargate) or env vars (dev).
  - Skill artifacts are text files (SKILL.md, references/*.md, templates/*.md).
    Binary assets (images, binaries) are not in scope for Phase D.
  - The local skills/ directory is used as a read-through cache: on cache miss,
    S3SkillSource downloads to a tmp dir and the caller reads from there.
  - Upload is explicit (push_skill): skills are NOT automatically synced from
    local → S3. This is intentional: skill publishing is a deliberate action.

Failure modes:
  - S3 bucket not found: raises ValueError with clear message + bucket name.
  - Object not found: list_skills returns [] (not an error); get_skill_file
    raises KeyError if the SKILL.md is missing.
  - Network timeout: boto3 default timeout applies; callers should retry.
  - Malformed SKILL.md (downloaded): returned as-is; validation is the caller's job.

Key implementation constraints:
  - s3_skills.py must NOT import from tools/ (circular dependency risk).
  - boto3 is a lazy import (only needed when HERMES_MODE=saas).
  - All S3 operations are synchronous (boto3 is sync); async wrappers use
    asyncio.to_thread to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional

logger = logging.getLogger(__name__)

# Default bucket name — override via S3_SKILLS_BUCKET env var.
DEFAULT_BUCKET = "hermes-saas-skills"

# Key prefix template: hermes-skills/{tenant_slug}/{skill_name}/{file}
_KEY_PREFIX = "hermes-skills"


# ---------------------------------------------------------------------------
# S3SkillSource
# ---------------------------------------------------------------------------

class S3SkillSource:
    """
    S3-backed skill artifact store.

    Usage::

        source = S3SkillSource(tenant_slug="slack_T0123ABC")
        skills = await source.list_skills()
        content = await source.get_skill_file("my-skill", "SKILL.md")
        await source.push_skill("my-skill", Path("/path/to/skills/my-skill"))

    Constructor parameters:
        tenant_slug: Slug identifying the tenant (e.g. "slack_T0123ABCDE").
                     Used as the S3 key prefix component for tenant isolation.
                     Must be URL-safe (no spaces, no slashes).
        bucket:     S3 bucket name.  Defaults to S3_SKILLS_BUCKET env var,
                    then DEFAULT_BUCKET ("hermes-saas-skills").
        scope:      "personal" | "team" | "global" (default "personal").
                    Included in the key prefix so personal/team skills are isolated.
                    Global skills use a fixed prefix (read-only from agent turns).
    """

    def __init__(
        self,
        tenant_slug: str,
        bucket: Optional[str] = None,
        scope: str = "personal",
    ) -> None:
        if not tenant_slug:
            raise ValueError("S3SkillSource: tenant_slug must be non-empty")
        if "/" in tenant_slug:
            raise ValueError(f"S3SkillSource: tenant_slug must not contain slashes: {tenant_slug!r}")
        if scope not in ("personal", "team", "global"):
            raise ValueError(f"S3SkillSource: scope must be one of personal/team/global, got {scope!r}")

        self._tenant_slug = tenant_slug
        self._bucket = bucket or os.environ.get("S3_SKILLS_BUCKET") or DEFAULT_BUCKET
        self._scope = scope
        self._s3: Optional[object] = None  # boto3 S3 client, lazy-initialised

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _client(self):
        """Lazy boto3 S3 client initialisation."""
        if self._s3 is None:
            import boto3  # noqa: PLC0415 — lazy import, only in SaaS mode
            self._s3 = boto3.client("s3")
        return self._s3

    def key_prefix(self, skill_name: Optional[str] = None) -> str:
        """
        Build the S3 key prefix for this tenant + scope + optional skill.

        Format: hermes-skills/{tenant_slug}/{scope}/{skill_name}/
        """
        parts = [_KEY_PREFIX, self._tenant_slug, self._scope]
        if skill_name:
            parts.append(skill_name)
        return "/".join(parts) + "/"

    def _skill_key(self, skill_name: str, file_path: str) -> str:
        """Build the full S3 key for a single skill file.

        Example: hermes-skills/slack_T0123/personal/my-skill/SKILL.md
        """
        # Normalise file_path to remove leading slashes.
        clean = file_path.lstrip("/")
        return f"{_KEY_PREFIX}/{self._tenant_slug}/{self._scope}/{skill_name}/{clean}"

    # ------------------------------------------------------------------
    # StorageBackend-style async interface
    # ------------------------------------------------------------------

    async def list_skills(self) -> list[str]:
        """
        Return a list of skill names for this tenant + scope.

        Each entry is the skill_name (top-level "directory" prefix in S3).
        Returns [] if the tenant has no skills yet.

        Assumption: skill names do not contain slashes.
        """
        prefix = self.key_prefix()

        def _list() -> list[str]:
            s3 = self._client()
            paginator = s3.get_paginator("list_objects_v2")
            pages = paginator.paginate(
                Bucket=self._bucket,
                Prefix=prefix,
                Delimiter="/",
            )
            skill_names = []
            for page in pages:
                for cp in page.get("CommonPrefixes", []):
                    # cp["Prefix"] = "hermes-skills/{slug}/{scope}/{skill_name}/"
                    # Strip the outer prefix and trailing slash.
                    relative = cp["Prefix"][len(prefix):]
                    skill_name = relative.rstrip("/")
                    if skill_name:
                        skill_names.append(skill_name)
            return skill_names

        return await asyncio.to_thread(_list)

    async def get_skill_file(self, skill_name: str, file_path: str = "SKILL.md") -> str:
        """
        Download and return the content of a skill file as a string.

        Args:
            skill_name: Skill directory name (e.g. "my-skill").
            file_path:  Relative path within the skill (default "SKILL.md").

        Returns: File content as UTF-8 string.
        Raises: KeyError if the key doesn't exist in S3.
        """
        key = self._skill_key(skill_name, file_path)

        def _get() -> str:
            s3 = self._client()
            try:
                response = s3.get_object(Bucket=self._bucket, Key=key)
                return response["Body"].read().decode("utf-8")
            except s3.exceptions.NoSuchKey:
                raise KeyError(f"S3SkillSource: key not found: s3://{self._bucket}/{key}")
            except Exception as exc:
                # ClientError for NoSuchKey on some boto3 versions.
                if "NoSuchKey" in str(exc) or "Not Found" in str(exc):
                    raise KeyError(f"S3SkillSource: key not found: s3://{self._bucket}/{key}")
                raise

        return await asyncio.to_thread(_get)

    async def push_skill(self, skill_name: str, skill_dir: Path) -> int:
        """
        Upload all files from skill_dir to S3 under the tenant prefix.

        Args:
            skill_name: Name used as the S3 "directory" key component.
            skill_dir:  Local Path to the skill directory (must contain SKILL.md).

        Returns: Number of files uploaded.
        Raises: ValueError if skill_dir doesn't exist or lacks SKILL.md.

        File discovery: uploads SKILL.md + all *.md files under references/
        and templates/ subdirectories.  Binary assets are skipped (Phase D scope).
        """
        if not skill_dir.exists():
            raise ValueError(f"S3SkillSource.push_skill: skill_dir not found: {skill_dir}")
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            raise ValueError(f"S3SkillSource.push_skill: SKILL.md missing in {skill_dir}")

        # Collect files to upload.
        files_to_upload: list[tuple[Path, str]] = []
        for path in skill_dir.rglob("*"):
            if path.is_file() and path.suffix in (".md", ".txt", ".yaml", ".yml"):
                relative = path.relative_to(skill_dir)
                files_to_upload.append((path, str(relative)))

        if not files_to_upload:
            logger.warning(
                "S3SkillSource.push_skill: no uploadable files found in %s", skill_dir
            )
            return 0

        def _upload() -> int:
            s3 = self._client()
            count = 0
            for local_path, relative_str in files_to_upload:
                key = self._skill_key(skill_name, relative_str)
                content = local_path.read_bytes()
                s3.put_object(
                    Bucket=self._bucket,
                    Key=key,
                    Body=content,
                    ContentType="text/markdown; charset=utf-8",
                    Metadata={
                        "hermes-skill-name": skill_name,
                        "hermes-tenant-slug": self._tenant_slug,
                        "hermes-scope": self._scope,
                    },
                )
                logger.debug("S3SkillSource: uploaded s3://%s/%s", self._bucket, key)
                count += 1
            return count

        count = await asyncio.to_thread(_upload)
        logger.info(
            "S3SkillSource: pushed skill %r (%d files) to s3://%s/%s",
            skill_name, count, self._bucket, self.key_prefix(skill_name),
        )
        return count

    async def delete_skill(self, skill_name: str) -> int:
        """
        Delete all S3 objects for a skill.  Returns number of deleted objects.

        This is a soft delete from S3 — local cache files are not touched.
        """
        prefix = self.key_prefix(skill_name)

        def _delete() -> int:
            s3 = self._client()
            paginator = s3.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self._bucket, Prefix=prefix)
            deleted = 0
            for page in pages:
                objects = page.get("Contents", [])
                if not objects:
                    continue
                delete_spec = {"Objects": [{"Key": obj["Key"]} for obj in objects]}
                s3.delete_objects(Bucket=self._bucket, Delete=delete_spec)
                deleted += len(objects)
            return deleted

        count = await asyncio.to_thread(_delete)
        logger.info(
            "S3SkillSource: deleted %d objects for skill %r from s3://%s/%s",
            count, skill_name, self._bucket, prefix,
        )
        return count

    async def download_to_local(
        self,
        skill_name: str,
        local_dir: Path,
    ) -> Path:
        """
        Download a skill from S3 and write it to local_dir/{skill_name}/.

        Creates the directory if it doesn't exist.  Existing files are
        overwritten (idempotent).

        Returns: Path to the local skill directory.
        """
        skill_local_dir = local_dir / skill_name
        skill_local_dir.mkdir(parents=True, exist_ok=True)
        prefix = self.key_prefix(skill_name)

        def _download() -> int:
            s3 = self._client()
            paginator = s3.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self._bucket, Prefix=prefix)
            count = 0
            for page in pages:
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    # Strip prefix to get relative path within skill.
                    relative = key[len(prefix):]
                    if not relative:
                        continue
                    local_path = skill_local_dir / Path(relative)
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    response = s3.get_object(Bucket=self._bucket, Key=key)
                    content = response["Body"].read()
                    local_path.write_bytes(content)
                    count += 1
            return count

        count = await asyncio.to_thread(_download)
        logger.info(
            "S3SkillSource: downloaded %d files for skill %r to %s",
            count, skill_name, skill_local_dir,
        )
        return skill_local_dir


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def skill_source_from_identity(
    identity: "HermesIdentity",  # type: ignore[name-defined]
    scope: str = "personal",
    bucket: Optional[str] = None,
) -> S3SkillSource:
    """
    Build an S3SkillSource from a HermesIdentity and a scope string.

    tenant_slug = "{platform}_{team_id}" — deterministic, URL-safe.
    scope must be "personal", "team", or "global".

    Raises: ValueError if called outside HERMES_MODE=saas (guard against
    accidental S3 usage in local dev).
    """
    mode = os.environ.get("HERMES_MODE", "local")
    if mode != "saas":
        raise ValueError(
            f"skill_source_from_identity: HERMES_MODE={mode!r}. "
            "S3SkillSource is only available in SaaS mode (HERMES_MODE=saas)."
        )
    tenant_slug = f"{identity.platform}_{identity.team_id}"
    return S3SkillSource(tenant_slug=tenant_slug, scope=scope, bucket=bucket)
