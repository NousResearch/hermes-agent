"""Disposable CI namespaces at the object-key boundary, never in bucket names."""
from __future__ import annotations

from dataclasses import dataclass
import os
import re

# A lease is the workflow run id alone: "re-run failed jobs" must re-enter the
# SAME namespace because succeeded jobs are skipped and their outputs persist.
_RUN_LEASE = r"[1-9][0-9]{0,19}"


def require_run(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(_RUN_LEASE, value):
        raise ValueError("Disposable R2 run must be <run-id>")
    return value


@dataclass(frozen=True)
class R2Scope:
    prefix: str = ""

    def __post_init__(self):
        if self.prefix and not re.fullmatch(
                r"ci-disposable/[1-9][0-9]{0,19}/" + _RUN_LEASE + r"/", self.prefix):
            raise ValueError("Invalid disposable R2 namespace")

    @classmethod
    def configured(cls, repository: str | None = None) -> R2Scope:
        run = os.environ.get("R2_DISPOSABLE_RUN", "")
        if not run:
            # Opt-in scoping: R2_DISPOSABLE_RUN selects the disposable
            # namespace regardless of repository. Unset means production
            # keys — every caller already holds the release-signing secret.
            return cls()
        require_run(run)
        repository_id = os.environ.get("GITHUB_REPOSITORY_ID", "")
        if not re.fullmatch(r"[1-9][0-9]{0,19}", repository_id):
            raise ValueError("Disposable R2 requires the GitHub repository ID")
        return cls(f"ci-disposable/{repository_id}/{run}/")

    def key(self, key: str) -> str:
        if not self.prefix:
            return key
        from scripts.releases.r2 import relative_artifact_path
        return self.prefix + relative_artifact_path(key)

    def listing_prefix(self, prefix: str) -> str:
        if not prefix:
            return self.prefix
        return self.key(prefix.removesuffix("/")) + ("/" if prefix.endswith("/") else "")

    def logical_key(self, key: str) -> str:
        if not key.startswith(self.prefix):
            raise ValueError("R2 listing escaped the disposable namespace")
        return key[len(self.prefix):]

    def bucket_url(self, base: str, bucket: str) -> str:
        # A bucket-with-slash silently scopes object calls but NOT ListObjectsV2.
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", bucket):
            raise ValueError("Invalid R2 bucket name")
        return f"{base.rstrip('/')}/{bucket}"

    def object_url(self, base: str, bucket: str, key: str) -> str:
        from scripts.releases.r2 import encode_key_path
        return f"{self.bucket_url(base, bucket)}/{encode_key_path(self.key(key))}"

    def public_base(self, base: str) -> str:
        root = base.rstrip("/")
        if self.prefix:
            from scripts.releases.r2 import public_artifact_url
            public_artifact_url(root, "scope-check")
            # The configured environment supplies the root. Callers cannot substitute
            # a different origin or a production-looking URL for scoped objects.
            if "/ci-disposable/" in root + "/":
                raise ValueError("Configured R2 public root must not contain a disposable namespace")
            return root + "/" + self.prefix.rstrip("/")
        return root


def channel_public_base(explicit: str | None = None) -> str:
    scope = R2Scope.configured()
    configured = os.environ.get("CLOUDFLARE_R2_PUBLIC_URL", "")
    if scope.prefix:
        if not configured:
            raise ValueError("Disposable channels require the configured R2 public root")
        expected = scope.public_base(configured)
        if explicit is not None and explicit.rstrip("/") != expected:
            raise ValueError("Disposable channel archive authority mismatch")
        return expected
    from hermes_cli.release_channels import public_base
    if explicit is not None:
        return public_base(explicit)
    if not configured:
        # The documented production origin is the default, exactly as the
        # commit-build path (r2.public_base_url) does, so a local command can
        # name the page it is about to publish without hand-setting the URL.
        from scripts.releases.r2 import DEFAULT_PUBLIC_URL
        configured = DEFAULT_PUBLIC_URL
    return public_base(configured)
