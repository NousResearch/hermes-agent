"""R2 URL authority: the bucket and object URLs the signer addresses, and the public channel base.

Bucket names are validated here because a bucket-with-slash silently scopes object calls but not
ListObjectsV2.
"""
from __future__ import annotations

import os
import re


def bucket_url(base: str, bucket: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", bucket):
        raise ValueError("Invalid R2 bucket name")
    return f"{base.rstrip('/')}/{bucket}"


def object_url(base: str, bucket: str, key: str) -> str:
    """Every signed object call goes through here, so every key is validated here: no absolute,
    dot-segment, empty-segment or encoded escape reaches the bucket."""
    from scripts.releases.r2 import encode_key_path, relative_artifact_path

    return f"{bucket_url(base, bucket)}/{encode_key_path(relative_artifact_path(key))}"


def channel_public_base(explicit: str | None = None) -> str:
    """The public origin channel records and archives are read from: the explicit one, else
    CLOUDFLARE_R2_PUBLIC_URL, else the documented production origin (as r2.public_base_url)."""
    from hermes_cli.release_channels import public_base

    if explicit is not None:
        return public_base(explicit)
    configured = os.environ.get("CLOUDFLARE_R2_PUBLIC_URL", "")
    if not configured:
        from scripts.releases.r2 import DEFAULT_PUBLIC_URL

        configured = DEFAULT_PUBLIC_URL
    return public_base(configured)
