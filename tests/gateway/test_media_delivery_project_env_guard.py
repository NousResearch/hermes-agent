"""Regression: a project-local ``.env`` / ``.envrc`` must not be deliverable as
native media when the same path is read-blocked by ``agent/file_safety``.

``gateway/platforms/base._media_delivery_denied_paths`` documents the invariant
that "a credential the agent is forbidden to ... read must also never be
auto-attached to a chat reply". ``agent/file_safety.get_read_block_error``
read-blocks ``.env`` / ``.envrc`` anywhere on disk by basename, but the media
denylist only enumerates ``<hermes-root>/.env``, so a user's *project* ``.env``
(e.g. ``/home/user/app/.env``) was deliverable via a prompt-injected ``MEDIA:``
tag in default (non-strict) single-user mode -> credential exfiltration.
"""

from pathlib import Path

import pytest

import gateway.platforms.base as base
from gateway.platforms.base import BasePlatformAdapter, validate_media_delivery_path
from agent import file_safety


def test_dotenv_basename_has_no_suffix():
    # Reachability of the extensionless ``MEDIA:`` branch hinges on this pathlib
    # fact: a leading-dot name has no suffix, so ``.env`` / ``.envrc`` are
    # extensionless and take the branch that calls validate_media_delivery_path.
    # Verified empirically, not from memory.
    assert Path("/home/u/app/.env").suffix == ""
    assert Path("/home/u/app/.envrc").suffix == ""
    # A real extension is still detected, so ``.env.production`` is NOT
    # extensionless and does not reach that branch (scope caveat).
    assert Path("/home/u/app/.env.production").suffix == ".production"


@pytest.mark.parametrize("basename", [".env", ".envrc"])
def test_project_env_gate_mirrors_read_guard(tmp_path, monkeypatch, basename):
    # Default single-user gateway = non-strict media delivery.
    monkeypatch.setattr(base, "_media_delivery_strict_mode", lambda: False)

    proj = tmp_path / "app"
    proj.mkdir()
    secret = proj / basename
    secret.write_text("OPENAI_API_KEY=sk-live-SECRET\nDATABASE_URL=postgres://u:p@h/db\n")
    p = str(secret.resolve())

    # Precondition: the read guard blocks reading this project env file.
    assert file_safety.get_read_block_error(p) is not None, (
        "read guard should block the project env file (basename rule)"
    )

    # The delivery gate must mirror the read guard: a read-blocked credential
    # must not be returned as a safe native attachment.
    assert validate_media_delivery_path(p) is None, (
        f"{basename} is read-blocked but validate_media_delivery_path approved "
        "it for native attachment -> prompt-injected MEDIA: exfil"
    )


@pytest.mark.parametrize("basename", [".env", ".envrc"])
def test_project_env_not_deliverable_via_media_tag(tmp_path, monkeypatch, basename):
    # End-to-end: a prompt-injected MEDIA: tag pointing at a project env file
    # must not surface it as a deliverable attachment.
    monkeypatch.setattr(base, "_media_delivery_strict_mode", lambda: False)

    proj = tmp_path / "app"
    proj.mkdir()
    secret = proj / basename
    secret.write_text("STRIPE_SECRET_KEY=sk_live_SECRET\n")
    p = str(secret.resolve())

    reply = f"Sure, here is the file:\nMEDIA:{p}\n"
    media, _cleaned = BasePlatformAdapter.extract_media(reply)
    delivered_basenames = [Path(mp).name.lower() for mp, _voice in media]
    assert basename not in delivered_basenames, (
        f"project {basename} auto-attached for exfil via MEDIA: tag: {media}"
    )
