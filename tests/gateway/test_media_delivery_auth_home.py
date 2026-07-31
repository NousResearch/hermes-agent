"""Outbound media delivery must not exfiltrate the credential residence.

The denylist is resolved per call under the current profile context, so the
multiplexing gateway denies the right profile's credential files on every
turn, and a genuinely distinct ``HERMES_AUTH_HOME`` is denied as a whole tree
while a path-equal override changes nothing.
"""

from __future__ import annotations

from pathlib import Path

from hermes_constants import reset_hermes_home_override, set_hermes_home_override

import gateway.platforms.base as base


def _fresh_file(path: Path, content: bytes = b"media") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_media_delivery_denies_the_whole_credential_residence(
    monkeypatch, tmp_path
):
    residence = tmp_path / "auth-residence"
    runtime = tmp_path / "runtime"
    monkeypatch.setenv("HERMES_HOME", str(runtime))
    monkeypatch.setenv("HERMES_AUTH_HOME", str(residence))

    for rel in (
        "auth.json",
        "auth.lock",
        "auth.json.tmp.4242.deadbeef",
        ".anthropic_oauth.json",
        "shared/nous_auth.json",
        "profiles/other/auth.json",
        "unclassified-report.pdf",
    ):
        target = _fresh_file(residence / rel)
        assert base.validate_media_delivery_path(str(target)) is None, rel

    # Ordinary runtime artifacts still deliver.
    artifact = _fresh_file(runtime / "render.png")
    assert base.validate_media_delivery_path(str(artifact)) == str(
        artifact.resolve()
    )


def test_path_equal_override_does_not_deny_the_runtime_tree(monkeypatch, tmp_path):
    runtime = tmp_path / "runtime"
    monkeypatch.setenv("HERMES_HOME", str(runtime))
    monkeypatch.setenv("HERMES_AUTH_HOME", str(runtime))

    artifact = _fresh_file(runtime / "render.png")
    assert base.validate_media_delivery_path(str(artifact)) == str(
        artifact.resolve()
    )
    # The per-file credential entries keep working.
    secret = _fresh_file(runtime / "auth.json")
    assert base.validate_media_delivery_path(str(secret)) is None


def test_media_denylist_follows_the_per_call_profile_context(monkeypatch, tmp_path):
    """The denied set is computed from the home active for THIS call.

    The multiplexer scopes each turn with a context-local HERMES_HOME; an
    import-time snapshot would deny the launch profile's paths forever and
    let another profile's credential files through.
    """
    monkeypatch.delenv("HERMES_AUTH_HOME", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "runtime"))
    scoped_home = tmp_path / "scoped-home"
    secret = _fresh_file(scoped_home / "google_token.json")

    # Outside any Hermes home this is an ordinary deliverable file.
    assert base.validate_media_delivery_path(str(secret)) == str(secret.resolve())

    token = set_hermes_home_override(str(scoped_home))
    try:
        assert base.validate_media_delivery_path(str(secret)) is None
    finally:
        reset_hermes_home_override(token)

    assert base.validate_media_delivery_path(str(secret)) == str(secret.resolve())


def test_sibling_profile_credentials_are_denied_from_the_root_gateway(
    monkeypatch, tmp_path
):
    root = tmp_path / "hermes-root"
    monkeypatch.delenv("HERMES_AUTH_HOME", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(root))

    sibling_secret = _fresh_file(root / "profiles" / "other" / "auth.json")
    assert base.validate_media_delivery_path(str(sibling_secret)) is None

    sibling_artifact = _fresh_file(root / "profiles" / "other" / "render.png")
    assert base.validate_media_delivery_path(str(sibling_artifact)) == str(
        sibling_artifact.resolve()
    )
