from __future__ import annotations

import hashlib
import itertools
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from scripts.canary import full_canary_owner_launcher as launcher
from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import owner_gate_inert_input_preparer as preparer
from scripts.canary import owner_gate_outer_stage0 as outer
from scripts.canary import owner_gate_release_author as release_author
from scripts.canary import owner_gate_trust as trust
from scripts.canary import owner_gate_trust_author as trust_author
from tests.gateway.test_canonical_writer_production_cutover import (
    _isolated_canary_goal_prerequisite,
)


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("/usr/bin/git", "-C", str(root), *arguments),
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TMPDIR": "/tmp"},
    )
    return completed.stdout.decode("ascii", errors="strict").strip()


def _release_repository(tmp_path: Path) -> tuple[Path, str, str]:
    source = tmp_path / "source"
    source.mkdir(mode=0o700)
    _git(source, "init", "-b", "main")
    _git(source, "config", "user.name", "Release Test")
    _git(source, "config", "user.email", "release@example.invalid")
    repository = Path(__file__).resolve().parents[3]
    for relative in set(outer.SOURCE_FILES.values()):
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repository / relative, destination)
    _git(source, "add", "--all")
    _git(source, "commit", "-m", "exact release")
    revision = _git(source, "rev-parse", "HEAD")
    tree = _git(source, "rev-parse", "HEAD^{tree}")
    _git(source, "remote", "add", "origin", release_author.FORK_ORIGIN)
    _git(source, "update-ref", "refs/remotes/origin/main", revision)
    return source, revision, tree


def _publication_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    owner = tmp_path / "owner"
    owner.mkdir(mode=0o700)
    hermes = owner / ".hermes"
    hermes.mkdir(mode=0o700)
    trusted = hermes / "trusted"
    release_base = trusted / "owner-gate-release-sources"
    monkeypatch.setattr(release_author, "OWNER_HOME", owner)
    monkeypatch.setattr(release_author, "TRUSTED_ROOT", trusted)
    monkeypatch.setattr(release_author, "RELEASE_SOURCE_BASE", release_base)
    monkeypatch.setattr(release_author.inert_inputs, "OWNER_HOME", owner)
    monkeypatch.setattr(release_author.inert_inputs, "_hermes_root", lambda: hermes)
    return release_base


def test_publish_release_source_uses_fixed_detached_no_hardlink_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, revision, tree = _release_repository(tmp_path)
    release_base = _publication_root(tmp_path, monkeypatch)

    receipt = release_author.publish_release_source(
        source_root=source,
        release_revision=revision,
    )

    destination = release_base / revision
    assert receipt == {
        "schema": release_author.SCHEMA,
        "action": "publish-release-source",
        "release_revision": revision,
        "source_tree_oid": tree,
        "publication_path": str(destination),
        "origin": release_author.FORK_ORIGIN,
        "detached_head": True,
        "created": True,
        "network_fetch_performed": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    assert stat.S_IMODE(destination.stat().st_mode) == 0o700
    assert _git(destination, "branch", "--show-current") == ""
    assert _git(destination, "rev-parse", "HEAD") == revision
    assert _git(destination, "rev-parse", "HEAD^{tree}") == tree
    assert _git(destination, "remote", "get-url", "origin") == (
        release_author.FORK_ORIGIN
    )
    assert (destination / ".git").is_dir()
    assert not (destination / ".git/objects/info/alternates").exists()
    _git(destination, "fsck", "--full", "--strict", "--no-dangling")
    replay = release_author.publish_release_source(
        source_root=source,
        release_revision=revision,
    )
    assert replay["created"] is False
    source.rename(tmp_path / "source-moved-away")
    assert _git(destination, "rev-parse", "HEAD") == revision
    _git(destination, "fsck", "--full", "--strict", "--no-dangling")


def test_publish_release_source_rejects_dirty_or_wrong_origin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, revision, _tree = _release_repository(tmp_path)
    _publication_root(tmp_path, monkeypatch)
    untracked = source / "untracked"
    untracked.write_text("not release\n", encoding="utf-8")

    with pytest.raises(
        release_author.OwnerGateReleaseAuthorError,
        match="owner_gate_release_author_source_invalid",
    ):
        release_author.publish_release_source(
            source_root=source,
            release_revision=revision,
        )
    untracked.unlink()
    _git(source, "remote", "set-url", "origin", "https://example.invalid/wrong.git")
    with pytest.raises(
        release_author.OwnerGateReleaseAuthorError,
        match="owner_gate_release_author_source_invalid",
    ):
        release_author.publish_release_source(
            source_root=source,
            release_revision=revision,
        )


def test_release_source_rejects_alternates_promisor_and_linked_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, revision, _tree = _release_repository(tmp_path)
    _publication_root(tmp_path, monkeypatch)
    shared = tmp_path / "shared"
    subprocess.run(
        (
            "/usr/bin/git",
            "clone",
            "--shared",
            "--",
            str(source),
            str(shared),
        ),
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TMPDIR": "/tmp"},
    )
    _git(shared, "remote", "set-url", "origin", release_author.FORK_ORIGIN)
    assert (shared / ".git/objects/info/alternates").exists()
    with pytest.raises(
        release_author.OwnerGateReleaseAuthorError,
        match="owner_gate_release_author_object_store_invalid",
    ):
        release_author.publish_release_source(
            source_root=shared,
            release_revision=revision,
        )

    _git(source, "config", "remote.origin.promisor", "true")
    with pytest.raises(
        release_author.OwnerGateReleaseAuthorError,
        match="owner_gate_release_author_object_store_invalid",
    ):
        release_author.publish_release_source(
            source_root=source,
            release_revision=revision,
        )
    _git(source, "config", "--unset", "remote.origin.promisor")

    linked = tmp_path / "linked-worktree"
    _git(source, "worktree", "add", "--detach", str(linked), revision)
    assert (linked / ".git").is_file()
    with pytest.raises(
        release_author.OwnerGateReleaseAuthorError,
        match="owner_gate_release_author_object_store_invalid",
    ):
        release_author.verify_exact_detached_release_source(
            linked,
            release_revision=revision,
        )


def test_release_source_rejects_missing_object_and_existing_hardlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, revision, _tree = _release_repository(tmp_path)
    release_base = _publication_root(tmp_path, monkeypatch)
    release_author.publish_release_source(
        source_root=source,
        release_revision=revision,
    )
    destination = release_base / revision
    selected = next(
        path
        for path in (destination / ".git/objects").rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    os.link(selected, tmp_path / "object-alias")
    with pytest.raises(
        release_author.OwnerGateReleaseAuthorError,
        match="owner_gate_release_author_object_store_invalid",
    ):
        release_author.publish_release_source(
            source_root=source,
            release_revision=revision,
        )

    missing_root = tmp_path / "missing"
    missing_root.mkdir()
    missing_source, missing_revision, _missing_tree = _release_repository(
        missing_root
    )
    relative = sorted(set(outer.SOURCE_FILES.values()))[0]
    blob = _git(
        missing_source,
        "rev-parse",
        f"{missing_revision}:{relative}",
    )
    loose = missing_source / ".git/objects" / blob[:2] / blob[2:]
    assert loose.is_file()
    loose.unlink()
    with pytest.raises(
        release_author.OwnerGateReleaseAuthorError,
        match="owner_gate_release_author_git_invalid",
    ):
        release_author.publish_release_source(
            source_root=missing_source,
            release_revision=missing_revision,
        )


def test_publish_release_source_fsync_failure_cleans_pending_and_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, revision, _tree = _release_repository(tmp_path)
    release_base = _publication_root(tmp_path, monkeypatch)
    real_fsync_tree = release_author._fsync_tree
    attempts = 0

    def fail_once(path: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise release_author.OwnerGateReleaseAuthorError(
                "owner_gate_release_author_publish_failed"
            )
        real_fsync_tree(path)

    monkeypatch.setattr(release_author, "_fsync_tree", fail_once)
    with pytest.raises(
        release_author.OwnerGateReleaseAuthorError,
        match="owner_gate_release_author_publish_failed",
    ):
        release_author.publish_release_source(
            source_root=source,
            release_revision=revision,
        )
    assert not (release_base / revision).exists()
    assert not (release_base / f".{revision}.pending").exists()

    receipt = release_author.publish_release_source(
        source_root=source,
        release_revision=revision,
    )
    assert receipt["created"] is True


def test_fixed_release_consumer_rejects_skip_worktree_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, revision, tree = _release_repository(tmp_path)
    release_base = _publication_root(tmp_path, monkeypatch)
    release_author.publish_release_source(
        source_root=source,
        release_revision=revision,
    )
    destination = release_base / revision
    relative = sorted(set(outer.SOURCE_FILES.values()))[0]
    _git(destination, "update-index", "--skip-worktree", "--", relative)
    with (destination / relative).open("ab") as stream:
        stream.write(b"\nmutated-after-publication\n")

    trusted = release_base.parent
    hermes = trusted.parent
    monkeypatch.setattr(preparer, "OWNER_HOME", hermes.parent)
    monkeypatch.setattr(preparer, "TRUSTED_ROOT", trusted)
    monkeypatch.setattr(preparer, "RELEASE_SOURCE_BASE", release_base)
    monkeypatch.setattr(preparer, "_hermes_root", lambda: hermes)
    monkeypatch.setattr(
        preparer.launcher,
        "require_trusted_owner_support_activation",
        lambda *_args, **_kwargs: None,
    )

    class Runtime:
        def trusted_owner_support_paths(self) -> tuple[str, str]:
            return ("/trusted/gcloud", "/trusted/root")

        def sealed_owner_support_manifest(
            self,
            *,
            expected_release_sha: str,
        ) -> dict[str, str]:
            return {
                "release_sha": expected_release_sha,
                "source_tree_oid": tree,
            }

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_inert_input_release_checkout_invalid",
    ):
        preparer._fixed_release_source(revision, Runtime())


def test_author_unsigned_trust_publishes_fixed_canonical_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_revision = "a" * 40
    source_tree = "b" * 40
    foundation_revision = "0" * 40
    foundation_tree = "8" * 40
    interpreter_sha256 = "f" * 64
    image_name = "debian-12-bookworm-v20260710"
    image = {
        "project": "debian-cloud",
        "image_name": image_name,
        "image_numeric_id": "1234567890123456789",
        "image_self_link": (
            "https://www.googleapis.com/compute/v1/projects/"
            f"debian-cloud/global/images/{image_name}"
        ),
        "python_version": "3.11.2",
        "interpreter_sha256": interpreter_sha256,
    }
    migration = {"fixture": "validated-migration"}
    inventory = {
        "source_tree_oid": source_tree,
        "interpreter_sha256": interpreter_sha256,
        "direct_iam_identity_authority_sha256": "e" * 64,
        "pre_foundation_authority_sha256": "4" * 64,
        "foundation_apply_receipt_sha256": "5" * 64,
        "resource_ancestor_chain": ["organizations/123456789012"],
    }
    chain = trust_author._ValidatedFoundationChain._create(
        final_release_revision=release_revision,
        pre_foundation_authority_sha256="4" * 64,
        foundation_apply_receipt_sha256="5" * 64,
        bootstrap_network_collector_public_key_id="9" * 64,
        foundation_source_revision=foundation_revision,
        foundation_source_tree_oid=foundation_tree,
        direct_iam_identity_authority_sha256="e" * 64,
        project_ancestry_evidence_sha256="6" * 64,
        project_ancestry_chain_sha256="7" * 64,
        resource_ancestor_chain=("organizations/123456789012",),
        interpreter_sha256=interpreter_sha256,
        interpreter_version="3.11.2",
        interpreter_image=image,
        attested_at_unix=1_784_300_000,
    )
    authority_parent = tmp_path / "authority"
    authority_parent.mkdir(mode=0o700)
    key_directory = authority_parent / "owner-gate-release-authority"
    key_directory.mkdir(mode=0o700)
    manifests = key_directory / "manifests"
    manifests.mkdir(mode=0o700)
    signing_key = Ed25519PrivateKey.generate()
    public_raw = signing_key.public_key().public_bytes_raw()
    public_path = key_directory / trust_author.PUBLIC_KEY_NAME
    public_path.write_bytes(public_raw)
    public_path.chmod(0o444)
    key_id = hashlib.sha256(public_raw).hexdigest()
    monkeypatch.setattr(trust, "PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256", key_id)
    monkeypatch.setattr(trust_author, "AUTHORITY_PARENT", authority_parent)
    monkeypatch.setattr(trust_author, "KEY_DIRECTORY", key_directory)
    monkeypatch.setattr(trust_author, "MANIFEST_DIRECTORY", manifests)
    monkeypatch.setattr(
        release_author,
        "_verify_release_source",
        lambda *_args, **_kwargs: source_tree,
    )
    monkeypatch.setattr(
        release_author,
        "_read_canonical_json",
        lambda *_args, **_kwargs: {"fixture": "wheelhouse"},
    )
    monkeypatch.setattr(
        release_author.package,
        "build_inventory",
        lambda _spec: inventory,
    )
    monkeypatch.setattr(
        trust_author,
        "_validate_foundation_chain_files",
        lambda **_kwargs: chain,
    )
    pre_foundation_path = tmp_path / "pre-foundation.json"
    migration_path = tmp_path / "migration.json"

    def read_immutable(path: Path, *, maximum: int) -> bytes:
        del maximum
        if path == pre_foundation_path:
            raise AssertionError("validated authority pathname was reread")
        if path == migration_path:
            return foundation.canonical_json_bytes(migration)
        raise AssertionError(path)

    monkeypatch.setattr(release_author, "_read_immutable", read_immutable)
    collector_ids = itertools.cycle(("1" * 64, "2" * 64, "3" * 64))
    monkeypatch.setattr(
        release_author,
        "_collector_key",
        lambda _path: (
            Ed25519PrivateKey.generate().public_key(),
            next(collector_ids),
        ),
    )
    monkeypatch.setattr(
        release_author.bootstrap,
        "validate_migration",
        lambda *_args, **_kwargs: migration,
    )
    dummy = tmp_path / "dummy"

    author_inputs = {
        "source_root": tmp_path,
        "release_revision": release_revision,
        "wheelhouse_root": tmp_path,
        "wheelhouse_manifest_path": dummy,
        "interpreter_sha256": interpreter_sha256,
        "foundation_source_revision": foundation_revision,
        "foundation_source_tree_oid": foundation_tree,
        "pre_foundation_authority_path": pre_foundation_path,
        "owner_reauthentication_receipt_path": dummy,
        "network_evidence_path": dummy,
        "foundation_collector_public_key_path": dummy,
        "project_ancestry_evidence_path": dummy,
        "direct_iam_identity_authority_path": dummy,
        "network_collector_public_key_path": dummy,
        "cloud_collector_public_key_path": dummy,
        "host_collector_public_key_path": dummy,
        "credential_migration_envelope_path": migration_path,
        "now_unix": 1_784_300_000,
    }
    receipt = release_author.author_unsigned_trust(**author_inputs)
    replay = release_author.author_unsigned_trust(**author_inputs)

    output = manifests / f"{release_revision}.trust.unsigned.json"
    value = json.loads(output.read_text(encoding="ascii"))
    trust._validate_unsigned(value)
    assert output.read_bytes() == foundation.canonical_json_bytes(value)
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    assert receipt["unsigned_manifest_path"] == str(output)
    assert replay["unsigned_manifest_sha256"] == receipt[
        "unsigned_manifest_sha256"
    ]
    assert receipt["private_key_loaded"] is False
    assert value["package_inventory_sha256"] == foundation.sha256_json(inventory)
    assert value["collector_public_key_ids"] == {
        "network": "1" * 64,
        "cloud": "2" * 64,
        "host": "3" * 64,
    }


def test_author_isolated_canary_prerequisite_uses_fixed_immutable_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _isolated_canary_goal_prerequisite()
    revision = expected["release_revision"]
    owner_root = tmp_path / "cutover-owner"
    owner_root.mkdir(mode=0o700)
    hermes = owner_root / ".hermes"
    hermes.mkdir(mode=0o700)
    authority_root = hermes / "owner-gate-production-cutover"
    prerequisite_root = authority_root / "isolated-canary-prerequisites"
    monkeypatch.setattr(release_author, "OWNER_HOME", owner_root)
    monkeypatch.setattr(
        release_author,
        "OWNER_CUTOVER_AUTHORITY_ROOT",
        authority_root,
    )
    monkeypatch.setattr(
        release_author,
        "ISOLATED_CANARY_PREREQUISITE_ROOT",
        prerequisite_root,
    )
    monkeypatch.setattr(release_author.inert_inputs, "OWNER_HOME", owner_root)
    monkeypatch.setattr(
        release_author.inert_inputs,
        "_hermes_root",
        lambda: hermes,
    )
    inputs = {
        "fixture": expected["fixture"],
        "workspace-gateway": expected["workspace_gateway"],
        "cleanup-receipt": expected["cleanup_receipt"],
        "production-diff": expected["production_diff"],
    }
    paths: dict[str, Path] = {}
    for name, value in inputs.items():
        path = tmp_path / f"{name}.json"
        path.write_bytes(release_author._cutover_canonical(value))
        path.chmod(0o444)
        paths[name] = path

    receipt = release_author.author_isolated_canary_goal_prerequisite(
        release_revision=revision,
        fixture_path=paths["fixture"],
        workspace_gateway_path=paths["workspace-gateway"],
        cleanup_receipt_path=paths["cleanup-receipt"],
        production_diff_path=paths["production-diff"],
    )

    output = prerequisite_root / f"{revision}.json"
    assert json.loads(output.read_text(encoding="utf-8")) == expected
    assert output.read_bytes() == release_author._cutover_canonical(expected)
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    assert receipt["publication_path"] == str(output)
    assert receipt["evidence_sha256"] == expected["evidence_sha256"]
    assert receipt["created"] is True
    replay = release_author.author_isolated_canary_goal_prerequisite(
        release_revision=revision,
        fixture_path=paths["fixture"],
        workspace_gateway_path=paths["workspace-gateway"],
        cleanup_receipt_path=paths["cleanup-receipt"],
        production_diff_path=paths["production-diff"],
    )
    assert replay["created"] is False


def test_release_author_cli_has_no_caller_selected_output() -> None:
    with pytest.raises(SystemExit):
        release_author.main([
            "publish-release-source",
            "--source-root",
            "/tmp/source",
            "--release-revision",
            "a" * 40,
            "--output",
            "/tmp/caller-selected",
        ])


@pytest.mark.parametrize(
    ("command", "forbidden"),
    (
        ("author-unsigned-trust", "--output"),
        ("sign-trust", "--output"),
        ("sign-trust", "--unsigned"),
    ),
)
def test_trust_authoring_cli_rejects_caller_selected_manifest_paths(
    command: str,
    forbidden: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    arguments = [
        command,
        "--source-root",
        "/tmp/source",
        "--release-revision",
        "a" * 40,
        "--wheelhouse-root",
        "/tmp/wheelhouse",
        "--wheelhouse-manifest",
        "/tmp/wheelhouse.json",
        "--interpreter-sha256",
        "f" * 64,
        "--foundation-source-revision",
        "0" * 40,
        "--foundation-source-tree-oid",
        "8" * 40,
        "--pre-foundation-authority",
        "/tmp/pre-foundation.json",
        "--owner-reauth-receipt",
        "/tmp/owner-reauth.json",
        "--network-evidence",
        "/tmp/network.json",
        "--foundation-collector-public-key",
        "/tmp/foundation.pub",
        "--project-ancestry-evidence",
        "/tmp/ancestry.json",
        "--direct-iam-identity-authority",
        "/tmp/direct-iam.json",
        "--network-collector-public-key",
        "/tmp/network.pub",
        "--cloud-collector-public-key",
        "/tmp/cloud.pub",
        "--host-collector-public-key",
        "/tmp/host.pub",
        "--credential-migration-envelope",
        "/tmp/migration.json",
        forbidden,
        "/tmp/caller-selected",
    ]
    with pytest.raises(SystemExit):
        release_author.main(arguments)
    assert "unrecognized arguments" in capsys.readouterr().err
