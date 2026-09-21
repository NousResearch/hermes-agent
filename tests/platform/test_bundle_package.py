"""Getting a bundle from a laptop onto a runtime host, reproducibly.

The transport step was a shell recipe, and the recipe had a trap. `tar czf -C bundles
yourfirm` produces a tenant-rooted archive; extracting it without `--strip-components=1`,
or one directory too high, leaves `/var/lib/nova/<tenant>/` beside `/var/lib/nova/bundle`.
Nothing reads the first, so the deployment looks complete and keeps serving what it
already had — which is exactly what the first field deployment did, with two stale copies
of the same `deployment.yaml`.
"""

from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path

import pytest

from nova.errors import SpecError
from nova.package import package, unpack
from nova.spec import load_bundle

from .conftest import EXAMPLE_BUNDLE


def test_the_archive_is_rooted_at_the_bundle_contents(tmp_path):
    """The whole point. No tenant-named directory means no strip to forget."""
    report = package(EXAMPLE_BUNDLE, tmp_path / "bundle.tgz")
    with tarfile.open(tmp_path / "bundle.tgz") as handle:
        names = handle.getnames()
    assert "organization.yaml" in names, f"bundle root is not at the archive root: {names[:5]}"
    assert not any(name.startswith(f"{report['tenant_id']}/") for name in names), (
        "the archive is tenant-rooted again; extracting it without --strip-components "
        "would strand a second bundle directory beside the real one"
    )


def test_packaging_the_same_bundle_twice_gives_the_same_bytes(tmp_path):
    """An archive hash that changes with the clock is not an identity."""
    first = package(EXAMPLE_BUNDLE, tmp_path / "a.tgz")
    second = package(EXAMPLE_BUNDLE, tmp_path / "b.tgz")
    assert first["archive_sha256"] == second["archive_sha256"]
    assert (tmp_path / "a.tgz").read_bytes() == (tmp_path / "b.tgz").read_bytes()


def test_the_output_filename_does_not_leak_into_the_bytes(tmp_path):
    """gzip takes its header filename from `fileobj.name` unless told otherwise, so the
    same bundle packaged to two paths hashed differently. Found by running it twice."""
    package(EXAMPLE_BUNDLE, tmp_path / "some-long-name.tgz")
    package(EXAMPLE_BUNDLE, tmp_path / "x.tgz")
    assert (tmp_path / "some-long-name.tgz").read_bytes() == (tmp_path / "x.tgz").read_bytes()


def test_the_reported_archive_hash_is_the_file_on_disk(tmp_path):
    """An operator compares this against `sha256sum` on the host; it has to be the same
    number or the check silently means nothing."""
    out = tmp_path / "bundle.tgz"
    report = package(EXAMPLE_BUNDLE, out)
    assert report["archive_sha256"] == "sha256:" + hashlib.sha256(out.read_bytes()).hexdigest()


def test_the_bundle_digest_is_the_parsed_configuration(tmp_path):
    """The other digest, and the one that answers "is this what I reviewed"."""
    report = package(EXAMPLE_BUNDLE, tmp_path / "bundle.tgz")
    assert report["bundle_digest"] == load_bundle(EXAMPLE_BUNDLE).digest()


def test_a_bundle_that_does_not_parse_is_never_packaged(tmp_path):
    """Shipping it would move the failure from a laptop to a runtime host."""
    import shutil

    broken = tmp_path / "broken"
    shutil.copytree(EXAMPLE_BUNDLE, broken)
    (broken / "organization.yaml").write_text("{{{ not yaml", encoding="utf-8")
    out = tmp_path / "bundle.tgz"
    with pytest.raises(SpecError):
        package(broken, out)
    assert not out.exists(), "a broken bundle left an archive behind"


def test_a_round_trip_reproduces_the_bundle(tmp_path):
    """package -> unpack -> load must give the same configuration, or transport is lossy."""
    package(EXAMPLE_BUNDLE, tmp_path / "bundle.tgz")
    destination = tmp_path / "onto-the-host"
    report = unpack(tmp_path / "bundle.tgz", destination)
    assert report["files"] > 0
    assert load_bundle(destination).digest() == load_bundle(EXAMPLE_BUNDLE).digest()


def test_unpack_refuses_a_member_that_escapes_the_destination(tmp_path):
    """NOVA writes the archive, but it travels through a bucket and a host first."""
    import gzip
    import io

    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        info = tarfile.TarInfo("../escaped.yaml")
        data = b"nope\n"
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    evil = tmp_path / "evil.tgz"
    with open(evil, "wb") as handle:
        with gzip.GzipFile(filename="", fileobj=handle, mode="wb", mtime=0) as gz:
            gz.write(payload.getvalue())

    with pytest.raises(SpecError, match="outside"):
        unpack(evil, tmp_path / "dest")
    assert not (tmp_path / "escaped.yaml").exists()


def test_editor_droppings_and_vcs_metadata_are_not_packaged(tmp_path):
    """A bundle is the declarations; `.git` in one would ship history to a runtime host."""
    import shutil

    source = tmp_path / "src"
    shutil.copytree(EXAMPLE_BUNDLE, source)
    (source / ".git").mkdir()
    (source / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (source / "organization.yaml~").write_text("stale\n", encoding="utf-8")

    report = package(source, tmp_path / "bundle.tgz")
    assert not any(".git" in name for name in report["members"])
    assert not any(name.endswith("~") for name in report["members"])


def test_a_symlink_is_refused_rather_than_packaged(tmp_path):
    """It either dangles on the host or points outside the bundle."""
    import shutil

    source = tmp_path / "src"
    shutil.copytree(EXAMPLE_BUNDLE, source)
    (source / "linked.yaml").symlink_to("/etc/passwd")
    with pytest.raises(SpecError, match="symbolic link"):
        package(source, tmp_path / "bundle.tgz")


# ---------------------------------------------------------------------------
# The deployment reconciles on start
# ---------------------------------------------------------------------------

DEPLOY = Path(__file__).resolve().parents[2] / "deploy"


def test_the_control_plane_applies_its_bundle_on_start():
    """Declaring an agent and materializing it are different acts, and the dispatcher
    refuses an unmaterialized assignee in silence. Applying at start removes the operator
    step that has to be remembered."""
    body = (DEPLOY / "aws" / "user_data.sh.tftpl").read_text(encoding="utf-8")
    control = body[body.index("name nova \\") : body.index("[Install]")]
    assert "NOVA_APPLY_ON_START=1" in control, (
        "the control plane no longer reconciles the bundle at start, so declared agents "
        "exist only as YAML and their work sits unclaimed"
    )


def test_a_failing_apply_does_not_crash_loop_the_control_plane():
    """`set -e` plus a malformed bundle would take away the surface that says WHY."""
    entrypoint = (DEPLOY / "docker" / "entrypoint.sh").read_text(encoding="utf-8")
    block = entrypoint[entrypoint.index("NOVA_APPLY_ON_START:-"):]
    block = block[: block.index("exec python -m nova serve")]
    assert "if python -m nova apply" in block, (
        "apply-on-start is unguarded; under `set -e` a bundle that does not parse turns "
        "into a restart loop with no Control API to diagnose it"
    )
    assert "APPLY FAILED" in block, "a failed apply must say so loudly"


def test_the_runbook_no_longer_tells_operators_to_strip_components():
    """The exact recipe that produced two bundle directories on the first deployment."""
    runbook = (DEPLOY / "FIRST_DEPLOYMENT.md").read_text(encoding="utf-8")
    transfer = runbook[runbook.index("## Steps 23–24"):]
    transfer = transfer[: transfer.index("## Step 25")]
    # Command lines only: the note below the block explains the old trap by name, and a
    # test that greps prose fails the next time somebody explains it better.
    commands = "\n".join(
        line for line in transfer.splitlines() if not line.lstrip().startswith(">")
    )
    assert "--strip-components" not in commands, (
        "the bundle transfer step still depends on --strip-components; forgetting it "
        "strands a tenant-named directory that nothing reads"
    )
    assert "nova bundle package" in commands and "nova bundle unpack" in commands


def test_the_bundle_digest_does_not_depend_on_where_the_bundle_sits(tmp_path):
    """The audit question the digest exists to answer is "is the deployed configuration
    the one I reviewed", and that comparison spans a laptop and a host.

    Knowledge source roots resolve against the bundle's location, so including them
    verbatim made the digest a property of the filesystem: the number printed locally
    could never match the number computed on the runtime host.
    """
    import shutil

    elsewhere = tmp_path / "some" / "other" / "place"
    shutil.copytree(EXAMPLE_BUNDLE, elsewhere)
    assert load_bundle(elsewhere).digest() == load_bundle(EXAMPLE_BUNDLE).digest(), (
        "the same declarations hash differently in two directories, so an operator "
        "comparing digests across a laptop and a host would never see them agree"
    )


def test_the_digest_does_not_depend_on_how_the_path_was_typed(tmp_path):
    """Relative and absolute must agree.

    A bundle loaded as `nova/examples/acme` carries a relative root while its knowledge
    sources resolved to absolute paths; a lexical relative_to between the two fails
    silently and falls back to the absolute path — the same bug one level down, and the
    one the CLI hit while the first version of this fix was passing its own tests.
    """
    import os

    absolute = load_bundle(EXAMPLE_BUNDLE).digest()
    previous = os.getcwd()
    os.chdir(EXAMPLE_BUNDLE.parents[2])
    try:
        relative = load_bundle(Path("nova/examples/acme")).digest()
    finally:
        os.chdir(previous)
    assert relative == absolute, (
        "the digest changes with how the bundle path was typed, so `nova bundle package "
        "bundles/test` and `nova validate /abs/bundles/test` disagree about the same files"
    )


def test_package_and_validate_report_the_same_digest(tmp_path):
    """The two commands an operator actually compares. They have to agree."""
    report = package(EXAMPLE_BUNDLE, tmp_path / "bundle.tgz")
    unpack(tmp_path / "bundle.tgz", tmp_path / "host")
    assert report["bundle_digest"] == load_bundle(tmp_path / "host").digest(), (
        "the digest printed at package time is not the one the host computes after "
        "unpacking, so comparing them proves nothing"
    )
