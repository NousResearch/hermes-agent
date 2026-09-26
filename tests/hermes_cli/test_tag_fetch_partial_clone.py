"""The post-update tag fetch must not convert a full clone into a partial one.

`fetch_full_commit_graph` passes `--filter=tree:0` so shallow/partial installs
don't pay for an unfiltered history transfer. But bare `git fetch --filter=...`
*writes* `remote.origin.promisor` / `remote.origin.partialclonefilter` —
silently converting a deliberately de-partialised checkout back into a partial
clone and re-arming the `should_include_obj` fetch failure (#122353). The fetch
therefore declares the promisor keys via `-c` for the invocation only: the
filter applies to the wire transfer while the config file stays untouched in
every state. A partial clone keeps repeating its own filter rather than being
silently tightened to tree:0. These tests pin the relationship between the
repo's promisor config before the fetch and its config after — not any current
git output.
"""

import os
import subprocess
from pathlib import Path

from hermes_cli.gitlock import fetch_full_commit_graph


def _server_repo(tmp_path: Path) -> Path:
    root = tmp_path / "server"
    root.mkdir()
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=root, env=env, check=True, capture_output=True)

    git("init", "-q", "-b", "main")
    (root / "tracked").write_text("release\n", encoding="utf-8")
    git("add", "tracked")
    git("commit", "-qm", "release")
    git("tag", "v0.21.5")
    git("commit", "-q", "--allow-empty", "-m", "after release")
    git("tag", "v0.21.6")
    return root


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout.strip()


def _config_values(root: Path, key: str) -> "list[str]":
    proc = subprocess.run(["git", "config", "--get-all", key], cwd=root,
                          capture_output=True, text=True, encoding="utf-8")
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _has_promisor_config(root: Path) -> bool:
    values = _config_values(root, "remote.origin.promisor")
    return any(value.lower() == "true" for value in values)


def _clone(tmp_path: Path, server: Path, name: str, *clone_args: str) -> Path:
    checkout = tmp_path / name
    subprocess.run(["git", "clone", "-q", *clone_args, server.as_uri(), str(checkout)],
                   check=True, capture_output=True)
    return checkout


def test_tag_fetch_never_writes_promisor_config_on_a_full_clone(tmp_path):
    # The #122353 report: a full clone that was deliberately de-partialised
    # (promisor keys removed) must not gain them back from the tag fetch.
    checkout = _clone(tmp_path, _server_repo(tmp_path), "full", "--no-tags")
    assert not _has_promisor_config(checkout)

    fetch_full_commit_graph(checkout)

    assert not _has_promisor_config(checkout)
    assert _config_values(checkout, "remote.origin.partialclonefilter") == []
    # The fetch still did its job: the tags arrived.
    assert "v0.21.6" in _git(checkout, "tag", "--list")


def test_tag_fetch_never_introduces_promisor_config_on_a_plain_clone(tmp_path):
    # A plain full clone (never partial) must not be converted either.
    checkout = _clone(tmp_path, _server_repo(tmp_path), "plain", "--no-tags")
    assert not _has_promisor_config(checkout)

    fetch_full_commit_graph(checkout)

    assert not _has_promisor_config(checkout)
    assert "v0.21.6" in _git(checkout, "tag", "--list")


def test_tag_fetch_keeps_partial_clone_semantics_and_tags(tmp_path):
    # Installer clones ship as partial; the filter must keep applying there so
    # trees stay on demand (the original intent of the flag).
    server = _server_repo(tmp_path)
    _git(server, "config", "uploadpack.allowFilter", "true")
    checkout = _clone(tmp_path, server, "partial", "--no-tags", "--filter=tree:0")
    assert _has_promisor_config(checkout)

    fetch_full_commit_graph(checkout)

    # Still a partial clone with the *same* filter, and the tags still arrived.
    assert _has_promisor_config(checkout)
    assert _git(checkout, "config", "--get", "remote.origin.partialclonefilter") == "tree:0"
    assert "v0.21.6" in _git(checkout, "tag", "--list")


def test_tag_fetch_preserves_a_non_default_partial_clone_filter(tmp_path):
    # A partial clone whose filter the user set themselves (blob:none) must not
    # be silently tightened to tree:0 by the tag fetch — repeat the clone's own
    # filter instead.
    server = _server_repo(tmp_path)
    _git(server, "config", "uploadpack.allowFilter", "true")
    checkout = _clone(tmp_path, server, "blobless", "--no-tags", "--filter=blob:none")
    assert _has_promisor_config(checkout)

    fetch_full_commit_graph(checkout)

    assert _git(checkout, "config", "--get", "remote.origin.partialclonefilter") == "blob:none"
    assert "v0.21.6" in _git(checkout, "tag", "--list")


def test_tag_fetch_respects_a_de_partialised_former_partial_clone(tmp_path):
    # The exact repair from the report: unsetting the promisor keys is the
    # supported escape from the `should_include_obj` failure, and the next
    # tag fetch must not undo it.
    server = _server_repo(tmp_path)
    _git(server, "config", "uploadpack.allowFilter", "true")
    checkout = _clone(tmp_path, server, "repaired", "--no-tags", "--filter=tree:0")
    assert _has_promisor_config(checkout)
    _git(checkout, "config", "--unset", "remote.origin.promisor")
    _git(checkout, "config", "--unset", "remote.origin.partialclonefilter")
    assert not _has_promisor_config(checkout)

    fetch_full_commit_graph(checkout)

    assert not _has_promisor_config(checkout)
    assert "v0.21.6" in _git(checkout, "tag", "--list")


def test_shallow_unshallow_fetch_leaves_promisor_config_alone(tmp_path):
    # A shallow clone is unshallowed by the same fetch; after --unshallow the
    # repo must still not gain promisor keys it did not have.
    server = _server_repo(tmp_path)
    _git(server, "config", "uploadpack.allowFilter", "true")
    checkout = _clone(tmp_path, server, "shallow", "--no-tags", "--depth", "1")
    assert not _has_promisor_config(checkout)

    assert fetch_full_commit_graph(checkout) is True

    assert not _has_promisor_config(checkout)
    assert _git(checkout, "rev-parse", "--is-shallow-repository") == "false"
    assert "v0.21.6" in _git(checkout, "tag", "--list")


def _server_repo_with_retired_blob(tmp_path: Path) -> Path:
    # History carrying a big blob that HEAD no longer references, so an
    # unfiltered history fetch must transfer it while --filter=tree:0 must not.
    root = tmp_path / "server-fat"
    root.mkdir()
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=root, env=env, check=True, capture_output=True)

    git("init", "-q", "-b", "main")
    (root / "tracked").write_text("release\n", encoding="utf-8")
    git("add", "tracked")
    git("commit", "-qm", "release")
    (root / "fat.bin").write_bytes(b"x" * 262_144)
    git("add", "fat.bin")
    git("commit", "-qm", "fat")
    git("tag", "v0.21.5")
    git("rm", "-q", "fat.bin")
    git("commit", "-qm", "drop fat")
    git("tag", "v0.21.6")
    return root


def _blob_sizes(checkout: Path) -> "list[int]":
    out = _git(checkout, "cat-file", "--batch-all-objects",
               "--batch-check=%(objecttype) %(objectsize)")
    return [int(line.split()[1]) for line in out.splitlines()
            if line.startswith("blob ")]


def test_shallow_non_partial_fetch_keeps_the_tree0_filter(tmp_path):
    # The installer shipped `git clone --depth 1` before the treeless switch:
    # a shallow, NON-partial checkout. Its --unshallow fetch must stay filtered
    # (tree:0), or every update pays for an unfiltered full-history transfer —
    # 303 MiB vs 33 MiB on the real graph (review blocker on #122430).
    server = _server_repo_with_retired_blob(tmp_path)
    _git(server, "config", "uploadpack.allowFilter", "true")
    checkout = _clone(tmp_path, server, "shallow-nonpartial", "--no-tags", "--depth", "1")
    assert not _has_promisor_config(checkout)
    assert _git(checkout, "rev-parse", "--is-shallow-repository") == "true"

    assert fetch_full_commit_graph(checkout) is True

    # The transfer stayed bounded by tree:0: the retired 256 KiB blob never
    # landed (the full history the commits describe stays undownloaded).
    assert _blob_sizes(checkout) and max(_blob_sizes(checkout)) < 262_144
    # ...and the fetch still did both of its jobs: unshallowed, tags arrived,
    # and the config file gained nothing.
    assert _git(checkout, "rev-parse", "--is-shallow-repository") == "false"
    assert not _has_promisor_config(checkout)
    assert _config_values(checkout, "remote.origin.partialclonefilter") == []
    assert "v0.21.6" in _git(checkout, "tag", "--list")
