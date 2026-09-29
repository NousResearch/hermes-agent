"""Prepared publication refuses incomplete or altered unprivileged handoffs."""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from pm import paths
from pm.lock import Lockfile
from pm.store import Store
from scripts.ci import prepared_tools


@pytest.mark.platforms("posix")
def test_real_stage_receipt_roundtrip_and_rejects_tampering(tmp_path, monkeypatch):
    # Use PM's actual stage-only and archive path with an AArch64 ELF pin. No
    # network or cross-target executable is needed to establish the receipt.
    target = "linux-arm64"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "runtime"))
    lock = Lockfile(tmp_path / "source-lock.json")
    monkeypatch.setattr(paths, "lockfile_path", lambda: lock.path)
    monkeypatch.setattr(prepared_tools, "selected", lambda _lock, _target: ["node"])
    elf = bytearray(b"\x7fELF" + b"\0" * 60)
    elf[4:7] = b"\x02\x01\x01"
    elf[18:20] = (0xB7).to_bytes(2, "little")
    raw_archive = io.BytesIO()
    with tarfile.open(fileobj=raw_archive, mode="w:gz") as archive:
        member = tarfile.TarInfo("node-v1.0-linux-arm64/bin/node")
        member.mode, member.size = 0o755, len(elf)
        archive.addfile(member, io.BytesIO(elf))
    raw = raw_archive.getvalue()
    sha = hashlib.sha256(raw).hexdigest()
    lock.set_pin("node", "1.0", {target: {
        "url": "https://example.invalid/node.tar.gz", "sha256": sha,
    }})
    lock.save()
    cache = Store(paths.store_root()).entry(f"fetch-{sha}")
    cache.mkdir(parents=True)
    (cache / "node.tar.gz").write_bytes(raw)
    receipts = tmp_path / "receipts"
    output = receipts / f"prepared-{target}"
    produced = prepared_tools.produce(target, output, lock=lock)
    assert produced["packages"][0]["name"] == "node"
    verified = prepared_tools.validate(receipts, lock, targets=(target,))
    assert len(verified) == 1
    assert verified[0][:2] == ("node", target)
    archive = verified[0][2]
    archive.write_bytes(archive.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="tampered archive"):
        prepared_tools.validate(receipts, lock, targets=(target,))
    archive.unlink()
    with pytest.raises(ValueError, match="missing or tampered archive"):
        prepared_tools.validate(receipts, lock, targets=(target,))


@pytest.mark.platforms("posix")
def test_receipt_rejects_source_change_and_missing_targets(tmp_path, monkeypatch):
    target = "linux-arm64"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "runtime"))
    lock = Lockfile(tmp_path / "pins.json")
    monkeypatch.setattr(paths, "lockfile_path", lambda: lock.path)
    monkeypatch.setattr(prepared_tools, "selected", lambda _lock, _target: ["node"])
    # Create a genuine PM archive roundtrip from a minimal tree, then exercise
    # the publisher's source closure rather than faking a hash comparison.
    from pm.prepare import archive_tree, source_identity

    entry = tmp_path / "entry"
    entry.mkdir()
    (entry / "node").write_bytes(b"tool")
    lock.set_pin("node", "1", {target: {"url": "https://example.invalid/node.tar.gz", "sha256": "a" * 64}})
    lock.save()
    receipts = tmp_path / "receipts"
    directory = receipts / f"prepared-{target}"
    directory.mkdir(parents=True)
    filename = "node--linux-arm64.tar.gz"
    pin = {**archive_tree(entry, target, directory / filename), "source": source_identity(lock, "node", target)}
    (directory / "receipt.json").write_text(json.dumps({
        "schema": 1, "target": target, "packages": [{"name": "node", "file": filename, "pin": pin}],
    }), encoding="utf-8")
    assert prepared_tools.validate(receipts, lock, targets=(target,))[0][3] == pin
    with pytest.raises(ValueError, match="missing or unexpected target"):
        prepared_tools.validate(receipts, lock, targets=(target, "linux-x64"))
    lock.set_pin("node", "2", {target: {"url": "https://example.invalid/node.tar.gz", "sha256": "b" * 64}})
    lock.save()
    with pytest.raises(ValueError, match="source identity"):
        prepared_tools.validate(receipts, lock, targets=(target,))


def test_publication_requires_current_head_review_by_repository_writer(monkeypatch):
    sha = "a" * 40
    reviews = [
        {"user": {"login": "pr-author"}, "state": "APPROVED", "commit_id": sha},
        {"user": {"login": "maintainer"}, "state": "APPROVED", "commit_id": "b" * 40},
    ]
    permission = "write"

    def response(*args, **_kwargs):
        if "reviews?" in args[-1]:
            return json.dumps([reviews]).encode()
        if "/permission" in args[-1]:
            return json.dumps({"permission": permission}).encode()
        raise AssertionError(args)

    monkeypatch.setattr(prepared_tools, "_run", response)
    approved = lambda: prepared_tools._approved_head(
        "ethernet8023/hermes-agent", 1, sha, "pr-author")
    assert not approved()
    reviews.append({"user": {"login": "maintainer"}, "state": "APPROVED", "commit_id": sha})
    assert approved()
    permission = "read"
    assert not approved()
    permission = "write"
    reviews.append({"user": {"login": "maintainer"}, "state": "DISMISSED", "commit_id": sha})
    assert not approved()
    assert prepared_tools._fork_owner_authorized("ethernet8023/hermes-agent", "ethernet8023")
    assert not prepared_tools._fork_owner_authorized("NousResearch/hermes-agent", "ethernet8023")
    assert not prepared_tools._fork_owner_authorized("ethernet8023/hermes-agent", "other-author")


@pytest.mark.platforms("posix")
def test_publisher_bot_commit_uses_git_objects_not_pr_checkout(tmp_path, monkeypatch):
    from pm.prepare import archive_tree, source_identity

    target = "linux-arm64"
    monkeypatch.setattr(prepared_tools, "ALL_TARGETS", (target,))
    monkeypatch.setattr(prepared_tools, "selected", lambda _lock, _target: ["node"])
    repo = tmp_path / "trusted"
    repo.mkdir()
    remote = tmp_path / "origin.git"

    def git(*args):
        return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True).stdout.strip().decode()

    git("init", "-b", "main")
    (repo / "pm").mkdir()
    lock = Lockfile(repo / "pm/lock.json")
    lock.set_pin("node", "1", {target: {"url": "https://example.invalid/node.tar.gz", "sha256": "a" * 64}})
    source_root = Path(prepared_tools.__file__).resolve().parents[2]
    shipped = json.loads((source_root / "pm/lock.json").read_text(encoding="utf-8-sig"))["packages"]
    for name in ("uv", "git"):
        lock.set_pin(name, shipped[name]["version"], shipped[name]["artifacts"])
    git_pin = {"sha256": "f" * 64, "digest": "e" * 64,
               "source": source_identity(lock, "git", "win32-x64")}
    lock.set_prepared("git", "win32-x64", git_pin)
    lock.save()
    (repo / "scripts").mkdir()
    for name in ("gen-bootstrap-pins.py", "install.sh", "install.ps1"):
        shutil.copy2(source_root / "scripts" / name, repo / "scripts" / name)
    (repo / "scripts/install.sh").chmod(0o755)
    git("add", "pm/lock.json", "scripts")
    git("-c", "user.name=test", "-c", "user.email=test@example.org", "commit", "-m", "main")
    trusted_sha = git("rev-parse", "HEAD")
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    git("remote", "add", "origin", str(remote))
    git("push", "origin", "main")
    git("switch", "-c", "feature/prepared")
    # This file would explode if imported; publication must never check it out.
    (repo / "pm/prepare.py").write_text("raise RuntimeError('untrusted execution')\n", encoding="utf-8")
    git("add", "pm/prepare.py")
    git("-c", "user.name=test", "-c", "user.email=test@example.org", "commit", "-m", "PR code")
    head = git("rev-parse", "HEAD")
    git("push", "origin", "feature/prepared")
    git("switch", "main")
    monkeypatch.chdir(repo)
    author = {"login": "fixture-author"}
    monkeypatch.setattr(prepared_tools, "_pr", lambda _repo, _number, sha: {
        "head": {"ref": "feature/prepared", "sha": sha, "repo": {"full_name": _repo}},
        "user": author, "state": "open",
    })
    receipts = tmp_path / "receipts"
    directory = receipts / f"prepared-{target}"
    directory.mkdir(parents=True)
    entry = tmp_path / "entry"
    entry.mkdir()
    (entry / "node").write_bytes(b"portable tool")
    filename = "node--linux-arm64.tar.gz"
    pin = {**archive_tree(entry, target, directory / filename), "source": source_identity(lock, "node", target)}
    (directory / "receipt.json").write_text(json.dumps({
        "schema": 1, "target": target, "packages": [{"name": "node", "file": filename, "pin": pin}],
    }), encoding="utf-8")

    class LocalMirror(prepared_tools.Mirror):
        @property
        def name(self) -> str:
            return "fixture"

        def __init__(self):
            self.objects: dict[str, Path] = {}

        def size(self, sha256: str) -> int | None:
            return self.objects[sha256].stat().st_size if sha256 in self.objects else None

        def put(self, sha256: str, local: Path) -> None:
            self.objects[sha256] = tmp_path / sha256
            shutil.copyfile(local, self.objects[sha256])

        def read_back(self, sha256: str, destination: Path, size: int) -> None:
            shutil.copyfile(self.objects[sha256], destination)
            assert destination.stat().st_size == size
            assert hashlib.sha256(destination.read_bytes()).hexdigest() == sha256

    mirror = LocalMirror()
    original_archive = (directory / filename).read_bytes()
    (directory / filename).write_bytes(original_archive + b"tamper")
    with pytest.raises(ValueError, match="tampered archive"):
        prepared_tools.publish(receipts, "ethernet8023/hermes-agent", 1, head, mirror=mirror)
    assert not mirror.objects
    assert git("ls-remote", "origin", "refs/heads/feature/prepared").split()[0] == head
    (directory / filename).write_bytes(original_archive)
    monkeypatch.setattr(prepared_tools, "_approved_head", lambda *_: False)
    with pytest.raises(ValueError, match="approve the exact PR head"):
        prepared_tools.publish(receipts, "ethernet8023/hermes-agent", 1, head, mirror=mirror)
    assert not mirror.objects
    author["login"] = "ethernet8023"
    commit = prepared_tools.publish(receipts, "ethernet8023/hermes-agent", 1, head, mirror=mirror)
    assert git("rev-parse", "HEAD") == trusted_sha
    assert git("rev-parse", f"{commit}^") == head
    assert git("ls-remote", "origin", "refs/heads/feature/prepared").split()[0] == commit
    published = json.loads(git("show", f"{commit}:pm/lock.json"))
    assert published["packages"]["node"]["prepared"][target] == pin
    generated = git("show", f"{commit}:scripts/install.ps1")
    assert f'PreparedSha256 = "{git_pin["sha256"]}"' in generated
    assert f'PreparedDigest = "{git_pin["digest"]}"' in generated
    assert generated != git("show", f"{head}:scripts/install.ps1")
    assert git("ls-tree", head, "--", "scripts/install.sh").split()[0] == "100755"
    assert git("ls-tree", commit, "--", "scripts/install.sh").split()[0] == "100755"
    with pytest.raises(ValueError, match="branch moved"):
        prepared_tools.publish(receipts, "ethernet8023/hermes-agent", 1, head, mirror=mirror)
    assert prepared_tools.publish(receipts, "ethernet8023/hermes-agent", 1, commit, mirror=mirror) == commit
