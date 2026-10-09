"""A real install served over real smart HTTP, and the real ``hermes update`` against it.

``World`` is one sandboxed machine (``_install_helpers.new_sandbox``: bwrap, empty fake HOME, the
git wrapper that reports the official remote URL) whose ``~/.gitconfig`` rewrites the official
GitHub URL to a loopback ``git http-backend`` origin (``_smart_http``). The install itself comes
from a real ``scripts/install.sh``: N-1's own copy for an N-1 install (it makes the depth-1
single-branch clone N-1 users have), HEAD's for a HEAD install (a ``--filter=blob:none`` clone).
``preclone`` seeds a checkout the installer then adopts, for the clone shapes users got another
way (a manual full clone, a ``blob:none`` fallback clone).

Everything a cell asserts is user-visible: the exit code and transcript of ``hermes update``, the
checkout's HEAD / branch / working tree / refs, the update receipt, and whether ``hermes
--version`` still answers. ``diag()`` renders all of it, plus the HTTP request log, for a failure
message that explains itself.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import pytest

from tests.e2e.core.upgrade import _helpers as H
from tests.e2e.core.upgrade import _install_helpers as I
from tests.e2e.core.upgrade.git._smart_http import GitHTTPServer, serve_bare

REPO = "hermes-agent.git"
SUCCESS = "Update complete"
TRACEBACK = I.TRACEBACK

PYTESTMARK = [
    pytest.mark.platforms("linux"),
    # Every `hermes update` runs inside the bwrap sandbox against a throwaway install in tmp_path.
    pytest.mark.live_system_guard_bypass,
    pytest.mark.skipif(H.sandbox_required_reason() is not None, reason=str(H.sandbox_required_reason())),
    pytest.mark.skipif(shutil.which("git") is None, reason="git required"),
    pytest.mark.skipif(I.real_uv() is None, reason="uv required"),
]


def output(cp: subprocess.CompletedProcess) -> str:
    return (cp.stdout or "") + (cp.stderr or "")


def reported_success(cp: subprocess.CompletedProcess) -> bool:
    return cp.returncode == 0 or SUCCESS in output(cp)


def n1_tag() -> str:
    """N-1: the newest release tag strictly before HEAD."""
    return I.git("describe", "--tags", "--abbrev=0", "--match", "v20[0-9][0-9].*", "HEAD~1", cwd=H.WORKTREE)


def n1_base() -> str:
    """The commit an N-1 install sits on: N-1's tag, or ``HERMES_E2E_N1_REF`` (a patched commit
    object, used to prove red a cell whose bug lives in N-1's own updater code)."""
    return commit_of(os.environ.get("HERMES_E2E_N1_REF") or n1_tag())


def commit_of(ref: str) -> str:
    return I.git("rev-parse", f"{ref}^{{commit}}", cwd=H.WORKTREE)


def installer_script(ref: str | None) -> str:
    """``scripts/install.sh`` as of ``ref`` (None: this worktree's copy)."""
    if ref is None:
        return (H.WORKTREE / "scripts" / "install.sh").read_text(encoding="utf-8")
    return I.git("show", f"{ref}:scripts/install.sh", cwd=H.WORKTREE) + "\n"


@dataclass
class World:
    root: Path
    srv: GitHTTPServer
    bare: Path
    sb: I.Sandbox
    transcripts: list[str] = field(default_factory=list)
    install_bytes: int = 0
    install_shape: dict[str, str] = field(default_factory=dict)

    # -- upstream ----------------------------------------------------------------------------
    def set_main(self, sha: str) -> None:
        I.git("update-ref", "refs/heads/main", sha, cwd=self.bare)

    def publish(self, message: str, files: dict[str, str]) -> str:
        """A new upstream commit on origin main (the release the user updates to)."""
        return I.publish_commit(self.bare, self.root, message, files)

    # -- the install ---------------------------------------------------------------------------
    @property
    def checkout(self) -> Path:
        return self.sb.checkout

    def git_env(self) -> dict[str, str]:
        """Host-side git with the sandbox's HOME: its URL rewrite, never the real GitHub."""
        return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(self.sb.home),
                "GIT_CONFIG_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C",
                "GIT_AUTHOR_NAME": "user", "GIT_AUTHOR_EMAIL": "user@example.invalid",
                "GIT_COMMITTER_NAME": "user", "GIT_COMMITTER_EMAIL": "user@example.invalid"}

    def git(self, *args: str, check: bool = True) -> str:
        return I.git(*args, cwd=self.checkout, check=check, env=self.git_env())

    def head(self) -> str:
        return self.git("rev-parse", "HEAD")

    def branch(self) -> str:
        return self.git("rev-parse", "--abbrev-ref", "HEAD")

    def config(self, key: str) -> str:
        return self.git("config", "--get", key, check=False)

    def shape(self) -> dict[str, str]:
        """The clone's shape: partial-clone filter and shallowness."""
        return {"promisor": self.config("remote.origin.promisor") or "-",
                "filter": self.config("remote.origin.partialclonefilter") or "-",
                "shallow": self.git("rev-parse", "--is-shallow-repository")}

    def status(self) -> str:
        return self.git("status", "--porcelain=v1", "--untracked-files=all")

    def refs(self, *patterns: str) -> list[str]:
        return self.git("for-each-ref", "--format=%(refname)", *patterns).split()

    def refs_containing(self, sha: str) -> list[str]:
        """Refs (branches, stash, backups, tags) that keep ``sha`` reachable; the reflog does not count."""
        return self.git("for-each-ref", "--contains", sha, "--format=%(refname)").split()

    def reset_clean(self) -> None:
        """Per-cell isolation on a shared install: on ``main`` at its current commit, a clean tree,
        no stash, no rescue refs, no git state files left by the previous cell."""
        for leftover in ("rebase-merge", "rebase-apply", "index.lock", "MERGE_HEAD"):
            p = self.checkout / ".git" / leftover
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
        self.git("checkout", "-q", "-f", "main")
        self.git("reset", "-q", "--hard", "refs/remotes/origin/main")
        self.git("clean", "-fdq")
        self.git("stash", "clear")
        for ref in self.refs("refs/hermes-update-backups", "refs/tags/pre-update-*"):
            self.git("update-ref", "-d", ref)
        assert not self.status(), f"reset_clean left a dirty tree:\n{self.status()}"

    def bytes_since(self, mark: int) -> int:
        return sum(r.body_bytes for r in self.srv.since(mark) if r.method == "POST")

    def update(self, *extra: str, timeout: float = 1200) -> subprocess.CompletedProcess:
        cp = self.sb.cli("update", "--yes", "--branch", "main", *extra, timeout=timeout)
        self.transcripts.append(f"$ hermes update --yes --branch main {' '.join(extra)} -> rc={cp.returncode}\n"
                                f"{(cp.stdout or '')[-8000:]}\n{(cp.stderr or '')[-4000:]}")
        return cp

    def version(self) -> subprocess.CompletedProcess:
        return self.sb.cli("--version", timeout=180)

    def receipt(self) -> dict:
        path = self.sb.hermes_home / "logs" / "update_receipts" / "latest.json"
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"unreadable": str(path)}

    def diag(self, cp: subprocess.CompletedProcess | None = None, mark: int = 0) -> str:
        out = []
        if cp is not None:
            out.append(H.describe(cp, limit=8000))
        with contextlib.suppress(Exception):
            out.append(f"HEAD={self.head()} branch={self.branch()} shape={self.shape()}")
            out.append("status:\n" + (self.status() or "  (clean)"))
            out.append("refs:\n" + self.git("for-each-ref", "--format=%(refname) %(objectname:short)",
                                            "refs/heads", "refs/stash", "refs/hermes-update-backups",
                                            "refs/remotes"))
        rec = self.receipt()
        if rec:
            out.append("receipt: " + json.dumps({k: rec.get(k) for k in ("outcome", "status", "exit_code", "steps")
                                                 if k in rec}, default=str)[:3000])
        logs = self.sb.hermes_home / "logs"
        for name in ("update.log", "errors.log"):
            p = logs / name
            if p.is_file():
                out.append(f"--- {p} (tail)\n" + p.read_text(encoding="utf-8", errors="replace")[-3000:])
        out.append("http requests:\n" + self.srv.describe(mark))
        return "\n".join(out)


def _preclone(sb: I.Sandbox, args: list[str], env: dict[str, str]) -> None:
    sb.hermes_home.mkdir(parents=True, exist_ok=True)
    cp = subprocess.run(["git", "clone", "-q", *args, "--branch", "main", I.OFFICIAL_HTTPS, str(sb.checkout)],
                        capture_output=True, text=True, env=env, timeout=900)
    assert cp.returncode == 0, f"pre-clone {args} failed: {cp.stderr[-2000:]}"


@contextlib.contextmanager
def world(root: Path, *, base: str, installer_ref: str | None = None,
          preclone: list[str] | None = None) -> Iterator[World]:
    """Serve origin ``main`` at ``base`` over smart HTTP and install from it with the installer
    of ``installer_ref`` (None: HEAD's). Yields the live ``World``; the server stops on exit."""
    root.mkdir(parents=True, exist_ok=True)
    srv_root = root / "srv"
    srv_root.mkdir(exist_ok=True)
    bare = serve_bare(srv_root, REPO, H.WORKTREE, base)
    with GitHTTPServer(srv_root) as srv:
        sb = I.new_sandbox(root / "sb")
        url = srv.url(REPO)
        (sb.home / ".gitconfig").write_text(
            f'[url "{url}"]\n  insteadOf = {I.OFFICIAL_HTTPS}\n  insteadOf = {I.OFFICIAL_SSH}\n'
            '[user]\n  name = user\n  email = user@example.invalid\n', encoding="utf-8")
        w = World(root=root, srv=srv, bare=bare, sb=sb)
        if preclone is not None:
            _preclone(sb, preclone, w.git_env())
        script = sb.root / "install.sh"
        script.write_text(installer_script(installer_ref), encoding="utf-8")
        cp = sb.run(["bash", str(script), "--non-interactive"], timeout=1800, input="")
        assert cp.returncode == 0, f"install ({installer_ref or 'HEAD'} installer, base {base[:12]}) failed:\n{w.diag(cp)}"
        head = w.head()
        assert head == base, f"installer left HEAD at {head}, expected {base}\n{w.diag(cp)}"
        w.install_bytes = w.bytes_since(0)
        w.install_shape = w.shape()
        yield w


def start_world(root: Path, **kw) -> tuple[contextlib.AbstractContextManager, World]:
    """``world()`` entered by hand, for fixtures that build several installs concurrently."""
    cm = world(root, **kw)
    return cm, cm.__enter__()


def in_parallel(jobs: dict[str, Callable[[], object]]) -> dict[str, object]:
    """Run independent scenarios concurrently; each value is the job's result or the exception it
    raised (re-raised by the cell that reads it, so one broken install fails only its own cells)."""
    out: dict[str, object] = {}
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {name: pool.submit(job) for name, job in jobs.items()}
        for name, fut in futures.items():
            try:
                out[name] = fut.result()
            except BaseException as exc:  # noqa: BLE001 - handed to the cell
                out[name] = exc
    return out
