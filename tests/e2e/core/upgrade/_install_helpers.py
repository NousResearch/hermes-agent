"""Helpers for the fresh-install / update-with-user-state / profile-update suites.

Every process runs through ``_helpers.run`` (bwrap sandbox, allowlisted env, fake HOME). On top
of that this module stages what a real user machine looks like to the installer and updater:

* a local bare ``origin`` (``--shared`` onto this checkout: no network, no object copy) that
  the official clone URLs are rewritten to by the sandbox's own ``~/.gitconfig``, plus a ``git``
  wrapper that reports the official URL for ``remote get-url origin`` (``insteadOf`` would
  otherwise expose the local path and send the updater down the fork path);
* a managed ``$HERMES_HOME/bin/uv`` pointing at the host's uv (warm cache, no download);
* ``TMPDIR`` inside the sandbox root (the host's is not writable in the sandbox).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path

from tests.e2e.core.upgrade import _helpers as H

OFFICIAL_HTTPS = "https://github.com/NousResearch/hermes-agent.git"
OFFICIAL_SSH = "git@github.com:NousResearch/hermes-agent.git"
TRACEBACK = "Traceback (most recent call last)"
FAKE_KEY = "sk-fake-e2e-install-update"


def real_uv() -> str | None:
    cand = shutil.which("uv") or str(H.REAL_HOME / ".hermes" / "bin" / "uv")
    return cand if cand and Path(cand).exists() else None


def git(*args: str, cwd: Path, check: bool = True, env: dict | None = None) -> str:
    cp = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True,
                        env=env or {**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    if check and cp.returncode != 0:
        raise AssertionError(f"git {args} failed in {cwd}: {cp.stderr}")
    return cp.stdout.strip()


def head_sha() -> str:
    """The commit the sandbox installs: HEAD, or HERMES_E2E_INSTALL_REF (a patched commit object,
    used to prove a scenario red against a sabotaged install without moving the branch)."""
    return git("rev-parse", os.environ.get("HERMES_E2E_INSTALL_REF") or "HEAD", cwd=H.WORKTREE)


def make_origin(root: Path, ref: str) -> Path:
    """Bare origin with ``main`` at ``ref``."""
    origin = root / "origin.git"
    git("clone", "-q", "--bare", "--shared", "--no-tags", str(H.WORKTREE), str(origin), cwd=root)
    git("update-ref", "refs/heads/main", ref, cwd=origin)
    git("symbolic-ref", "HEAD", "refs/heads/main", cwd=origin)
    git("config", "uploadpack.allowAnySHA1InWant", "true", cwd=origin)
    return origin


def publish_commit(origin: Path, scratch: Path, message: str, files: dict[str, str]) -> str:
    """Add one upstream commit on origin/main (a new release the user updates to); returns its sha."""
    work = scratch / f"publish-{hashlib.sha1(message.encode()).hexdigest()[:8]}"
    if work.exists():
        shutil.rmtree(work)
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_AUTHOR_NAME": "e2e", "GIT_AUTHOR_EMAIL": "e2e@example.invalid",
           "GIT_COMMITTER_NAME": "e2e", "GIT_COMMITTER_EMAIL": "e2e@example.invalid"}
    git("clone", "-q", "--shared", "--no-checkout", "-b", "main", str(origin), str(work), cwd=scratch, env=env)
    git("reset", "-q", "main", cwd=work, env=env)
    for rel, text in files.items():
        p = work / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        git("add", "--", rel, cwd=work, env=env)
    git("commit", "-q", "-m", message, cwd=work, env=env)
    git("push", "-q", "origin", "HEAD:main", cwd=work, env=env)
    return git("rev-parse", "HEAD", cwd=work)


@dataclass
class Sandbox:
    root: Path
    env: dict[str, str]

    @property
    def home(self) -> Path:
        return Path(self.env["HOME"])

    @property
    def hermes_home(self) -> Path:
        return Path(self.env["HERMES_HOME"])

    @property
    def checkout(self) -> Path:
        return self.hermes_home / "hermes-agent"

    @property
    def hermes(self) -> str:
        """The command the installer put on PATH, as a user's shell resolves it."""
        return str(self.home / ".local" / "bin" / "hermes")

    def run(self, argv: list[str], *, timeout: float = 600, cwd: Path | None = None,
            input: str | None = None) -> subprocess.CompletedProcess:
        return H.run(argv, env=self.env, cwd=cwd or self.root, writable=[self.root], timeout=timeout, input=input)

    def cli(self, *args: str, timeout: float = 600) -> subprocess.CompletedProcess:
        return self.run([self.hermes, *args], timeout=timeout)


def new_sandbox(root: Path, origin: Path | None = None, *, pythonpath: Path | None = None) -> Sandbox:
    """Empty fake HOME: no ``~/.hermes`` at all, only what a fresh machine with uv/git/node has."""
    root.mkdir(parents=True, exist_ok=True)
    wrap = root / "wrap"
    env = H.isolated_env(root, extra_path=[wrap], pythonpath=pythonpath)
    home = Path(env["HOME"])
    shutil.rmtree(home / ".hermes")
    (root / "tmp").mkdir(exist_ok=True)
    env["TMPDIR"] = str(root / "tmp")
    env["SHELL"] = "/bin/bash"
    # A fresh machine: ~/.local/bin is NOT on PATH yet; the installer must wire it up.
    managed = home / ".hermes" / "bin"
    managed.mkdir(parents=True)
    uv = managed / "uv"
    uv.write_text(f'#!/bin/sh\nif [ "$1" = self ]; then exit 0; fi\nexec "{real_uv()}" "$@"\n', encoding="utf-8")
    uv.chmod(0o755)
    if origin is not None:
        (home / ".gitconfig").write_text(
            f'[url "file://{origin}"]\n  insteadOf = {OFFICIAL_HTTPS}\n  insteadOf = {OFFICIAL_SSH}\n', encoding="utf-8")
    wrap.mkdir(exist_ok=True)
    real_git = shutil.which("git")
    shim = wrap / "git"
    shim.write_text(
        "#!/usr/bin/env bash\n"
        'p2=""; p1=""\n'
        'for a in "$@"; do\n'
        f'  if [ "$p2" = remote ] && [ "$p1" = get-url ] && [ "$a" = origin ]; then echo "{OFFICIAL_HTTPS}"; exit 0; fi\n'
        '  p2="$p1"; p1="$a"\n'
        "done\n"
        f'exec "{real_git}" "$@"\n', encoding="utf-8")
    shim.chmod(0o755)
    return Sandbox(root=root, env=env)


def run_installer(sb: Sandbox, *, timeout: float = 1800) -> subprocess.CompletedProcess:
    """HEAD's scripts/install.sh, non-interactive, as the documented `curl | bash` run does it."""
    script = sb.root / "install.sh"
    shutil.copy(H.WORKTREE / "scripts" / "install.sh", script)
    return sb.run(["bash", str(script), "--skip-setup", "--skip-browser", "--skip-computer-use", "--non-interactive"],
                  timeout=timeout, input="")


def provider_config(base_url: str, version: int | None, extra: str = "") -> str:
    """A hand-edited user config: comments, a key HEAD does not know, and the fake provider."""
    ver = f"_config_version: {version}\n" if version is not None else ""
    return (
        "# My Hermes config. Hand-edited; comments must survive.\n"
        "model:\n"
        "  provider: custom\n"
        f"  base_url: {base_url}  # local fake provider\n"
        "  default: fake-model\n"
        "  context_length: 128000\n"
        "agent:\n"
        "  api_max_retries: 1   # keep it snappy\n"
        "# A key this Hermes version does not know about (from a plugin or a newer release).\n"
        "my_future_section:\n"
        "  nested_flag: true\n"
        "  words: \"unicode é 漢字 and a # that is not a comment\"\n"
        f"{extra}{ver}"
    )


def tree_digest(path: Path) -> dict[str, str]:
    """{relative path: sha256} of every regular file / symlink target under ``path``.
    ``__pycache__`` is skipped: Python writes it whenever the code is imported (a plugin the
    config migration enabled loads on the next run); it is not the user's content."""
    out: dict[str, str] = {}
    for p in sorted(path.rglob("*")):
        if "__pycache__" in p.relative_to(path).parts:
            continue
        rel = str(p.relative_to(path))
        if p.is_symlink():
            out[rel] = "link:" + os.readlink(p)
        elif p.is_file():
            out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def db_state(db: Path) -> dict:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        integrity = con.execute("PRAGMA integrity_check").fetchall()
        sessions = sorted(r[0] for r in con.execute("SELECT id FROM sessions"))
        msgs = con.execute("SELECT session_id, role, content FROM messages ORDER BY id").fetchall()
    finally:
        con.close()
    return {"integrity": integrity, "sessions": sessions,
            "messages": hashlib.sha256(repr(msgs).encode()).hexdigest(), "n_messages": len(msgs)}


def cron_jobs(profile_home: Path) -> list[dict]:
    f = profile_home / "cron" / "jobs.json"
    if not f.exists():
        return []
    data = json.loads(f.read_text(encoding="utf-8"))
    items = data.get("jobs", data) if isinstance(data, dict) else data
    return [j for j in items if isinstance(j, dict)]


def describe(cp: subprocess.CompletedProcess) -> str:
    return H.describe(cp)
