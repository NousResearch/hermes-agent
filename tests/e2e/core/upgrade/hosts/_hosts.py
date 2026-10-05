"""Shared staging for the host-shape suites (odd paths, PATH shapes, permissions, libc).

Everything runs through ``tests/e2e/core/upgrade/_helpers.py`` (bwrap sandbox, allowlisted env)
and ``_install_helpers.py`` (local bare origin + URL rewrite, git wrapper). This module only adds
what a host shape needs on top: a sandbox rooted at an arbitrary path, the configured fake
provider, a one-shot turn, a background ``hermes gateway run`` and a login-shell probe.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from tests.e2e.core.upgrade import _helpers as H
from tests.e2e.core.upgrade import _install_helpers as I
from tests.fakes.fake_llm_provider import FakeLLMServer

UPDATE_TIMEOUT = 1500
# What `prepare_launch` prints when it decides the install is stale and re-runs the source-update
# completion. A healthy install prints neither on a plain launch.
COMPLETION_BANNERS = ("completing source-update dependencies", "finishing an interrupted source update")


def reran_completion(cp: subprocess.CompletedProcess) -> bool:
    return any(b in cp.stdout + cp.stderr for b in COMPLETION_BANNERS)


def make_origin(root: Path) -> Path:
    """``I.make_origin`` at the commit under test, serving partial clones like GitHub does.

    install.sh clones ``--filter=blob:none``. Without ``uploadpack.allowFilter`` the local origin
    ignores the filter and must pack every blob in history, which a blobless developer
    checkout (the ``--shared`` object store) cannot serve.
    """
    origin = I.make_origin(root, I.head_sha())
    I.git("config", "uploadpack.allowFilter", "true", cwd=origin)
    return origin


def new_sandbox(root: Path, origin: Path) -> I.Sandbox:
    """``I.new_sandbox`` with the launch-time source-update check a real user's process runs.

    The shared sandbox env sets ``HERMES_DISABLE_LAZY_INSTALLS=1``, which also short-circuits
    ``prepare_launch`` (the per-launch "is this install current?" check). Host-shape bugs live in
    exactly that check (path strings that differ from one launch to the next), so these
    sandboxes run without it, as users do.
    """
    sb = I.new_sandbox(root, origin)
    sb.env.pop("HERMES_DISABLE_LAZY_INSTALLS", None)
    return sb


def ok(cp: subprocess.CompletedProcess) -> subprocess.CompletedProcess:
    assert cp.returncode == 0 and I.TRACEBACK not in cp.stdout + cp.stderr, I.describe(cp)
    return cp


def configure(sb: I.Sandbox, provider: FakeLLMServer) -> None:
    """Point the install at the fake provider with a current-version config."""
    ver = ok(sb.run([sb.python, "-c", "from hermes_cli.config_defaults import DEFAULT_CONFIG as D; print(D['_config_version'])"]))
    version = int(ver.stdout.strip().splitlines()[-1])
    (sb.hermes_home / "config.yaml").write_text(I.provider_config(provider.base_url, version), encoding="utf-8")
    (sb.hermes_home / ".env").write_text(f"OPENAI_API_KEY={I.FAKE_KEY}\n", encoding="utf-8")


def turn(sb: I.Sandbox, provider: FakeLLMServer, marker: str, *, env: dict | None = None) -> subprocess.CompletedProcess:
    """One ``hermes -z`` turn that must reach the provider exactly once and print its reply."""
    n = len(provider.main_requests())
    cp = H.run([sb.hermes, "-z", marker], env=env or sb.env, cwd=sb.root, writable=[sb.root], timeout=900)
    assert cp.returncode == 0 and I.TRACEBACK not in cp.stdout + cp.stderr, I.describe(cp)
    assert provider.default_text in cp.stdout, "the reply never reached stdout:\n" + I.describe(cp)
    new = provider.main_requests()[n:]
    assert len(new) == 1 and marker in json.dumps(new[0]["messages"]), (
        f"one-shot turn reached the provider {len(new)} times (want 1):\n" + I.describe(cp))
    return cp


def login_shell(sb: I.Sandbox, script: str, *, path: str = "/usr/local/bin:/usr/bin:/bin") -> subprocess.CompletedProcess:
    """Run ``script`` in a new interactive login bash: PATH comes only from the rc files."""
    env = dict(sb.env, PATH=path)
    bash = shutil.which("bash")
    assert bash is not None, "bash required"
    return H.run([bash, "-lic", script], env=env, cwd=sb.root, writable=[sb.root], timeout=120)


def interpreter_paths(sb: I.Sandbox, python: str | None = None) -> dict:
    """sys.path / PATH / sys.executable as the installed interpreter sees them."""
    code = ("import json, os, sys; print(json.dumps({'sys_path': sys.path, 'path': os.environ.get('PATH', ''),"
            " 'executable': sys.executable}))")
    cp = ok(sb.run([python or sb.python, "-c", code]))
    return json.loads(cp.stdout.strip().splitlines()[-1])


def missing_entries(entries: list[str]) -> list[str]:
    """Entries that name a location that does not exist (a mangled or half-quoted path)."""
    return [e for e in entries
            if e and not os.path.exists(e) and not e.endswith(".zip") and not e.endswith(".__path_hook__")]


class Gateway:
    """``hermes gateway run`` in the foreground of its own sandbox, as a user starts it by hand."""

    def __init__(self, sb: I.Sandbox):
        self.sb = sb
        self.log = sb.root / "gateway-run.log"
        self.proc: subprocess.Popen | None = None

    def state(self) -> dict:
        f = self.sb.hermes_home / "gateway_state.json"
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def start(self, timeout: float = 180) -> dict:
        fh = open(self.log, "w", encoding="utf-8")  # noqa: SIM115 - handed to the child
        self.proc = subprocess.Popen(
            H.sandbox_argv([self.sb.hermes, "gateway", "run"], writable=[self.sb.root]),
            env=self.sb.env, cwd=str(self.sb.root), stdin=subprocess.DEVNULL, stdout=fh,
            stderr=subprocess.STDOUT, start_new_session=True)
        fh.close()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            st = self.state()
            if st.get("gateway_state") == "running":
                return st
            if self.proc.poll() is not None:
                break
            time.sleep(0.5)
        raise AssertionError(
            f"gateway did not reach gateway_state=running (rc={self.proc.poll()}, state={self.state()})\n"
            + self.tail())

    def tail(self, n: int = 6000) -> str:
        parts = []
        for label, p in (("gateway stdout", self.log), ("agent.log", self.sb.hermes_home / "logs" / "agent.log"),
                         ("errors.log", self.sb.hermes_home / "logs" / "errors.log")):
            try:
                parts.append(f"--- {label} ---\n{p.read_text(encoding='utf-8', errors='replace')[-n:]}")
            except OSError:
                parts.append(f"--- {label} --- (missing)")
        return "\n".join(parts)

    def stop(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                pass
            except ProcessLookupError:
                pass
        H.kill_tree(self.proc)
