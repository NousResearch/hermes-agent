"""Shared staging for the PM lifecycle suite (``tests/e2e/core/upgrade/pm``).

Built on ``_helpers`` (bwrap sandbox, allowlisted env) and ``_install_helpers`` (local bare origin,
HEAD's own ``scripts/install.sh``). Every cell starts from a real install and then drives the real
user entry points (``hermes update``, ``hermes pm ...``, ``hermes doctor``, ``hermes gateway ...``).

What a PM install looks like on disk, and what the cells read back:

* ``$HERMES_HOME/installs/<key>/facts.json`` names the selected dependency generation
  (``packages.venv.environment``) and its recorded extras;
* ``$HERMES_HOME/installs/<key>/environments/<gen>/{venv,workspace}`` are the generations;
* ``$HERMES_HOME/installs/<key>/source-completion-pending`` is the owed update tail.

A PM generation is keyed on the bytes of ``uv.lock`` (plus extras, Python, plugin members), so a
release that changes ``uv.lock`` is what makes ``hermes update`` build a NEW generation.
``publish_dependency_release`` publishes exactly that: ``uv.lock`` plus a trailing TOML comment,
which ``uv sync --locked`` accepts unchanged but which moves the generation key.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

from tests.e2e.core.upgrade import _helpers as H
from tests.e2e.core.upgrade import _install_helpers as I

UPDATE_TIMEOUT = 1500
TRACEBACK = I.TRACEBACK


def ok(cp: subprocess.CompletedProcess, what: str = "") -> subprocess.CompletedProcess:
    assert cp.returncode == 0 and TRACEBACK not in cp.stdout + cp.stderr, (
        (f"{what}\n" if what else "") + I.describe(cp))
    return cp


def install_head(root: Path) -> tuple[I.Sandbox, Path]:
    """HEAD installed by its own ``scripts/install.sh`` into an empty sandbox HOME."""
    origin = I.make_origin(root, I.head_sha())
    sb = I.new_sandbox(root / "sb", origin)
    ok(I.run_installer(sb), "install.sh failed on an empty HOME")
    return sb, origin


def configure(sb: I.Sandbox, base_url: str, extra: str = "", env_extra: str = "") -> None:
    """The user's config at HEAD's schema version, pointing at the fake provider."""
    ver = ok(sb.run([sb.python, "-c",
                     "from hermes_cli.config_defaults import DEFAULT_CONFIG as D; print(D['_config_version'])"]))
    version = int(ver.stdout.strip().splitlines()[-1])
    (sb.hermes_home / "config.yaml").write_text(I.provider_config(base_url, version, extra), encoding="utf-8")
    (sb.hermes_home / ".env").write_text(f"OPENAI_API_KEY={I.FAKE_KEY}\n{env_extra}", encoding="utf-8")


def state_dir(sb: I.Sandbox) -> Path:
    from pm.environments import install_key

    return sb.hermes_home / "installs" / install_key(sb.checkout)


def facts(sb: I.Sandbox) -> dict:
    return json.loads((state_dir(sb) / "facts.json").read_text(encoding="utf-8"))


def selected_generation(sb: I.Sandbox) -> Path:
    """``environments/<gen>`` that facts.json selects (the parent of its ``venv``)."""
    return Path(facts(sb)["packages"]["venv"]["environment"]).parent


def generations(sb: I.Sandbox) -> list[str]:
    envs = state_dir(sb) / "environments"
    return sorted(p.name for p in envs.iterdir() if p.is_dir()) if envs.is_dir() else []


def pending_marker(sb: I.Sandbox) -> Path:
    return state_dir(sb) / "source-completion-pending"


def lazy_env(sb: I.Sandbox) -> dict[str, str]:
    """The sandbox env WITHOUT the suite-wide ``HERMES_DISABLE_LAZY_INSTALLS``: a real user's
    launch, which finishes an owed source-update tail before it imports the app."""
    env = dict(sb.env)
    env.pop("HERMES_DISABLE_LAZY_INSTALLS", None)
    return env


def run_env(sb: I.Sandbox, argv: list[str], env: dict[str, str], *, timeout: float = 600,
            input: str | None = None) -> subprocess.CompletedProcess:
    return H.run(argv, env=env, cwd=sb.root, writable=[sb.root], timeout=timeout, input=input)


def managed_imports(sb: I.Sandbox, *modules: str) -> dict[str, str]:
    """{module: "ok" | "<error>"} imported by the selected generation's interpreter, booted the way
    the launcher boots it (its own site; nothing from the test process)."""
    code = ("import importlib, json, sys\nout = {}\n"
            "for m in sys.argv[1:]:\n"
            "    try:\n        importlib.import_module(m); out[m] = 'ok'\n"
            "    except BaseException as e:\n        out[m] = f'{type(e).__name__}: {e}'\n"
            "print(json.dumps(out))\n")
    cp = ok(sb.run([sb.python, "-c", code, *modules]))
    return json.loads(cp.stdout.strip().splitlines()[-1])


def update(sb: I.Sandbox, *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return run_env(sb, [sb.hermes, "update", "--yes", "--branch", "main"], env or sb.env, timeout=UPDATE_TIMEOUT)


def publish_dependency_release(origin: Path, scratch: Path, n: int) -> str:
    """Publish a release whose ``uv.lock`` bytes differ (same resolution): the next update must
    build and select a new dependency generation."""
    lock = (I.git("show", "main:uv.lock", cwd=origin) + f"\n# e2e dependency release {n}\n")
    return I.publish_commit(origin, scratch, f"release: e2e dependency release {n}", {"uv.lock": lock})


def diagnostics(sb: I.Sandbox, *cps: subprocess.CompletedProcess) -> str:
    """Everything needed to judge a PM failure without re-running it."""
    parts = [I.describe(cp) for cp in cps]
    try:
        parts.append(f"--- facts.json ---\n{json.dumps(facts(sb), indent=1)}")
    except (OSError, ValueError) as exc:
        parts.append(f"--- facts.json unreadable: {exc}")
    parts.append(f"--- generations --- {generations(sb)}")
    parts.append(f"--- pending marker --- {pending_marker(sb).exists()}")
    receipts = sorted((sb.hermes_home / "logs").glob("*.log")) if (sb.hermes_home / "logs").is_dir() else []
    for log in receipts:
        text = log.read_text(encoding="utf-8", errors="replace")
        parts.append(f"--- {log.name} (tail) ---\n{text[-2500:]}")
    return "\n".join(parts)


def kill_when(proc: subprocess.Popen, predicate, *, timeout: float, what: str) -> float:
    """SIGKILL the whole sandbox the moment ``predicate()`` is true; returns when it fired.

    The sandbox runs in its own session and PID namespace (``--die-with-parent``), so killing its
    process group takes every descendant with it: a power cut at that exact phase."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise AssertionError(f"process exited rc={proc.returncode} before {what}")
        if predicate():
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=60)
            return time.monotonic()
        time.sleep(0.02)
    H.kill_tree(proc)
    raise AssertionError(f"timed out after {timeout}s waiting for {what}")


def arm_pending_tail(sb: I.Sandbox) -> None:
    """The marker an update killed between publish and tail leaves (``arm_completion``'s bytes;
    ``test_interrupted_update`` produces it for real)."""
    pending_marker(sb).write_text("source update tail not finished\n", encoding="utf-8")


class Gateway:
    """``hermes gateway run`` in the sandbox (the user's foreground/systemd ``ExecStart``)."""

    def __init__(self, sb: I.Sandbox, env: dict[str, str], log: Path, argv0: list[str] | None = None):
        self.sb, self.log = sb, log
        (sb.hermes_home / "gateway_state.json").unlink(missing_ok=True)
        self._out = log.open("w")
        self.proc = subprocess.Popen(
            H.sandbox_argv([*(argv0 or []), sb.hermes, "gateway", "run"], writable=[sb.root]),
            env=env, cwd=str(sb.root), stdin=subprocess.DEVNULL, stdout=self._out, stderr=subprocess.STDOUT,
            text=True, start_new_session=True)

    def state(self) -> str:
        try:
            return json.loads((self.sb.hermes_home / "gateway_state.json").read_text(encoding="utf-8")).get(
                "gateway_state", "")
        except (OSError, ValueError):
            return ""

    def wait_running(self, timeout: float = 600) -> float:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self.state() == "running":
                return time.monotonic() - t0
            if self.proc.poll() is not None:
                break
            time.sleep(0.25)
        rc = self.proc.poll()
        self.stop()
        raise AssertionError(f"gateway never reported running (rc={rc}, state={self.state()!r})\n"
                             f"--- gateway output ---\n{self.output()[-5000:]}")

    def pids(self) -> list[int]:
        """Host pids of every process in the gateway's sandbox (bwrap and its descendants)."""
        children: dict[int, list[int]] = {}
        for d in Path("/proc").iterdir():
            if d.name.isdigit():
                try:
                    ppid = int((d / "stat").read_text().rsplit(")", 1)[1].split()[1])
                except (OSError, ValueError, IndexError):
                    continue
                children.setdefault(ppid, []).append(int(d.name))
        out, todo = [], [self.proc.pid]
        while todo:
            pid = todo.pop()
            out.append(pid)
            todo += children.get(pid, [])
        return out

    def output(self) -> str:
        if not self._out.closed:
            self._out.flush()
        return self.log.read_text(encoding="utf-8", errors="replace")

    def stop(self) -> None:
        """SIGTERM like ``systemctl stop``; SIGKILL the sandbox if it does not exit."""
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=60)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                H.kill_tree(self.proc)
        self._out.close()


def process_files(pid: int) -> tuple[str, list[str], dict[str, str]]:
    """(exe, mapped files, environ) of a host pid; empty when it already exited."""
    base = Path(f"/proc/{pid}")
    try:
        exe = os.readlink(base / "exe")
        maps = sorted({line.split(None, 5)[5].strip() for line in (base / "maps").read_text().splitlines()
                       if len(line.split(None, 5)) == 6 and line.split(None, 5)[5].startswith("/")})
        environ = dict(kv.split("=", 1) for kv in (base / "environ").read_bytes().decode(errors="replace").split("\0")
                       if "=" in kv)
    except (OSError, ValueError):
        return "", [], {}
    return exe, maps, environ


def turn(sb: I.Sandbox, provider, marker: str) -> subprocess.CompletedProcess:
    """One real ``hermes -z`` turn through the loopback provider; the request must carry ``marker``."""
    n = len(provider.main_requests())
    cp = run_env(sb, [sb.hermes, "-z", marker], lazy_env(sb), timeout=600)
    new = provider.main_requests()[n:]
    assert cp.returncode == 0 and len(new) == 1 and marker in json.dumps(new[0]["messages"]), (
        f"`hermes -z` did not reach the provider ({len(new)} requests)\n"
        + diagnostics(sb, cp))
    return cp
