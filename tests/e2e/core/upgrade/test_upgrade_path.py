"""Upgrade-path E2E: a real release N-1 git install updated to HEAD by the real `hermes update`.

Class C6 (bricked installs, stale code, lost state after `hermes update`). Each leg:

1. stages a local bare ``origin`` (``--shared`` onto this repository, so no network and no
   object copy) with ``main`` parked at release N-1 (``git describe --tags --abbrev=0 HEAD~1``);
2. clones it as a git-mode install with its own venv (``uv sync --locked --extra all`` from N-1's
   own uv.lock and the warm uv cache: the installer's tier 0 and its editable layout; the installer
   script itself is covered by ``.github/workflows/install-e2e*.yml``);
3. gives it user state created BY THE N-1 CLI ITSELF: sessions in state.db from real one-shot
   turns against the scripted fake provider, a named profile with its own state.db, a cron
   job, and a hand-edited config.yaml at the N-1 schema version with comments, long quoted
   unicode values and a legacy MCP ``disabled: true`` entry (the one documented N-1 -> HEAD
   migration);
4. moves ``origin/main`` to HEAD and runs the real ``hermes update --yes`` non-interactively.

Legs: ``clean``; ``autostash`` (local edits + an orphan update autostash from an earlier run);
``kill_mid_pull`` (SIGKILL while the fast-forward holds ``index.lock``, then the user retries
later; ``torn-tree`` variant: half the changed files already swapped, a strict xfail); ``kill_before_deps``
(SIGKILL after the code swap, when the dependency sync starts: new code on the old venv);
``offline`` (origin unreachable: must fail loudly and leave N-1 working).

Invariants after every leg (fresh processes only): the final update's exit code matches reality
(0 and HEAD checked out and the editable install serves HEAD's tree, or non-zero and nothing
changed); ``hermes --version``, ``hermes doctor`` and a one-shot turn succeed; state.db and the
profile's state.db pass ``integrity_check`` with unchanged row counts and byte-identical
pre-existing messages (also after HEAD's first real use of the DB); the cron job survives; the
venv satisfies HEAD's dependency set; config.yaml of both profiles equals exactly what HEAD's own
non-interactive ``migrate_config`` produces from the pre-update bytes (so the updater never
rewrites anything beyond documented migrations), user values and comments survive independently
of that oracle, and (strict xfail, #119844) long-value lines survive verbatim; no
service-manager restart was attempted against anything outside the sandbox.

The install's origin is the official URL rewritten (``url.<store>.insteadOf``) to the local
store, so the updater takes the normal non-fork path and never touches the network.

All processes run in the bubblewrap sandbox from ``_helpers`` (own PID namespace, no user
systemd bus, real ~/.hermes read-only): the updater's all-profile gateway scan cannot reach a
live gateway on the host.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import pytest
import yaml

from tests.e2e.core.upgrade import _helpers as H
from tests.fakes.fake_llm_provider import FakeLLMServer

pytestmark = [
    pytest.mark.linux_only,
    # The real updater runs against a throwaway local origin + install inside the sandbox, never
    # this checkout (the guard this bypasses exists to stop `hermes update` on the real repo).
    pytest.mark.live_system_guard_bypass,
    pytest.mark.skipif(H.sandbox_required_reason() is not None, reason=str(H.sandbox_required_reason())),
    pytest.mark.skipif(shutil.which("git") is None, reason="git required"),
]

TRACEBACK = "Traceback (most recent call last)"
UPDATE_TIMEOUT = 1500  # seconds; a cold dependency sync on a loaded CI box is minutes
CLI_TIMEOUT = 600


def _git(*args: str, cwd: Path, check: bool = True) -> str:
    cp = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True,
                        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    if check and cp.returncode != 0:
        raise AssertionError(f"git {args} failed in {cwd}: {cp.stderr}")
    return cp.stdout.strip()


def _real_uv() -> str | None:
    cand = shutil.which("uv") or str(H.REAL_HOME / ".hermes" / "bin" / "uv")
    return cand if cand and Path(cand).exists() else None


OFFICIAL_URL = "https://github.com/NousResearch/hermes-agent.git"


class _Refs(NamedTuple):
    head: str
    base_tag: str
    base: str


@functools.cache
def _refs() -> _Refs:
    """HEAD and release N-1, resolved on first use: collection (every CI shard) runs no git.

    N-1 is ``git describe --tags --abbrev=0 HEAD~1``; HERMES_E2E_UPGRADE_BASE=<ref> starts from any
    older ref instead (e.g. the pre-handoff v2026.9.14, or a patched base when proving a leg red
    against the N-1 side).
    """
    head = _git("rev-parse", "HEAD", cwd=H.WORKTREE)
    try:
        tag = os.environ.get("HERMES_E2E_UPGRADE_BASE") or _git("describe", "--tags", "--abbrev=0", "HEAD~1",
                                                                  cwd=H.WORKTREE)
        return _Refs(head, tag, _git("rev-parse", f"{tag}^{{commit}}", cwd=H.WORKTREE))
    except AssertionError:  # shallow CI checkout without tags
        return _Refs(head, "", "")


@pytest.fixture(scope="module", autouse=True)
def _upgrade_prerequisites() -> None:
    if not _refs().base_tag:
        pytest.skip("no release tag reachable before HEAD (fetch tags)")
    if _real_uv() is None:
        pytest.skip("uv required to build the N-1 venv")


IMPORT_PROBE = r"""
import importlib, subprocess, sys, tomllib
from pathlib import Path
root, base = Path(sys.argv[1]), sys.argv[2]
cfg = tomllib.load(open(root / "pyproject.toml", "rb"))
tops = [p for p in cfg["tool"]["setuptools"]["packages"]["find"]["include"] if "*" not in p]
added = subprocess.run(["git", "-C", str(root), "diff", "--name-only", "--diff-filter=A", base, "HEAD", "--",
                        *[f"{t}/*.py" for t in tops]], capture_output=True, text=True).stdout.split()
added = [a[:-3].replace("/", ".") for a in added if "/tests/" not in a and not a.endswith("__init__.py")][:3]
bad = []
for name in tops + ["hermes_cli.main", "run_agent", "hermes_state"] + added:
    try:
        mod = importlib.import_module(name)
    except BaseException as exc:
        bad.append(f"{name}: {type(exc).__name__}: {exc}")
        continue
    where = getattr(mod, "__file__", None) or str(list(getattr(mod, "__path__", [""]))[0])
    if not str(Path(where).resolve()).startswith(str(root.resolve())):
        bad.append(f"{name}: resolved outside the checkout: {where}")
if bad:
    sys.exit("\n".join(bad))
"""


DEPS_PROBE = r"""
import sys, tomllib
from importlib import metadata
from packaging.markers import default_environment
from packaging.requirements import Requirement
cfg = tomllib.load(open(sys.argv[1], "rb"))
proj = cfg["project"]
env = default_environment()
# [tool.uv] override-dependencies REPLACE the matching requirement during resolution.
overrides = {}
for spec in cfg.get("tool", {}).get("uv", {}).get("override-dependencies", []):
    o = Requirement(spec)
    if o.marker is None or o.marker.evaluate(env):
        overrides[o.name.lower()] = o
bad = []
for spec in proj["dependencies"]:
    req = Requirement(spec)
    if req.marker is not None and not req.marker.evaluate(env):
        continue
    req = overrides.get(req.name.lower(), req)
    try:
        have = metadata.version(req.name)
    except metadata.PackageNotFoundError:
        bad.append(f"{req.name}: missing (wants {req.specifier})")
        continue
    if req.specifier and not req.specifier.contains(have, prereleases=True):
        bad.append(f"{req.name}: {have} does not satisfy {req.specifier}")
dist = metadata.distribution("hermes-agent")
declared = sorted(str(Requirement(r)) for r in (dist.requires or []) if "extra ==" not in r)
wanted = sorted(str(Requirement(r)) for r in proj["dependencies"])
if declared != wanted:
    bad.append(f"installed hermes-agent metadata is stale: {sorted(set(declared) ^ set(wanted))}")
if bad:
    sys.exit("\n".join(bad))
"""


# ---------------------------------------------------------------------------
# User state
# ---------------------------------------------------------------------------

LONG_VALUE = (
    "You are a meticulous reviewer.  Keep  double  spaces, unicode (é, 漢字, emoji-free), "
    "colons: like this, a hash # that is not a comment, and 'single quotes' intact; this value "
    "is deliberately longer than any YAML emitter's default fold width so a re-fold would "
    "change the stored string."
)
QUICK_CMD = "echo 'deploy: step one' && echo \"step two # still a string\" && printf '%s' ok"


def user_config(base_url: str, version: int) -> str:
    return (
        "# Hand-edited by the user. Every comment and value below must survive `hermes update`.\n"
        "model:\n"
        "  provider: custom\n"
        f"  base_url: {base_url}  # local fake provider\n"
        "  default: fake-model\n"
        "  context_length: 128000\n"
        "agent:\n"
        "  api_max_retries: 1   # keep tests fast\n"
        "\n"
        "# Personalities carry long quoted prose.\n"
        "personalities:\n"
        f"  reviewer: \"{LONG_VALUE}\"\n"
        "quick_commands:\n"
        "  deploy:\n"
        "    type: exec\n"
        f"    command: \"{QUICK_CMD.replace(chr(34), chr(92) + chr(34))}\"\n"
        "mcp_servers:\n"
        "  legacy-off:  # switched off by the old profile editor\n"
        "    command: /bin/true\n"
        "    disabled: true\n"
        "  kept-on:\n"
        "    command: /bin/echo\n"
        "    args: [\"hello\"]\n"
        "    enabled: true\n"
        f"_config_version: {version}\n"
    )


def _base_config_version() -> int:
    src = _git("show", f"{_refs().base}:hermes_cli/config_defaults.py", cwd=H.WORKTREE)
    for line in src.splitlines():
        if '"_config_version"' in line:
            return int(line.split(":", 1)[1].split(",", 1)[0].strip())
    raise AssertionError("could not read the N-1 _config_version")


# ---------------------------------------------------------------------------
# Install staging
# ---------------------------------------------------------------------------


@dataclass
class Leg:
    root: Path
    origin: Path
    install: Path
    env: dict[str, str]
    hermes_home: Path
    wrap_dir: Path
    snapshot: dict = field(default_factory=dict)

    @property
    def hermes(self) -> str:
        return str(self.install / "venv" / "bin" / "hermes")

    @property
    def python(self) -> str:
        return str(self.install / "venv" / "bin" / "python")

    def run(self, *args: str, timeout: float = CLI_TIMEOUT, argv0: str | None = None,
            cwd: Path | None = None) -> subprocess.CompletedProcess:
        return H.run([argv0 or self.hermes, *args], env=self.env, cwd=cwd or self.install,
                     writable=[self.root], timeout=timeout)

    def popen(self, *args: str) -> subprocess.Popen:
        log = open(self.root / f"popen-{int(time.monotonic() * 1000)}.log", "w")  # noqa: SIM115
        return subprocess.Popen(
            H.sandbox_argv([*_KILLED_RUN_PREFIX, self.hermes, *args], writable=[self.root]),
            env=self.env, cwd=str(self.install), stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True,
        )


# Each sandboxed command gets a fresh PID namespace, so pids restart at 2 every time. The update
# marker a killed run leaves (``$HERMES_HOME/.hermes-update-in-progress``) holds its pid; if the
# retry came up under the SAME pid it would see a "live" holder that is really itself, an
# artefact of the harness, not of a user retrying later (the killed pid is dead for them). So
# the killed run's updater is pid 3 (a non-exec shell is pid 2) and every retry's updater is
# pid 2 with pid 3 a process that already exited.
_KILLED_RUN_PREFIX = ["/bin/sh", "-c", '"$0" "$@"; exit $?']
_RETRY_PREFIX = ["/bin/sh", "-c", '/bin/true; exec "$0" "$@"']


def _make_origin(root: Path) -> Path:
    origin = root / "origin.git"
    _git("clone", "-q", "--bare", "--shared", "--no-tags", str(H.WORKTREE), str(origin), cwd=root)
    _git("update-ref", "refs/heads/main", _refs().base, cwd=origin)
    _git("symbolic-ref", "HEAD", "refs/heads/main", cwd=origin)
    _git("tag", "-f", "e2e-upgrade-base", _refs().base, cwd=origin)
    return origin


def _write_wrappers(leg_root: Path, install: Path, hermes_home: Path) -> Path:
    """Managed uv (as the installer provisions it) and a PATH git wrapper; both armable to freeze."""
    real_uv = _real_uv()
    real_git = shutil.which("git")
    bin_dir = hermes_home / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    uv = bin_dir / "uv"
    uv.write_text(
        "#!/bin/sh\n"
        # `uv self update` would reach the network and the real binary; the pinned copy is current.
        'if [ "$1" = self ]; then exit 0; fi\n'
        f'if [ "$1" = pip ] && [ "$2" = install ] && [ -f "{leg_root}/arm-deps" ]; then\n'
        f'  rm -f "{leg_root}/arm-deps"; : > "{leg_root}/frozen-deps"; exec sleep 3600\n'
        "fi\n"
        f'exec "{real_uv}" "$@"\n',
        encoding="utf-8",
    )
    uv.chmod(0o755)
    wrap = leg_root / "wrap"
    wrap.mkdir(exist_ok=True)
    git = wrap / "git"
    git.write_text(
        "#!/bin/sh\n"
        f'if [ -f "{leg_root}/arm-pull" ] && echo " $* " | grep -q " merge --ff-only "; then\n'
        f'  rm -f "{leg_root}/arm-pull"\n'
        # Model a SIGKILL half-way through the fast-forward: git writes the tree in index order,
        # so a prefix of the changed paths is already new, HEAD still names N-1, and the index
        # lock the dead git held stays behind.
        f'  xargs -r -a "{leg_root}/torn-paths" "{real_git}" -C "{install}" checkout origin/main -- >/dev/null 2>&1\n'
        f'  : > "{install}/.git/index.lock"; : > "{leg_root}/frozen-pull"; exec sleep 3600\n'
        "fi\n"
        f'exec "{real_git}" "$@"\n',
        encoding="utf-8",
    )
    git.chmod(0o755)
    return wrap


def make_leg(root: Path, template_home: Path | None) -> Leg:
    root.mkdir(parents=True, exist_ok=True)
    origin = _make_origin(root)
    install = root / "install"
    _git("clone", "-q", "--shared", "-b", "main", str(origin), str(install), cwd=root)
    assert _git("rev-parse", "HEAD", cwd=install) == _refs().base
    # Look like a normal (non-fork) install: origin is the official URL, rewritten in this repo's
    # own config to the local store, so the updater takes the common path and never hits the network.
    _git("remote", "set-url", "origin", OFFICIAL_URL, cwd=install)
    _git("config", f"url.{origin}.insteadOf", OFFICIAL_URL, cwd=install)
    uv = _real_uv()
    py = H.WORKTREE / ".venv" / "bin" / "python"
    base_python = str(Path(os.path.realpath(py))) if py.exists() else "python3"
    # The installer's tier 0: N-1's own uv.lock (hash-pinned, `--extra all`) into install/venv, with the
    # user's uv config hidden, so the N-1 venv is the one users of that release actually have.
    no_cfg = root / "uv-config"
    no_cfg.mkdir()
    uv_env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "UV_NO_CONFIG", "UV_CONFIG_FILE")}
    uv_env.update(UV_PROJECT_ENVIRONMENT=str(install / "venv"), XDG_CONFIG_HOME=str(no_cfg), XDG_CONFIG_DIRS=str(no_cfg))
    cp = subprocess.run([uv, "sync", "-q", "--locked", "--extra", "all", "--python", base_python], cwd=str(install),
                        env=uv_env, capture_output=True, text=True, timeout=1800)
    assert cp.returncode == 0, f"N-1 venv install from its uv.lock failed:\n{cp.stderr[-4000:]}"
    env_probe = H.isolated_env(root)
    hermes_home = Path(env_probe["HERMES_HOME"])
    if template_home is not None:
        shutil.rmtree(hermes_home)
        shutil.copytree(template_home, hermes_home, symlinks=True)
    wrap = _write_wrappers(root, install, hermes_home)
    env = H.isolated_env(root, extra_path=[wrap])
    return Leg(root=root, origin=origin, install=install, env=env, hermes_home=hermes_home, wrap_dir=wrap)


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


def db_fingerprint(db: Path) -> dict:
    """integrity + per-table row counts + a digest of every pre-existing message."""
    assert db.exists(), f"{db} missing"
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        integrity = con.execute("PRAGMA integrity_check").fetchall()
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            " AND name NOT LIKE '%_fts%' AND name NOT LIKE '%fts_%'")]
        counts = {t: con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] for t in tables}
        sessions = {r[0] for r in con.execute("SELECT id FROM sessions")}
        msg_cols = {r[1] for r in con.execute("PRAGMA table_info(messages)")}
        key_cols = [c for c in ("id", "session_id", "role", "content") if c in msg_cols]
        rows = con.execute(f"SELECT {', '.join(key_cols)} FROM messages ORDER BY id").fetchall()
    finally:
        con.close()
    return {
        "integrity": integrity,
        "counts": counts,
        "sessions": sessions,
        "messages": {r[0]: hashlib.sha256(repr(r).encode()).hexdigest() for r in rows},
    }


def _profile_homes(leg: Leg) -> dict[str, Path]:
    return {"default": leg.hermes_home, "work": leg.hermes_home / "profiles" / "work"}


def snapshot_state(leg: Leg) -> dict:
    snap = {}
    for name, home in _profile_homes(leg).items():
        snap[name] = {
            "config": (home / "config.yaml").read_bytes(),
            "env": (home / ".env").read_bytes() if (home / ".env").exists() else None,
            "db": db_fingerprint(home / "state.db"),
        }
    snap["cron"] = json.loads((leg.hermes_home / "cron" / "jobs.json").read_text(encoding="utf-8"))
    return snap


def _job_ids(jobs) -> set[str]:
    items = jobs.get("jobs", jobs) if isinstance(jobs, dict) else jobs
    if isinstance(items, dict):
        return set(items)
    return {j.get("id") for j in items if isinstance(j, dict)}


def migrate_oracle(leg: Leg, name: str, before: bytes) -> bytes:
    """HEAD's own non-interactive migration of the pre-update bytes, in a fresh sandboxed process.

    This is the documented migration path (``migrate_config(interactive=False)``, what the updater
    runs for ``--yes`` and for sibling profiles), so ``after == oracle`` means the updater wrote
    nothing beyond documented migrations.
    """
    oroot = leg.root / f"oracle-{name}"
    if oroot.exists():
        shutil.rmtree(oroot)
    env = H.isolated_env(oroot, pythonpath=H.WORKTREE)
    home = Path(env["HERMES_HOME"])
    (home / "config.yaml").write_bytes(before)
    src_env = _profile_homes(leg)[name] / ".env"
    if src_env.exists():
        shutil.copy2(src_env, home / ".env")
    code = "from hermes_cli.config import migrate_config; migrate_config(interactive=False, quiet=True)"
    cp = H.run([str(H.WORKTREE / ".venv" / "bin" / "python"), "-c", code],
               env=env, cwd=H.WORKTREE, writable=[oroot], timeout=CLI_TIMEOUT)
    assert cp.returncode == 0 and TRACEBACK not in cp.stderr, H.describe(cp)
    return (home / "config.yaml").read_bytes()


def assert_healthy_at_head(leg: Leg, provider: FakeLLMServer, final: subprocess.CompletedProcess) -> None:
    before = leg.snapshot
    # 1. exit code matches reality
    assert final.returncode == 0, "final `hermes update` failed:\n" + H.describe(final)
    assert _git("rev-parse", "HEAD", cwd=leg.install) == _refs().head, "update exited 0 but HEAD is not the target"
    assert not (leg.install / ".git" / "index.lock").exists(), "update left .git/index.lock behind"
    # The editable install must serve the pulled tree (stale editable finder, #119466): every
    # top-level package HEAD ships plus a module that only exists at HEAD import from the venv's
    # own interpreter in a fresh process whose cwd is OUTSIDE the checkout (no implicit sys.path).
    cp = leg.run("-c", IMPORT_PROBE, str(leg.install), _refs().base, argv0=leg.python, cwd=leg.root)
    assert cp.returncode == 0, "the updated venv does not serve HEAD's tree:\n" + H.describe(cp)
    # The venv satisfies HEAD's declared dependency set (not just "the old release still imports"):
    # every core requirement in the pulled pyproject is installed at a satisfying version, and the
    # installed hermes-agent distribution was built from the pulled pyproject.
    cp = leg.run("-c", DEPS_PROBE, str(leg.install / "pyproject.toml"), argv0=leg.python)
    assert cp.returncode == 0, "venv does not satisfy HEAD's dependencies:\n" + H.describe(cp)
    # 2. state integrity: nothing lost, nothing corrupted (measured BEFORE any new turn)
    for name, home in _profile_homes(leg).items():
        fp = db_fingerprint(home / "state.db")
        assert fp["integrity"] == [("ok",)], f"{name} state.db integrity: {fp['integrity']}"
        for table, n in before[name]["db"]["counts"].items():
            assert fp["counts"].get(table) == n, f"{name} state.db `{table}` rows {n} -> {fp['counts'].get(table)}"
        assert fp["sessions"] == before[name]["db"]["sessions"], f"{name} sessions changed"
        assert fp["messages"] == before[name]["db"]["messages"], f"{name} pre-existing messages changed"
        if before[name]["env"] is not None:
            assert (home / ".env").read_bytes() == before[name]["env"], f"{name} .env rewritten by the update"
    jobs_after = json.loads((leg.hermes_home / "cron" / "jobs.json").read_text(encoding="utf-8"))
    assert _job_ids(jobs_after) == _job_ids(before["cron"]) and _job_ids(jobs_after), "cron jobs lost/changed"
    # 3. config: exactly HEAD's documented migrations, nothing else
    for name, home in _profile_homes(leg).items():
        orig = before[name]["config"]
        after = (home / "config.yaml").read_bytes()
        expected = migrate_oracle(leg, name, orig)
        assert after == expected, (
            f"{name} config.yaml differs from HEAD's own migration of the pre-update file\n"
            f"--- after update ---\n{after.decode()}\n--- expected ---\n{expected.decode()}")
        # Independent of the oracle (a clobbering migration would be mirrored by it): the user's
        # values survive semantically and every comment line survives verbatim.
        o, a = yaml.safe_load(orig), yaml.safe_load(after)
        for section in ("model", "agent", "personalities", "quick_commands"):
            assert a.get(section) == o.get(section), f"{name}: `{section}` changed by the update: {a.get(section)!r}"
        assert a["personalities"]["reviewer"] == LONG_VALUE and a["quick_commands"]["deploy"]["command"] == QUICK_CMD
        off = a["mcp_servers"]["legacy-off"]
        assert off.get("enabled", True) is False or off.get("disabled") is True, f"{name}: MCP server switched back on"
        assert a["mcp_servers"]["kept-on"] == o["mcp_servers"]["kept-on"]
        after_lines = after.decode("utf-8").splitlines()
        for line in orig.decode("utf-8").splitlines():
            if line.lstrip().startswith("#"):
                assert line in after_lines, f"{name}: comment line lost: {line!r}"
    # 4. fresh-process entrypoints on the updated install
    ver = leg.run("--version")
    assert ver.returncode == 0 and TRACEBACK not in ver.stdout + ver.stderr, H.describe(ver)
    doc = leg.run("doctor")
    assert doc.returncode in (0, 1) and TRACEBACK not in doc.stdout + doc.stderr, H.describe(doc)
    n_before = len(provider.main_requests())
    marker = f"post-update turn {leg.root.name}"
    turn = leg.run("-z", marker)
    assert turn.returncode == 0 and TRACEBACK not in turn.stderr, H.describe(turn)
    assert provider.default_text in turn.stdout, H.describe(turn)
    new = provider.main_requests()[n_before:]
    assert len(new) == 1 and marker in json.dumps(new[0]["messages"]), "one-shot turn did not reach the provider once"
    fp = db_fingerprint(leg.hermes_home / "state.db")
    assert fp["integrity"] == [("ok",)]
    assert len(fp["sessions"]) == len(before["default"]["db"]["sessions"]) + 1, "post-update turn not persisted"
    # HEAD's first real use of the DB (its schema migrations run on open) keeps every old row intact.
    old = before["default"]["db"]["messages"]
    assert {k: fp["messages"].get(k) for k in old} == old, "pre-update messages lost/changed by HEAD's first use"
    assert before["default"]["db"]["sessions"] <= fp["sessions"], "pre-update sessions lost by HEAD's first use"
    # 5. no supervisor restart reached outside the sandbox
    calls = (leg.root / "shims" / "shim-calls.log")
    text = calls.read_text() if calls.exists() else ""
    assert not any(w in text for w in (" restart ", " stop ", " kill ", " kickstart ")), text


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def provider():
    with FakeLLMServer(default_text="fake reply for the upgrade suite") as srv:
        yield srv


@pytest.fixture(scope="module")
def template_home(tmp_path_factory, provider) -> Path:
    """HERMES_HOME populated by the N-1 CLI itself (sessions, profile, cron, config)."""
    seed = make_leg(tmp_path_factory.mktemp("seed"), None)
    version = _base_config_version()
    cfg = user_config(provider.base_url, version)
    (seed.hermes_home / "config.yaml").write_text(cfg, encoding="utf-8")
    (seed.hermes_home / ".env").write_text("OPENAI_API_KEY=sk-fake-e2e\n", encoding="utf-8")
    for prompt in ("first session before the upgrade", "second session before the upgrade"):
        cp = seed.run("-z", prompt)
        assert cp.returncode == 0, H.describe(cp)
    cp = seed.run("profile", "create", "work", "--no-alias")
    assert cp.returncode == 0, H.describe(cp)
    work = seed.hermes_home / "profiles" / "work"
    (work / "config.yaml").write_text(cfg, encoding="utf-8")
    (work / ".env").write_text("OPENAI_API_KEY=sk-fake-e2e-work\n", encoding="utf-8")
    cp = seed.run("-p", "work", "-z", "work profile session before the upgrade")
    assert cp.returncode == 0, H.describe(cp)
    cp = seed.run("cron", "create", "--name", "nightly", "0 3 * * *", "summarize the day")
    assert cp.returncode == 0, H.describe(cp)
    # Configs were written by hand AFTER the N-1 CLI touched them; re-pin them to the user's bytes.
    (seed.hermes_home / "config.yaml").write_text(cfg, encoding="utf-8")
    (work / "config.yaml").write_text(cfg, encoding="utf-8")
    template = seed.root / "template-home"
    shutil.copytree(seed.hermes_home, template, symlinks=True)
    return template


@pytest.fixture
def leg(tmp_path, template_home) -> Leg:
    lg = make_leg(tmp_path / "leg", template_home)
    lg.snapshot = snapshot_state(lg)
    assert lg.snapshot["default"]["db"]["integrity"] == [("ok",)]
    assert len(lg.snapshot["default"]["db"]["sessions"]) >= 2 and len(lg.snapshot["work"]["db"]["sessions"]) >= 1
    return lg


def _publish_head(leg: Leg) -> None:
    _git("update-ref", "refs/heads/main", _refs().head, cwd=leg.origin)


def _update(leg: Leg) -> subprocess.CompletedProcess:
    return leg.run(*_RETRY_PREFIX[1:], leg.hermes, "update", "--yes", argv0=_RETRY_PREFIX[0], timeout=UPDATE_TIMEOUT)


def _freeze_update_at(leg: Leg, marker: str) -> None:
    proc = leg.popen("update", "--yes")
    try:
        H.wait_for(lambda: (leg.root / marker).exists() or proc.poll() is not None,
                   timeout=UPDATE_TIMEOUT, interval=0.5, what=marker)
        assert (leg.root / marker).exists(), (
            f"update finished (rc={proc.returncode}) without reaching the {marker} point")
    finally:
        H.kill_tree(proc)  # SIGKILL the whole sandbox: power loss / closed terminal


def _age_git_locks(install: Path, seconds: float = 3600) -> None:
    """The user retries later: locks left by the dead git are now old (heal policy is age-gated)."""
    old = time.time() - seconds
    for p in (install / ".git").glob("*.lock"):
        os.utime(p, (old, old))


# ---------------------------------------------------------------------------
# Legs
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def clean_leg(tmp_path_factory, template_home):
    lg = make_leg(tmp_path_factory.mktemp("clean") / "leg", template_home)
    lg.snapshot = snapshot_state(lg)
    _publish_head(lg)
    return lg, _update(lg)


def test_clean_update(clean_leg, provider):
    leg, final = clean_leg
    assert_healthy_at_head(leg, provider, final)
    assert _git("status", "--porcelain", "--untracked-files=no", cwd=leg.install) == "", "update left a dirty tree"


@pytest.mark.xfail(strict=True, reason="#119844: ruamel re-folds long scalars when the updater's migration saves config.yaml")
def test_clean_update_keeps_long_scalar_lines_verbatim(clean_leg):
    leg, final = clean_leg
    assert final.returncode == 0, H.describe(final)
    for name, home in _profile_homes(leg).items():
        after_lines = (home / "config.yaml").read_text(encoding="utf-8").splitlines()
        for line in leg.snapshot[name]["config"].decode("utf-8").splitlines():
            if LONG_VALUE in line or "deploy: step one" in line:
                assert line in after_lines, f"{name}: long user value re-folded: {line!r}"


def test_update_with_local_edits_and_orphan_autostash(leg, provider):
    # A file that does not change N-1 -> HEAD, so restoring the user's edit is conflict-free.
    unchanged = next(p for p in ("README.md", "LICENSE", "AGENTS.md")
                     if (leg.install / p).exists()
                     and not _git("diff", "--name-only", _refs().base, _refs().head, "--", p, cwd=H.WORKTREE))
    # An orphan autostash from an earlier update that never restored it (#63717).
    (leg.install / unchanged).write_text((leg.install / unchanged).read_text() + "\norphan edit\n")
    _git("stash", "push", "-m", "hermes-update-autostash-20260101-000000", cwd=leg.install)
    orphan = _git("rev-parse", "refs/stash", cwd=leg.install)
    # The user's current local work: a tracked edit and an untracked file.
    user_line = "\nuser's local note that must survive the update\n"
    (leg.install / unchanged).write_text((leg.install / unchanged).read_text() + user_line)
    (leg.install / "my_notes.txt").write_text("untracked user file\n")
    _publish_head(leg)
    final = _update(leg)
    assert_healthy_at_head(leg, provider, final)
    stash_commits = _git("log", "-g", "--format=%H", "refs/stash", cwd=leg.install, check=False).split()
    assert orphan in stash_commits, "the pre-existing autostash (maybe the only copy of user work) was dropped"
    # The user's work is either restored in the tree or parked in a stash entry: never lost.
    restored = user_line.strip() in (leg.install / unchanged).read_text()
    parked = any(user_line.strip() in _git("stash", "show", "-p", "--include-untracked", c, cwd=leg.install, check=False)
                 for c in stash_commits if c != orphan)
    assert restored or parked, "local tracked edit lost by the update"
    untracked_ok = (leg.install / "my_notes.txt").exists() or any(
        "my_notes.txt" in _git("show", "--name-only", "--format=", f"{c}^3", cwd=leg.install, check=False)
        for c in stash_commits if c != orphan)
    assert untracked_ok, "untracked user file lost by the update"


@pytest.mark.parametrize("torn", [
    pytest.param(False, id="lock-only"),
    pytest.param(True, id="torn-tree", marks=pytest.mark.xfail(strict=True, reason=(
        "NEW: a fast-forward killed half-way leaves a prefix of the changed files at HEAD while HEAD "
        "still names N-1; every entry point, including `hermes update`, then dies at import "
        "(cannot import name ... from 'utils'), so nothing can heal the install without manual git"))),
])
def test_kill_mid_pull_then_retry_heals(leg, provider, torn):
    changed = sorted(_git("diff", "--name-only", "--no-renames", "--diff-filter=AM", _refs().base, _refs().head,
                          cwd=H.WORKTREE).splitlines())
    assert changed, "N-1 and HEAD have no file differences"
    torn_paths = changed[: max(1, len(changed) // 2)] if torn else []
    (leg.root / "torn-paths").write_text("".join(p + "\n" for p in torn_paths))
    (leg.root / "arm-pull").touch()
    _publish_head(leg)
    _freeze_update_at(leg, "frozen-pull")
    assert _git("rev-parse", "HEAD", cwd=leg.install) == _refs().base, "kill point was not mid-pull"
    assert (leg.install / ".git" / "index.lock").exists()
    _age_git_locks(leg.install)
    final = _update(leg)
    assert_healthy_at_head(leg, provider, final)


def test_kill_before_deps_then_retry_heals(leg, provider):
    (leg.root / "arm-deps").touch()
    _publish_head(leg)
    _freeze_update_at(leg, "frozen-deps")
    assert _git("rev-parse", "HEAD", cwd=leg.install) == _refs().head, "kill point was not after the code swap"
    _age_git_locks(leg.install)
    final = _update(leg)
    assert_healthy_at_head(leg, provider, final)


def test_offline_update_fails_loudly_and_changes_nothing(leg, provider):
    _publish_head(leg)
    # Origin goes unreachable (network down / repo moved). The install borrows its objects from
    # the origin store (--shared), so the store moves and the install's alternates follow it.
    store = leg.root / "offline-store.git"
    leg.origin.rename(store)
    (leg.install / ".git" / "objects" / "info" / "alternates").write_text(f"{store / 'objects'}\n")
    cp = _update(leg)
    assert cp.returncode != 0, "update against an unreachable origin reported success:\n" + H.describe(cp)
    assert TRACEBACK not in cp.stdout + cp.stderr, H.describe(cp)
    assert _git("rev-parse", "HEAD", cwd=leg.install) == _refs().base
    assert _git("status", "--porcelain", "--untracked-files=no", cwd=leg.install) == ""
    after = snapshot_state(leg)
    for name in ("default", "work"):
        assert after[name]["config"] == leg.snapshot[name]["config"], f"{name} config touched by a failed update"
        assert after[name]["db"] == leg.snapshot[name]["db"], f"{name} state.db touched by a failed update"
    ver = leg.run("--version")
    assert ver.returncode == 0, H.describe(ver)
    n = len(provider.main_requests())
    turn = leg.run("-z", "turn after a failed update")
    assert turn.returncode == 0 and provider.default_text in turn.stdout, H.describe(turn)
    assert len(provider.main_requests()) == n + 1
