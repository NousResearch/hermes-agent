"""C6: every shipped module and every entrypoint works in a FRESH process after an update.

The ``hermes update`` failure class this guards (issue_classes.md C6): after the pull every
command dies with ``ImportError: cannot import name 'file_signature'`` (#111942, #111943,
#112522, #114616), a module-level ``NameError``/circular import only shows up in a clean
interpreter, or the restart phase of a pre-hand-off updater imports new code into a stale
``sys.modules`` graph. Nothing here mocks the import system or the CLI:

* **Import smoke** – the module list is derived from the packaging config itself
  (``setup.py::_root_py_modules()`` + ``[tool.setuptools.packages.find]``), and each module
  is imported in its own ``fork()`` of ONE fresh interpreter that has never imported
  first-party code, so an import-order bug ("X only imports if Y ran first") cannot hide
  behind an earlier import. Only a missing *optional-extra* third-party dependency is
  tolerated; every first-party ``ImportError``/``NameError``/``AttributeError``/
  ``SyntaxError``/circular import fails.
* **Stale graph** – the newest release whose updater still finished in the pre-pull
  interpreter (no ``hermes_cli/update_handoff.py``) is extracted from git, its updater graph
  is imported, the checkout is swapped in place, the OLD release's own reload + purge run,
  and every module that updater imports after its purge must import from the new tree.
* **Entrypoints** – ``python -m hermes_cli.main`` and every ``[project.scripts]`` console
  script run as real sandboxed processes (isolated HOME, only a loopback fake provider)
  with shared invariants: bounded termination, documented rc, no traceback, no
  service-manager calls, no model call unless the command is a turn. ``-z`` must persist
  exactly what was sent and rendered; ``serve`` must announce READY on stdout, bind the
  announced port, and stop on SIGTERM leaving no descendant and no host record behind.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import tomllib
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.e2e.core.upgrade._helpers import (
    WORKTREE,
    describe,
    isolated_env,
    kill_tree,
    run,
    sandbox_argv,
    sandbox_required_reason,
    wait_for,
)
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, write_hermes_home

pytestmark = [
    pytest.mark.skipif(sys.platform != "linux", reason="fork()-based import isolation and bwrap sandbox are Linux-only"),
]

PY = sys.executable
PYPROJECT = tomllib.loads((WORKTREE / "pyproject.toml").read_text(encoding="utf-8"))

TRACEBACK = "Traceback (most recent call last)"

# Third-party import name -> distribution. A module failing ONLY because one of these is
# absent is tolerated, and only while that distribution is an optional extra (or a core
# requirement whose environment marker excludes this platform) — the table is re-validated
# against pyproject.toml, so it cannot be used to hide a missing core dependency.
_OPTIONAL_IMPORTS = {
    "acp": "agent-client-protocol", "aiohttp": "aiohttp", "aiohttp_socks": "aiohttp-socks",
    "aiosqlite": "aiosqlite", "alibabacloud_dingtalk": "alibabacloud-dingtalk", "anthropic": "anthropic",
    "asyncpg": "asyncpg", "azure": "azure-identity", "boto3": "boto3", "botocore": "boto3",
    "brotlicffi": "brotlicffi", "daytona": "daytona", "defusedxml": "defusedxml",
    "dingtalk_stream": "dingtalk-stream", "discord": "discord.py", "edge_tts": "edge-tts",
    "elevenlabs": "elevenlabs", "exa_py": "exa-py", "fal_client": "fal-client",
    "faster_whisper": "faster-whisper", "firecrawl": "firecrawl-py", "google": "google-auth",
    "google_auth_oauthlib": "google-auth-oauthlib", "googleapiclient": "google-api-python-client",
    "honcho": "honcho-ai", "httplib2": "httplib2", "lark_oapi": "lark-oapi", "mautrix": "mautrix",
    "mcp": "mcp", "mem0": "mem0ai", "microsoft_teams": "microsoft-teams-apps", "mistralai": "mistralai",
    "modal": "modal", "numpy": "numpy", "pyopen_wakeword": "pyopen-wakeword",
    "opentelemetry": "opentelemetry-sdk", "parallel": "parallel-web", "pvporcupine": "pvporcupine",
    "pyasn1": "pyasn1", "qrcode": "qrcode", "sentencepiece": "sentencepiece", "sherpa_onnx": "sherpa-onnx",
    "slack_bolt": "slack-bolt", "slack_sdk": "slack-sdk", "sounddevice": "sounddevice",
    "supermemory": "supermemory", "telegram": "python-telegram-bot", "uvloop": "uvloop",
    "vercel": "vercel", "youtube_transcript_api": "youtube-transcript-api",
    # Windows-only core requirements (their markers exclude Linux).
    "pywintypes": "pywin32", "win32api": "pywin32", "win32con": "pywin32", "win32event": "pywin32",
    "win32file": "pywin32", "win32job": "pywin32", "win32process": "pywin32", "win32security": "pywin32",
    "winerror": "pywin32", "winpty": "pywinpty",
}

# Shipped modules that cannot import on this platform for a reason other than a missing
# dependency. Every entry needs a reason; an entry that imports fine is reported as stale.
_PLATFORM_ALLOWLIST: dict[str, str] = {}

# Third-party modules imported in the parent BEFORE forking, purely to amortise their import
# cost across ~1.8k children. Never first-party (the runner refuses to fork if any is loaded).
_PRELOAD = (
    "openai", "httpx", "pydantic", "rich", "prompt_toolkit", "yaml", "ruamel.yaml", "requests", "jinja2",
    "fastapi", "starlette", "uvicorn", "anthropic", "cryptography", "psutil", "websockets", "dotenv",
    "tenacity", "fire", "croniter", "markdown", "jwt", "packaging", "PIL",
)

# Import sites the pre-hand-off updater (v2026.9.14 update_cmd_fleet/_maint) reaches AFTER
# its purge in the pre-pull interpreter: the restart phase (``hermes_cli.gateway`` and what
# it pulls), the maintenance/summary steps, and the atexit browser cleanup that re-imports
# ``tools.browser_tool`` -> ``hermes_cli.config`` (#112522). Entry points that only ever start
# in a NEW process (gateway.run, tui_gateway.entry, cron.scheduler) are covered by the
# fresh-interpreter smoke instead.
_POST_PURGE_IMPORTS = (
    "hermes_cli.config", "hermes_cli.managed_scope", "hermes_cli.gateway", "gateway.status",
    "hermes_cli.gateway_migrate", "hermes_cli.profiles", "hermes_cli.backup", "hermes_cli.model_catalog",
    "hermes_cli.plugin_compat", "agent.curator", "tools.skills_sync", "tools.browser_tool",
)
# What the pre-hand-off ``hermes update`` process had imported before the pull.
_OLD_UPDATER_GRAPH = ("hermes_cli.main", "hermes_cli.update_cmd", "hermes_cli.config", "hermes_cli.gateway")

_READY_RE = re.compile(r"^HERMES_(?:BACKEND|DASHBOARD)_READY port=(\d+)", re.M)  # electron/backend-ready.ts


# --------------------------------------------------------------------------- packaging model


def _canonical(dist: str) -> str:
    return re.sub(r"[-_.]+", "-", dist).lower()


def _requirements():
    from packaging.requirements import Requirement

    core = [Requirement(r) for r in PYPROJECT["project"]["dependencies"]]
    extras = {_canonical(Requirement(r).name) for reqs in PYPROJECT["project"]["optional-dependencies"].values()
              for r in reqs if not r.startswith("hermes-agent")}
    return core, extras


def _dist_is_optional_here(dist: str) -> bool:
    core, extras = _requirements()
    core_here = [r for r in core if _canonical(r.name) == _canonical(dist)]
    if core_here:
        return all(r.marker is not None and not r.marker.evaluate() for r in core_here)
    return _canonical(dist) in extras


def _git(*args: str, cwd: Path = WORKTREE) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, encoding="utf-8",
                          stdin=subprocess.DEVNULL, timeout=120)


def _tracked(prefix: str = "") -> set[str] | None:
    cp = _git("ls-files", "-z", *(["--", prefix] if prefix else []))
    if cp.returncode != 0:
        return None
    return {p for p in cp.stdout.split("\0") if p}


def _root_py_modules() -> list[str]:
    """``setup.py::_root_py_modules()`` — the list the wheel build ships."""
    spec = importlib.util.spec_from_file_location("_hermes_setup_py_c6", WORKTREE / "setup.py")
    mod = importlib.util.module_from_spec(spec)
    saved = sys.argv
    sys.argv = ["setup.py", "--name"]  # setup() must not build anything on import
    try:
        try:
            spec.loader.exec_module(mod)
        except SystemExit:
            pass
    finally:
        sys.argv = saved
    return list(mod._root_py_modules())


def _package_dirs() -> list[str]:
    setuptools = pytest.importorskip("setuptools")
    find = PYPROJECT["tool"]["setuptools"]["packages"]["find"]
    finder = setuptools.find_namespace_packages if find.get("namespaces", True) else setuptools.find_packages
    return sorted(finder(where=str(WORKTREE), include=find.get("include", ("*",)), exclude=find.get("exclude", ())))


def _shipped_modules() -> tuple[list[dict], set[str]]:
    """Every importable module the packaging config ships, as runner entries.

    Tests are excluded (``tests`` package dirs, ``test_*.py``, ``conftest.py``), as are
    ``__main__`` modules (executed, never imported). Files a parallel test drops at the repo
    root (``_test_*``) and untracked scratch files are not part of a release checkout.
    Directory plugins whose path is not a Python identifier (``plugins/model-providers/nous``)
    are imported the way ``hermes_cli.plugins_loader`` imports them.
    """
    tracked = _tracked()
    roots = [n for n in _root_py_modules()
             if not n.startswith("_test_") and (tracked is None or f"{n}.py" in tracked)]
    entries = [{"id": n, "kind": "name", "name": n} for n in roots]
    pkgs = _package_dirs()
    for pkg in pkgs:
        parts = pkg.split(".")
        if "tests" in parts:
            continue
        pdir = WORKTREE / pkg.replace(".", "/")
        for f in sorted(pdir.glob("*.py")):
            rel = f.relative_to(WORKTREE).as_posix()
            if f.name.startswith("test_") or f.name == "conftest.py" or f.stem == "__main__":
                continue
            # This frozen old-updater shim exits at import time to force a relaunch.
            # Importing it is neither safe nor an import-graph smoke test.
            if rel == "hermes_cli/psutil_android.py":
                continue
            if tracked is not None and rel not in tracked:
                continue
            if all(p.isidentifier() for p in parts):
                name = pkg if f.stem == "__init__" else f"{pkg}.{f.stem}"
                entries.append({"id": name, "kind": "name", "name": name})
                continue
            plugin_root = next((a for a in (pdir, *pdir.parents)
                                if (a / "plugin.yaml").exists() or (a / "plugin.yml").exists()), None)
            if plugin_root is not None and (plugin_root / "__init__.py").exists():
                slug = plugin_root.relative_to(WORKTREE / "plugins").as_posix().replace("/", "__").replace("-", "_")
                sub = None if f.stem == "__init__" else ".".join((*f.parent.relative_to(plugin_root).parts, f.stem))
                entries.append({"id": rel, "kind": "plugin", "root": str(plugin_root), "slug": slug, "sub": sub})
            else:
                entries.append({"id": rel, "kind": "file", "file": str(f), "slug": re.sub(r"\W", "_", rel[:-3])})
    first_party = set(roots) | {p.split(".")[0] for p in pkgs} | {"hermes_plugins"}
    return entries, first_party


# --------------------------------------------------------------------------- in-sandbox runners

# Shared by both runners: drop the venv's PEP 660 editable finder so a module missing from the
# tree under test can never be satisfied from whatever checkout the venv was installed from.
_STRIP_EDITABLE = r'''
import sys
sys.meta_path[:] = [f for f in sys.meta_path if not type(f).__module__.startswith("__editable__")]
sys.path_hooks[:] = [h for h in sys.path_hooks if not getattr(h, "__module__", "").startswith("__editable__")]
sys.path_importer_cache.clear()
'''

_IMPORT_RUNNER = _STRIP_EDITABLE + r'''
import importlib, importlib.util, json, os, select, time, traceback, types

spec = json.load(open(sys.argv[1]))
tree = os.path.realpath(spec["tree"])
sys.path.insert(0, spec["tree"])
first_party = set(spec["first_party"])
for name in spec["preload"]:
    try:
        importlib.import_module(name)
    except BaseException:
        pass
leaked = sorted(n for n in sys.modules if n.split(".")[0] in first_party)
if leaked:
    sys.exit("first-party modules loaded before forking: %s" % leaked)


def _load_file_module(name, init, search):
    s = importlib.util.spec_from_file_location(name, init, submodule_search_locations=search)
    m = importlib.util.module_from_spec(s)
    if search is not None:
        m.__package__ = name
        m.__path__ = search
    sys.modules[name] = m
    s.loader.exec_module(m)


def _import(entry):
    if entry["kind"] == "name":
        importlib.import_module(entry["name"])
        return
    ns = types.ModuleType("hermes_plugins")  # plugins_loader's synthetic namespace parent
    ns.__path__ = []
    ns.__package__ = "hermes_plugins"
    sys.modules.setdefault("hermes_plugins", ns)
    name = "hermes_plugins." + entry["slug"]
    if entry["kind"] == "plugin":
        _load_file_module(name, os.path.join(entry["root"], "__init__.py"), [entry["root"]])
        if entry["sub"]:
            importlib.import_module(name + "." + entry["sub"])
    else:
        _load_file_module(name, entry["file"], None)


def child(entry, wfd):
    devnull = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(devnull, fd)
    res = {"id": entry["id"], "ok": True}
    try:
        _import(entry)
        outside = sorted(
            n for n, m in list(sys.modules.items())
            if n.split(".")[0] in first_party and getattr(m, "__file__", None)
            and not os.path.realpath(m.__file__).startswith(tree + os.sep))
        if outside:
            res = {"id": entry["id"], "ok": False, "type": "ForeignFirstParty",
                   "msg": "first-party modules resolved outside the tree: %s" % outside[:5]}
    except BaseException as exc:
        res = {"id": entry["id"], "ok": False, "type": type(exc).__name__, "msg": str(exc)[:600],
               "missing": exc.name if isinstance(exc, ModuleNotFoundError) else None,
               "tb": "".join(traceback.format_exception(exc))[-2500:]}
    os.write(wfd, json.dumps(res).encode())
    os._exit(0)


queue = list(spec["modules"])
running = {}
results = []
while queue or running:
    while queue and len(running) < spec["workers"]:
        entry = queue.pop(0)
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(r)
            child(entry, w)
        os.close(w)
        running[r] = [pid, entry, time.monotonic(), []]
    ready, _, _ = select.select(list(running), [], [], 0.5)
    for r in ready:
        chunk = os.read(r, 65536)
        if chunk:
            running[r][3].append(chunk)
            continue
        pid, entry, _start, chunks = running.pop(r)
        os.close(r)
        _, status = os.waitpid(pid, 0)
        raw = b"".join(chunks)
        results.append(json.loads(raw) if raw else {
            "id": entry["id"], "ok": False, "type": "ChildDied", "msg": "wait status %d, no result" % status})
    now = time.monotonic()
    for r, (pid, entry, start, _chunks) in list(running.items()):
        if now - start > spec["timeout"]:
            os.kill(pid, 9)
            os.waitpid(pid, 0)
            os.close(r)
            running.pop(r)
            results.append({"id": entry["id"], "ok": False, "type": "Timeout",
                            "msg": "import did not finish within %ss" % spec["timeout"]})
with open(spec["out"], "w") as fh:
    json.dump(results, fh)
'''

_STALE_RUNNER = _STRIP_EDITABLE + r'''
import ast, importlib, json, os, traceback

spec = json.load(open(sys.argv[1]))
checkout = spec["checkout"]
sys.path.insert(0, checkout)
for name in spec["old_graph"]:
    importlib.import_module(name)
main = sys.modules["hermes_cli.main"]

# The pull, in place: the same path now holds the new tree (running frames keep old objects).
os.rename(checkout, checkout + ".pre-pull")
os.rename(spec["new_tree"], checkout)

# The OLD updater's own post-pull steps, called the way it calls them (``_m()._...``).
for step in ("_reload_updated_runtime_modules", "_purge_stale_hermes_modules"):
    fn = getattr(main, step, None)
    if fn is not None:
        fn()


def _top_level_names(path):
    names = set()
    for node in ast.parse(open(path, encoding="utf-8").read()).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, (ast.AnnAssign,)) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


stale_roots = {}
for name in spec["root_modules"]:
    mod = sys.modules.get(name)
    new_src = os.path.join(checkout, name + ".py")
    if mod is not None and os.path.exists(new_src):
        missing = sorted(_top_level_names(new_src) - set(vars(mod)))
        if missing:
            stale_roots[name] = missing[:20]

results = []
for target in spec["targets"]:
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        dn = os.open(os.devnull, os.O_RDWR)
        os.dup2(dn, 1)
        os.dup2(dn, 2)
        res = {"target": target, "ok": True}
        try:
            importlib.import_module(target)
        except BaseException as exc:
            res = {"target": target, "ok": False, "type": type(exc).__name__, "msg": str(exc)[:800],
                   "tb": "".join(traceback.format_exception(exc))[-3000:]}
        os.write(w, json.dumps(res).encode())
        os._exit(0)
    os.close(w)
    chunks = []
    while True:
        c = os.read(r, 65536)
        if not c:
            break
        chunks.append(c)
    os.close(r)
    os.waitpid(pid, 0)
    results.append(json.loads(b"".join(chunks)) if chunks else {"target": target, "ok": False, "type": "ChildDied"})
json.dump({"results": results, "stale_roots": stale_roots}, open(spec["out"], "w"))
'''

# Reaps and reports every descendant that outlives the process it runs (a child subreaper),
# so "serve leaves nothing behind" is observable even inside a PID namespace.
_SUBREAPER = r'''
import ctypes, json, os, signal, subprocess, sys, time
import psutil

out = sys.argv[1]
ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0)  # PR_SET_CHILD_SUBREAPER
# Survive whatever the watched process sends its process group on the way out. A handler (not
# SIG_IGN) so the child starts with default dispositions: handlers reset across exec.
for _sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    signal.signal(_sig, lambda *_: None)
proc = subprocess.Popen(sys.argv[2:], stdin=subprocess.DEVNULL)
status = proc.wait()
deadline = time.monotonic() + 20
orphans = []
while True:
    orphans = [p for p in psutil.Process().children(recursive=True) if p.status() != psutil.STATUS_ZOMBIE]
    if not orphans or time.monotonic() > deadline:
        break
    time.sleep(0.1)
report = {"returncode": status, "orphans": [" ".join(p.cmdline()) for p in orphans]}
for p in orphans:
    try:
        p.kill()
    except psutil.Error:
        pass
json.dump(report, open(out, "w"))
'''


def _write_script(dirpath: Path, name: str, body: str) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    path = dirpath / name
    path.write_text(body, encoding="utf-8")
    return path


def _sandbox_or_skip() -> None:
    reason = sandbox_required_reason()
    if reason:
        pytest.skip(reason)


# --------------------------------------------------------------------------- (a) import smoke


def test_relaunch_shim_is_excluded_only_while_it_exits_on_import():
    shim = WORKTREE / "hermes_cli" / "psutil_android.py"
    top_level = ast.parse(shim.read_text(encoding="utf-8")).body
    assert any(isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
               and isinstance(node.value.func, ast.Name) and node.value.func.id == "stop_for_relaunch"
               for node in top_level)
    entries, _ = _shipped_modules()
    assert "hermes_cli.psutil_android" not in {entry["id"] for entry in entries}

def test_optional_import_table_only_names_optional_or_platform_deps():
    """The tolerance table cannot launder a missing CORE dependency into a skip."""
    core, extras = _requirements()
    core_names = {_canonical(r.name) for r in core}
    bad = {}
    for mod, dist in _OPTIONAL_IMPORTS.items():
        c = _canonical(dist)
        if c in core_names:
            if _dist_is_optional_here(dist):
                continue
            bad[mod] = f"{dist} is a core dependency on this platform"
        elif c not in extras:
            bad[mod] = f"{dist} is not declared in any optional extra"
    assert not bad, bad
    assert all(reason.strip() for reason in _PLATFORM_ALLOWLIST.values()), "every allowlist entry needs a reason"


def test_every_shipped_module_imports_from_a_clean_first_party_graph(tmp_path):
    _sandbox_or_skip()
    entries, first_party = _shipped_modules()
    ids = [e["id"] for e in entries]
    # Non-vacuous: the list really is the packaging config (every root module, every package).
    assert len(ids) == len(set(ids)), "duplicate module ids in the enumeration"
    assert {"hermes_cli.main", "run_agent", "gateway.run", "tui_gateway.entry", "acp_adapter.entry"} <= set(ids)
    find_tops = {p.split(".")[0] for p in PYPROJECT["tool"]["setuptools"]["packages"]["find"]["include"]}
    covered_tops = {i.split(".")[0] if not i.endswith(".py") else i.split("/")[0] for i in ids}
    assert find_tops <= covered_tops, f"packages.find tops with no module enumerated: {find_tops - covered_tops}"

    runner = _write_script(tmp_path / "runner", "import_runner.py", _IMPORT_RUNNER)
    out = tmp_path / "results.json"
    spec = {
        "tree": str(WORKTREE), "first_party": sorted(first_party), "modules": entries,
        "workers": max(2, min(8, (os.cpu_count() or 4) // 2)), "timeout": 180,
        "out": str(out), "preload": list(_PRELOAD),
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    env = isolated_env(tmp_path / "sbx")
    cp = run([PY, str(runner), str(spec_path)], env=env, cwd=tmp_path, writable=[tmp_path], timeout=900)
    assert cp.returncode == 0 and out.exists(), describe(cp)
    results = {r["id"]: r for r in json.loads(out.read_text(encoding="utf-8"))}
    assert set(results) == set(ids), f"runner lost modules: {sorted(set(ids) - set(results))[:10]}"

    failures, tolerated = {}, {}
    for mid, res in sorted(results.items()):
        if res["ok"]:
            if mid in _PLATFORM_ALLOWLIST:
                failures[mid] = "stale _PLATFORM_ALLOWLIST entry: module imports fine here"
            continue
        missing_top = (res.get("missing") or "").split(".")[0]
        if (res["type"] == "ModuleNotFoundError" and missing_top
                and missing_top not in first_party and missing_top not in sys.stdlib_module_names
                and missing_top in _OPTIONAL_IMPORTS and _dist_is_optional_here(_OPTIONAL_IMPORTS[missing_top])):
            tolerated[mid] = missing_top
            continue
        if mid in _PLATFORM_ALLOWLIST:
            continue
        failures[mid] = f"{res['type']}: {res['msg']}\n{res.get('tb', '')}"
    assert not failures, (
        f"{len(failures)} shipped module(s) fail to import from a clean first-party graph "
        f"(tolerated optional-extra misses: {len(tolerated)}):\n\n"
        + "\n\n".join(f"== {k}\n{v}" for k, v in list(failures.items())[:15])
    )


def _pre_handoff_tag() -> tuple[str, str] | None:
    """Newest release tag whose updater still ran post-pull phases in the pre-pull interpreter."""
    cp = _git("for-each-ref", "--merged=HEAD", "--sort=-version:refname",
              "--format=%(refname:short)", "refs/tags/v20*")
    for tag in cp.stdout.split()[:15] if cp.returncode == 0 else []:
        if _git("cat-file", "-e", f"{tag}:hermes_cli/update_handoff.py").returncode == 0:
            continue
        grep = _git("grep", "-l", "def _purge_stale_hermes_modules", tag, "--", "hermes_cli")
        return (tag, grep.stdout.strip()) if grep.returncode == 0 and grep.stdout.strip() else None
    return None


def _copy_tree(src: Path, dst: Path, names: list[str]) -> None:
    def _link_or_copy(s, d):
        try:
            os.link(s, d)
        except OSError:
            shutil.copy2(s, d)

    dst.mkdir(parents=True)
    for name in names:
        s = src / name
        if s.is_dir():
            shutil.copytree(s, dst / name, copy_function=_link_or_copy, ignore=shutil.ignore_patterns("__pycache__"))
        elif s.is_file():
            _link_or_copy(s, dst / name)


def test_pre_handoff_updater_stale_graph_imports_post_update_modules(tmp_path):
    """#114616 / #112522 shape: v2026.9.14's updater purges package prefixes only, keeps root
    modules (``utils``, ``hermes_constants``) cached, then imports new restart-phase code.
    Retire this leg together with ``hermes_cli/stale_modules.py``."""
    _sandbox_or_skip()
    found = _pre_handoff_tag()
    if found is None:
        pytest.skip("no pre-hand-off release tag reachable from HEAD (shallow clone without tags?)")
    tag, _purge_file = found
    old_manifest = _git("show", f"{tag}:pyproject.toml")
    assert old_manifest.returncode == 0, old_manifest.stderr
    old_deps = tomllib.loads(old_manifest.stdout)["project"]["dependencies"]
    # The old updater starts under its installed dependencies, then switches code in place.
    # The current test environment deliberately lacks PyYAML; provision the legacy direct
    # requirement in an isolated PM environment rather than mutating the test interpreter.
    from packaging.requirements import Requirement
    from pm import ensure_environment

    legacy_yaml = [r for r in old_deps if Requirement(r).name.lower() == "pyyaml"]
    assert len(legacy_yaml) == 1, f"{tag} no longer declares exactly one PyYAML requirement"
    legacy_python = ensure_environment(
        "pre-handoff-updater", [*PYPROJECT["project"]["dependencies"], *legacy_yaml],
        root=tmp_path / "legacy-deps", explicit=True,
    )
    pkg_tops = sorted({p.split(".")[0] for p in PYPROJECT["tool"]["setuptools"]["packages"]["find"]["include"]})
    old_entries = [n for n in _git("ls-tree", "--name-only", tag).stdout.split()
                   if n.endswith(".py") or n in pkg_tops]
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    archive = subprocess.run(["git", "archive", "--format=tar", tag, "--", *old_entries], cwd=str(WORKTREE),
                             capture_output=True, stdin=subprocess.DEVNULL, timeout=120)
    assert archive.returncode == 0, archive.stderr.decode(errors="replace")
    subprocess.run(["tar", "-x", "-C", str(checkout)], input=archive.stdout, check=True, timeout=120)
    roots = [n for n in _root_py_modules() if not n.startswith("_test_")]
    _copy_tree(WORKTREE, tmp_path / "new", [f"{n}.py" for n in roots] + pkg_tops)

    runner = _write_script(tmp_path / "runner", "stale_runner.py", _STALE_RUNNER)
    out = tmp_path / "stale.json"
    spec = {
        "checkout": str(checkout), "new_tree": str(tmp_path / "new"), "old_graph": list(_OLD_UPDATER_GRAPH),
        "targets": list(_POST_PURGE_IMPORTS), "root_modules": roots, "out": str(out),
    }
    spec_path = tmp_path / "stale-spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    env = isolated_env(tmp_path / "sbx")
    cp = run([str(legacy_python), str(runner), str(spec_path)], env=env, cwd=tmp_path, writable=[tmp_path], timeout=600)
    assert cp.returncode == 0 and out.exists(), describe(cp)
    report = json.loads(out.read_text(encoding="utf-8"))
    # Non-vacuous: the purge really left root modules cached that lack symbols the new tree defines.
    assert report["stale_roots"], f"{tag} left no stale root module behind; the scenario no longer exercises anything"
    broken = [r for r in report["results"] if not r["ok"]]
    assert not broken, (
        f"after a {tag} updater's purge (stale roots: {sorted(report['stale_roots'])}), post-update imports fail:\n\n"
        + "\n\n".join(f"== {r['target']}: {r['type']}: {r['msg']}\n{r.get('tb', '')}" for r in broken)
    )


# --------------------------------------------------------------------------- (b) entrypoints


def _console_script(bin_dir: Path, name: str) -> Path:
    """The wrapper pip/uv generate for a ``[project.scripts]`` entry, byte-for-byte in behaviour."""
    module, func = PYPROJECT["project"]["scripts"][name].split(":")
    return _write_script(bin_dir, name, (
        f"#!{PY}\nimport re\nimport sys\nfrom {module} import {func}\n"
        "if __name__ == '__main__':\n"
        "    sys.argv[0] = re.sub(r'(-script\\.pyw|\\.exe)?$', '', sys.argv[0])\n"
        f"    sys.exit({func}())\n"
    ))


@dataclass(frozen=True)
class Entry:
    via: str                 # "module" (python -m hermes_cli.main) or a [project.scripts] name
    args: tuple[str, ...]
    rcs: frozenset[int]
    prints_version: bool = False
    turn: bool = False
    tty: bool = False        # interactive: run on a pseudo-terminal and leave with EOF (Ctrl-D)


_ENTRIES = {
    "module-version": Entry("module", ("--version",), frozenset({0}), prints_version=True),
    "module-doctor": Entry("module", ("doctor",), frozenset({0, 1})),  # docs: 1 when problems remain
    "module-oneshot": Entry("module", ("-z",), frozenset({0}), turn=True),
    # The #111942 symptom: the interactive CLI died at startup whenever config.yaml existed.
    "module-interactive-cli": Entry("module", (), frozenset({0}), tty=True),
    "hermes-version": Entry("hermes", ("--version",), frozenset({0}), prints_version=True),
    "hermes-help": Entry("hermes", ("--help",), frozenset({0})),
    "hermes-acp-version": Entry("hermes-acp", ("--version",), frozenset({0}), prints_version=True),
    "hermes-acp-help": Entry("hermes-acp", ("--help",), frozenset({0})),
    "hermes-agent-help": Entry("hermes-agent", ("--help",), frozenset({0})),
}
def _run_on_tty(argv: list[str], *, env: dict[str, str], cwd: Path, writable: list[Path],
                timeout: float) -> subprocess.CompletedProcess:
    """Run ``argv`` on a pty; once it has drawn anything, send EOF every second until it exits.

    EOF typed before the line editor switched the tty to raw mode can be swallowed, so it is
    re-sent (bounded by ``timeout``) rather than timed with a sleep.
    """
    import pty
    import select
    import time

    master, slave = pty.openpty()
    proc = subprocess.Popen(sandbox_argv(argv, writable=writable), env=env, cwd=str(cwd),
                            stdin=slave, stdout=slave, stderr=slave, start_new_session=True)
    os.close(slave)
    buf, last_eof, deadline = b"", 0.0, time.monotonic() + timeout
    try:
        while proc.poll() is None:
            if time.monotonic() > deadline:
                raise AssertionError(f"{argv} still running after {timeout}s on a tty:\n"
                                     f"{buf.decode(errors='replace')[-4000:]}")
            if select.select([master], [], [], 0.2)[0]:
                try:
                    buf += os.read(master, 65536)
                except OSError:  # EIO: every slave fd closed, the process is exiting
                    pass
            if buf and time.monotonic() - last_eof > 1.0:
                os.write(master, b"\x04")
                last_eof = time.monotonic()
        while select.select([master], [], [], 0)[0]:
            try:
                chunk = os.read(master, 65536)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
    finally:
        kill_tree(proc)
        os.close(master)
    return subprocess.CompletedProcess(argv, proc.returncode, buf.decode(errors="replace"), "")


def test_every_console_script_is_in_the_entrypoint_matrix():
    assert set(PYPROJECT["project"]["scripts"]) <= {e.via for e in _ENTRIES.values()}


def _db_rows(db: Path, sql: str) -> list[tuple]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


@pytest.mark.parametrize("case", list(_ENTRIES))
def test_entrypoint_in_a_fresh_process(case, tmp_path):
    _sandbox_or_skip()
    entry = _ENTRIES[case]
    canary = f"C6-{uuid.uuid4().hex[:12]}"
    reply = f"FAKE-REPLY-{uuid.uuid4().hex[:12]}"
    env = isolated_env(tmp_path, pythonpath=WORKTREE,
                       extra={"TERM": "xterm-256color", "COLUMNS": "120", "LINES": "40"} if entry.tty else None)
    hermes_home = Path(env["HERMES_HOME"])
    with FakeLLMServer([Text(reply)]) as srv:
        write_hermes_home(hermes_home, srv.base_url)
        config_before = (hermes_home / "config.yaml").read_bytes()
        args = [*entry.args, canary] if entry.turn else list(entry.args)
        if entry.via == "module":
            argv = [PY, "-m", "hermes_cli.main", *args]
        else:
            argv = [PY, str(_console_script(tmp_path / "bin", entry.via)), *args]
        launcher = _run_on_tty if entry.tty else run
        cp = launcher(argv, env=env, cwd=WORKTREE, writable=[tmp_path], timeout=300)
        main_requests = srv.main_requests()

    out = cp.stdout + cp.stderr
    assert cp.returncode in entry.rcs, describe(cp)
    assert TRACEBACK not in out, describe(cp)
    shim_log = tmp_path / "shims" / "shim-calls.log"
    assert not shim_log.exists() or not shim_log.read_text(encoding="utf-8").strip(), (
        f"{case} called a service manager: {shim_log.read_text(encoding='utf-8')}")
    if entry.prints_version:
        from hermes_cli.version_info import get_version_info
        commit = get_version_info().commit
        assert commit and commit[:7] in cp.stdout, describe(cp)
    db = hermes_home / "state.db"
    if db.exists():
        assert _db_rows(db, "PRAGMA integrity_check") == [("ok",)], f"{case} left a corrupt state.db"
    if case == "module-doctor":
        assert cp.stdout.strip(), describe(cp)
        assert (hermes_home / "config.yaml").read_bytes() == config_before, "doctor without --fix rewrote config.yaml"
    if not entry.turn:
        assert main_requests == [], f"{case} made {len(main_requests)} model call(s)"
        return

    # One-shot turn: sent == rendered == persisted, and the DB is intact.
    assert len(main_requests) == 1, main_requests
    users = [m for m in main_requests[0]["messages"] if m["role"] == "user"]
    assert users and users[-1]["content"] == canary, main_requests[0]["messages"]
    assert reply in cp.stdout, describe(cp)
    assert db.exists(), "the one-shot turn persisted nothing"
    sessions = _db_rows(db, "SELECT id FROM sessions")
    assert len(sessions) == 1, sessions
    rows = _db_rows(db, f"SELECT role, content FROM messages WHERE session_id = '{sessions[0][0]}' ORDER BY id")
    assert ("user", canary) in rows and ("assistant", reply) in rows, rows


def test_serve_announces_ready_and_stops_cleanly_on_sigterm(tmp_path):
    _sandbox_or_skip()
    psutil = pytest.importorskip("psutil")
    lock_dir = tmp_path / "host-locks"
    env = isolated_env(tmp_path, pythonpath=WORKTREE, extra={"HERMES_GATEWAY_LOCK_DIR": str(lock_dir)})
    reaper = _write_script(tmp_path / "runner", "subreaper.py", _SUBREAPER)
    report_path = tmp_path / "reaper.json"
    stdout_path, stderr_path = tmp_path / "serve.out", tmp_path / "serve.err"
    with FakeLLMServer() as srv, open(stdout_path, "w", encoding="utf-8") as out, open(stderr_path, "w", encoding="utf-8") as err:
        write_hermes_home(Path(env["HERMES_HOME"]), srv.base_url)
        argv = [PY, str(reaper), str(report_path),
                PY, "-m", "hermes_cli.main", "serve", "--host", "127.0.0.1", "--port", "0"]
        proc = subprocess.Popen(sandbox_argv(argv, writable=[tmp_path]), env=env, cwd=str(WORKTREE),
                                stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)

        def _logs() -> str:
            return f"--- stdout ---\n{stdout_path.read_text(encoding='utf-8')[-4000:]}\n--- stderr ---\n{stderr_path.read_text(encoding='utf-8')[-4000:]}"

        try:
            def _ready():
                m = _READY_RE.search(stdout_path.read_text(encoding="utf-8"))
                if m:
                    return int(m.group(1))
                assert proc.poll() is None, f"serve exited rc={proc.returncode} before READY\n{_logs()}"
                return None

            port = wait_for(_ready, timeout=180, what="HERMES_BACKEND_READY on stdout")
            socket.create_connection(("127.0.0.1", port), timeout=10).close()
            record_file = lock_dir / "host-serve.json"
            record = json.loads(record_file.read_text(encoding="utf-8"))
            assert record.get("port") == port, record

            # With bwrap the reaper is a child; without it (CI) it is proc itself.
            root = psutil.Process(proc.pid)
            reaper_proc = next(p for p in [root, *root.children(recursive=True)]
                               if p.cmdline()[1:2] == [str(reaper)])
            serve = next(p for p in reaper_proc.children() if p.cmdline()[1:3] == ["-m", "hermes_cli.main"])
            descendants = [p.pid for p in serve.children(recursive=True)]
            serve.send_signal(signal.SIGTERM)
            wait_for(lambda: report_path.exists() and report_path.stat().st_size > 0, timeout=120,
                     what=f"serve to exit after SIGTERM (descendants at signal time: {descendants})")
            proc.wait(timeout=60)
        finally:
            kill_tree(proc)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["returncode"] in (0, -signal.SIGTERM), f"{report}\n{_logs()}"
    assert report["orphans"] == [], f"serve left descendants running after SIGTERM: {report['orphans']}"
    assert not record_file.exists() and not (lock_dir / "host-serve.token").exists(), (
        "serve's host record/token outlived the process (a Desktop would attach to a dead port)")
    assert TRACEBACK not in stdout_path.read_text(encoding="utf-8") + stderr_path.read_text(encoding="utf-8"), _logs()
    shim_log = tmp_path / "shims" / "shim-calls.log"
    assert not shim_log.exists() or not shim_log.read_text(encoding="utf-8").strip(), shim_log.read_text(encoding="utf-8")
    assert srv.main_requests() == [], "serve made a model call while idle"
