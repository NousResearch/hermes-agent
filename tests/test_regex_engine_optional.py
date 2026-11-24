"""Code that runs where google-re2 is not installed must not need it.

Product code matches through the ``hermes_re`` facade, which needs the third-party package
google-re2 at the first MATCH (compiling works without it). Two rules follow, and each case
below checks one of them in a fresh interpreter that refuses to import ``re2``:

* **Importing never matches.** After ``hermes update`` swaps the checkout, the old release's
  process imports modules of the new tree with its old dependencies: lazily, and in its
  import-health probe, which reads an ``ImportError`` from a core module as a broken checkout.
  So no product module may run a regex match while it is being imported, and the error a match
  raises without the engine is a ``ModuleNotFoundError`` naming ``re2``, the one shape that
  probe reads as "dependencies not installed yet".
* **Code that runs without dependencies stays on stdlib ``re``:** what runs before dependencies
  exist (launcher, bootstrap, the package manager and its worker runtime), what an older
  release's still-running process calls from the new checkout, and files copied out of the
  package and imported alone.

The child reports, as JSON, every step that raised and every match the facade was asked for.
The control step proves the block is real: a facade match in that interpreter must raise.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

pytest.importorskip("hermes_re")  # the facade lands with the migration; nothing to guard before it

REPO = Path(__file__).resolve().parents[1]
ENGINE_HINT = "google-re2"

# Frozen by the releases already shipped: from v0.21.6 on, `hermes update` imports exactly these
# from the new checkout to judge its health, before the new dependencies are installed.
UPDATE_CRITICAL_MODULES = ("hermes_cli.main", "run_agent", "model_tools", "toolsets")

# Modules whose import ran a regex match before the facade arrived (module-level table building,
# a helper called at module level, a catalog entry that reaches a matcher). The match was moved
# out of import time in each; they stay listed so that it cannot come back unnoticed.
MATCHED_AT_IMPORT_BEFORE = (
    "agent.model_metadata",
    "agent.redact",
    "cli",
    "gateway.hosted_rooms",
    "gateway.run",
    "gateway.run_status_filter",
    "hermes_cli.commands",
    "hermes_cli.models_catalog_static",
    "persistence.sqlite.conversation._store.messages_sql",
    "persistence.sqlite.gateway._hosted_rooms_schema",
    "tools.discord_tool",
    "tools.threat_patterns",
)

_DRIVER = r'''
import importlib, importlib.abc, json, os, sys, traceback

REPO, CASE, WORK, ENGINE, EXTRA = sys.argv[1:6]
# What the shipped import-health probe sets before it imports: hermes_cli.main reads the command line
# while it is imported, and `update` is the command that keeps that import free of side effects.
sys.argv = ["hermes", "update"]
sys.path.insert(0, REPO)


class NoEngine(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("re2", "_re2"):
            raise ImportError(f"{name} is blocked: this interpreter models an install without google-re2")
        return None


if ENGINE == "blocked":
    for loaded in [m for m in sys.modules if m.split(".")[0] in ("re2", "_re2")]:
        del sys.modules[loaded]
    sys.meta_path.insert(0, NoEngine())

import hermes_re
from hermes_re import _engine

steps = []
matches = []      # [step label, "file:line: source line"] for every match the facade could not serve
current = ["control"]
_policy = _engine.engine_missing
_FACADE = os.path.dirname(os.path.abspath(hermes_re.__file__)) + os.sep


def recording_policy(exc):
    """Called by the facade each time a match needs the missing engine, also when the caller
    swallows the error: name the product line that asked."""
    outside = [f for f in traceback.extract_stack()[:-1] if not os.path.abspath(f.filename).startswith(_FACADE)]
    asked = outside[-1]
    matches.append([current[0], f"{asked.filename}:{asked.lineno}: {asked.line}"])
    return _policy(exc)


_engine.engine_missing = recording_policy


def step(label, fn):
    current[0] = label
    try:
        fn()
        steps.append([label, "ok", ""])
    except BaseException as exc:
        steps.append([label, type(exc).__name__, f"{exc}\n{traceback.format_exc(limit=-14)}"])


def control():
    try:
        hermes_re.compile("a").search("a")
    except ImportError as exc:
        return ["blocked", type(exc).__name__, getattr(exc, "name", None), str(exc)]
    return ["matched", "", None, ""]


verdict = control()
del matches[:]

if CASE == "imports":
    from hermes_agent.compat.required_names import REQUIRED_NAMES
    names = [name for name, kind in REQUIRED_NAMES.items() if kind in ("module", "package") and "/" not in name]
    for name in dict.fromkeys(json.loads(EXTRA) + names):
        step(f"import {name}", lambda name=name: importlib.import_module(name))

if CASE == "calls":
    def status():
        from gateway.status import looks_like_gateway_command_line
        from hermes_cli._launchers import runtime_command
        import shlex
        looks_like_gateway_command_line(f"{sys.executable} -m hermes_cli.main gateway run")
        looks_like_gateway_command_line(shlex.join(runtime_command(REPO, ["gateway", "run"])))

    def manifest():
        from hermes_cli.plugins_manifest import requires_hermes_error, version_satisfies
        assert version_satisfies(">=0.1.0,<999", "0.23.3")
        requires_hermes_error({"requires_hermes": ">=0.1.0"})

    def discord_home():
        from gateway.config import discord_channel_id_from_link
        assert discord_channel_id_from_link("https://discord.com/channels/1/2") == "2"

    def env_file():
        from pathlib import Path
        from agent.secret_scope import load_env_file
        path = Path(WORK) / "sample.env"
        path.write_text("PLAIN=1\nCOMMENTED=two # trailing comment\n", encoding="utf-8")
        assert load_env_file(path)["PLAIN"] == "1"

    def fleet_scope():
        from hermes_cli.update_fleet_scope import gateway_pid_foreign_checkout
        gateway_pid_foreign_checkout(os.getpid())

    def pid_namespace():
        from process_identity import persistent_record_pidns_checkable
        persistent_record_pidns_checkable(None)
        persistent_record_pidns_checkable("pid:[1]")

    def acp_edit():
        from acp_adapter.edit_approval import build_edit_proposal, should_auto_approve_edit
        body = "*** Begin Patch\n*** Update File: a.py\n@@\n-x\n+y\n*** End Patch\n"
        proposal = build_edit_proposal("patch", {"mode": "patch", "patch": body})
        if proposal is not None:
            should_auto_approve_edit(proposal, "ask", WORK)

    def restart_env():
        os.environ["HERMES_BRIDGED_TELEGRAM_REACTIONS"] = "sha256:0"
        from tools.environments.local import served_profile_child_env
        served_profile_child_env(target_home=os.environ["HERMES_HOME"], inherit_credentials=True)

    def frozen_patterns():
        from agent.skill_utils import _NAMESPACE_RE
        from hermes_cli.profiles import _PROFILE_ID_RE
        _NAMESPACE_RE.match("plugin:skill")
        assert _PROFILE_ID_RE.match("default")

    for fn in (status, manifest, discord_home, env_file, fleet_scope, pid_namespace, acp_edit,
               restart_env, frozen_patterns):
        step(fn.__name__, fn)

print("RESULT " + json.dumps({"control": verdict, "steps": steps, "matches": matches}))
sys.stdout.flush()
os._exit(0)  # an imported module may have started a thread that would keep the interpreter alive
'''


def _run(case: str, tmp_path: Path, *, engine: bool = False, extra: tuple[str, ...] = ()) -> dict:
    """Run one driver case in a fresh interpreter; ``engine=True`` leaves google-re2 importable."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    home = tmp_path / "home"
    hermes_home = tmp_path / "hermes-home"
    work = tmp_path / "work"
    for folder in (home, hermes_home, work):
        folder.mkdir()
    driver = tmp_path / "driver.py"
    driver.write_text(_DRIVER, encoding="utf-8")
    env = {"HOME": str(home), "USERPROFILE": str(home), "HERMES_HOME": str(hermes_home),
           "PATH": os.environ.get("PATH", ""), "LANG": "C.UTF-8", "TZ": "UTC",
           "XDG_STATE_HOME": str(tmp_path / "state"), "XDG_CONFIG_HOME": str(tmp_path / "config"),
           "LOCALAPPDATA": str(home / "AppData" / "Local"), "APPDATA": str(home / "AppData" / "Roaming"),
           "PYTHONDONTWRITEBYTECODE": "1", "HERMES_DISABLE_LAZY_INSTALLS": "1",
           "HERMES_TEST_ISOLATION": str(hermes_home)}
    if "SYSTEMROOT" in os.environ:  # Windows cannot start an interpreter without it
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    proc = subprocess.run(
        [sys.executable, str(driver), str(REPO), case, str(work), "present" if engine else "blocked", json.dumps(extra)],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=120, check=False, stdin=subprocess.DEVNULL)
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    assert lines, f"driver failed (exit {proc.returncode}):\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    result = json.loads(lines[-1][len("RESULT "):])
    outcome, kind, _, text = result["control"]
    if engine:
        assert outcome == "matched", f"google-re2 does not work in the test interpreter ({kind}: {text})"
    else:
        assert outcome == "blocked" and ENGINE_HINT in text, (
            f"the engine was not blocked in the child ({outcome}: {text!r}): this test would pass vacuously")
    result["steps"] = {label: (outcome, detail) for label, outcome, detail in result["steps"]}
    return result


def _engine_failures(steps: dict[str, tuple[str, str]]) -> dict[str, str]:
    return {label: detail for label, (outcome, detail) in steps.items()
            if outcome != "ok" and ENGINE_HINT in detail}


def test_a_match_without_the_engine_raises_what_old_updaters_tolerate(tmp_path):
    """The import-health probe of releases that shipped without google-re2 skips a
    ``ModuleNotFoundError`` whose ``name`` is a third-party module; any other ``ImportError``
    from a core module makes it report a broken checkout."""
    outcome, kind, name, text = _run("control", tmp_path)["control"]
    assert (outcome, kind, name) == ("blocked", "ModuleNotFoundError", "re2"), text


def test_modules_import_without_the_engine(tmp_path):
    """Every module an old release imports from the new checkout, and every module that used to
    match at import, loads where google-re2 is absent, and asks the facade for no match while
    loading (a match whose error is swallowed leaves a half-initialised module behind)."""
    extra = UPDATE_CRITICAL_MODULES + MATCHED_AT_IMPORT_BEFORE
    with_engine = _run("imports", tmp_path / "with-engine", engine=True, extra=extra)["steps"]
    blocked = _run("imports", tmp_path / "blocked", extra=extra)
    # A REQUIRED_NAMES entry this environment cannot import even with the engine (an optional
    # dependency that is not installed here) says nothing about the engine. The modules listed in
    # this file must always import: a renamed one would otherwise drop out unjudged.
    importable = [label for label, (outcome, _) in with_engine.items() if outcome == "ok"]
    not_importable = {f"import {name}": with_engine[f"import {name}"][1] for name in extra
                      if f"import {name}" not in importable}
    assert not not_importable, (
        "listed modules do not import even with google-re2 present (renamed? fix the list in this file):\n"
        + "\n".join(f"{label}\n{detail[-2500:]}" for label, detail in not_importable.items()))
    problems = [f"{label} raised {blocked['steps'][label][0]} without google-re2:\n{blocked['steps'][label][1][-2500:]}"
                for label in importable if blocked["steps"][label][0] != "ok"]
    problems += [f"during `{label}` a regex match ran at {where}" for label, where in blocked["matches"]]
    assert not problems, (
        "importing a product module must not need google-re2: a process of an older release imports "
        "these with its old dependencies. Compile at module level and match inside functions (or use "
        "string code for import-time tables); the file to change is the last one outside hermes_re/ "
        "in each traceback:\n\n" + "\n\n".join(problems))


def test_functions_old_releases_call_run_without_the_engine(tmp_path):
    steps = _run("calls", tmp_path)["steps"]
    failures = _engine_failures(steps)
    assert not failures, (
        "these calls reached a google-re2 match in a process that does not have it:\n"
        + "\n".join(f"{label}\n{detail[-1500:]}" for label, detail in failures.items()))
    broken = {label: detail for label, (outcome, detail) in steps.items() if outcome != "ok"}
    assert not broken, (
        "a call failed for another reason, so it may not have reached its regex:\n"
        + "\n".join(f"{label}\n{detail[-1500:]}" for label, detail in broken.items()))


def test_copied_files_import_alone(tmp_path):
    """A ``copy`` file is run with nothing else on the path (``nix/desktop.nix`` copies the desktop
    entry module next to its build script): it cannot import a first-party module."""
    from hermes_agent.compat.required_names import REQUIRED_NAMES

    copied = [name for name, kind in REQUIRED_NAMES.items() if kind == "copy" and name.endswith(".py")]
    assert copied, "REQUIRED_NAMES lists no copied Python file"
    for name in copied:
        alone = tmp_path / Path(name).stem
        alone.mkdir()
        shutil.copyfile(REPO / name, alone / Path(name).name)
        code = f"import sys; sys.path.insert(0, {str(alone)!r}); import {Path(name).stem}"
        proc = subprocess.run([sys.executable, "-I", "-S", "-c", code], cwd=tmp_path, capture_output=True,
                              text=True, timeout=60, check=False, env={"PATH": os.environ.get("PATH", ""),
                                                                       **({"SYSTEMROOT": os.environ["SYSTEMROOT"]}
                                                                          if "SYSTEMROOT" in os.environ else {})})
        assert proc.returncode == 0, f"{name} does not import alone:\n{proc.stderr[-2000:]}"
