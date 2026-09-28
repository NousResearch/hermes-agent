"""Install-scoped dependency selection is readable before third-party imports."""
import json
from pathlib import Path

import pytest


def test_install_runtime_selection_is_scoped_and_read_only(tmp_path, monkeypatch):
    from pm import environments as runtime_paths

    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    first, second = tmp_path / "first", tmp_path / "second"
    for root in (first, second):
        (root / ".venv").mkdir(parents=True)
    assert runtime_paths.selected_venv(first) == first / ".venv"
    assert not home.exists()
    state = runtime_paths.install_state_dir(first)
    assert state != runtime_paths.install_state_dir(second)
    generation = state / "environments" / "candidate" / "venv"
    generation.mkdir(parents=True)
    (generation / "pyvenv.cfg").write_text("home = test\n")
    (state / "facts.json").write_text(json.dumps({
        "schema": 1, "packages": {"venv": {"environment": str(generation), "stamp": "verified"}},
    }))
    assert runtime_paths.selected_venv(first) == generation
    assert runtime_paths.selected_venv(second) == second / ".venv"
    monkeypatch.setenv("HERMES_HOME", str(home / "profiles" / "work"))
    assert runtime_paths.selected_venv(first) == generation


def test_boot_uses_one_selected_dependency_tree_in_fresh_process(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys
    from pm import environments as runtime_paths

    root = tmp_path / "repo"
    base = root / "venv"
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    state = runtime_paths.install_state_dir(root)
    selected = state / "environments" / "new" / "venv"
    def site_of(venv):
        return venv / ("Lib/site-packages" if os.name == "nt" else f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")
    for venv, version in [(base, "old"), (selected, "new")]:
        site = site_of(venv)
        site.mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text("home = test")
        (site / "probe_package.py").write_text(f"version = {version!r}")
    (site_of(base) / "base_only.py").write_text("version = 'must-not-leak'")
    (state / "facts.json").write_text(json.dumps({"schema": 1, "packages": {
        "venv": {"environment": str(selected)}
    }}))
    code = (
        "import sys; from pathlib import Path; from pm.environments import activate_dependencies; "
        "sys.path.insert(0, sys.argv[2]); activate_dependencies(Path(sys.argv[1])); "
        "import probe_package, importlib.util; print(probe_package.version); "
        "print(importlib.util.find_spec('base_only') is None)"
    )
    process = subprocess.run([sys.executable, "-c", code, str(root), str(site_of(base))],
                             env=dict(os.environ), text=True, capture_output=True, timeout=30)
    assert process.returncode == 0, process.stderr
    assert process.stdout.splitlines() == ["new", "True"]


@pytest.mark.parametrize("command,allowed", [(["pm", "install", "--help"], True), (["pm", "doctor"], True),
    (["-p", "default", "pm", "repair"], True), (["chat"], False), (["chat", "pm", "install"], False)])
def test_broken_environment_keeps_explicit_repair_entry_reachable(tmp_path, monkeypatch, command, allowed):
    import os
    import subprocess
    import sys
    from pm.environments import runtime_facts_path

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    repo = Path(__file__).resolve().parents[2]
    record = runtime_facts_path(repo)
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"packages": {"venv": {"environment": str(tmp_path / "missing")}}}))
    code = "import sys; sys.argv = ['hermes', *sys.argv[1:]]; import hermes_bootstrap; print('bootstrap-ready')"
    result = subprocess.run([sys.executable, "-c", code, *command], env=dict(os.environ),
                            capture_output=True, text=True, timeout=30)
    assert (result.returncode == 0) is allowed, result.stderr
    if not allowed:
        assert "hermes pm repair" in result.stderr
        assert "Traceback" not in result.stderr


def test_manual_repair_bypasses_damaged_generation_activation(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys
    from pm.environments import install_state_dir, runtime_facts_path, site_packages

    repo = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    generation = install_state_dir(repo) / "environments" / "damaged"
    environment = generation / "venv"
    site_packages(environment).mkdir(parents=True)
    (environment / "pyvenv.cfg").write_text("home = test", encoding="utf-8")
    (generation / ".lease-managed").touch()
    (generation / ".leases").write_text("not a directory", encoding="utf-8")
    runtime_facts_path(repo).write_text(json.dumps({"schema": 1, "packages": {"venv": {
        "environment": str(environment), "extras": [], "stamp": "old",
    }}}), encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(repo)}
    result = subprocess.run([sys.executable, "-S", "-m", "hermes_cli.main", "pm", "repair", "--help"],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "hermes pm repair" in result.stdout


@pytest.mark.parametrize("interpreter", ["store", "venv"])
@pytest.mark.parametrize("with_state", [True, False])
def test_boot_never_activates_the_pre_pm_venv(tmp_path, monkeypatch, interpreter, with_state):
    """Nothing committed must not mean "load the in-tree venv": it was built for another
    interpreter, so PM's store Python lost every compiled module from it after an update."""
    import os
    import subprocess
    import sys
    from pm import environments as runtime_paths

    base_python = getattr(sys, "_base_executable", sys.executable)
    base_prefix = Path(sys.base_prefix).resolve()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    # The store interpreter is PM's: a non-venv Python living under the runtime dir.
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(base_prefix.parent))
    root = tmp_path / "repo"
    legacy = root / "venv"
    (legacy / "pyvenv.cfg").parent.mkdir(parents=True)
    (legacy / "pyvenv.cfg").write_text("home = test\n")
    runtime_paths.site_packages(legacy).mkdir(parents=True)
    (runtime_paths.site_packages(legacy) / "legacy_only.py").write_text("")
    if with_state:
        runtime_paths.install_state_dir(root).mkdir(parents=True)
    python = base_python
    if interpreter == "venv":
        subprocess.run([base_python, "-m", "venv", "--without-pip", str(tmp_path / "dev")], check=True, timeout=60)
        python = str(runtime_paths.venv_python(tmp_path / "dev"))
    repo = Path(__file__).resolve().parents[2]
    code = (
        "import sys, importlib.util; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
        "from pm.environments import activate_dependencies\n"
        "try:\n    activate_dependencies(Path(sys.argv[2]))\n"
        "except RuntimeError as exc:\n    print('refused:', exc); raise SystemExit(0)\n"
        "print('legacy importable:', importlib.util.find_spec('legacy_only') is not None)"
    )
    result = subprocess.run([python, "-I", "-c", code, str(repo), str(root)], env=dict(os.environ),
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    expected = ("refused: no dependency environment is committed" if interpreter == "store"
                else "legacy importable: False")
    assert result.stdout.strip().startswith(expected), result.stdout


@pytest.mark.parametrize("data", [[], {"packages": []}, {"packages": {"venv": []}}])
def test_malformed_selection_has_actionable_error(tmp_path, monkeypatch, data):
    from pm import environments as runtime_paths
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    record = runtime_paths.runtime_facts_path(tmp_path / "repo")
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="dependency environment"):
        runtime_paths.selected_venv(tmp_path / "repo")


@pytest.mark.parametrize("bad_path", ["outside", "missing"])
def test_invalid_selected_environment_never_silently_falls_back(tmp_path, monkeypatch, bad_path):
    from pm import environments as runtime_paths

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    root = tmp_path / "repo"
    (root / "venv").mkdir(parents=True)
    state = runtime_paths.install_state_dir(root)
    state.mkdir(parents=True)
    candidate = tmp_path / "outside" if bad_path == "outside" else state / "environments" / "missing"
    if bad_path == "outside":
        candidate.mkdir()
        (candidate / "pyvenv.cfg").write_text("home = test\n")
    (state / "facts.json").write_text(json.dumps({
        "schema": 1, "packages": {"venv": {"environment": str(candidate)}},
    }))
    with pytest.raises(RuntimeError, match="environment"):
        runtime_paths.selected_venv(root)


@pytest.mark.parametrize("published", ["checkout", "home", "none"])
@pytest.mark.parametrize("preactivated", [False, True])
@pytest.mark.parametrize("launcher_last", [False, True])
def test_activated_path_keeps_own_launchers_ahead_of_snapshot_scripts(
        tmp_path, monkeypatch, published, preactivated, launcher_last):
    """The dependency venv ships ``hermes`` console scripts bound to its build snapshot.

    Activation prepends the venv's bin directory to PATH, and the snapshot the venv was built
    from is only refreshed when the dependency graph changes — so a launcher this install
    publishes must keep its position ahead of that entry. Otherwise an activated shell answers
    ``hermes`` with a stale copy of the checkout (observed as ``Not a git repository`` out of
    ``hermes update``), which is the convergence the installer's user-PATH registration owns.
    """
    import os
    import shutil
    import subprocess
    import sys
    from pm import environments as runtime_paths

    root = tmp_path / "repo"
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    state = runtime_paths.install_state_dir(root)
    venv = state / "environments" / "gen" / "venv"
    runtime_paths.site_packages(venv).mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = test\n", encoding="utf-8")
    scripts = runtime_paths.venv_bin_dir(venv)
    scripts.mkdir(parents=True)
    (state / "facts.json").write_text(json.dumps({"schema": 1, "packages": {
        "venv": {"environment": str(venv)}}}), encoding="utf-8")
    entry = "hermes.cmd" if os.name == "nt" else "hermes"
    (scripts / entry).write_text("snapshot\n", encoding="utf-8")
    launcher = {"checkout": root / ".hermes" / "bin", "home": home / "bin"}.get(published)
    # ``preactivated`` reproduces the shape that actually broke: the venv's bin dir already on
    # PATH ahead of our launcher, as every process spawned by an activated backend sees it.
    ambient = [str(scripts)] if preactivated else []
    if launcher is not None:
        launcher.mkdir(parents=True)
        (launcher / entry).write_text(
            f"exec '{root / '.hermes' / 'bin' / 'hermes'}' \"$@\"  # checkout\n", encoding="utf-8")
        os.chmod(launcher / entry, 0o755)
    ambient.append(str(tmp_path / "system"))
    if launcher is not None and not launcher_last:
        ambient.insert(0, str(launcher))
    elif launcher is not None:
        # The breaking shape: the launcher sits behind ambient entries, so an order that only
        # looks at the venv's bin dir leaves it where it was.
        ambient.append(str(launcher))
    code = (
        "import os, shutil, sys; from pathlib import Path; sys.path.insert(0, sys.argv[2]); "
        "from pm.environments import activate_dependencies; activate_dependencies(Path(sys.argv[1])); "
        "first = os.environ['PATH']; activate_dependencies(Path(sys.argv[1])); "
        "found = shutil.which('hermes'); print(os.environ['PATH']); "
        "print(os.environ['PATH'] == first); "
        "print(Path(found).read_text(encoding='utf-8').strip() if found else 'missing')"
    )
    process = subprocess.run(
        [sys.executable, "-c", code, str(root), str(Path(__file__).resolve().parents[2])],
        env={**os.environ, "PATH": os.pathsep.join(ambient)},
        text=True, capture_output=True, timeout=30,
    )
    assert process.returncode == 0, process.stderr
    ordered, stable, resolved = process.stdout.splitlines()
    entries = ordered.split(os.pathsep)
    # Activation prepends the venv's bin dir, so the input here is exactly [venv, <ambient>…]
    # — the shape in which a position computed from the unordered input pushes the venv dir
    # behind the ambient entries. Both invariants are asserted rather than the raw order.
    assert stable == "True", "a second activation must not grow or reorder PATH"
    assert entries.count(str(scripts)) == 1, entries
    system = str(tmp_path / "system")
    assert entries.index(str(scripts)) < entries.index(system), entries
    if published == "none":
        # Nothing of ours is on PATH: the venv's bin dir still goes first, unchanged.
        assert entries[0] == str(scripts)
        assert resolved == "snapshot"
    else:
        assert entries[0] == str(launcher)
        assert "checkout" in resolved


@pytest.mark.platforms("posix")
def test_posix_shared_bin_counts_only_while_it_owns_this_checkout(tmp_path, monkeypatch):
    """``~/.local/bin`` and ``/usr/local/bin`` are shared: a launcher for another install there
    must not be hoisted ahead of the dependency environment. POSIX-gated — the candidate list
    only exists there, and faking ``os.name`` cannot work because ``pathlib`` binds its concrete
    path class from it at instantiation."""
    import os
    from pm import environments as runtime_paths

    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "userhome")
    shared = Path.home() / ".local" / "bin"
    shared.mkdir(parents=True)
    assert shared not in runtime_paths.launcher_dirs(root)
    launcher = root / ".hermes" / "bin" / "hermes"
    (shared / "hermes").write_text(f"#!/bin/sh\nexec '{launcher}' \"$@\"\n", encoding="utf-8")
    os.chmod(shared / "hermes", 0o755)
    assert shared in runtime_paths.launcher_dirs(root)
    # A launcher that merely mentions the checkout in a comment is not ownership.
    (shared / "hermes").write_text(f"#!/bin/sh\n# for {root / 'other'}\nexec /bin/false\n", encoding="utf-8")
    assert shared not in runtime_paths.launcher_dirs(root)
    # A sealed bundle links its shim into the payload bin, which lives *outside* the checkout, so
    # ownership is judged against the payload — the root never contains that target.
    payload_bin = root.parent / "bin"
    payload_bin.mkdir()
    (payload_bin / "hermes").write_text("shim\n", encoding="utf-8")
    (shared / "hermes").unlink()
    (shared / "hermes").symlink_to(payload_bin / "hermes")
    monkeypatch.setattr("hermes_cli._launchers._is_bundled_payload", lambda *_: True)
    assert payload_bin in runtime_paths.launcher_dirs(root)
    assert shared in runtime_paths.launcher_dirs(root)


def test_shared_home_bin_counts_only_while_its_launcher_serves_this_install(tmp_path, monkeypatch):
    """``$HERMES_HOME/bin`` is shared the same way: a second checkout installed into one home must
    not have its launcher hoisted ahead of this install's dependency environment, while this
    install's own launcher keeps its place — including the Windows executable, whose embedded
    bootstrap names the checkout instead of being readable as a script."""
    import os
    from pm import environments as runtime_paths

    root = tmp_path / "repo"
    (root / ".hermes" / "bin").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    shared = tmp_path / "home" / "bin"
    shared.mkdir(parents=True)
    entry = "hermes.cmd" if os.name == "nt" else "hermes"
    assert shared not in runtime_paths.launcher_dirs(root)
    # Another install's launcher, and a wrapper whose only mention of this checkout is a comment:
    # neither can certify the directory — what resolution picks there has to *lead* here.
    (shared / entry).write_text(
        f"#!/bin/sh\n# {root}\nexec '{tmp_path / 'other' / 'hermes'}' \"$@\"\n", encoding="utf-8")
    assert shared not in runtime_paths.launcher_dirs(root)
    (shared / entry).write_text(
        f"#!/bin/sh\nexec '{root / '.hermes' / 'bin' / 'hermes'}' \"$@\"\n", encoding="utf-8")
    assert shared in runtime_paths.launcher_dirs(root)


def test_shared_home_bin_recognizes_generated_windows_commands(tmp_path, monkeypatch):
    """The generated Windows commands keep the bootstrap where no text scan reaches it: the
    ``distlib`` ``.exe`` is a ZIP whose ``__main__.py`` *is* the bootstrap, and the fallback
    ``.cmd`` carries the same bootstrap base64-encoded. Both must certify the directory — and the
    same files for another install must not."""
    import base64
    import os
    import zipfile
    from pm import environments as runtime_paths

    if os.name != "nt":
        pytest.skip("the generated command formats only exist on Windows")

    root = tmp_path / "repo"
    (root / ".hermes" / "bin").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    shared = tmp_path / "home" / "bin"
    shared.mkdir(parents=True)
    script = f"sys.path.insert(0, {str(root)!r})\n"
    encoded = base64.b64encode(script.encode("utf-8")).decode("ascii")
    (shared / "hermes.cmd").write_text(
        f'@echo off\n"python.exe" -I -c "import base64; exec(base64.b64decode(\'{encoded}\'))"\n',
        encoding="utf-8")
    assert shared in runtime_paths.launcher_dirs(root)
    (shared / "hermes.cmd").unlink()
    with zipfile.ZipFile(shared / "hermes.exe", "w") as archive:
        archive.writestr("__main__.py", script)
    assert shared in runtime_paths.launcher_dirs(root)
    (shared / "hermes.exe").unlink()
    with zipfile.ZipFile(shared / "hermes.exe", "w") as archive:
        archive.writestr("__main__.py", f"sys.path.insert(0, {str(tmp_path / 'other')!r})\n")
    assert shared not in runtime_paths.launcher_dirs(root)
