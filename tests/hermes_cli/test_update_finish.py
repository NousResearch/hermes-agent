"""Historical takeover completion in a fresh, dependency-selected process.

PM preparation has its own worker tests. Here the tool provisioner supplies
the real test Node/npm and icons are pre-prepared. npm ci, both compilers,
application imports, maintenance/config migration and receipts are real.
Only fleet discovery/restarts and machine-wide post-update repairs are seams.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _put(root, name, content):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def completion(tmp_path, monkeypatch):
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    from hermes_cli.runtime_paths import install_state_dir, runtime_facts_path, site_packages

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(home / ".hermes"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "store"))
    source = tmp_path / "selected source"
    source.mkdir()
    # Copies give bootstrap/main a genuine selected checkout identity. No file
    # in the working checkout is modified, even by import-time self-heals.
    for path in ROOT.glob("*.py"):
        shutil.copy2(path, source / path.name)
    for name in ("hermes_cli", "pm", "agent", "gateway", "tools", "cron"):
        shutil.copytree(ROOT / name, source / name,
                        ignore=shutil.ignore_patterns("__pycache__", "web_dist", "tui_dist"))
    shutil.copytree(ROOT / "scripts/build", source / "scripts/build",
                    ignore=shutil.ignore_patterns("__pycache__"))
    _put(source, "pyproject.toml", '[project]\nname="takeover-fixture"\nversion="2.0"\n')
    # The selected interpreter is dependency-free, not the pytest environment.
    python = tmp_path / "store/python/bin/python3"
    python.parent.mkdir(parents=True)
    shutil.copy2(Path(sys._base_executable).resolve(), python)
    _put(tmp_path / "store", "facts.json", json.dumps({"packages": {"python": {"entry": "python"}}}))
    generation = install_state_dir(source) / "environments/prepared"
    site = site_packages(generation)
    site.mkdir(parents=True)
    _put(generation, "pyvenv.cfg", "include-system-site-packages = false\n")
    # Add the already prepared *real* test dependencies through a selected
    # generation .pth, as editable PM generations do. -I ignores PYTHONPATH.
    dependency_sites = [p for p in sys.path if Path(p).name in ("site-packages", "dist-packages")]
    assert dependency_sites
    _put(site, "dependencies.pth", "\n".join(dependency_sites) + "\n")
    _put(site, "selected_dependency.py", "VALUE = 'selected generation'\n")
    _put(runtime_facts_path(source).parent, "facts.json", json.dumps({
        "packages": {"venv": {"environment": str(generation)}}}))
    _put(home / ".hermes", "config.yaml",
         f"_config_version: {DEFAULT_CONFIG['_config_version'] - 1}\nmodel:\n  default: retained-model\n")
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("HERMES_", "PYTHON", "PYTEST_", "VIRTUAL_ENV", "npm_", "NPM_"))}
    env.update(HOME=str(home), HERMES_HOME=str(home / ".hermes"),
               HERMES_RUNTIME_DIR=str(tmp_path / "store"),
               PYTHONPATH="/obsolete/pre-pm/site-packages", PYTHONHOME="/obsolete/python",
               NPM_CONFIG_OFFLINE="true", NPM_CONFIG_CACHE=str(tmp_path / "npm-cache"))
    probe = subprocess.run([str(python), "-I", "-c",
                            "import importlib.util; assert importlib.util.find_spec('yaml') is None"],
                           env=env, capture_output=True, text=True)
    assert probe.returncode == 0, probe.stderr
    request = {
        "root": str(source), "desktop": False, "assume_yes": True, "gateway_mode": False,
        "pre_update_snapshot_id": "original-snapshot", "pre_update_version": "1.0",
        "update_id": "historical-parent-correlation",
        "receipt": {"update_id": "historical-parent-correlation", "started_at": "before-the-swap",
                    "pre_update": {"sha": "old-sha"},
                    "steps": [{"name": "pull", "ok": True, "detail": "original", "at": "before"}],
                    "plan": {"expected_sha": "old-sha"}},
        "pm_receipt": {"update_id": "historical-parent-correlation", "outcome": "success",
                       "exit_code": 0, "venv_rebuild": {"environment": str(generation)}},
        "plan": {"install_method": "git", "expected_sha": "old-sha", "expected_version": "1.0",
                 "profiles": ["default"], "runtimes": [{
                     "kind": "serve", "profile": "default", "pid": 99999999,
                     "supervisor": "manual-serve", "code_sha": "old-sha", "code_version": "1.0",
                     "restart_via": "respawn-argv", "detail": {"argv": ["original", "serve"]}}]},
        "windows_resume": {"resume_needed": False, "profiles": {"default": {"argv": ["original"]}},
                           "unmapped": [], "services": [], "service_profiles": {}},
    }
    context = _put(tmp_path, "context.json", json.dumps(request))
    result = tmp_path / "result.json"
    runner = _put(tmp_path, "run-completion.py", textwrap.dedent('''
        import importlib.abc
        import importlib.machinery
        import json
        from pathlib import Path
        import runpy
        import sys

        context, result, fault = sys.argv[1:]
        request = json.loads(Path(context).read_text())
        root = Path(request['root'])
        sys.path.insert(0, str(root))

        class CompletionImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'pre_pm_only':
                    raise AssertionError('historical process module leaked into completion')
                if fullname == 'pm.recovery':
                    raise RuntimeError('PM recovery is unavailable in this completion fixture')
                if fullname == 'pm.receipt' and fault == 'receipt':
                    spec = importlib.machinery.PathFinder.find_spec(fullname, path)
                    original = spec.loader.exec_module
                    def execute(module):
                        original(module)
                        def broken(*args, **kwargs):
                            raise RuntimeError('cannot accept prepared receipt')
                        module.accept_worker_receipt = broken
                    spec.loader.exec_module = execute
                    return spec

                if fullname != 'hermes_cli.main':
                    return None
                import selected_dependency
                assert selected_dependency.VALUE == 'selected generation'
                if fault == 'import':
                    raise RuntimeError('application import failed before build')
                spec = importlib.machinery.PathFinder.find_spec(fullname, path)
                original = spec.loader.exec_module
                def execute(module):
                    original(module)
                    # Install only service/machine boundaries after REAL CLI
                    # startup; all dependencies must be available by now.
                    from hermes_cli import update_cmd as update
                    from hermes_cli import update_cmd_maint as maint
                    from hermes_cli import update_cmd_fleet as fleet
                    from hermes_cli import gateway_migrate
                    from hermes_cli import macos_tcc_anchor
                    from hermes_cli import source_build
                    from hermes_cli.update_inventory import UpdatePlan, RuntimeRecord
                    from pm.package import Runner
                    import pm
                    import os
                    import shutil

                    def provision(name, *, base_env, explicit):
                        assert name == 'npm' and explicit
                        assert shutil.which('node', path=base_env['PATH'])
                        assert shutil.which('npm', path=base_env['PATH'])
                        assert base_env['HERMES_PYTHON'] == sys.executable
                        assert base_env['PYTHON'] == sys.executable
                        (root / 'build-environment.json').write_text(json.dumps({
                            'python': sys.executable, 'selected': selected_dependency.__file__,
                            'argv': sys.argv, 'old_module': 'pre_pm_only' in sys.modules,
                            'pid': os.getpid(), 'parent': os.getppid(),
                        }))
                        return Runner(name, base_env)
                    pm.ensure = provision
                    web = source_build.build_source_web
                    source_build.build_source_web = lambda root, **kwargs: web(root, icons=root, **kwargs)
                    # Ancillary system changes are not the acceptance target.
                    macos_tcc_anchor.ensure_tcc_anchor = lambda: None
                    maint._print_post_update_notices_and_self_heals = lambda: None
                    maint._sync_profiles_after_update = lambda: None
                    maint._print_bundled_skills_sync_report = lambda: None
                    fleet._print_legacy_units_warning = lambda: None
                    maint._refresh_dashboard_after_update = lambda **kwargs: None
                    gateway_migrate.maybe_auto_migrate_after_update = lambda: None
                    update._surviving_pre_update_serve_runtimes = lambda plan: []
                    fleet._collect_fleet_snapshot = lambda *args: []
                    def restart(plan, gateway_mode):
                        assert isinstance(plan, UpdatePlan)
                        assert isinstance(plan.runtimes[0], RuntimeRecord)
                        assert plan.to_dict() == request['plan'] | {
                            'updatable_in_place': True, 'update_mechanism': 'hermes update'}
                        (root / 'restarted-plan.json').write_text(json.dumps(plan.to_dict()))
                        return fleet._GatewayRestartOutcome(
                            incomplete=False, phase_errors=[], pre_restart_gateway_pids=[],
                            restarted_services=[], failed_or_stale_units=[], relaunched_profiles=[],
                            externally_supervised_profiles=[], killed_pids=set())
                    update._restart_gateway_fleet_after_update = restart
                    merge = update._resume_windows_gateways_and_merge_outcome
                    def resume(outcome, token, gateway_mode):
                        assert token == request['windows_resume']
                        (root / 'resumed-token.json').write_text(json.dumps(token))
                        return merge(outcome, token, gateway_mode)
                    update._resume_windows_gateways_and_merge_outcome = resume
                spec.loader.exec_module = execute
                return spec
        sys.meta_path.insert(0, CompletionImports())
        sys.argv = [str(root / 'hermes_cli/update_finish.py'), context, result]
        runpy.run_path(sys.argv[0], run_name='__main__')
    '''))
    def run(fault=""):
        return subprocess.run([str(python), "-I", "-B", str(runner), str(context), str(result), fault],
                              cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90)
    return source, home / ".hermes", request, context, result, run


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("fault", ["import", "activation", "plan", "receipt"])
def test_failure_preserves_original_receipt_before_build(completion, fault):
    source, home, request, context, result, run = completion
    request["gateway_mode"] = True
    context.write_text(json.dumps(request), encoding="utf-8")
    message = "application import failed before build"
    if fault == "activation":
        from hermes_cli.runtime_paths import selected_venv, site_packages

        shutil.rmtree(site_packages(selected_venv(source)))
        message = "dependency environment has no site-packages"
    elif fault == "plan":
        request["plan"]["runtimes"] = [{"profile": "missing required kind"}]
        context.write_text(json.dumps(request), encoding="utf-8")
        message = "kind"
    elif fault == "receipt":
        message = "cannot accept prepared receipt"
    before = context.read_bytes()
    child = run(fault)
    assert child.returncode == 1, child.stdout + child.stderr
    assert message in child.stderr
    receipt = json.loads((home / "logs/update_receipts/latest.json").read_text())
    assert receipt["update_id"] == request["update_id"]
    assert receipt["started_at"] == request["receipt"]["started_at"]
    assert receipt["pre_update"] == request["receipt"]["pre_update"]
    assert receipt["steps"][0] == request["receipt"]["steps"][0]
    if fault != "receipt":
        assert receipt["pm_venv_rebuild"] == request["pm_receipt"]["venv_rebuild"]
    assert receipt["outcome"] == "failed"
    assert receipt["exit_code"] == child.returncode
    assert json.loads(result.read_text())["receipt_handled"] is True
    assert (home / ".update_exit_code").read_text().strip() == "1"
    assert context.read_bytes() == before
    assert not (source / "build-environment.json").exists()


def _npm_graph(source):
    # Use the same prepared compilers as the JS builder acceptance suite,
    # while npm itself installs a tiny, entirely local workspace graph.
    tools = {}
    node = shutil.which("node")
    assert node, "source completion acceptance requires Node and npm"
    for name in ("esbuild", "typescript", "vite"):
        probe = subprocess.run([node, "-p", f"require.resolve('{name}/package.json')"],
                               cwd=ROOT, capture_output=True, text=True, check=True)
        tools[name] = str(Path(probe.stdout.strip()).parent)
    root = {"name": "completion-graph", "version": "1.0.0", "private": True, "type": "module",
            "workspaces": ["ui-tui", "web", "packages/value"],
            "scripts": {"postinstall": "node prepare-tools.mjs"}}
    packages = {"": root}
    for directory, name in (("ui-tui", "fixture-tui"), ("web", "fixture-web"),
                            ("packages/value", "fixture-value")):
        manifest = {"name": name, "version": "1.0.0", "type": "module"}
        if name == "fixture-value":
            manifest["exports"] = "./index.js"
        else:
            manifest["dependencies"] = {"fixture-value": "*"}
        _put(source, f"{directory}/package.json", json.dumps(manifest))
        packages[directory] = manifest
        packages[f"node_modules/{name}"] = {"resolved": directory, "link": True}
    _put(source, "package.json", json.dumps(root))
    _put(source, "package-lock.json", json.dumps({"name": root["name"], "version": "1.0.0",
                                                 "lockfileVersion": 3, "packages": packages}))
    _put(source, "prepare-tools.mjs", textwrap.dedent(f'''
        import {{ mkdirSync, symlinkSync, writeFileSync }} from 'node:fs';
        import {{ execFileSync }} from 'node:child_process';
        const tools = {json.dumps(tools)};
        for (const [name, target] of Object.entries(tools)) {{
          const workspace = name === 'esbuild' ? 'ui-tui' : 'web';
          mkdirSync(workspace + '/node_modules', {{recursive: true}});
          symlinkSync(target, workspace + '/node_modules/' + name, 'dir');
        }}
        // A real npm lifecycle child, not an assertion about a constructed env.
        writeFileSync('npm-python.json', execFileSync(process.env.HERMES_PYTHON,
          ['-I', '-c', 'import json, os, sys; print(json.dumps(dict(python=sys.executable, configured=os.environ["PYTHON"])))']));
    '''))
    _put(source, "packages/value/index.js", "export const value = 'compiled local graph';\n")
    _put(source, "packages/value/index.d.ts", "export const value: string;\n")
    _put(source, "ui-tui/src/entry.tsx", "import { value } from 'fixture-value'; console.log(value);\n")
    _put(source, "web/src/main.ts", "import { value } from 'fixture-value'; document.body.textContent = value;\n")
    _put(source, "web/index.html", '<html><body><script type="module" src="/src/main.ts"></script></body></html>')
    _put(source, "web/tsconfig.json", json.dumps({"compilerOptions": {
        "target": "ES2022", "module": "ESNext", "moduleResolution": "bundler",
        "types": [], "noEmit": True}, "include": ["src"]}))
    _put(source, "web/vite.config.ts", "export default {};\n")
    _put(source, "web/public/favicon.ico", "prepared icon input")
    return node


@pytest.mark.platforms("posix")
def test_selected_child_builds_and_finalizes_under_parent_lock(completion):
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    from hermes_cli.update_lock import UpdateLock
    import hermes_yaml

    source, home, request, context, result, run = completion
    node = _npm_graph(source)
    before = context.read_bytes()
    with UpdateLock() as lock:
        assert lock.acquired
        marker = lock.path.read_bytes()
        child = run()
        assert child.returncode == 0, child.stdout + child.stderr
        assert lock.path.read_bytes() == marker, "completion must not release its parent's lock"
    built = subprocess.run([node, str(source / "ui-tui/dist/entry.js")],
                           capture_output=True, text=True, check=True)
    assert built.stdout.strip() == "compiled local graph"
    assert (source / "hermes_cli/web_dist/index.html").is_file()
    assert any("compiled local graph" in p.read_text() for p in (source / "hermes_cli/web_dist/assets").glob("*.js"))
    assert (source / "node_modules/.hermes-node-deps").is_file()
    environment = json.loads((source / "build-environment.json").read_text())
    assert "repairing the recorded dependency environment" not in child.stderr
    npm_python = json.loads((source / "npm-python.json").read_text())
    assert npm_python["python"] == npm_python["configured"] == environment["python"]
    assert environment["parent"] == os.getpid()
    assert environment["old_module"] is False
    assert environment["argv"] == [str(source / "hermes"), "update"]
    assert context.read_bytes() == before
    assert json.loads((source / "restarted-plan.json").read_text())["expected_sha"] == "old-sha"
    assert json.loads((source / "resumed-token.json").read_text()) == request["windows_resume"]
    config = hermes_yaml.safe_load((home / "config.yaml").read_text())
    assert config["_config_version"] == DEFAULT_CONFIG["_config_version"], child.stdout + child.stderr
    assert config["model"]["default"] == "retained-model"
    receipt = json.loads((home / "logs/update_receipts/latest.json").read_text())
    assert receipt["outcome"] == "success"
    assert receipt["update_id"] == request["update_id"]
    assert receipt["pre_update"] == request["receipt"]["pre_update"]
    assert receipt["steps"][0] == request["receipt"]["steps"][0]
    assert receipt["pm_venv_rebuild"] == request["pm_receipt"]["venv_rebuild"]
    assert json.loads(result.read_text()) == {"resume_handled": True, "receipt_handled": True}


@pytest.mark.platforms("posix")
def test_real_compiler_failure_retains_receipt_and_skips_completion(completion):
    source, home, request, context, result, run = completion
    _npm_graph(source)
    # A real TypeScript error: dependency installation and TUI succeed, but
    # neither the web product nor config/fleet completion may claim success.
    _put(source, "web/src/main.ts", "const broken: string = 42; document.body.textContent = broken;\n")
    config_before = (home / "config.yaml").read_bytes()
    context_before = context.read_bytes()
    child = run()
    assert child.returncode == 1, child.stdout + child.stderr
    assert "TypeScript build failed" in child.stderr
    assert (source / "npm-python.json").is_file()
    assert (source / "ui-tui/dist/entry.js").is_file()
    assert not (source / "hermes_cli/web_dist/index.html").exists()
    assert not (source / "restarted-plan.json").exists()
    assert (home / "config.yaml").read_bytes() == config_before
    assert context.read_bytes() == context_before
    receipt = json.loads((home / "logs/update_receipts/latest.json").read_text())
    assert receipt["outcome"] == "failed"
    assert receipt["exit_code"] == child.returncode
    assert receipt["update_id"] == request["update_id"]
    assert receipt["steps"][0] == request["receipt"]["steps"][0]
    assert receipt["pm_venv_rebuild"] == request["pm_receipt"]["venv_rebuild"]
    assert json.loads(result.read_text()) == {"resume_handled": True, "receipt_handled": True}


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("argument, code", [("--help", 0), ("--no-such-update-option", 2)])
def test_pre_pull_restart_enters_current_cli_with_original_arguments(completion, argument, code):
    source, home, request, context, result, run = completion
    # Exercise the REAL parser without contacting a repository or supervisor.
    # The full update command's apply path is covered by its own tests.
    request.update(restart_update=True, argv=[str(source / "hermes"), "update", argument])
    context.write_text(json.dumps(request), encoding="utf-8")
    child = run()
    assert child.returncode == code, child.stdout + child.stderr
    output = child.stdout + child.stderr
    assert "usage:" in output and "update" in output
    assert not (source / "build-environment.json").exists()
    assert json.loads(result.read_text()) == {"resume_handled": True, "receipt_handled": True}