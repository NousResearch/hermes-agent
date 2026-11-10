"""The three-part hosted code-provenance guarantee, as behaviour.

(A) PRIMARY, the loader census: every Hermes code path that loads Python from a path
consults the ONE predicate (:func:`hermes_cli.hosted_code_fence.hosted_code_refusal`)
before it loads — the structural census test (``test_hosted_loader_census.py``) keeps
the repository honest; the tests here prove the GUARDS themselves refuse, in a process
whose verdict is frozen hosted but whose audit hook is NOT installed (the hook is
defence in depth, so each loader's own refusal is exactly what these tests observe).

(B) INTERPRETER, isolated startup: the hosted start refuses to arm — and exits
non-zero — unless the interpreter started isolated from user and data site paths:
a user-site-enabled interpreter (a ``PYTHONUSERBASE`` in tenant-writable storage
executes its ``.pth`` and ``usercustomize`` before any Hermes code runs), a
``usercustomize``/``sitecustomize`` defined outside the image, or an outside-defined
finder or path hook present at arm time each refuse the arm.

(C) DEFENCE IN DEPTH, the audit hook: the five standard-library loader calls a review
drove through the armed fence — sourceless bytecode under a ``.bin`` name, an
``import_module`` of a zip put on ``sys.path``, a direct ``zipimporter.load_module``,
and ``runpy.run_path`` of bytecode or a zip — execute. That is the documented
boundary, not a bug to fix: they are raw in-process standard-library calls no audit
event names the real file for, and the guarantee against them is (A) — the census
found no Hermes loader that can be driven to any of them with a data-directory path —
plus (B). If a Hermes loader ever can be, the census forces its predicate guard and
the refusal test belongs with the guards above.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import py_compile
import runpy
import subprocess
import sys
import types
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import hosted_mode
from hermes_cli.hosted_code_fence import (
    HOSTED_BOOT_REFUSED_EXIT,
    HostedCodeRefused,
    hosted_fence_installed,
    hosted_code_refusal,
    install_hosted_code_fence,
    startup_isolation_refusal,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def hosted_verdict(monkeypatch):
    """A frozen hosted verdict with NO audit hook installed in this process (the fence
    tests own the hook): what each loader guard does on its own is exactly what the
    census tests hold the repository to."""
    monkeypatch.setitem(hosted_mode._VERDICT, "hosted", True)
    monkeypatch.setattr("hermes_cli.plugin_isolation.isolation_mode", lambda *a, **k: "in_process")


@pytest.fixture
def hosted(monkeypatch):
    """A frozen hosted verdict AND the process-wide fence installed (idempotent): the
    full hosted state the boundary tests run under."""
    monkeypatch.setitem(hosted_mode._VERDICT, "hosted", True)
    monkeypatch.setattr("hermes_cli.plugin_isolation.isolation_mode", lambda *a, **k: "in_process")
    install_hosted_code_fence()
    assert hosted_fence_installed()


def _payload_dir(root: Path, name: str, marker: Path, init_name: str = "__init__.py") -> Path:
    """A plugin-like directory whose package body writes *marker* when it executes."""
    directory = root / name
    directory.mkdir(parents=True)
    (directory / init_name).write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n",
        encoding="utf-8")
    return directory


# --------------------------------------------------------------------------------------------------
# (A) Every Hermes loader consults the one predicate before it loads — and refuses


def test_a_model_provider_plugin_directory_is_refused_before_import(tmp_path, hosted_verdict, caplog):
    """The model-provider discovery imports a plugin directory in-process; in hosted
    mode the predicate refuses a directory outside the deployment image BEFORE the
    spec exists, so the plugin's package body never runs."""
    import providers

    marker = tmp_path / "executed"
    plugin_dir = _payload_dir(tmp_path, "evil_provider", marker)
    with caplog.at_level("ERROR", logger="providers"):
        providers._exec_plugin_module(plugin_dir, "user", "_census_evil_provider")
    assert not marker.exists()
    assert "_census_evil_provider" not in sys.modules
    assert "refused in hosted mode" in caplog.text and "deployment image" in caplog.text


def test_a_gateway_hook_directory_is_refused_before_its_handler_loads(tmp_path, hosted_verdict, capsys):
    """Hooks live under the deployment home — tenant-writable code. In hosted mode the
    hook registry's directory loader refuses the directory before ``handler.py`` is
    ever read, printing the skip reason, and never raises: one bad hook never blocks
    the pipeline, but a hosted process never runs it."""
    import gateway.hooks as gateway_hooks

    marker = tmp_path / "executed"
    hook_dir = tmp_path / "evilhook"
    hook_dir.mkdir()
    (hook_dir / "HOOK.yaml").write_text("name: evilhook\nevents: [gateway:startup]\n", encoding="utf-8")
    (hook_dir / "handler.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n\n\ndef handle(*a, **k):\n    pass\n",
        encoding="utf-8")
    loaded = gateway_hooks._load_hook_dir(hook_dir)
    out = capsys.readouterr().out
    assert loaded is None
    assert not marker.exists()
    assert "[hooks] Skipping evilhook:" in out and "deployment image" in out


def test_a_directory_plugin_is_refused_before_its_module_loads(tmp_path, hosted_verdict):
    """The general plugin loader's directory-module seam (the one ``hermes_plugins.<slug>``
    imports go through): an out-of-image plugin directory raises the predicate's
    refusal before a spec — or any code — exists."""
    from hermes_cli.plugins_loader import PluginLoaderMixin

    marker = tmp_path / "executed"
    plugin_dir = _payload_dir(tmp_path, "evilplug", marker)
    manifest = SimpleNamespace(path=str(plugin_dir))
    stub_self = SimpleNamespace(scope_key="test-scope",
                                _directory_module_name=lambda _m: "hermes_plugins.census_evilplug")
    with pytest.raises(HostedCodeRefused, match="deployment image"):
        PluginLoaderMixin._load_directory_module(stub_self, manifest,
                                                  module_name="hermes_plugins.census_evilplug")
    assert not marker.exists()
    assert "hermes_plugins.census_evilplug" not in sys.modules


def test_the_category_plugin_loader_choke_point_refuses_out_of_image_files(tmp_path, hosted_verdict, caplog):
    """:func:`plugins.plugin_loader._new_module` is the choke point every category
    plugin file load (memory providers, context engines, cron and computer-use providers)
    goes through; the predicate answers each file — the plugin's own, its parents, its
    siblings — before a spec exists, skipping it loudly like every other load failure."""
    from plugins import plugin_loader

    marker = tmp_path / "executed"
    module_file = tmp_path / "evil_plugin.py"
    module_file.write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n",
                           encoding="utf-8")
    with caplog.at_level("WARNING", logger="plugins.plugin_loader"):
        module = plugin_loader._new_module("census_evil_plugin", module_file)
    assert module is None
    assert not marker.exists()
    assert "census_evil_plugin" not in sys.modules
    assert "Refusing to load" in caplog.text and "deployment image" in caplog.text


def test_a_memory_provider_cli_module_is_refused(tmp_path, monkeypatch, hosted_verdict, caplog):
    """The memory provider's ``cli.py`` (a user plugin's own code): in hosted mode the
    CLI command discovery refuses the provider directory before the file is read, so
    the surface is simply absent."""
    import plugins.memory as memory_plugins

    marker = tmp_path / "executed"
    plugin_dir = tmp_path / "evilprov"
    plugin_dir.mkdir()
    (plugin_dir / "cli.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n\n\ndef register_cli(subparsers):\n    pass\n",
        encoding="utf-8")
    monkeypatch.setattr(memory_plugins, "_get_active_memory_provider", lambda: "evilprov")
    monkeypatch.setattr(memory_plugins, "find_provider_dir", lambda _n: plugin_dir)
    with caplog.at_level("ERROR", logger="plugins.memory"):
        commands = memory_plugins.discover_plugin_cli_commands()
    assert commands == []
    assert not marker.exists()
    assert "refused in hosted mode" in caplog.text
    assert not any(name.startswith("_hermes_user_memory") for name in sys.modules)


def test_a_memory_provider_config_schema_is_refused(tmp_path, monkeypatch, hosted_verdict, caplog):
    """The memory provider's ``config_schema.py``: the settings panel loader refuses an
    out-of-image schema file in hosted mode before it is read — the panel simply has
    no schema."""
    import plugins.memory as memory_plugins
    from plugins.memory import config_schema

    marker = tmp_path / "executed"
    plugin_dir = tmp_path / "evilprov"
    plugin_dir.mkdir()
    (plugin_dir / "config_schema.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\nCONFIG_SCHEMA = None\n",
        encoding="utf-8")
    monkeypatch.setattr(memory_plugins, "find_provider_dir", lambda _n: plugin_dir)
    monkeypatch.setattr("hermes_cli.plugin_isolation.user_plugin_host", lambda: None)
    with caplog.at_level("ERROR", logger="plugins.memory.config_schema"):
        schema = config_schema.get_provider_config_schema("evilprov")
    assert schema is None
    assert not marker.exists()
    assert "refused in hosted mode" in caplog.text


def test_a_dashboard_plugin_api_file_is_refused(tmp_path, monkeypatch, hosted_verdict, caplog):
    """A dashboard plugin's backend ``api.py``: even with the plugin enabled in config,
    a hosted process refuses an out-of-image api file before it is read — the dashboard
    of a hosted deployment mounts only the plugins its image bundles."""
    from hermes_cli import web_server_dashboard

    marker = tmp_path / "executed"
    plugin_dir = tmp_path / "evilplug"
    plugin_dir.mkdir()
    (plugin_dir / "api.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\nrouter = object()\n",
        encoding="utf-8")
    app = SimpleNamespace(router=SimpleNamespace(routes=[]))
    plugin = {"name": "evilplug", "source": "user", "_dir": str(plugin_dir), "_api_file": "api.py"}
    with caplog.at_level("ERROR", logger="hermes_cli.web_server_dashboard"):
        mounted = web_server_dashboard._mount_one_plugin_api(app, plugin, {"evilplug"}, set())
    assert mounted is False
    assert not marker.exists()
    assert "refused in hosted mode" in caplog.text


def test_the_plugin_host_child_refuses_paths_from_the_wire(tmp_path, hosted_verdict):
    """The plugin host child loads whatever directory, api file, schema file or base
    class the parent process sends over the wire; the wire is not provenance, so in a
    hosted host process the predicate answers each of them before any code runs."""
    import hermes_cli.plugin_host_child as child

    marker = tmp_path / "executed"
    plugin_dir = _payload_dir(tmp_path, "evilhostplug", marker)
    with pytest.raises(HostedCodeRefused, match="deployment image"):
        child._import_plugin({"path": str(plugin_dir), "module_name": "census_evil_hostplug"})

    api_dir = tmp_path / "dashboard"
    api_dir.mkdir()
    (api_dir / "api.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n", encoding="utf-8")
    with pytest.raises(HostedCodeRefused, match="deployment image"):
        child.HostRuntime.op_asgi(SimpleNamespace(asgi_apps={}),
                                  {"dashboard_dir": str(api_dir), "api_file": "api.py", "plugin": "x"})

    schema_dir = tmp_path / "schemadir"
    schema_dir.mkdir()
    (schema_dir / "config_schema.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n", encoding="utf-8")
    with pytest.raises(HostedCodeRefused, match="deployment image"):
        child.HostRuntime.op_config_schema(object(), {"path": str(schema_dir / "config_schema.py")})

    with pytest.raises(HostedCodeRefused, match="deployment image"):
        child._import_base("census_evil_base:Thing")
    assert not marker.exists()


def test_pm_package_definitions_from_the_wire_are_refused(tmp_path, hosted_verdict):
    """The package-manager worker restores package definitions a client process
    recorded; a definition names a module and a file, and in hosted mode the predicate
    answers both before anything imports them."""
    from pm import registry as pm_registry

    marker = tmp_path / "executed"
    payload = tmp_path / "census_evil_pm.py"
    payload.write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n",
                       encoding="utf-8")
    with pytest.raises(HostedCodeRefused, match="deployment image"):
        pm_registry.load_package_definitions(
            [{"name": "x", "module": "census_evil_pm", "qualname": "C", "path": str(payload)}])
    assert not marker.exists()
    assert "census_evil_pm" not in sys.modules


@pytest.mark.parametrize("flow", ["prepare", "takeover_main", "finish", "serve_resume"])
def test_the_update_flow_refuses_a_foreign_checkout_root(tmp_path, monkeypatch, hosted_verdict, flow,
                                                         capsys):
    """The update takeover pulls a NEW checkout root onto the import path. A hosted
    process never swaps its image in-process: the predicate refuses the root before
    the insert, whatever stage of the update flow reaches for it."""
    root = tmp_path / "new-checkout"
    root.mkdir()
    context = tmp_path / "context.json"
    context.write_text(json.dumps({"root": str(root)}), encoding="utf-8")

    def assert_root_never_on_path():
        assert str(root) not in sys.path

    if flow == "prepare":
        from hermes_cli import _update_takeover

        with pytest.raises(HostedCodeRefused, match="deployment image"):
            _update_takeover.prepare({"root": str(root)})
        assert_root_never_on_path()
    elif flow == "takeover_main":
        from hermes_cli import _update_takeover

        monkeypatch.setattr(sys, "argv", ["prog", str(context), str(tmp_path / "result.json")])
        with pytest.raises(HostedCodeRefused, match="deployment image"):
            _update_takeover.main()
        assert_root_never_on_path()
    elif flow == "finish":
        from hermes_cli import update_finish

        with pytest.raises(HostedCodeRefused, match="deployment image"):
            update_finish.main(context, tmp_path / "result.json")
        assert_root_never_on_path()
    else:
        from hermes_cli import update_serve_resume

        context.write_text(
            json.dumps({"root": str(root), "stopped_serves": {"pending": True, "entries": []}}),
            encoding="utf-8")
        code = update_serve_resume.main(context, tmp_path / "result.json")
        captured = capsys.readouterr().err
        assert code == 1  # the recovery refuses cleanly instead of relaunching anything
        assert "deployment image" in captured
        assert_root_never_on_path()


def test_the_guards_are_dormant_on_a_local_install(tmp_path, monkeypatch, capsys):
    """The same loaders, with the verdict NOT frozen hosted (a local install): the
    predicate answers nothing, user plugins load exactly as before, and no fence
    exists. The positive controls for every guard above."""
    import gateway.hooks as gateway_hooks

    assert hosted_code_refusal() == ""  # the predicate is dormant
    marker = tmp_path / "executed"
    hook_dir = tmp_path / "goodhook"
    hook_dir.mkdir()
    (hook_dir / "HOOK.yaml").write_text("name: goodhook\nevents: [gateway:startup]\n", encoding="utf-8")
    (hook_dir / "handler.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n\n\ndef handle(*a, **k):\n    pass\n",
        encoding="utf-8")
    loaded = gateway_hooks._load_hook_dir(hook_dir)
    assert loaded is not None and loaded[0] == "goodhook" and marker.exists()


# --------------------------------------------------------------------------------------------------
# (B) The hosted start refuses to arm on an interpreter that did not start isolated


def test_this_isolated_interpreter_has_no_startup_isolation_refusal():
    """The test environment's own interpreter (an isolated venv: user site disabled, no
    sitecustomize, image-defined finders) passes the isolation check — the same
    guarantee a deployment image's interpreter must satisfy."""
    assert startup_isolation_refusal() == ""


def test_user_site_enabled_is_a_refusal(monkeypatch):
    """An interpreter that started with user-site loading enabled (no ``-I``/``-s``, no
    isolated venv): a ``PYTHONUSERBASE`` in tenant-writable storage would execute its
    ``.pth`` files and ``usercustomize`` before the fence arms, so the start refuses."""
    import site

    monkeypatch.setattr(site, "ENABLE_USER_SITE", True)
    refusal = startup_isolation_refusal()
    assert "user-site" in refusal and "PYTHONUSERBASE" in refusal


def test_a_usercustomize_defined_outside_the_image_is_a_refusal(tmp_path, monkeypatch):
    """A ``usercustomize`` module this interpreter already loaded, defined by a file
    outside the image roots: site initialization that already ran cannot be trusted,
    so the start refuses rather than arm behind it."""
    file = tmp_path / "usercustomize.py"
    file.write_text("# site initialization outside the image\n", encoding="utf-8")
    module = types.ModuleType("usercustomize")
    module.__file__ = str(file)
    monkeypatch.setitem(sys.modules, "usercustomize", module)
    refusal = startup_isolation_refusal()
    assert "usercustomize" in refusal and "outside the deployment image" in refusal


def test_an_outside_defined_finder_at_arm_time_is_a_refusal(tmp_path, monkeypatch):
    """A meta-path finder (or path hook) whose defining module's file lies outside the
    roots at arm time: pruning removes search entries, but a finder such a module
    installed keeps consulting whatever it wants — refused rather than trusted."""
    finder_file = tmp_path / "outside_finder.py"
    finder_file.write_text("class OutsideFinder:\n    pass\n", encoding="utf-8")
    spec = importlib.util.spec_from_file_location("census_outside_finder", finder_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules["census_outside_finder"] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "meta_path", [*sys.meta_path, module.OutsideFinder()])
    refusal = startup_isolation_refusal()
    assert "finder or path hook" in refusal and "outside the deployment image" in refusal
    monkeypatch.setattr(sys, "path_hooks", [*sys.path_hooks, module.OutsideFinder()])
    assert "finder or path hook" in startup_isolation_refusal()


def test_the_arm_exits_non_zero_on_a_non_isolated_interpreter(monkeypatch):
    """The arm itself (the first statement of the gateway's entry modules): with the
    startup-isolation check failing, :func:`arm_from_start_argv` exits the process with
    the fatal-config exit code and installs nothing."""
    import site

    from hermes_cli import hosted_code_fence

    monkeypatch.setattr(site, "ENABLE_USER_SITE", True)
    monkeypatch.setattr(sys, "argv", [sys.argv[0], "--hosted"])
    monkeypatch.setattr(hosted_code_fence, "_installed", False)
    with pytest.raises(SystemExit) as excinfo:
        hosted_code_fence.arm_from_start_argv()
    assert excinfo.value.code == HOSTED_BOOT_REFUSED_EXIT == 78
    assert not hosted_code_fence.hosted_fence_installed()
    assert not hosted_code_fence.armed_by_start_flag()


def _arm_subprocess(tmp_path, script: str, *, env_extra: dict | None = None):
    """A fresh interpreter running *script* (column-0 lines, no dedenting involved) with
    the repo importable and a scratch home — the closest a unit test gets to a real
    hosted start."""
    home = tmp_path / "arm-home"
    home.mkdir(parents=True, exist_ok=True)
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HERMES_HOME": str(home),
           "HERMES_RUNTIME_DIR": str(tmp_path / "run"), "UV_OFFLINE": "1",
           "PYTHONPATH": str(REPO_ROOT), **(env_extra or {})}
    return subprocess.run([sys.executable, "-c", script],
                           cwd=str(REPO_ROOT), env=env, capture_output=True, text=True,
                           check=False, timeout=180)


_ARM_TAIL = (
    "import sys\n"
    "sys.argv = [sys.argv[0], '--hosted']\n"
    "from hermes_cli.hosted_code_fence import arm_from_start_argv, hosted_fence_installed\n"
    "print('ARMED', arm_from_start_argv(), hosted_fence_installed())\n")


def test_a_real_arm_refuses_a_user_site_enabled_interpreter(tmp_path):
    """End to end: a fresh interpreter with user-site loading enabled plants the
    ``--hosted`` flag, and the arm exits 78 without installing the fence."""
    result = _arm_subprocess(tmp_path,
                             "import site\n"
                             "site.ENABLE_USER_SITE = True  # a base interpreter without -I/-s\n"
                             + _ARM_TAIL)
    assert result.returncode == HOSTED_BOOT_REFUSED_EXIT
    assert "Hosted start refused" in result.stderr and "user-site" in result.stderr
    assert "ARMED" not in result.stdout


def test_a_real_arm_refuses_a_planted_usercustomize(tmp_path):
    """End to end: a fresh interpreter carrying a ``usercustomize`` whose file lies
    outside the image refuses the arm — whatever that module already did, a hosted
    start will not serve behind it."""
    usercustomize = tmp_path / "data" / "usercustomize.py"
    usercustomize.parent.mkdir()
    usercustomize.write_text("# tenant-writable site initialization\n", encoding="utf-8")
    result = _arm_subprocess(
        tmp_path,
        "import sys, types\n"
        f"module = types.ModuleType('usercustomize')\n"
        f"module.__file__ = {str(usercustomize)!r}\n"
        "sys.modules['usercustomize'] = module\n"
        + _ARM_TAIL)
    assert result.returncode == HOSTED_BOOT_REFUSED_EXIT
    assert "usercustomize" in result.stderr and "outside the deployment image" in result.stderr
    assert "ARMED" not in result.stdout


def test_a_real_arm_refuses_an_outside_finder_at_arm_time(tmp_path):
    """End to end: a fresh interpreter with a meta-path finder defined outside the
    image refuses the arm instead of installing the fence behind it."""
    finder_file = tmp_path / "outside_finder.py"
    finder_file.write_text("class OutsideFinder:\n    pass\n", encoding="utf-8")
    result = _arm_subprocess(
        tmp_path,
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('census_outside_finder', {str(finder_file)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "sys.modules['census_outside_finder'] = module\n"
        "spec.loader.exec_module(module)\n"
        "sys.meta_path.append(module.OutsideFinder())\n"
        + _ARM_TAIL)
    assert result.returncode == HOSTED_BOOT_REFUSED_EXIT
    assert "finder or path hook" in result.stderr
    assert "ARMED" not in result.stdout


def test_an_isolated_interpreter_arms_despite_a_planted_pythonuserbase(tmp_path):
    """The deployment prerequisite, working: with the isolated venv interpreter (user
    site disabled), a ``PYTHONUSERBASE`` pointing into tenant-writable storage changes
    nothing — its ``.pth`` and ``usercustomize`` never run, the payload marker stays
    absent, and the arm proceeds. The non-isolated interpreter is the refusal case
    above; this is why the image entrypoint must pass ``-I`` or ``-s`` plus
    ``PYTHONNOUSERSITE``."""
    userbase = tmp_path / "userbase"
    env = {"PYTHONUSERBASE": str(userbase)}
    probe = _arm_subprocess(tmp_path, "import site; print(site.getusersitepackages())", env_extra=env)
    usersite = Path(probe.stdout.strip())
    usersite.mkdir(parents=True)
    marker = tmp_path / "executed"
    (usersite / "usercustomize.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n", encoding="utf-8")
    (usersite / "evil.pth").write_text("import usercustomize\n", encoding="utf-8")
    result = _arm_subprocess(tmp_path, _ARM_TAIL, env_extra=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ARMED True True" in result.stdout
    assert not marker.exists()  # user site is disabled: the .pth and usercustomize never ran


# --------------------------------------------------------------------------------------------------
# (C) The audit hook is defence in depth — its documented boundary, pinned


def _bytecode_payload(tmp_path: Path, marker: Path, name: str) -> Path:
    """A valid ``.pyc`` (compiled from source that writes *marker*), returned as a file
    the fence's suffix-gated read check does not recognize."""
    source = tmp_path / f"{name}.py"
    source.write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n",
                      encoding="utf-8")
    bytecode = tmp_path / f"{name}.pyc"
    py_compile.compile(str(source), cfile=str(bytecode))
    return bytecode


@pytest.fixture(scope="module")
def boundary_payloads(tmp_path_factory):
    """The five payloads, built ONCE and BEFORE the fence fixture ever arms (module
    scope instantiates before the first requesting test's function-scoped fixtures, and
    the first boundary test is what installs the process-wide — irremovable — hook):
    compiling the payload source and copying its bytecode are themselves code loads
    the armed hook would refuse, so every file the boundary tests touch exists first."""
    tmp_path = tmp_path_factory.mktemp("stdlib_boundary_payloads")
    marker = tmp_path / "executed"
    bytecode = _bytecode_payload(tmp_path, marker, "census_evil_payload")
    renamed = tmp_path / "bytecode.bin"
    renamed.write_bytes(bytecode.read_bytes())
    evil_zip = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil_zip, "w") as zf:
        zf.write(bytecode, "census_evil_zip.pyc")
    direct_zip = tmp_path / "evil_direct.zip"
    with zipfile.ZipFile(direct_zip, "w") as zf:
        zf.write(bytecode, "census_evil_direct.pyc")
    run_zip = tmp_path / "run.zip"
    with zipfile.ZipFile(run_zip, "w") as zf:
        zf.write(bytecode, "__main__.pyc")
    return SimpleNamespace(marker=marker, bytecode=bytecode, renamed=renamed,
                           evil_zip=evil_zip, direct_zip=direct_zip, run_zip=run_zip)


def _boundary_sourceless_bin(payloads) -> None:
    """A direct SourcelessFileLoader on bytecode under a ``.bin`` name."""
    from importlib.machinery import SourcelessFileLoader

    spec = importlib.util.spec_from_loader(
        "census_evil_bin", SourcelessFileLoader("census_evil_bin", str(payloads.renamed)))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def _boundary_zip_importlib(payloads) -> None:
    """``importlib.import_module`` of a module in a zip added to ``sys.path``."""
    sys.path.insert(0, str(payloads.evil_zip))
    importlib.import_module("census_evil_zip")


def _boundary_zip_direct(payloads) -> None:
    """A direct ``zipimport.zipimporter(...).load_module``."""
    import zipimport

    zipimport.zipimporter(str(payloads.direct_zip)).load_module("census_evil_direct")


def _boundary_runpy_bin(payloads) -> None:
    """``runpy.run_path`` of bytecode under a ``.bin`` name."""
    runpy.run_path(str(payloads.renamed), run_name="__main__")


def _boundary_runpy_zip(payloads) -> None:
    """``runpy.run_path`` of a zip archive holding ``__main__.pyc``."""
    runpy.run_path(str(payloads.run_zip), run_name="__main__")


_BOUNDARY_CASES = {
    "sourceless_bin": _boundary_sourceless_bin,
    "zip_importlib": _boundary_zip_importlib,
    "zip_direct": _boundary_zip_direct,
    "runpy_bin": _boundary_runpy_bin,
    "runpy_zip": _boundary_runpy_zip,
}


@pytest.mark.parametrize("case", sorted(_BOUNDARY_CASES))
def test_in_process_stdlib_loader_calls_are_the_documented_hook_boundary(boundary_payloads,
                                                                        hosted, case):
    """The five loader calls a review drove through the armed fence execute — pinned
    here as the documented boundary of the defence-in-depth hook, not a regression:
    they are raw in-process standard-library calls that raise no audit event naming
    the real file (no compile for bytecode, no open for a ``.pyc`` under a ``.bin``
    name, no import event for ``importlib.import_module``, the archive itself not a
    bytecode file). No Hermes loader can be driven to any of them with a
    data-directory path — the census test holds that line — and the isolated-startup
    check keeps tenant site directories from ever reaching the interpreter."""
    try:
        _BOUNDARY_CASES[case](boundary_payloads)
        assert boundary_payloads.marker.exists(), (
            "the documented boundary moved — update the fence docstring and these tests")
    finally:
        for module_name in ("census_evil_bin", "census_evil_zip", "census_evil_direct"):
            sys.modules.pop(module_name, None)
        for entry in (str(boundary_payloads.evil_zip), str(boundary_payloads.direct_zip)):
            while entry in sys.path:
                sys.path.remove(entry)


def test_the_fence_docstring_describes_the_three_part_guarantee():
    """The contract docstring must say what the guarantee actually is: the census is
    primary, the start must be isolated, and the hook is defence in depth with its
    known limits named — so a reader never mistakes layer three for the whole story.
    (Whitespace is normalized first: the phrases must survive any rewrapping.)"""
    from hermes_cli import hosted_code_fence

    doc = " ".join((hosted_code_fence.__doc__ or "").split())
    assert "(A) PRIMARY, the loader census" in doc
    assert "(B) INTERPRETER, isolated startup" in doc
    assert "(C) DEFENCE IN DEPTH, the audit hook" in doc
    assert "PYTHONNOUSERSITE" in doc  # the deployment prerequisite, named
    assert "zip bytecode raises no" in doc  # known limit: no compile or open event for the member
    assert "non-``.pyc``/``.pyo`` name" in doc  # known limit: the suffix-gated read check
    assert "in-process calls to the standard-library loaders" in doc  # known limit: no event at all
