"""Prepared inputs become runnable products without acquiring dependencies."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def inputs_fixture(tmp_path):
    source = tmp_path / "prepared source"
    source.mkdir()
    (source / "pyproject.toml").write_text(
        '[project]\nname="assembly-fixture"\nversion="1.2.3"\n'
        '[project.scripts]\nprobe="entry:main"\n', encoding="utf-8")
    (source / "entry.py").write_text(
        "import json, os, sys\nfrom pathlib import Path\nimport dependency\n"
        "from importlib.metadata import version\n"
        "def main():\n print(json.dumps([sys.argv[1:], dependency.VALUE, "
        "version('assembly-fixture'), (Path(__file__).parent/'skills/data').read_text()]))\n"
        " return 7\n", encoding="utf-8")
    (source / "skills").mkdir()
    (source / "skills/data").write_text("resource", encoding="utf-8")
    out = tmp_path / "payload"
    python = out / "tools/python/bin/python3"
    python.parent.mkdir(parents=True)
    shutil.copy2(Path(sys._base_executable).resolve(), python)
    site = out / "venv/lib/python3.14/site-packages"
    site.mkdir(parents=True)
    (site / "dependency.py").write_text("VALUE = 'prepared'\n", encoding="utf-8")
    (site / "entry.py").write_text("raise RuntimeError('dependency shadowed source')\n", encoding="utf-8")
    pm = out / "pm-runtime"
    (pm / "lib/site-packages").mkdir(parents=True)
    (pm / "pm-runtime.json").write_text(json.dumps({
        "python": "../tools/python/bin/python3", "sitePackages": "lib/site-packages"}), encoding="utf-8")
    tui, web = tmp_path / "tui", tmp_path / "web"
    (tui / "dist").mkdir(parents=True)
    web.mkdir()
    (tui / "dist/entry.js").write_text("export const built = true;", encoding="utf-8")
    (tui / "package.json").write_text('{"type":"module"}', encoding="utf-8")
    (web / "index.html").write_text("built web", encoding="utf-8")
    return out, {
        "project": str(source / "pyproject.toml"), "code": str(source),
        "repo": "app", "target": "linux-x64", "placement": "contained",
        "python": str(python), "site_packages": str(site),
        "environment": str(out / "venv"), "tools": str(out / "tools"),
        "pm_runtime": str(pm), "resources": {"skills": str(source / "skills")},
        "frontends": {"tui": str(tui), "web": str(web)}, "ref": "fixture-revision",
    }


def build_cli(data, out, tmp_path):
    document = tmp_path / "inputs.json"
    document.write_text(json.dumps(data), encoding="utf-8")
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    return subprocess.run(
        [sys.executable, "-B", "-m", "scripts.build.agent", "--inputs", str(document), "--out", str(out)],
        cwd=tmp_path, env={"PATH": os.environ["PATH"], "HOME": str(home),
                          "HERMES_HOME": str(home / ".hermes"), "PYTHONPATH": str(ROOT)},
        capture_output=True, text=True, timeout=30,
    )


@pytest.mark.platforms("posix")
def test_contained_cli_assembly_runs_after_move_and_preserves_prepared_state(tmp_path):
    out, data = inputs_fixture(tmp_path)
    (out / "provider-state").write_text("retain", encoding="utf-8")
    before = {p.relative_to(data["code"]): p.read_bytes() for p in Path(data["code"]).rglob("*") if p.is_file()}
    result = build_cli(data, out, tmp_path)
    assert result.returncode == 0, result.stderr
    assert (out / "provider-state").read_text() == "retain"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["ref"] == data["ref"]
    assert manifest["runtime"] == {
        "repoDir": "app", "toolsDir": "tools", "storePython": "tools/python/bin/python3",
        "sitePackages": "venv/lib/python3.14/site-packages", "commands": {"probe": "bin/probe"}}
    assert (out / "app/hermes_cli/tui_dist/entry.js").is_file()
    assert json.loads((out / "app/hermes_cli/tui_dist/package.json").read_text())["type"] == "module"
    assert (out / "app/hermes_cli/web_dist/index.html").is_file()
    assert not (tmp_path / "home/.hermes").exists()
    moved = tmp_path / "relocated payload"
    out.rename(moved)
    (tmp_path / "entry.py").write_text("raise RuntimeError('cwd shadowed source')", encoding="utf-8")
    run = subprocess.run([str(moved / manifest["runtime"]["commands"]["probe"]), "two words", "$(no)", ""],
                         cwd=tmp_path, env={"PATH": os.environ["PATH"], "HOME": str(tmp_path / "home"),
                                            "PYTHONPATH": "/foreign", "PYTHONHOME": "/foreign"},
                         text=True, capture_output=True, timeout=30)
    assert run.returncode == 7, run.stderr
    assert json.loads(run.stdout) == [["two words", "$(no)", ""], "prepared", "1.2.3", "resource"]
    assert before == {p.relative_to(data["code"]): p.read_bytes() for p in Path(data["code"]).rglob("*") if p.is_file()}
    assert not list(moved.rglob("__pycache__"))
    assert (moved / "pm-runtime/pm-runtime.json").is_file()


@pytest.mark.platforms("posix")
def test_referenced_installed_code_uses_derived_command_map_without_copying(tmp_path):
    _, data = inputs_fixture(tmp_path)
    out = tmp_path / "nix output"
    source = Path(data["code"])
    (source / "entry.py").write_text(
        "import json, os\nfrom pathlib import Path\n"
        "def main():\n print(json.dumps([Path(os.environ['HERMES_BUNDLED_SKILLS'],'data').read_text(),"
        "os.environ['HERMES_NODE']]))\n", encoding="utf-8")
    bindir = tmp_path / "installed env/bin"
    bindir.mkdir(parents=True)
    command = bindir / "probe"
    command.write_text(f"#!{sys.executable}\nimport sys\nsys.path.insert(0, {str(source)!r})\nfrom entry import main\nmain()\n", encoding="utf-8")
    command.chmod(0o755)
    data.update(placement="references", repo="share/hermes-agent", command_dir=str(bindir),
                env={"HERMES_NODE": "/prepared/node"})
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    result = build_cli(data, out, tmp_path)
    assert result.returncode == 0, result.stderr
    mapping = json.loads((out / "command-map.json").read_text())
    assert mapping["commands"] == {"probe": {
        "source": str(command), "destination": "bin/probe", "entry": "entry:main"}}
    assert Path(mapping["env"]["HERMES_TUI_DIR"]).resolve() == Path(data["frontends"]["tui"])
    assert Path(mapping["env"]["HERMES_WEB_DIST"]).resolve() == Path(data["frontends"]["web"])
    assert mapping["env"]["HERMES_INSTALL_ROOT"] == str(out / "share/hermes-agent")
    assert not list(out.rglob("entry.py"))
    assert (out / "share/hermes-agent/skills").is_symlink()
    assert (out / "ui-tui").is_symlink()
    assert (out / "share/hermes-agent/web_dist").is_symlink()
    assert not (out / "bin/probe").exists(), "Nix owns native makeWrapper"
    run = subprocess.run([mapping["commands"]["probe"]["source"]], cwd=tmp_path,
                         env={"PATH": os.environ["PATH"], "HOME": str(tmp_path / "home"),
                              "PYTHONDONTWRITEBYTECODE": "1", **mapping["env"]},
                         text=True, capture_output=True, check=True)
    assert json.loads(run.stdout) == ["resource", "/prepared/node"]
    assert before == {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}


@pytest.mark.platforms("posix")
def test_fixed_root_keeps_privilege_shim_and_resolves_venv_command_symlink(tmp_path):
    out, data = inputs_fixture(tmp_path)
    # Docker source already occupies /opt/hermes; its bin/hermes belongs to s6.
    shutil.copytree(data["code"], out, dirs_exist_ok=True)
    data.update(placement="fixed", repo=".", code=str(out), project=str(out / "pyproject.toml"), bin_dir="libexec")
    bindir = out / "bin"
    bindir.mkdir()
    (bindir / "probe").write_text("privilege shim", encoding="utf-8")
    result = build_cli(data, out, tmp_path)
    assert result.returncode == 0, result.stderr
    assert (bindir / "probe").read_text() == "privilege shim"
    link = out / "venv/bin/probe"
    link.parent.mkdir()
    link.symlink_to("../../libexec/probe")
    run = subprocess.run([str(link), "ok"], cwd=tmp_path,
                         env={"PATH": os.environ["PATH"], "HOME": str(tmp_path / "home")},
                         text=True, capture_output=True)
    assert run.returncode == 7, run.stderr
    assert json.loads(run.stdout)[0] == ["ok"]


@pytest.mark.platforms("posix")
def test_source_metadata_keeps_declared_requirements_extras_and_entrypoints(tmp_path):
    from importlib.metadata import distributions
    from packaging.requirements import Requirement

    out, data = inputs_fixture(tmp_path)
    project = Path(data["project"])
    project.write_text(
        '[project]\nname="assembly-fixture"\nversion="1.2.3"\nrequires-python=">=3.11"\n'
        'dependencies=["dependency>=1,<2; python_version >= \'3.11\'"]\n'
        '[project.optional-dependencies]\nextra=["optional>=2; sys_platform == \'linux\'"]\n'
        '[project.scripts]\nprobe="entry:main"\n', encoding="utf-8")
    result = build_cli(data, out, tmp_path)
    assert result.returncode == 0, result.stderr
    metadata = next(distributions(path=[str(out / "app")]))
    assert metadata.metadata["Requires-Python"] == ">=3.11"
    reqs = {Requirement(r).name: Requirement(r) for r in metadata.requires}
    assert str(reqs["dependency"].specifier) == "<2,>=1"
    optional = reqs["optional"]
    assert optional.marker.evaluate({"extra": "extra", "sys_platform": "linux"})
    assert not optional.marker.evaluate({"extra": "", "sys_platform": "linux"})
    assert not optional.marker.evaluate({"extra": "extra", "sys_platform": "win32"})
    assert [(e.name, e.value) for e in metadata.entry_points] == [("probe", "entry:main")]


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("fault", ["missing-pm", "missing-resource", "missing-tui", "unknown-field", "bad-repo", "bad-target"])
def test_failed_inputs_cannot_leave_a_completion_claim(tmp_path, fault):
    out, data = inputs_fixture(tmp_path)
    (out / "manifest.json").write_text('{"runtime": {"commands": {}}}', encoding="utf-8")
    if fault == "missing-pm":
        (Path(data["pm_runtime"]) / "pm-runtime.json").unlink()
    elif fault == "missing-resource":
        data["resources"]["skills"] = str(tmp_path / "missing skills")
    elif fault == "missing-tui":
        (Path(data["frontends"]["tui"]) / "dist/entry.js").unlink()
    elif fault == "unknown-field":
        data["typo"] = True
    elif fault == "bad-repo":
        data["repo"] = "../escape"
    else:
        data["target"] = "linux-typo"
    result = build_cli(data, out, tmp_path)
    assert result.returncode != 0
    assert not (out / "manifest.json").exists()
    assert not (out / "bin/probe").exists()


@pytest.mark.platforms("posix")
def test_incremental_copy_drops_removed_source_without_deleting_provider_files(tmp_path):
    out, data = inputs_fixture(tmp_path)
    source = Path(data["code"])
    (source / "obsolete.py").write_text("old", encoding="utf-8")
    assert build_cli(data, out, tmp_path).returncode == 0
    sentinel = out / "provider-file"
    sentinel.write_text("retain", encoding="utf-8")
    (source / "obsolete.py").unlink()
    assert build_cli(data, out, tmp_path).returncode == 0
    assert not (out / "app/obsolete.py").exists()
    assert sentinel.read_text() == "retain"


@pytest.mark.platforms("posix")
def test_fixed_launcher_accepts_explicit_external_python_without_path_guessing(tmp_path):
    out, data = inputs_fixture(tmp_path)
    data.update(placement="fixed", python=sys.executable)
    result = build_cli(data, out, tmp_path)
    assert result.returncode == 0, result.stderr
    run = subprocess.run([str(out / "bin/probe")], cwd=tmp_path,
                         env={"PATH": os.environ["PATH"], "HOME": str(tmp_path / "home")},
                         capture_output=True, text=True)
    assert run.returncode == 7, run.stderr


@pytest.mark.platforms("posix")
def test_stage_passes_explicit_products_to_the_single_assembler(tmp_path, monkeypatch):
    from scripts.bundles import stage
    from scripts.build.agent import assemble
    from scripts.build.inputs import AgentInputs

    out, data = inputs_fixture(tmp_path)
    calls = []
    def native(args):
        calls.append(args.frontends)
        assemble(AgentInputs.from_dict({**data, "frontends": args.frontends}), out)
        return 0
    monkeypatch.setattr(stage, "stage_native", native)
    assert stage.main(["--out", str(out), "--ref", "fixture", "--tui", data["frontends"]["tui"],
                       "--web", data["frontends"]["web"]]) == 0
    assert calls == [{k: Path(v) for k, v in data["frontends"].items()}]
    assert json.loads((out / "manifest.json").read_text())["runtime"]["commands"] == {"probe": "bin/probe"}


@pytest.mark.platforms("posix")
def test_prepared_environment_python_imports_app_from_unrelated_script(tmp_path):
    out, data = inputs_fixture(tmp_path)
    environment = out / ".venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(environment)], check=True)
    python = environment / "bin/python"
    site = subprocess.check_output([str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"], text=True).strip()
    Path(site, "dependency.py").write_text("VALUE='prepared'", encoding="utf-8")
    data.update(placement="fixed", environment=str(environment), python=str(python), site_packages=site)
    result = build_cli(data, out, tmp_path)
    assert result.returncode == 0, result.stderr
    script = tmp_path / "external-probe.py"
    script.write_text("import entry\nfrom importlib.metadata import version\nprint(version('assembly-fixture'))\n", encoding="utf-8")
    run = subprocess.run([str(python), "-I", str(script)], cwd=tmp_path,
                         env={"PATH": os.environ["PATH"], "HOME": str(tmp_path / "home"), "PYTHONDONTWRITEBYTECODE": "1"},
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "1.2.3"


def test_site_layout_uses_explicit_target_version_not_builder_python(tmp_path):
    from scripts.build.inputs import dependency_site
    assert dependency_site(tmp_path, "3.99.1+vendor", "linux-x64") == tmp_path / "lib/python3.99/site-packages"
    assert dependency_site(tmp_path, "3.99.1", "win32-arm64") == tmp_path / "Lib/site-packages"


def test_automatic_frontends_use_requested_snapshot_not_current_checkout(tmp_path, monkeypatch):
    from scripts.bundles import stage
    repo = tmp_path / "checkout"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    project = repo / "pyproject.toml"
    project.write_text('[project]\nversion="1"\n', encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=f@example.test", "commit", "-m", "selected"], cwd=repo, check=True, capture_output=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    project.write_text('[project]\nversion="dirty-current"\n', encoding="utf-8")
    monkeypatch.setattr(stage, "ROOT", repo)
    monkeypatch.setattr(stage.shutil, "which", lambda name: "/prepared/node")
    real_run = subprocess.run
    built_sources = []
    def run(command, **kwargs):
        if command[0] != "/prepared/node":
            return real_run(command, **kwargs)
        source = Path(command[command.index("--source") + 1])
        built_sources.append(source)
        assert source != repo
        assert 'version="1"' in (source / "pyproject.toml").read_text()
    monkeypatch.setattr(stage.subprocess, "run", run)
    monkeypatch.setattr(stage, "stage_native", lambda args: 0)
    assert stage.main(["--out", str(tmp_path / "out"), "--ref", revision]) == 0
    assert len(built_sources) == 4
