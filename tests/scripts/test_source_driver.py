"""Cheap source-driver checks: real shells, disposable installs, no installer."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import sysconfig

import pytest


ASSETS = Path(__file__).resolve().parents[1] / "install/e2e-assets"


@pytest.mark.platforms("posix")
def test_shell_selects_the_installed_command_not_the_phase_or_path(tmp_path):
    root = tmp_path / "installed source"
    legacy = root / "venv/bin/hermes"
    published = root / ".hermes/bin/hermes"
    for command, label in ((legacy, "legacy"), (published, "published")):
        command.parent.mkdir(parents=True)
        command.write_text(f'#!/bin/sh\nprintf "{label}:%s\\n" "$*"\n', encoding="utf-8")
        command.chmod(0o755)
    env = dict(os.environ, INSTALL_DIR=str(root), ASSETS=str(ASSETS), HOME=str(tmp_path))
    result = subprocess.run(["bash", "-euc", '''
source "$ASSETS/source-driver.sh"
command=$(source_hermes "$INSTALL_DIR")
"$command" 'literal argument'
rm "$INSTALL_DIR/venv/bin/hermes"
test "$(source_hermes "$INSTALL_DIR")" = "$command"
printf '#!/bin/sh\nexit 23\n' > "$command"
"$command" || test "$?" = 23
rm "$command"
if source_hermes "$INSTALL_DIR"; then exit 91; fi
printf '#!/bin/sh\nprintf legacy\n' > "$INSTALL_DIR/venv/bin/hermes"
chmod +x "$INSTALL_DIR/venv/bin/hermes"
"$(source_hermes "$INSTALL_DIR")"
# A PM checkout cannot hide missing publication behind its old venv.
mkdir -p "$INSTALL_DIR/pm"
printf '{}' > "$INSTALL_DIR/pm/lock.json"
if source_hermes "$INSTALL_DIR"; then exit 92; fi
'''], env=env, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines() == ["published:literal argument", "legacy"]


@pytest.mark.platforms("windows", "posix")
def test_powershell_selects_exact_exe_or_cmd_and_legacy_fallback(tmp_path):
    # These are file-selection rules, not Windows execution emulation. Run
    # them in real PowerShell on any host that supplies it (native CI included).
    path = os.pathsep.join(p for p in os.get_exec_path() if ".hermes" not in Path(p).parts)
    pwsh = shutil.which("pwsh", path=path) or shutil.which("powershell", path=path)
    if not pwsh:
        if os.name == "nt":
            pytest.fail("native Windows acceptance requires PowerShell")
        pytest.skip("PowerShell is not available on this host")
    root = tmp_path / "installed source"
    root.mkdir()
    harness = tmp_path / "probe.ps1"
    harness.write_text('''$ErrorActionPreference = 'Stop'
. (Join-Path $env:ASSETS 'source-driver.ps1')
$root = $env:INSTALL_DIR
$legacy = Join-Path $root 'venv/Scripts/hermes.exe'
$exe = Join-Path $root '.hermes/bin/hermes.exe'
$cmd = Join-Path $root '.hermes/bin/hermes.cmd'
New-Item -ItemType Directory -Path (Split-Path $legacy), (Split-Path $exe) -Force | Out-Null
Set-Content $legacy 'legacy'
if ((Get-SourceHermes $root) -ne $legacy) { throw 'legacy fallback' }
Set-Content $cmd 'cmd'
if ((Get-SourceHermes $root) -ne $cmd) { throw 'published cmd' }
Set-Content $exe 'exe'
if ((Get-SourceHermes $root) -ne $exe) { throw 'exe precedence' }
Remove-Item $legacy
if ((Get-SourceHermes $root) -ne $exe) { throw 'requires legacy venv' }
Remove-Item $exe, $cmd
$refused = $false
try { Get-SourceHermes $root } catch { $refused = $true }
if (-not $refused) { throw 'missing command accepted' }
Set-Content $legacy 'stale legacy'
New-Item -ItemType Directory -Path (Join-Path $root 'pm') | Out-Null
Set-Content (Join-Path $root 'pm/lock.json') '{}'
$refused = $false
try { Get-SourceHermes $root } catch { $refused = $true }
if (-not $refused) { throw 'missing PM launcher accepted' }
Write-Output 'selection verified'
''', encoding="utf-8")
    result = subprocess.run([pwsh, "-NoProfile", "-NonInteractive", "-File", str(harness)],
                            env=dict(os.environ, INSTALL_DIR=str(root), ASSETS=str(ASSETS), HOME=str(tmp_path)),
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "selection verified"


def test_observer_preserves_no_desktop_and_refuses_incomplete_app(tmp_path):
    verify = runpy.run_path(str(ASSETS / "source_driver.py"))["verify_products"]
    root = tmp_path / "source"
    root.mkdir()
    verify(root, "absent")
    assert list(root.iterdir()) == []
    with pytest.raises(RuntimeError, match="desktop"):
        verify(root, "present")
    release = root / "apps/desktop/release/linux-unpacked"
    release.mkdir(parents=True)
    (release / "hermes").write_bytes(b"incomplete fixture")
    with pytest.raises(RuntimeError, match="desktop"):
        verify(root, "absent")
    with pytest.raises(RuntimeError, match="desktop"):
        verify(root, "present")
    assert (release / "hermes").read_bytes() == b"incomplete fixture"


@pytest.mark.platforms("posix")
def test_observer_checks_real_compiler_receipts_without_repairing(tmp_path):
    verify = runpy.run_path(str(ASSETS / "source_driver.py"))["verify_products"]
    path = os.pathsep.join(p for p in os.get_exec_path() if ".hermes" not in Path(p).parts)
    node = shutil.which("node", path=path)
    assert node, "source-driver tests require the prepared Node tool"
    root = tmp_path / "source"
    build = root / "scripts/build"
    build.mkdir(parents=True)
    repo = ASSETS.parents[2]
    for name in ("freshness.mjs", "frontend-common.mjs"):
        shutil.copy2(repo / "scripts/build" / name, build / name)
    outputs = {"tui": root / "ui-tui/dist", "web": root / "hermes_cli/web_dist",
               "desktop": root / "apps/desktop/release/linux-unpacked/resources/app.asar.unpacked/dist"}
    for out in outputs.values():
        out.mkdir(parents=True)
        (out / "index.html").write_text("fixture renderer", encoding="utf-8")
    (root / "apps/desktop/release/linux-unpacked/hermes").write_text("fixture executable", encoding="utf-8")
    record = '''import { buildInputs, recordProduct } from './scripts/build/freshness.mjs';
const source = process.cwd();
for (const [product, out] of Object.entries(JSON.parse(process.argv[1]))) {
  recordProduct({ source, product, out, inputs: buildInputs(source, product) });
}
'''
    subprocess.run([node, "--input-type=module", "-e", record, json.dumps({k: str(v) for k, v in outputs.items()})],
                   cwd=root, check=True, timeout=30)
    def snapshot():
        return {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
                for p in root.rglob("*") if p.is_file()}
    before = snapshot()
    verify(root, "present", Path(node))
    assert snapshot() == before
    # The successful receipts cannot bless damaged or missing outputs.
    damaged = outputs["web"] / "index.html"
    damaged.write_text("damaged renderer", encoding="utf-8")
    before = snapshot()
    with pytest.raises(RuntimeError, match="web output"):
        verify(root, "present", Path(node))
    assert snapshot() == before
    damaged.unlink()
    with pytest.raises(RuntimeError, match="web output"):
        verify(root, "present", Path(node))
    assert not damaged.exists()


@pytest.mark.platforms("posix")
def test_pm_probe_rejects_foreign_launcher_before_any_bootstrap(tmp_path, monkeypatch):
    from hermes_cli import _launchers

    root = tmp_path / "installed source"
    foreign = tmp_path / "other source"
    repo = ASSETS.parents[2]
    # Real published-launcher protocol, with only the stdlib pre-boot closure.
    for tree in (root, foreign):
        for name in ("hermes_constants.py", "hermes_cli/__init__.py", "hermes_cli/_launchers.py",
                     "hermes_cli/runtime_paths.py"):
            dest = tree / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo / name, dest)
        (tree / "pm").mkdir()
        (tree / "pm/lock.json").write_text("{}", encoding="utf-8")
        (tree / "hermes_bootstrap.py").write_text("raise RuntimeError('bootstrap must not run')", encoding="utf-8")
    home = tmp_path / "home"
    store = home / "tools"
    store.mkdir(parents=True)
    interpreter = Path(sys._base_executable).resolve()
    (store / "facts.json").write_text(json.dumps({"packages": {"python": {
        "entry": str(interpreter.parents[1])}}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(store))
    bin_dir = root / ".hermes/bin"
    bin_dir.mkdir(parents=True)
    launcher = _launchers.mint_launcher("hermes", foreign, bin_dir, interpreter, None)
    assert launcher is not None
    result = subprocess.run([sys.executable, "-B", str(ASSETS / "source_driver.py"),
                             "--root", str(root), "--launcher", str(launcher), "--desktop", "absent"],
                            cwd=tmp_path, env=dict(os.environ, HOME=str(tmp_path)),
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "belongs to another installation" in result.stderr
    assert "bootstrap must not run" not in result.stderr
    assert not (root / "venv").exists()


@pytest.mark.platforms("posix")
def test_pm_observer_accepts_ready_fixture_and_leaves_failed_fixture_untouched(tmp_path):
    # This exercises the complete observer process, not a whole install. The
    # fixture borrows prepared test dependencies and records real PM input
    # stamps / compiler receipts. It never resolves or acquires a package.
    repo = ASSETS.parents[2]
    root = tmp_path / "source"
    for directory in ("pm", "hermes_cli"):
        shutil.copytree(repo / directory, root / directory, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(repo / "hermes_constants.py", root / "hermes_constants.py")
    (root / "hermes_bootstrap.py").write_text("raise RuntimeError('bootstrap must not run')", encoding="utf-8")
    (root / "uv.lock").write_text("fixture locked graph", encoding="utf-8")
    home = tmp_path / "home"
    store = tmp_path / "store"
    env = dict(os.environ, HOME=str(home), HERMES_HOME=str(home), HERMES_RUNTIME_DIR=str(store),
               PYTHONDONTWRITEBYTECODE="1", HERMES_DISABLE_LAZY_INSTALLS="1")
    env.pop("HERMES_INSTALL_ROOT", None)
    path = os.pathsep.join(p for p in os.get_exec_path() if ".hermes" not in Path(p).parts)
    node = shutil.which("node", path=path)
    assert node
    setup = '''import sys
from pathlib import Path
root, store, node, deps = map(Path, sys.argv[1:])
sys.path.insert(0, str(root))
from pm.lock import Facts, Lockfile
from pm.packages import Venv
from hermes_cli.runtime_paths import install_state_dir, runtime_facts_path, site_packages
from hermes_cli._launchers import ensure_install_launchers
lock = Lockfile(root / 'pm/lock.json')
lock.set_pin('node', 'fixture', {})
lock.save()
for name, executable in [('python', Path(sys.executable)), ('node', node)]:
    binary = store / name / 'bin' / ('python3' if name == 'python' else name)
    binary.parent.mkdir(parents=True)
    binary.symlink_to(executable)
    Facts(store / 'facts.json').record(name, 'fixture', name, {}, store)
selected = install_state_dir(root) / 'environments/fixture/venv'
selected.mkdir(parents=True)
(selected / 'pyvenv.cfg').write_text('home = fixture')
site = site_packages(selected)
site.parent.mkdir(parents=True)
site.symlink_to(deps, target_is_directory=True)
Facts(runtime_facts_path(root)).record_state('venv', Venv(root).expected_stamp([]), [], environment=selected)
ensure_install_launchers(root, root / '.hermes/bin')
'''
    subprocess.run([sys.executable, "-I", "-B", "-c", setup, str(root), str(store), node,
                    sysconfig.get_path("purelib")], env=env, cwd=tmp_path, check=True, timeout=60)
    build = root / "scripts/build"
    build.mkdir(parents=True)
    for name in ("freshness.mjs", "frontend-common.mjs"):
        shutil.copy2(repo / "scripts/build" / name, build / name)
    record = '''import { mkdirSync, writeFileSync } from 'node:fs';
import { buildInputs, recordProduct } from './scripts/build/freshness.mjs';
const source = process.cwd();
for (const [product, out] of [['tui', 'ui-tui/dist'], ['web', 'hermes_cli/web_dist']]) {
  mkdirSync(out, { recursive: true }); writeFileSync(out + '/index.html', 'fixture product');
  recordProduct({source, product, out, inputs: buildInputs(source, product)});
}
'''
    subprocess.run([node, "--input-type=module", "-e", record], cwd=root, env=env, check=True, timeout=30)
    command = [sys.executable, "-B", str(ASSETS / "source_driver.py"), "--root", str(root),
               "--launcher", str(root / ".hermes/bin/hermes"), "--desktop", "absent"]
    # The passive query must not write even import caches: -I ignores the
    # environment's bytecode switch, so the published query enforces it.
    def snapshot():
        return {p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) for tree in (root, home, store)
                for p in tree.rglob("*") if p.is_file()}
    before = snapshot()
    result = subprocess.run(command, env=env, cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert snapshot() == before
    (root / "uv.lock").write_text("changed graph without preparation", encoding="utf-8")
    before = snapshot()
    result = subprocess.run(command, env=env, cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert "dependency generation is not current" in result.stderr
    assert snapshot() == before