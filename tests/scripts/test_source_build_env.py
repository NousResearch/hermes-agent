"""Source E2E children stamp their own checkout, not the workflow's NEW ref."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "tests/install/e2e-assets"


def _stamp_probe(tmp_path, shell):
    repo = tmp_path / "installed source"
    for relative in (
        "scripts/write_install_stamp.py", "hermes_cli/__init__.py",
        "hermes_cli/update_channel.py", "hermes_cli/release_channels.py",
        "pm/paths.py", "pm/environments.py",
        "hermes_cli/steward.py", "hermes_constants.py",
    ):
        dest = repo / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, dest)
    # distance -> versioning -> semver import each other; copy the whole
    # package so a new sibling import cannot break the fixture.
    shutil.copytree(ROOT / "scripts/releases", repo / "scripts/releases",
                    ignore=shutil.ignore_patterns("__pycache__"))
    env = dict(os.environ, HOME=str(tmp_path), HERMES_HOME=str(tmp_path / "home"),
               GIT_CONFIG_GLOBAL=str(tmp_path / "gitconfig"), GIT_CONFIG_NOSYSTEM="1")
    (tmp_path / "gitconfig").write_text('[url "file:///staged/serve.git"]\n'
                                      '\tinsteadOf = https://github.com/NousResearch/hermes-agent.git\n',
                                      encoding="utf-8")

    def git(*args):
        return subprocess.run(["git", *args], cwd=repo, env=env, check=True,
                              capture_output=True, text=True, timeout=30).stdout.strip()

    git("init", "-q", "-b", "installed")
    git("config", "user.name", "Fixture")
    git("config", "user.email", "fixture@example.invalid")
    git("add", ".")
    git("commit", "-qm", "OLD")
    old = git("rev-parse", "HEAD")
    git("tag", "v1.0.0", old)
    git("commit", "--allow-empty", "-qm", "NEW")
    new = git("rev-parse", "HEAD")
    # Control: this is the real stamp writer's CI preference, not a source-text assertion.
    stamp_command = [sys.executable, "-I", "-S", str(repo / "scripts/write_install_stamp.py"),
                     "--output", str(tmp_path / "contaminated.json"),
                     "--base-version", "1.0.0", "--distance", "0",
                     "--update-mechanism", "self"]
    git("checkout", "-q", "-B", "installed", old)
    subprocess.run(stamp_command, env={**env, "GITHUB_SHA": new}, check=True,
                   capture_output=True, text=True, timeout=30)
    assert json.loads((tmp_path / "contaminated.json").read_text(encoding="utf-8-sig"))["commit"] == new
    overrides = dict(GITHUB_SHA=new, GITHUB_REF="refs/heads/workflow", GITHUB_REF_NAME="workflow",
                     GITHUB_HEAD_REF="workflow", GITHUB_BASE_REF="main", HERMES_BUILD_COMMIT=new,
                     HERMES_PAYLOAD_TAG="v1.2.3", HERMES_PAYLOAD_VERSION="1.2.3",
                     HERMES_DESKTOP_VARIANT="bundled")
    env.update(overrides, ASSETS=str(ASSETS), PROBE_PYTHON=sys.executable,
               PROBE_ROOT=str(repo), PROBE_OUT=str(tmp_path / "stamp.json"),
               PROBE_ENV=str(tmp_path / "child-env.json"))
    probe = tmp_path / "probe.py"
    probe.write_text('''import json, os, runpy
from pathlib import Path
stamp = runpy.run_path(str(Path(os.environ['PROBE_ROOT']) / 'scripts/write_install_stamp.py'))
stamp['write_stamp'](os.environ['PROBE_OUT'], update_mechanism='self', base_version='1.0.0', distance=0)
Path(os.environ['PROBE_ENV']).write_text(json.dumps(dict(os.environ)), encoding='utf-8')
''', encoding="utf-8")
    env["PROBE_SCRIPT"] = str(probe)
    # A file, not `-c 'import json, ...'`: pwsh 7.6 on the Windows lane re-quotes native
    # argv and the venv launcher receives a truncated -c body (`import` → SyntaxError).
    dump = tmp_path / "dump_env.py"
    dump.write_text("import json, os; print(json.dumps(dict(os.environ)))\n", encoding="utf-8")
    env["PROBE_DUMP"] = str(dump)
    if shell == "bash":
        command = [shell, "-euc", '''
source "$ASSETS/source-build-env.sh"
source_build_env "$PROBE_PYTHON" -I -S "$PROBE_SCRIPT"
if source_build_env "$PROBE_PYTHON" -I -S -c 'raise SystemExit(23)'; then exit 99; else test "$?" = 23; fi
"$PROBE_PYTHON" -I -S -c 'import json, os; print(json.dumps(dict(os.environ)))'
''']
    elif shell == "node":
        env["HERMES_DESKTOP_USER_DATA_DIR"] = str(tmp_path / "user-data")
        env["HERMES_PYTHON_SRC_ROOT"] = str(repo)
        command = [shell, "--input-type=module", "-e", '''
import { pathToFileURL } from 'node:url';
import { spawnSync } from 'node:child_process';
const { updateWindowEnvironment } = await import(pathToFileURL(process.env.ASSETS + '/smoke-env.mjs'));
const env = updateWindowEnvironment(process.env, process.env.PROBE_ROOT, 'source');
const result = spawnSync(process.env.PROBE_PYTHON, ['-I', '-S', process.env.PROBE_SCRIPT], { env, stdio: 'inherit' });
if (result.error || result.status !== 0) throw new Error(`stamp child failed: ${result.error || result.status}`);
console.log(JSON.stringify(process.env));
''']
    else:
        # -File, not -Command: Windows PowerShell 5.1 reads a multi-line -Command argument as
        # far as the first line break and exits 0 having run only the preference line.
        script = tmp_path / "probe.ps1"
        script.write_text('''$ErrorActionPreference = 'Stop'
[Console]::Error.WriteLine("probe start: $($PSVersionTable.PSVersion) assets=$env:ASSETS python=$env:PROBE_PYTHON")
trap { [Console]::Error.WriteLine("probe trap: $_"); [Console]::Error.WriteLine($_.ScriptStackTrace); exit 97 }
. (Join-Path $env:ASSETS 'source-build-env.ps1')
if ($env:OS -eq 'Windows_NT') {
    # Diagnostics (stderr, shown only on failure): `&` on the lane returns at once with $LASTEXITCODE
    # unset and no output — PowerShell's behaviour for a non-console image — while Start-Process
    # gets a real exit code. Ask what image the launcher is and whether a known console exe behaves.
    $subsystem = try {
        $bytes = [System.IO.File]::ReadAllBytes($env:PROBE_PYTHON)
        $pe = [BitConverter]::ToInt32($bytes, 0x3C)
        [BitConverter]::ToUInt16($bytes, $pe + 24 + 68)
    } catch { "unreadable: $_" }
    [Console]::Error.WriteLine("probe: launcher subsystem=$subsystem (2=GUI 3=console) passing=$PSNativeCommandArgumentPassing")
    & "$env:SystemRoot\System32\cmd.exe" /c "exit 7"
    [Console]::Error.WriteLine("probe: control cmd.exe exit=$LASTEXITCODE")
    $home_ = (Get-Content -LiteralPath (Join-Path (Split-Path -Parent (Split-Path -Parent $env:PROBE_PYTHON)) 'pyvenv.cfg') |
        Where-Object { $_ -match '^home\s*=' }) -replace '^home\s*=\s*', ''
    [Console]::Error.WriteLine("probe: pyvenv home=$home_ base exists=$(Test-Path -LiteralPath (Join-Path $home_ 'python.exe'))")
    & (Join-Path $home_ 'python.exe') -I -S -c "import sys; sys.exit(5)"
    [Console]::Error.WriteLine("probe: base interpreter exit=$LASTEXITCODE")
    & $env:PROBE_PYTHON -I -S -c "import sys; sys.exit(6)"
    [Console]::Error.WriteLine("probe: launcher exit=$LASTEXITCODE")
}
Invoke-SourceBuild {
    & $env:PROBE_PYTHON -I -S $env:PROBE_SCRIPT
    [Console]::Error.WriteLine("probe: child dollar-question=$? exit=$LASTEXITCODE out-exists=$(Test-Path -LiteralPath $env:PROBE_OUT)")
    if ($LASTEXITCODE) { throw 'stamp failed' }
}
try { Invoke-SourceBuild { throw 'child failure' }; throw 'lost exception' }
catch { if ($_.Exception.Message -ne 'child failure') { throw } }
& $env:PROBE_PYTHON -I -S $env:PROBE_DUMP
if ($LASTEXITCODE) { exit $LASTEXITCODE }
''', encoding="utf-8-sig")
        command = [shell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(script)]
        print("powershell probe:", shell, file=sys.stderr)
    for sha in (old, new):
        git("checkout", "-q", "-B", "installed", sha)
        result = subprocess.run(command, env=env, cwd=tmp_path, capture_output=True, stdin=subprocess.DEVNULL,
                                text=True, encoding="utf-8", errors="replace", timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
        assert Path(env["PROBE_OUT"]).is_file(), (
            f"the build child wrote no stamp (shell={command[0]} rc={result.returncode})\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}")
        stamp = json.loads(Path(env["PROBE_OUT"]).read_text(encoding="utf-8-sig"))
        assert (stamp["commit"], stamp["branch"], stamp["source"], stamp["payload"]) == (
            sha, "installed", "local", "bootstrap")
        assert (stamp["baseVersion"], stamp["distance"], stamp["displayVersion"]) == (
            "1.0.0", 0, "1.0.0")
        child = json.loads(Path(env["PROBE_ENV"]).read_text(encoding="utf-8-sig"))
        assert not overrides.keys() & child.keys()
        assert child["GIT_CONFIG_GLOBAL"] == env["GIT_CONFIG_GLOBAL"]
        redirect = subprocess.run(["git", "ls-remote", "--get-url",
                                   "https://github.com/NousResearch/hermes-agent.git"], env=child,
                                  cwd=tmp_path, check=True, capture_output=True, text=True, timeout=30)
        assert redirect.stdout.strip() == "file:///staged/serve.git"
        assert child["HERMES_HOME"] == env["HERMES_HOME"]
        parent = json.loads(result.stdout.splitlines()[-1])
        assert {key: parent[key] for key in overrides} == overrides


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("shell", ["bash", "node"])
def test_posix_build_children_stamp_old_and_new_without_changing_parent(tmp_path, shell):
    _stamp_probe(tmp_path, shell)


@pytest.mark.platforms("windows", "posix")
def test_powershell_build_children_stamp_old_and_new_without_changing_parent(tmp_path):
    path = os.pathsep.join(p for p in os.get_exec_path() if ".hermes" not in Path(p).parts)
    # Windows PowerShell 5.1 is what install-e2e-windows-run.yml drives windows-e2e.ps1 (the
    # asset's real consumer) with; pwsh is the only PowerShell a POSIX host offers.
    order = ("powershell", "pwsh") if os.name == "nt" else ("pwsh", "powershell")
    shell = next((found for name in order if (found := shutil.which(name, path=path))), None)
    if not shell:
        if os.name == "nt":
            pytest.fail("native Windows acceptance requires PowerShell")
        pytest.skip("PowerShell is not available on this host")
    _stamp_probe(tmp_path, shell)
