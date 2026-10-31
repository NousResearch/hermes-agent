"""Real Termux shell boundaries; only the unavailable dpkg executable is adapted."""
import io
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile

from scripts.termux import stage_apt_repo
from tests.ci.test_desktop_release_tag_admission import _child_env, _native_tool, _workflow

ROOT = Path(__file__).resolve().parents[2]
COMMIT = "a" * 40


def stamp_from_shell(tmp_path, tag, *, substitutions=()):
    script = (ROOT / "scripts/termux/build_deb.sh").read_text(encoding="utf-8")
    block = script[script.index('HERMES_PAYLOAD_TAG="$TAG" \\\n'):]
    block = block[:block.index('|| fail "stamp write failed"')]
    for old, new in substitutions:
        assert old in block
        block = block.replace(old, new)
    # The candidate's --release-commit path derives PY_VERSION through this real heredoc.
    version_block = script[script.index('    PY_VERSION="$(python3 - '):]
    version_block = version_block[:version_block.index('|| fail "commit version admission failed"')]
    env = _child_env(TAG=tag, COMMIT=COMMIT, REPO_ABS=ROOT.as_posix(), REPO_ROOT=ROOT.as_posix(),
                     PAYLOAD_ABS=tmp_path.as_posix(),
                     HERMES_BUILD_COMMIT="", HERMES_PAYLOAD_TAG="", HERMES_DESKTOP_VARIANT="")
    launcher = f'python3() {{ {shlex.quote(Path(sys.executable).as_posix())} "$@"; }}\n'
    return subprocess.run([_native_tool("bash"), "-e", "-o", "pipefail", "-c", launcher + version_block + "\n" + block],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30, check=False)


def dpkg_run(argv, *args, **kwargs):
    """Exercise real archive bytes without requiring host dpkg (not shipped on Windows)."""
    if argv[0] != "dpkg-deb" or shutil.which("dpkg-deb"):
        return REAL_RUN(argv, *args, **kwargs)
    package = Path(argv[2])
    if argv[1] == "--field":
        fields, _ = stage_apt_repo.deb_control_fields_and_bytes(package)
        output = "".join(f"{name}: {fields[name]}\n" for name in argv[3:])
        return subprocess.CompletedProcess(argv, 0, output, "")
    assert argv[1] == "--extract", argv
    for name, _, body in stage_apt_repo.read_ar_entries(package.read_bytes()):
        if name.startswith("data.tar."):
            with tarfile.open(fileobj=io.BytesIO(body)) as archive:
                archive.extractall(argv[3], filter="data")
    return subprocess.CompletedProcess(argv, 0, "", "")


REAL_RUN = subprocess.run


def stage_from_workflow(tmp_path, server, tag, public_base):
    driver = tmp_path / "driver.py"
    driver.write_text(
        "import runpy, subprocess, sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from tests.ci.channel_termux_helpers import dpkg_run\n"
        "from scripts.releases import r2\n"
        f"r2.s3_endpoint = lambda _: 'http://127.0.0.1:{server.server_port}'\n"
        "subprocess.run = dpkg_run\n"
        "assert sys.argv[1] == '-m', sys.argv\n"
        "module = sys.argv[2]\nsys.argv = sys.argv[2:]\n"
        "runpy.run_module(module, run_name='__main__')\n", encoding="utf-8")
    steps = [step for job in _workflow()["jobs"].values() for step in job.get("steps", [])]
    script, = [step["run"] for step in steps if "record --platform termux" in step.get("run", "")]
    # Replay the entire record/handoff tail, including the candidate exit that protects APT heads.
    script = script[script.index("args=(--include 'deb/*.deb'"):]
    launcher = f'python() {{ {shlex.quote(Path(sys.executable).as_posix())} {shlex.quote(driver.as_posix())} "$@"; }}\n'
    env = _child_env(RELEASE_PHASE="candidate", RELEASE_COMMIT=COMMIT, HERMES_PAYLOAD_TAG=tag,
                     HERMES_ARCHIVE_TAG=tag, HERMES_BUILD_COMMIT="", R2_DISPOSABLE_RUN="",
                     CLOUDFLARE_R2_PUBLIC_URL=public_base)
    return subprocess.run([_native_tool("bash"), "-e", "-o", "pipefail", "-c", launcher + script],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30, check=False)


def read_stamp(tmp_path):
    return json.loads((tmp_path / "app/install-stamp.json").read_text(encoding="utf-8"))
