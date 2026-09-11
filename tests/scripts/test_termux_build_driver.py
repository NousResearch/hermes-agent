"""Exercise the Termux sequencing boundary with real child executables."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from ruamel.yaml import YAML


ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "scripts/termux/build.py"
pytestmark = pytest.mark.platforms("posix")


@pytest.fixture
def build_fixture(tmp_path):
    repo = tmp_path / "source with spaces"
    payload = tmp_path / "prepared payload"
    out = tmp_path / "deb output"
    tools = tmp_path / "tools"
    tools.mkdir()
    payload.mkdir()
    for name in ("scripts/build/node-deps.mjs", "scripts/build/tui.mjs", "scripts/termux/build_deb.sh"):
        script = repo / name
        script.parent.mkdir(parents=True, exist_ok=True)
        script.touch()
    log = tmp_path / "calls.jsonl"
    recorder = f"""#!{sys.executable}
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
assert Path(args[0]).is_file(), args
stage = Path(args[0]).name
with Path(os.environ['CALL_LOG']).open('a') as stream:
    stream.write(json.dumps({{'tool': Path(sys.argv[0]).name, 'args': args,
        'cwd': os.getcwd(), 'image': os.environ.get('TERMUX_BUILDER_IMAGE')}}) + '\\n')
if stage == os.environ.get('FAIL_STAGE'):
    sys.exit(37)
if stage == 'tui.mjs':
    product = Path(args[args.index('--out') + 1])
    (product / 'dist').mkdir(parents=True, exist_ok=True)
    (product / 'dist/entry.js').write_text('fixture TUI product')
    (product / 'package.json').write_text('{{"type":"module"}}')
if stage == 'build_deb.sh':
    product = Path(args[args.index('--tui-product') + 1])
    assert (product / 'dist/entry.js').read_text() == 'fixture TUI product'
    destination = Path(args[args.index('--out') + 1])
    destination.mkdir(parents=True)
    (destination / 'assembly-received-product').write_text(str(product))
"""
    for name in ("node", "bash"):
        tool = tools / name
        tool.write_text(recorder, encoding="utf-8")
        tool.chmod(0o755)
    env = dict(os.environ, PATH=f"{tools}{os.pathsep}{os.environ['PATH']}",
               CALL_LOG=str(log), TERMUX_BUILDER_IMAGE="fixture-pinned-image")
    return repo, payload, out, log, env


def invoke(build_fixture, *identity, extra=()):
    repo, payload, out, _, env = build_fixture
    return subprocess.run(
        [sys.executable, str(DRIVER), "--repo", str(repo), "--payload", str(payload),
         "--out", str(out), *identity, *extra],
        cwd=repo.parent, env=env, capture_output=True, text=True, timeout=30,
    )


def recorded_calls(log):
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


@pytest.mark.parametrize("identity", [("--tag", "v1.2.3"), ("--commit", "a" * 40)])
def test_driver_prepares_only_tui_then_passes_product_to_deb(build_fixture, identity):
    repo, payload, out, log, _ = build_fixture
    result = invoke(build_fixture, *identity)
    assert result.returncode == 0, result.stdout + result.stderr
    product = repo / ".build/termux/tui"
    assert recorded_calls(log) == [
        {"tool": "node", "args": [str(repo / "scripts/build/node-deps.mjs"),
          "--source", str(repo), "--workspace", "ui-tui"],
         "cwd": str(repo), "image": "fixture-pinned-image"},
        {"tool": "node", "args": [str(repo / "scripts/build/tui.mjs"),
          "--source", str(repo), "--out", str(product)],
         "cwd": str(repo), "image": "fixture-pinned-image"},
        {"tool": "bash", "args": [str(repo / "scripts/termux/build_deb.sh"),
          "--repo", str(repo), "--payload", str(payload), "--out", str(out),
          "--tui-product", str(product), *identity],
         "cwd": str(repo), "image": "fixture-pinned-image"},
    ]
    assert (out / "assembly-received-product").read_text() == str(product)


@pytest.mark.parametrize("identity", [
    (), ("--tag", "v1.2.3", "--commit", "a" * 40),
    ("--ta", "v1.2.3"), ("--commit", "short"),
    ("--commit", "A" * 40), ("--tag", ""), ("--tag", "latest"),
    ("--tag", "v1.2.3", "--skip-build"),
])
def test_invalid_cli_never_starts_a_build(build_fixture, identity):
    _, payload, out, log, _ = build_fixture
    result = invoke(build_fixture, *identity)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "error:" in result.stderr
    assert recorded_calls(log) == []
    assert not (payload / ".work").exists()
    assert not out.exists()


@pytest.mark.parametrize("missing", ["--repo", "--payload", "--out"])
def test_all_paths_are_explicit(build_fixture, missing):
    repo, payload, out, log, env = build_fixture
    paths = {"--repo": repo, "--payload": payload, "--out": out}
    args = [value for option, path in paths.items() if option != missing
            for value in (option, str(path))]
    result = subprocess.run([sys.executable, str(DRIVER), *args, "--tag", "v1.2.3"],
                            cwd=repo.parent, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert missing in result.stderr
    assert recorded_calls(log) == []


def test_missing_prepared_payload_is_rejected_before_frontend_work(build_fixture):
    _, payload, out, log, _ = build_fixture
    payload.rmdir()
    result = invoke(build_fixture, "--tag", "v1.2.3")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "--payload" in result.stderr
    assert recorded_calls(log) == []
    assert not payload.exists()
    assert not out.exists()


@pytest.mark.parametrize("stage, reached", [
    ("node-deps.mjs", ["node-deps.mjs"]),
    ("tui.mjs", ["node-deps.mjs", "tui.mjs"]),
    ("build_deb.sh", ["node-deps.mjs", "tui.mjs", "build_deb.sh"]),
])
def test_child_failure_stops_sequence_even_with_stale_product(build_fixture, stage, reached):
    repo, payload, out, log, env = build_fixture
    stale = repo / ".build/termux/tui/dist/entry.js"
    stale.parent.mkdir(parents=True)
    stale.write_text("stale product")
    env["FAIL_STAGE"] = stage
    result = invoke(build_fixture, "--tag", "v1.2.3")
    assert result.returncode == 37, result.stdout + result.stderr
    assert [Path(call["args"][0]).name for call in recorded_calls(log)] == reached
    assert not out.exists()


@pytest.mark.parametrize("identity", [("--tag", "v1.2.3"), ("--commit", "a" * 40)])
def test_release_workflow_runs_the_shared_sequence(build_fixture, identity):
    repo, payload, _, log, env = build_fixture
    (repo / "scripts/termux/build.py").symlink_to(DRIVER)
    (repo / "termux-build").mkdir()
    (repo / "termux-build/payload").symlink_to(payload, target_is_directory=True)
    tools = log.parent / "tools"
    (tools / "python3").symlink_to(sys.executable)
    env.update(HERMES_BUILD_COMMIT=identity[1] if identity[0] == "--commit" else "",
               HERMES_PAYLOAD_TAG=identity[1] if identity[0] == "--tag" else "")
    workflow = YAML(typ="safe").load((ROOT / ".github/workflows/desktop-bundled-release.yml").read_text())
    step = next(step for step in workflow["jobs"]["termux-deb"]["steps"]
                if step.get("name") == "Assemble the .deb")
    bash = shutil.which("bash")
    assert bash is not None
    result = subprocess.run([bash, "-euo", "pipefail", "-c", step["run"]],
                            cwd=repo, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = recorded_calls(log)
    assert [Path(call["args"][0]).name for call in calls] == [
        "node-deps.mjs", "tui.mjs", "build_deb.sh",
    ]
    assert calls[-1]["args"][-2:] == list(identity)
    assert (repo / "termux-build/deb/assembly-received-product").read_text() == str(
        repo / ".build/termux/tui")
