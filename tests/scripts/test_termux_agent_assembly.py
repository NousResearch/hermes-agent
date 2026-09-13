"""Bionic assembly consumes prepared bytes without running foreign binaries."""
import json
from pathlib import Path
import shutil

from scripts.build.inputs import RESOURCE_ENV, dependency_site
from scripts.termux.payload_facts import write_facts


def test_bionic_facts_and_agent_keep_fixed_prefix_and_tui_only(tmp_path):
    from pm.lock import Lockfile, Facts
    from pm.registry import get_package

    payload = tmp_path / "payload"
    repo = payload / "app"
    (repo / "pm").mkdir(parents=True)
    lock = repo / "pm/lock.json"
    shutil.copy2(Path(__file__).resolve().parents[2] / "pm/lock.json", lock)
    (repo / "pyproject.toml").write_text('[project]\nname="fixture"\nversion="1"\n[project.scripts]\nprobe="entry:main"\n', encoding="utf-8")
    (repo / "install-stamp.json").write_text('{"distribution":"apt-termux","payload":"bundled"}', encoding="utf-8")
    for name in RESOURCE_ENV:
        (repo / name).mkdir()
    target = "linux-arm64-bionic"
    tools = ("python", "node", "uv", "npm", "ffmpeg", "ripgrep")
    for name in tools:
        executable = get_package(name).binary(payload / "tools" / name, target)
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.write_bytes(b"foreign binary must not be executed")
        executable.chmod(0o755)
    python = get_package("python").binary(payload / "tools/python", target)
    site = dependency_site(payload / "venv", Lockfile(lock).version("python"), target)
    site.mkdir(parents=True)
    pm = payload / "pm-runtime"
    (pm / "deps").mkdir(parents=True)
    (pm / "pm-runtime.json").write_text(json.dumps({"python": "../" + python.relative_to(payload).as_posix(), "sitePackages": "deps"}), encoding="utf-8")
    tui = tmp_path / "tui-product"
    (tui / "dist").mkdir(parents=True)
    (tui / "dist/entry.js").write_text("built", encoding="utf-8")
    (tui / "package.json").write_text('{"type":"module"}', encoding="utf-8")
    build_set = tmp_path / "build-set.txt"
    build_set.write_text("native-fixture\n", encoding="utf-8")
    write_facts(payload, lock, build_set, tui)
    manifest = json.loads((payload / "manifest.json").read_text())
    assert manifest["runtime"]["commands"] == {"probe": "bin/probe"}
    assert manifest["runtime"]["sitePackages"] == site.relative_to(payload).as_posix()
    assert (repo / "hermes_cli/tui_dist/entry.js").read_text() == "built"
    assert not (repo / "hermes_cli/web_dist").exists()
    assert json.loads((payload / "native-wheels.json").read_text()) == ["native-fixture"]
    facts = Facts(payload / "tools/facts.json", strict=True)
    assert all(facts.get(name)["target"] == target for name in tools)
