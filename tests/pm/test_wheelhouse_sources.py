"""Windows ARM64 wheel URLs must stay disjoint from other platform resolutions."""

import re
import tomllib
from pathlib import Path

from packaging.markers import Marker, default_environment


ROOT = Path(__file__).resolve().parents[2]


def test_wheelhouse_sources_are_locked_only_for_windows_arm64():
    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8-sig"))
    packages = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8-sig"))["package"]
    sources = manifest["tool"]["uv"]["sources"]
    wheelhouse = {name: entries for name, entries in sources.items()
                  if any("/releases/download/wheelhouse/" in entry.get("url", "") for entry in entries)}
    assert wheelhouse

    for name, entries in wheelhouse.items():
        assert len(entries) == 1
        entry = entries[0]
        assert entry["url"].startswith("https://github.com/ethernet8023/hermes-agent/releases/download/wheelhouse/")
        marker = Marker(entry["marker"])
        for system, machine, selected in (("win32", "ARM64", True), ("win32", "AMD64", False),
                                          ("linux", "aarch64", False), ("darwin", "arm64", False)):
            env = default_environment()
            env.update(sys_platform=system, platform_machine=machine)
            assert marker.evaluate(env) is selected, (name, system, machine)
        rows = [row for row in packages if row["name"] == name]
        direct = [row for row in rows if row["source"].get("url") == entry["url"]]
        registry = [row for row in rows if "registry" in row["source"]]
        assert len(direct) == len(registry) == 1, name
        assert direct[0]["version"] == registry[0]["version"]
        assert any(wheel["url"] == entry["url"] and re.fullmatch(r"sha256:[0-9a-f]{64}", wheel["hash"])
                   for wheel in direct[0]["wheels"]), name
