"""One browser payload serves both headed and headless callers."""
from pathlib import Path

import pytest

from pm.lock import Facts
from pm.packages import Chromium
from pm.registry import walk
from scripts.bundles.native import _bundle_package_names


def test_browser_closure_and_bundle_ship_full_chromium_only():
    closure = {package.name for package in walk(["agent-browser"])}
    bundled = set(_bundle_package_names())
    assert "chromium" in closure & bundled
    assert "chromium-headless-shell" not in closure | bundled


@pytest.mark.parametrize(("target", "relative"), [
    ("linux-x64", "chrome-linux64/chrome"),
    ("linux-arm64", "chrome-linux/chrome"),
    ("darwin-arm64", "chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"),
    ("win32-x64", "chrome-win64/chrome.exe"),
    ("win32-arm64", "chrome-win64/chrome.exe"),
])
def test_chromium_binary_and_env_survive_store_relocation(tmp_path, target, relative):
    package = Chromium()
    store = tmp_path / "store"
    entry = store / package.store_entry("1234+145.0.0.0", target)
    executable = entry / relative
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"browser-fixture")
    executable.chmod(0o755)
    assert package.binary(entry, target) == executable
    facts = Facts(store / "facts.json")
    facts.record("chromium", "1234+145.0.0.0", entry.name, package.env(entry, target), store)
    moved = tmp_path / "relocated"
    store.rename(moved)
    env = Facts(moved / "facts.json").env_for("chromium", moved)
    assert Path(env["AGENT_BROWSER_EXECUTABLE_PATH"]) == moved / entry.name / relative
    assert env["PLAYWRIGHT_BROWSERS_PATH"] == str(moved)
