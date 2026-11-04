"""A plugin pack must hand its fan-out the same scan decision a single install gets (#148858).

``hermes plugins install <source> --ref <sha>`` shows a caution scan's findings and asks
``Install anyway? [y/N]``; ``hermes plugins pack install`` called ``_install_plugin_core`` without
a ``scan_decision_cb``, so the same plugin at the same commit died with "Requires confirmation"
and the question was never asked. A bare catalog ``name:`` entry additionally lost the catalog's
reviewed sha, so the pack refused a tree that ``hermes plugins install <name>`` admits silently.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import hermes_yaml as yaml

from hermes_cli import plugin_catalog
from hermes_cli.plugin_packs import install_pack_plugins, parse_pack, resolve_pack_plugins

SHA_A = "a" * 40
CATALOG_REPO = "https://github.com/o/r"


class FakeConsole:
    """Minimal Rich-console stand-in recording printed lines."""

    def __init__(self):
        self.lines: list[str] = []

    def print(self, *args, **_kwargs):
        self.lines.append(" ".join(str(a) for a in args))

    def input(self, prompt=""):
        return "y"

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def _catalog_entry():
    return plugin_catalog.PluginCatalogEntry(
        name="caution-plugin", repo=CATALOG_REPO, sha=SHA_A, description="fixture", maintainer="fixture")


def test_catalog_pack_entry_keeps_the_reviewed_pin(monkeypatch):
    """A bare ``name:`` entry resolved through the catalog carries the catalog's reviewed sha."""
    monkeypatch.setattr(plugin_catalog, "load_catalog_live", lambda: [_catalog_entry()])
    pack = parse_pack(yaml.safe_dump({"name": "p", "plugins": [{"name": "caution-plugin", "ref": SHA_A}]}))

    (resolved,) = resolve_pack_plugins(pack)

    assert resolved.identifier == CATALOG_REPO
    assert resolved.catalog_sha == SHA_A


def test_pack_fan_out_passes_a_scan_decision_and_the_reviewed_pin(monkeypatch):
    """Every fan-out install gets the interactive decision and, for catalog rows, the reviewed pin."""
    from hermes_cli import plugins_cmd

    pack = parse_pack(
        yaml.safe_dump(
            {
                "name": "p",
                "plugins": [
                    {"repo": "o/a", "ref": SHA_A},
                    {"name": "caution-plugin", "ref": SHA_A},
                ],
            }
        )
    )
    monkeypatch.setattr(plugin_catalog, "load_catalog_live", lambda: [_catalog_entry()])
    resolved = resolve_pack_plugins(pack)

    installer_calls: list[dict] = []

    def installer(identifier, **kwargs):
        installer_calls.append({"identifier": identifier, **kwargs})
        return (Path("/tmp/x"), {"name": "x"}, "x")

    monkeypatch.setattr(plugins_cmd, "_install_plugin_core", installer)
    monkeypatch.setattr(plugins_cmd, "_prompt_plugin_env_vars", lambda *a, **k: None)
    monkeypatch.setattr(plugins_cmd, "_set_plugin_enabled", lambda *a, **k: None)
    monkeypatch.setattr(plugins_cmd, "_declared_capabilities_from_manifest", lambda *a, **k: [])
    monkeypatch.setattr(plugins_cmd, "_run_capability_consent", lambda *a, **k: True)

    results = install_pack_plugins(pack, resolved, FakeConsole())

    assert [r.ok for r in results] == [True, True]
    # A repo entry is a custom source with no reviewed pin; the catalog row carries the catalog sha.
    assert [call["reviewed_pin"] for call in installer_calls] == [None, SHA_A]
    assert all(callable(call["scan_decision_cb"]) for call in installer_calls), installer_calls


# ---------------------------------------------------------------------------
# End to end: a caution-verdict plugin installs through a pack once confirmed.
# ---------------------------------------------------------------------------

def _git_plugin(tmp_path) -> tuple[str, str]:
    """A pinned local plugin repo whose only finding is a caution-level ``eval``."""
    repo = tmp_path / "caution-plugin"
    repo.mkdir()
    (repo / "plugin.yaml").write_text(
        "name: caution-plugin\nversion: 1.0.0\nmanifest_version: 1\n", encoding="utf-8")
    (repo / "__init__.py").write_text("def register(ctx):\n    pass\n", encoding="utf-8")
    (repo / "helper.py").write_text("eval('1 + 1')\n", encoding="utf-8")
    identity = {
        "GIT_AUTHOR_NAME": "fixture", "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
        "GIT_COMMITTER_NAME": "fixture", "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
    }
    env = {**os.environ, **identity}
    for argv in (["git", "init", "-qb", "main"], ["git", "add", "--all"], ["git", "commit", "-qm", "v1"]):
        subprocess.run(argv, cwd=repo, env=env, check=True, capture_output=True)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    return f"file://{repo}", sha


@pytest.fixture
def interactive(monkeypatch, tmp_path):
    """A TTY session that answers "y" to every question and records what was asked."""
    from hermes_cli import plugins_cmd as pc

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(pc, "_scan_on_install_enabled", lambda: True)
    monkeypatch.setattr(pc, "_is_tty", lambda: True)
    console = FakeConsole()
    monkeypatch.setattr(pc, "_console", lambda: console)
    asked: list[str] = []
    monkeypatch.setattr(pc, "_ask_yes", lambda prompt, reader=None: asked.append(prompt) or True)
    return asked, console


def test_pack_install_asks_before_accepting_a_caution_verdict(tmp_path, interactive):
    """Same plugin, same commit, through a pack: the confirmed pack asks about the caution verdict.

    The publication step after the scan needs PM runtime assets that the test fence blocks, so this
    stops at the scan decision: the findings are shown, the question is asked, and the tree is not
    refused. On ``main`` the pack printed "Security scan blocked ... Requires confirmation" and asked
    nothing.
    """
    from hermes_cli.plugin_packs import cmd_pack_install

    asked, console = interactive
    source, sha = _git_plugin(tmp_path)
    pack = tmp_path / "hermes-pack.yaml"
    pack.write_text(
        yaml.safe_dump({"name": "demo-pack", "plugins": [{"repo": source, "ref": sha}]}), encoding="utf-8")

    with pytest.raises(SystemExit):
        cmd_pack_install(str(pack))

    assert any("Install anyway?" in prompt for prompt in asked), asked
    assert "Security scan flagged this plugin" in console.text, console.text
    assert "Security scan blocked" not in console.text, console.text
