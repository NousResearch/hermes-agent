"""Tests for reproducible setup: default profile export carries MCP and sync config.

Issue: the default profile's export allow-list omitted ``mcp.json``,
``import-sync.json`` and ``profile.yaml``, so exporting ``default`` produced
an archive that could not recreate the same setup on another machine, while
exporting a named profile did. The fix adds these three files to the allow-list.
Secrets inside them are redacted by the existing scrubber before packing.
"""

from __future__ import annotations

from hermes_cli.profiles import _DEFAULT_EXPORT_INCLUDE_ROOT


def test_default_export_includes_mcp_and_sync_config() -> None:
    """MCP servers and sync agents are part of a portable setup snapshot."""
    for name in ("mcp.json", "import-sync.json", "profile.yaml"):
        assert name in _DEFAULT_EXPORT_INCLUDE_ROOT, f"{name} missing from default export"


def test_default_export_still_excludes_runtime_state() -> None:
    """Adding config files must not pull runtime/install trees into the archive."""
    for name in ("installs", "tools", "cache", "state.db"):
        assert name not in _DEFAULT_EXPORT_INCLUDE_ROOT, f"runtime state {name} leaked into export"
