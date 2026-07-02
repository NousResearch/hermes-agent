"""Regression tests for issue #54217.

The WhatsApp adapter's Python side talks to the Node.js bridge over HTTP via
aiohttp, which is not a core dependency. Before the fix `platform.whatsapp` was
missing from LAZY_DEPS (and pyproject had no `whatsapp` extra), so a gateway
with WhatsApp configured crashed with ``ModuleNotFoundError: No module named
'aiohttp'`` and fell back to cron-only mode.
"""

import tomllib
from pathlib import Path

from tools import lazy_deps


def test_whatsapp_feature_registered_in_lazy_deps():
    assert "platform.whatsapp" in lazy_deps.LAZY_DEPS
    specs = lazy_deps.LAZY_DEPS["platform.whatsapp"]
    assert any(s.startswith("aiohttp==") for s in specs), specs


def test_whatsapp_extra_declared_in_pyproject():
    root = Path(__file__).resolve().parents[2]
    data = tomllib.loads((root / "pyproject.toml").read_text())
    extras = data["project"]["optional-dependencies"]
    assert "whatsapp" in extras
    assert any(s.startswith("aiohttp==") for s in extras["whatsapp"])


def test_whatsapp_and_teams_pin_the_same_aiohttp():
    """The lazy WhatsApp aiohttp pin must match the other messaging platforms
    so a CVE bump can't leave WhatsApp on a vulnerable already-installed
    aiohttp."""
    wa = [s for s in lazy_deps.LAZY_DEPS["platform.whatsapp"] if s.startswith("aiohttp==")]
    teams = [s for s in lazy_deps.LAZY_DEPS["platform.teams"] if s.startswith("aiohttp==")]
    assert wa and teams
    assert wa[0] == teams[0]
