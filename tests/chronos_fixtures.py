"""A hosted-shaped Hermes root whose named profile carries a stale copy of the Chronos keys.

NAS seeds ``cron.chronos.*`` into the root ``config.yaml`` (``/opt/data`` on hosted agents); a
profile cloned or imported from another agent keeps that agent's values in its own config.yaml.
"""

import json
from pathlib import Path

ROOT_CHRONOS = {
    "portal_url": "https://portal.nous.test",
    "callback_url": "https://this-agent.agents.nous.test",
    "expected_audience": "agent:this-instance",
    "nas_jwks_url": "https://portal.nous.test/.well-known/jwks.json",
}
STALE_CHRONOS = {
    "portal_url": "https://old-portal.nous.test",
    "callback_url": "https://old-agent.agents.nous.test",
    "expected_audience": "agent:old-instance",
    "nas_jwks_url": "https://old-portal.nous.test/.well-known/jwks.json",
}


def write_chronos_config(home: Path, chronos: dict) -> None:
    home.mkdir(parents=True, exist_ok=True)
    # JSON is valid YAML, and the test environment has no PyYAML.
    (home / "config.yaml").write_text(
        json.dumps({"cron": {"provider": "chronos", "chronos": chronos}}), encoding="utf-8")


def root_and_stale_profile(tmp_path: Path) -> tuple[Path, Path]:
    """``<root>/config.yaml`` with ``ROOT_CHRONOS`` and ``<root>/profiles/jack`` with ``STALE_CHRONOS``."""
    root = tmp_path / "data"
    profile = root / "profiles" / "jack"
    write_chronos_config(root, ROOT_CHRONOS)
    write_chronos_config(profile, STALE_CHRONOS)
    return root, profile


def enter_stale_profile(tmp_path: Path, monkeypatch) -> Path:
    """Run as the stale profile (``HERMES_HOME=<root>/profiles/jack``), as a process launched with
    ``-p jack`` does."""
    _root, profile = root_and_stale_profile(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    return profile
