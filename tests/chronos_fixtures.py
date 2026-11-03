"""A hosted-shaped Hermes root whose named profile carries a stale copy of the Chronos keys.

NAS seeds ``cron.chronos.*`` into the root ``config.yaml`` (``/opt/data`` on hosted agents); a
profile cloned or imported from another agent keeps that agent's values in its own config.yaml.
"""

from pathlib import Path

import yaml

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


def _write_config(home: Path, chronos: dict) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        yaml.safe_dump({"cron": {"provider": "chronos", "chronos": chronos}}), encoding="utf-8")


def enter_stale_profile(tmp_path: Path, monkeypatch) -> Path:
    """Build ``<root>/config.yaml`` (``ROOT_CHRONOS``) and ``<root>/profiles/jack`` (``STALE_CHRONOS``),
    then run as that profile (``HERMES_HOME=<root>/profiles/jack``), as a process launched with
    ``-p jack`` does."""
    root = tmp_path / "data"
    profile = root / "profiles" / "jack"
    _write_config(root, ROOT_CHRONOS)
    _write_config(profile, STALE_CHRONOS)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    return profile
