"""
REHOBOAM Storage Layer
Directory management, profile I/O, index maintenance.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

BASE_DIR = Path.home() / ".hermes" / "rehoboam"
PROFILES_DIR = BASE_DIR / "profiles"
POPULATIONS_DIR = BASE_DIR / "populations"
SIMULATIONS_DIR = BASE_DIR / "simulations"
MONITORING_DIR = BASE_DIR / "monitoring"
CONFIG_DIR = BASE_DIR / "config"


def init_storage():
    """Create all required directories."""
    for d in [PROFILES_DIR, POPULATIONS_DIR, SIMULATIONS_DIR,
              MONITORING_DIR, MONITORING_DIR / "alerts", CONFIG_DIR,
              BASE_DIR / "db"]:
        d.mkdir(parents=True, exist_ok=True)

    # Create default configs if they don't exist
    staleness_path = CONFIG_DIR / "staleness_policy.json"
    if not staleness_path.exists():
        staleness_path.write_text(json.dumps({
            "thresholds": {
                "fresh": {"max_age_hours": 72},
                "stale": {"max_age_hours": 336},
                "expired": {"max_age_hours": 2160},
                "archived": {"max_age_hours": 8760}
            },
            "per_field_decay": {
                "psychometrics": {"half_life_days": 180},
                "stances": {"half_life_days": 30},
                "posting_patterns": {"half_life_days": 60},
                "relationships": {"half_life_days": 45},
                "influence": {"half_life_days": 90},
                "voice_fingerprint": {"half_life_days": 365}
            },
            "auto_refresh_on_simulation": True,
            "auto_refresh_threshold": "stale"
        }, indent=2))

    config_path = CONFIG_DIR / "rehoboam.json"
    if not config_path.exists():
        config_path.write_text(json.dumps({
            "version": "7.0",
            "default_model": "claude-opus-4-20250514",
            "max_thread_age_days": 30,
            "monitoring_enabled": False,
            "auto_thread": True,
            "auto_profile_update": True
        }, indent=2))

    # Create indexes if they don't exist
    for idx_path in [PROFILES_DIR / "_index.json", POPULATIONS_DIR / "_index.json",
                     SIMULATIONS_DIR / "_index.json"]:
        if not idx_path.exists():
            idx_path.write_text("{}")


def normalize_handle(handle: str) -> str:
    """Normalize a handle to a filesystem-safe directory name."""
    h = handle.lstrip("@").lower().strip()
    # Replace characters that are problematic in filenames
    return h.replace("/", "_").replace("\\", "_")


# -- Profile I/O --

def get_profile_dir(handle: str) -> Path:
    return PROFILES_DIR / normalize_handle(handle)


def profile_exists(handle: str) -> bool:
    return (get_profile_dir(handle) / "profile.json").exists()


def load_profile(handle: str) -> Optional[dict]:
    path = get_profile_dir(handle) / "profile.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def save_profile(handle: str, profile: dict, snapshot: bool = True):
    """Save a profile, optionally snapshotting the old one."""
    pdir = get_profile_dir(handle)
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "history").mkdir(exist_ok=True)
    (pdir / "raw").mkdir(exist_ok=True)
    (pdir / "predictions").mkdir(exist_ok=True)

    profile_path = pdir / "profile.json"

    # Snapshot old profile before overwriting
    if snapshot and profile_path.exists():
        old = json.loads(profile_path.read_text())
        ts = old.get("last_updated", datetime.utcnow().isoformat()).replace(":", "-")
        snapshot_path = pdir / "history" / f"profile_{ts[:10]}.json"
        shutil.copy2(profile_path, snapshot_path)

    profile_path.write_text(json.dumps(profile, indent=2))
    _update_profile_index(handle, profile)


def _update_profile_index(handle: str, profile: dict):
    idx_path = PROFILES_DIR / "_index.json"
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    idx[normalize_handle(handle)] = {
        "platform": profile.get("platform", "x"),
        "last_updated": profile.get("last_updated", ""),
        "staleness": compute_staleness(profile.get("last_updated", "")),
        "has_star_thread": (get_profile_dir(handle) / "star_thread.json").exists(),
        "simulation_count": idx.get(normalize_handle(handle), {}).get("simulation_count", 0),
        "display_name": profile.get("display_name", "")
    }
    idx_path.write_text(json.dumps(idx, indent=2))


# -- Star Thread I/O --

def load_star_thread(handle: str) -> Optional[dict]:
    path = get_profile_dir(handle) / "star_thread.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def save_star_thread(handle: str, thread: dict):
    path = get_profile_dir(handle) / "star_thread.json"
    get_profile_dir(handle).mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(thread, indent=2))
    # Update index to reflect thread existence
    idx_path = PROFILES_DIR / "_index.json"
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
        key = normalize_handle(handle)
        if key in idx:
            idx[key]["has_star_thread"] = True
            idx_path.write_text(json.dumps(idx, indent=2))


# -- Staleness --

def compute_staleness(last_updated: str) -> str:
    """Determine staleness level from a timestamp string."""
    if not last_updated:
        return "expired"
    try:
        dt = datetime.fromisoformat(last_updated.rstrip("Z"))
    except ValueError:
        return "expired"

    age = datetime.utcnow() - dt
    hours = age.total_seconds() / 3600

    policy = _load_staleness_policy()
    thresholds = policy.get("thresholds", {})

    if hours <= thresholds.get("fresh", {}).get("max_age_hours", 72):
        return "fresh"
    elif hours <= thresholds.get("stale", {}).get("max_age_hours", 336):
        return "stale"
    elif hours <= thresholds.get("expired", {}).get("max_age_hours", 2160):
        return "expired"
    else:
        return "archived"


def _load_staleness_policy() -> dict:
    path = CONFIG_DIR / "staleness_policy.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"thresholds": {"fresh": {"max_age_hours": 72}, "stale": {"max_age_hours": 336},
                           "expired": {"max_age_hours": 2160}, "archived": {"max_age_hours": 8760}}}


def needs_thread_recompute(handle: str) -> bool:
    """Check if a star thread needs recomputation."""
    thread = load_star_thread(handle)
    if thread is None:
        return True

    profile = load_profile(handle)
    if profile is None:
        return True

    # Thread is stale if profile was updated after thread was computed
    thread_time = thread.get("based_on_profile_version", "")
    profile_time = profile.get("last_updated", "")
    if thread_time < profile_time:
        return True

    # Thread is stale if older than max_thread_age_days
    config = json.loads((CONFIG_DIR / "rehoboam.json").read_text()) if (CONFIG_DIR / "rehoboam.json").exists() else {}
    max_age = config.get("max_thread_age_days", 30)
    try:
        computed = datetime.fromisoformat(thread.get("computed_at", "").rstrip("Z"))
        if (datetime.utcnow() - computed).days > max_age:
            return True
    except ValueError:
        return True

    return False


# -- Simulation I/O --

def save_simulation(sim_id: str, config: dict, output: dict, analytics: dict, audit: dict):
    sdir = SIMULATIONS_DIR / sim_id
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "config.json").write_text(json.dumps(config, indent=2))
    (sdir / "output.json").write_text(json.dumps(output, indent=2))
    (sdir / "analytics.json").write_text(json.dumps(analytics, indent=2))
    (sdir / "audit.json").write_text(json.dumps(audit, indent=2))

    # Update index
    idx_path = SIMULATIONS_DIR / "_index.json"
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    idx[sim_id] = {
        "created_at": config.get("created_at", datetime.utcnow().isoformat() + "Z"),
        "scenario": config.get("scenario", ""),
        "participant_count": len(config.get("participants", [])),
    }
    idx_path.write_text(json.dumps(idx, indent=2))


# -- Population I/O --

def save_population(group_id: str, definition: dict, aggregate: dict = None):
    pdir = POPULATIONS_DIR / group_id
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "history").mkdir(exist_ok=True)
    (pdir / "definition.json").write_text(json.dumps(definition, indent=2))
    if aggregate:
        (pdir / "aggregate.json").write_text(json.dumps(aggregate, indent=2))

    idx_path = POPULATIONS_DIR / "_index.json"
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    idx[group_id] = {
        "name": definition.get("name", group_id),
        "member_count": len(definition.get("resolved_members", definition.get("explicit_members", []))),
        "last_updated": definition.get("last_updated", "")
    }
    idx_path.write_text(json.dumps(idx, indent=2))


def load_population(group_id: str) -> Optional[dict]:
    path = POPULATIONS_DIR / group_id / "definition.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


# -- Listing --

def list_profiles() -> dict:
    idx_path = PROFILES_DIR / "_index.json"
    return json.loads(idx_path.read_text()) if idx_path.exists() else {}


def list_populations() -> dict:
    idx_path = POPULATIONS_DIR / "_index.json"
    return json.loads(idx_path.read_text()) if idx_path.exists() else {}


def list_simulations() -> dict:
    idx_path = SIMULATIONS_DIR / "_index.json"
    return json.loads(idx_path.read_text()) if idx_path.exists() else {}


if __name__ == "__main__":
    init_storage()
    print(f"Storage initialized at {BASE_DIR}")
