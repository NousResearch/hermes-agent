"""Radio config persistence.

Saves/loads radio settings to ~/.hermes/radio/config.yaml.
Separate from the main hermes config so radio state doesn't
pollute the core config file.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

RADIO_DIR = Path(os.path.expanduser("~/.hermes/radio"))
CONFIG_PATH = RADIO_DIR / "config.yaml"


def _ensure_dir():
    RADIO_DIR.mkdir(parents=True, exist_ok=True)


def load() -> Dict[str, Any]:
    """Load radio config. Returns empty dict if not found."""
    if not CONFIG_PATH.exists():
        return {}
    try:
        with open(CONFIG_PATH) as f:
            data = yaml.safe_load(f) or {}
        return data
    except Exception:
        return {}


def save(config: Dict[str, Any]) -> None:
    """Save radio config to disk."""
    _ensure_dir()
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_decades() -> Set[int]:
    """Get saved active decades."""
    cfg = load()
    decades = cfg.get("decades")
    if decades and isinstance(decades, list):
        return set(decades)
    return {1950, 1960, 1970, 1980, 1990}


def set_decades(decades: Set[int]) -> None:
    cfg = load()
    cfg["decades"] = sorted(decades)
    save(cfg)


def get_moods() -> Set[str]:
    cfg = load()
    moods = cfg.get("moods")
    if moods and isinstance(moods, list):
        return set(moods)
    return {"slow", "fast", "weird"}


def set_moods(moods: Set[str]) -> None:
    cfg = load()
    cfg["moods"] = sorted(moods)
    save(cfg)


def get_mic_breaks() -> bool:
    return load().get("mic_breaks", True)


def set_mic_breaks(enabled: bool) -> None:
    cfg = load()
    cfg["mic_breaks"] = enabled
    save(cfg)


def get_volume() -> int:
    return load().get("volume", 80)


def set_volume(vol: int) -> None:
    cfg = load()
    cfg["volume"] = vol
    save(cfg)


def get_visualizer() -> str:
    return load().get("visualizer", "braille")


def set_visualizer(name: str) -> None:
    cfg = load()
    cfg["visualizer"] = name
    save(cfg)


def get_presets() -> Dict[str, Dict[str, Any]]:
    return load().get("presets", {})


def save_preset(name: str, preset: Dict[str, Any]) -> None:
    cfg = load()
    presets = cfg.setdefault("presets", {})
    presets[name] = preset
    save(cfg)


def delete_preset(name: str) -> bool:
    cfg = load()
    presets = cfg.get("presets", {})
    if name in presets:
        del presets[name]
        save(cfg)
        return True
    return False


def get_country_weights() -> Dict[str, float]:
    return load().get("country_weights", {})


def set_country_weights(weights: Dict[str, float]) -> None:
    cfg = load()
    cfg["country_weights"] = weights
    save(cfg)


def get_mood_weights() -> Dict[str, float]:
    return load().get("mood_weights", {})


def get_decade_weights() -> Dict[int, float]:
    return load().get("decade_weights", {})


def get_recent_stations() -> List[Dict[str, Any]]:
    """Get recently listened stations (most recent first, max 10)."""
    return load().get("recent_stations", [])


def add_recent_station(name: str, url: str, source: str = "stream") -> None:
    """Add a station to recently listened. Deduplicates by URL."""
    cfg = load()
    recent = cfg.get("recent_stations", [])

    # Remove existing entry with same URL
    recent = [s for s in recent if s.get("url") != url]

    # Add to front
    recent.insert(0, {"name": name, "url": url, "source": source})

    # Keep max 10
    cfg["recent_stations"] = recent[:10]
    save(cfg)
