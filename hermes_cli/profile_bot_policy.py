"""Live per-profile Bot Mode execution gate, independent of presentation metadata."""

from pathlib import Path

import yaml


def read_bot_enabled(profile_dir: Path) -> bool:
    """Legacy/no field defaults on; unknown or invalid authority defaults off."""
    path = profile_dir / "profile.yaml"
    try:
        if not profile_dir.is_dir():
            return False
        try:
            path.lstat()
        except FileNotFoundError:
            return True
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return False
        if "bot" not in data:
            return True
        bot = data["bot"]
        return isinstance(bot, dict) and bot.get("enabled", True) is True
    except (OSError, UnicodeError, yaml.YAMLError):
        return False
