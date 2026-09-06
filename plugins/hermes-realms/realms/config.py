"""Profile-local realm settings; never change the user's desktop config."""

from dataclasses import dataclass
from pathlib import Path
import os


def effective_home(home=None):
    if home is not None:
        return Path(home).expanduser().resolve()
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home()).resolve()
    except ImportError:
        return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes").resolve()


def driver_path(home=None):
    """Writable per-profile binary location; installed plugin code may be sealed."""
    return effective_home(home) / "plugin-data" / "hermes-realms" / "bin" / "cua-driver"


@dataclass(frozen=True)
class Config:
    default_mode: str = "realm"
    size: str = "1920x1080"
    idle_ttl: float = 1800
    renderer: str = "gles2"
    overlay: bool = True
    cursor_theme: str = "cua.default"

    def __post_init__(self):
        import math

        parse_size(self.size)
        if self.default_mode not in ("realm", "host", "ask"):
            raise ValueError("default_mode must be realm, host or ask")
        if self.renderer not in ("gles2", "pixman"):
            raise ValueError("renderer must be gles2 or pixman")
        if (
            isinstance(self.idle_ttl, bool)
            or not isinstance(self.idle_ttl, (int, float))
            or not math.isfinite(self.idle_ttl)
            or self.idle_ttl <= 0
        ):
            raise ValueError("idle_ttl must be a positive finite number of seconds")
        if not isinstance(self.overlay, bool):
            raise ValueError("overlay must be a YAML boolean")
        if not isinstance(self.cursor_theme, str) or not self.cursor_theme:
            raise ValueError("cursor_theme must be a nonempty string")

    @classmethod
    def load(cls, home):
        import yaml
        from dataclasses import fields
        from hermes_cli.config import load_config_readonly

        try:
            data = load_config_readonly(home=Path(home))
            settings = data.get("plugins", {}).get("realms", {}) or {}
            return cls(
                **{f.name: settings[f.name] for f in fields(cls) if f.name in settings}
            )
        except (TypeError, AttributeError, yaml.YAMLError) as exc:
            raise ValueError("invalid plugins.realms configuration") from exc


def parse_size(size):
    import re

    if not isinstance(size, str) or not re.fullmatch(
        r"[1-9][0-9]{1,4}x[1-9][0-9]{1,4}", size
    ):
        raise ValueError("size must be WIDTHxHEIGHT")
    width, height = map(int, size.split("x"))
    if not (64 <= width <= 8192 and 64 <= height <= 8192):
        raise ValueError("size dimensions must be between 64 and 8192")
    return width, height
