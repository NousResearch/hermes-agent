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


def vm_data_path(home=None):
    """Writable per-profile Omarchy VM images; installed plugin code may be sealed."""
    return effective_home(home) / "plugin-data" / "hermes-realms" / "vm"


# A realm kind selects an implementation, not a routing mode. ``realm`` is the
# labwc compositor that has always existed; ``omarchy-vm`` is a QEMU guest.
KINDS = ("realm", "omarchy-vm")


@dataclass(frozen=True)
class VmConfig:
    """Omarchy VM guest sizing and exposure. Applies at the next guest boot."""

    memory: int = 3072
    network: bool = True
    disk_size: str = "40G"
    omarchy_vm_path: str = ""
    boot_timeout: float = 180

    def __post_init__(self):
        import math
        import re

        if isinstance(self.memory, bool) or not isinstance(self.memory, int):
            raise ValueError("vm.memory must be an integer number of MiB")
        if not (1024 <= self.memory <= 262144):
            raise ValueError("vm.memory must be between 1024 and 262144 MiB")
        if not isinstance(self.network, bool):
            raise ValueError("vm.network must be a YAML boolean")
        if not isinstance(self.disk_size, str) or not re.fullmatch(
            r"[1-9][0-9]{0,3}G", self.disk_size
        ):
            raise ValueError("vm.disk_size must look like 40G")
        if not isinstance(self.omarchy_vm_path, str) or "\0" in self.omarchy_vm_path:
            raise ValueError("vm.omarchy_vm_path must be a NUL-free string")
        if self.omarchy_vm_path and not Path(self.omarchy_vm_path).is_absolute():
            raise ValueError("vm.omarchy_vm_path must be an absolute path")
        if (
            isinstance(self.boot_timeout, bool)
            or not isinstance(self.boot_timeout, (int, float))
            or not math.isfinite(self.boot_timeout)
            or not (10 <= self.boot_timeout <= 1800)
        ):
            raise ValueError("vm.boot_timeout must be 10 to 1800 seconds")


@dataclass(frozen=True)
class Config:
    default_mode: str = "realm"
    default_kind: str = "realm"
    size: str = "1920x1080"
    idle_ttl: float = 1800
    renderer: str = "gles2"
    overlay: bool = True
    cursor_theme: str = "cua.default"
    vm: VmConfig = VmConfig()

    def __post_init__(self):
        import math

        parse_size(self.size)
        if self.default_mode not in ("realm", "host", "ask"):
            raise ValueError("default_mode must be realm, host or ask")
        if self.default_kind not in KINDS:
            raise ValueError("default_kind must be " + " or ".join(KINDS))
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
        # Accept the asdict() round-trip used for a realm's frozen launch spec.
        if isinstance(self.vm, dict):
            object.__setattr__(self, "vm", VmConfig(**self.vm))
        elif not isinstance(self.vm, VmConfig):
            raise ValueError("vm must be a mapping of Omarchy VM settings")

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
