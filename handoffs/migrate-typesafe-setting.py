"""Copy a legacy TypeSafe model override to its safe plugin setting key.

Run from this checkout's managed environment with --home <explicit profile home>.
The old key is retained; managed-key refusals and the config writer stay authoritative.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_cli.config import load_config_readonly
from hermes_cli.plugins_state import _plugin_settings_entry, _plugin_relative_segments, save_plugin_setting


def migrate(home: Path) -> bool:
    token = set_hermes_home_override(home.resolve())
    try:
        entry = _plugin_settings_entry(load_config_readonly(), "typesafe-skill-router") or {}
        canonical = entry.get("settings") or {}
        legacy = entry.get("config") or {}
        if "router_model" in canonical or "router_model" in legacy:
            return False
        value = canonical.get("model", legacy.get("model"))
        if value is None:
            return False
        save_plugin_setting("typesafe-skill-router", _plugin_relative_segments("router_model"), value)
        return True
    finally:
        reset_hermes_home_override(token)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", required=True, type=Path)
    print("Legacy override copied." if migrate(parser.parse_args().home) else "No migration needed.")
