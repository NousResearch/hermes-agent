"""hermes provider CLI commands — list and switch provider presets."""
from __future__ import annotations
import os
import sys
from pathlib import Path


def cmd_provider(args):
    from hermes_cli.runtime_provider import (
        list_provider_presets,
        get_provider_preset,
        get_default_preset_name,
    )
    action = getattr(args, "provider_action", None)

    if action == "list" or action is None:
        presets = list_provider_presets()
        default_name = get_default_preset_name()
        if not presets:
            print("No provider presets configured.")
            print("Add a 'providers' section to ~/.hermes/config.yaml")
            print("")
            print("Example:")
            print("  providers:")
            print("    local:")
            print("      type: openai-compatible")
            print("      model: meta-llama/Llama-3.3-70B-Instruct")
            print("      base_url: http://localhost:8000/v1")
            print("      api_key: dummy")
            print("  default_provider: local")
            return
        print("Provider presets:")
        for name, cfg in presets.items():
            ptype = cfg.get("type", "openai-compatible")
            pmodel = cfg.get("model", "")
            pbase = cfg.get("base_url", "")
            marker = " [default]" if name == default_name else ""
            print("  {}{}".format(name, marker))
            print("    type:  {}".format(ptype))
            if pmodel:
                print("    model: {}".format(pmodel))
            if pbase:
                print("    url:   {}".format(pbase))

    elif action == "set":
        name = args.name
        from hermes_cli.runtime_provider import get_provider_preset
        preset = get_provider_preset(name)
        if preset is None:
            presets = list_provider_presets()
            available = ", ".join(presets.keys()) if presets else "none"
            print("Unknown provider preset: {}".format(name))
            print("Available: {}".format(available))
            sys.exit(1)
        ptype = preset.get("type", "openai-compatible")
        pmodel = preset.get("model", "")
        pbase_url = preset.get("base_url", "")
        papi_key = preset.get("api_key", "")
        # Apply to env
        if pmodel:
            os.environ["HERMES_MODEL"] = pmodel
        if papi_key:
            os.environ["OPENAI_API_KEY"] = papi_key
        if pbase_url:
            os.environ["OPENAI_BASE_URL"] = pbase_url
        else:
            os.environ.pop("OPENAI_BASE_URL", None)
        os.environ["HERMES_INFERENCE_PROVIDER"] = ptype
        # Persist to config
        try:
            import yaml
            hermes_home = Path(os.getenv("HERMES_HOME", str(Path.home() / ".hermes")))
            config_path = hermes_home / "config.yaml"
            user_config = {}
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    user_config = yaml.safe_load(f) or {}
            if "model" not in user_config or not isinstance(user_config["model"], dict):
                user_config["model"] = {}
            if pmodel:
                user_config["model"]["default"] = pmodel
            user_config["model"]["provider"] = ptype
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(user_config, f, default_flow_style=False, sort_keys=False)
            print("Switched to provider preset: {} ({} / {})".format(name, ptype, pmodel))
            print("Saved to config.")
        except Exception as e:
            print("Switched to preset {} (session only — could not save: {})".format(name, e))

    elif action == "show":
        name = args.name
        preset = get_provider_preset(name)
        if preset is None:
            print("Unknown provider preset: {}".format(name))
            sys.exit(1)
        print("Preset: {}".format(name))
        for k, v in preset.items():
            if k == "api_key" and v:
                print("  {}: {}...".format(k, str(v)[:8]))
            else:
                print("  {}: {}".format(k, v))
    else:
        print("Usage: hermes provider [list|set <name>|show <name>]")
