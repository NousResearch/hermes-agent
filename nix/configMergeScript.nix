# nix/configMergeScript.nix — Deep-merge Nix settings into existing config.yaml
#
# Used by the NixOS module activation script and by checks.nix tests.
# Nix keys override; user-added keys (skills, streaming, etc.) are preserved.
{ pkgs }:
pkgs.writeScript "hermes-config-merge" ''
  #!${pkgs.python3.withPackages (ps: [ ps.ruamel-yaml ])}/bin/python3
  import json, sys
  from pathlib import Path
  from ruamel.yaml import YAML

  yaml = YAML(typ="safe", pure=True)
  # Existing configs use YAML 1.1 booleans such as yes and off.
  yaml.version = (1, 1)
  yaml.default_flow_style = False
  yaml.sort_base_mapping_type_on_output = False

  nix_json, config_path = sys.argv[1], Path(sys.argv[2])

  with open(nix_json) as f:
      nix = json.load(f)

  existing = {}
  if config_path.exists():
      with open(config_path) as f:
          existing = yaml.load(f) or {}

  def deep_merge(base, override):
      result = dict(base)
      for k, v in override.items():
          if k in result and isinstance(result[k], dict) and isinstance(v, dict):
              result[k] = deep_merge(result[k], v)
          else:
              result[k] = v
      return result

  merged = deep_merge(existing, nix)
  with open(config_path, "w") as f:
      yaml.dump(merged, f)
''
