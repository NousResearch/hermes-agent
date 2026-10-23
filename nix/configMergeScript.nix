# nix/configMergeScript.nix — Deep-merge Nix settings into existing config.yaml
#
# Used by the NixOS module activation script and by checks.nix tests.
# Nix keys override; user-added keys (skills, streaming, etc.) are preserved.
{ pkgs }:
pkgs.writeScript "hermes-config-merge" ''
  #!${pkgs.python3.withPackages (ps: [ ps.ruamel-yaml ])}/bin/python3
  import json, os, shutil, sys, tempfile
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

  # Serialize to a sibling temporary file and replace the destination only
  # after the write succeeded: opening the destination with mode "w" first
  # would truncate the active config when a later dump/flush fails.
  tmp_fd, tmp_name = tempfile.mkstemp(
      dir=config_path.parent, prefix=config_path.name + ".", suffix=".tmp"
  )
  try:
      with os.fdopen(tmp_fd, "w") as f:
          yaml.dump(merged, f)
          f.flush()
          os.fsync(f.fileno())
      if config_path.exists():
          shutil.copymode(config_path, tmp_name)
      else:
          umask = os.umask(0)
          os.umask(umask)
          os.chmod(tmp_name, 0o666 & ~umask)
      os.replace(tmp_name, config_path)
  except BaseException:
      try:
          os.unlink(tmp_name)
      except OSError:
          pass
      raise
''
