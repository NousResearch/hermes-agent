# nix/configMergeScript.nix — Deep-merge Nix settings into existing config.yaml
#
# Used by the NixOS module activation script and by checks.nix tests.
# Nix keys override; user-added keys (skills, streaming, etc.) are preserved.
{ pkgs }:
pkgs.writeScript "hermes-config-merge" ''
  #!${pkgs.python3.withPackages (ps: [ ps.pyyaml ])}/bin/python3
  import grp, json, os, pwd, tempfile, yaml, sys
  from pathlib import Path

  nix_json, config_path = sys.argv[1], Path(sys.argv[2])
  owner = group = mode = None
  if len(sys.argv) == 6:
      owner = pwd.getpwnam(sys.argv[3]).pw_uid
      group = grp.getgrnam(sys.argv[4]).gr_gid
      mode = int(sys.argv[5], 8)
  elif len(sys.argv) != 3:
      raise SystemExit("usage: hermes-config-merge NIX_JSON CONFIG_PATH [OWNER GROUP MODE]")

  with open(nix_json) as f:
      nix = json.load(f)

  def reject_symlink(path):
      if path.exists() or path.is_symlink():
          if path.is_symlink():
              raise SystemExit(f"refusing to follow symlink: {path}")

  reject_symlink(config_path.parent)
  reject_symlink(config_path)

  existing = {}
  if config_path.exists():
      with open(config_path) as f:
          existing = yaml.safe_load(f) or {}

  def deep_merge(base, override):
      result = dict(base)
      for k, v in override.items():
          if k in result and isinstance(result[k], dict) and isinstance(v, dict):
              result[k] = deep_merge(result[k], v)
          else:
              result[k] = v
      return result

  merged = deep_merge(existing, nix)
  config_path.parent.mkdir(parents=True, exist_ok=True)
  reject_symlink(config_path.parent)
  fd, tmp_name = tempfile.mkstemp(prefix=".config.yaml.", suffix=".tmp", dir=config_path.parent)
  try:
      if owner is not None:
          os.fchown(fd, owner, group)
          os.fchmod(fd, mode)
      with os.fdopen(fd, "w") as f:
          yaml.dump(merged, f, default_flow_style=False, sort_keys=False)
      os.replace(tmp_name, config_path)
  finally:
      try:
          os.unlink(tmp_name)
      except FileNotFoundError:
          pass
''
