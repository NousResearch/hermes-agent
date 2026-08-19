# nix/configMergeScript.nix — Deep-merge Nix settings into existing config.yaml
#
# Used by the NixOS module activation script and by checks.nix tests.
# Nix keys override; user-added keys (skills, streaming, etc.) are preserved.
{ pkgs }:
pkgs.writeScript "hermes-config-merge" ''
  #!${pkgs.python3.withPackages (ps: [ ps.pyyaml ])}/bin/python3
  import json, os, stat, sys, tempfile, yaml
  from pathlib import Path

  nix_json, config_path = sys.argv[1], Path(sys.argv[2])
  requested_mode = int(sys.argv[3], 8) if len(sys.argv) > 3 else None

  # Match the previous in-place writer's behavior for symlinked configs: write
  # the symlink target rather than replacing the link itself.
  target_path = config_path.resolve() if config_path.is_symlink() else config_path
  existing_stat = None
  try:
      existing_stat = target_path.stat()
  except FileNotFoundError:
      pass

  with open(nix_json) as f:
      nix = json.load(f)

  existing = {}
  if existing_stat is not None:
      with open(target_path) as f:
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

  final_mode = requested_mode
  final_owner = None
  if requested_mode is None:
      if existing_stat is not None:
          # Two-argument callers historically preserved both mode and ownership
          # by rewriting in place. Keep that compatibility while making the
          # write atomic.
          final_mode = stat.S_IMODE(existing_stat.st_mode)
          final_owner = (existing_stat.st_uid, existing_stat.st_gid)
      else:
          # open(path, "w") historically created a missing config with the
          # process umask applied to 0666; mkstemp would otherwise tighten it
          # unexpectedly to 0600 for legacy two-argument callers.
          current_umask = os.umask(0)
          os.umask(current_umask)
          final_mode = 0o666 & ~current_umask

  fd, temporary = tempfile.mkstemp(dir=target_path.parent, prefix=".hermes-config.")
  try:
      with os.fdopen(fd, "w") as f:
          yaml.dump(merged, f, default_flow_style=False, sort_keys=False)
          f.flush()
          os.fsync(f.fileno())
          if final_owner is not None:
              try:
                  os.fchown(f.fileno(), *final_owner)
              except PermissionError:
                  # A non-root two-argument caller cannot restore a different
                  # prior owner; retaining the writer is the only writable result.
                  pass
          if final_mode is not None:
              os.fchmod(f.fileno(), final_mode)
      os.replace(temporary, target_path)
  except BaseException:
      try:
          os.unlink(temporary)
      except FileNotFoundError:
          pass
      raise
''
