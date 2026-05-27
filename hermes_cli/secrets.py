"""
Hermes secrets CLI — SSOT governance for runtime secrets.

Commands:
    hermes secrets check       [--target TARGET] [--manifest PATH]
    hermes secrets sync        --target TARGET [--apply] [--dry-run]
    hermes secrets preflight   [--profile PROFILE] [--strict]
    hermes secrets install-systemd [--profile PROFILE] [--strict] [--dry-run]
    hermes secrets provider-check [--profile PROFILE]
    hermes secrets break-glass [--target TARGET] [--list|--revoke|create opts]
    hermes secrets retire      --target TARGET --key KEY

BWS (Bitwarden Secrets Manager) is the canonical SSOT.
Local .env files are generated runtime projections.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from hermes_constants import get_hermes_home, get_default_hermes_root

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolvers (no hardcoded paths)
# ---------------------------------------------------------------------------

DEFAULT_MANIFEST_PATH = os.path.join(get_default_hermes_root(), "secrets-manifest.yaml")
BWS_ENV_FILE = os.path.expanduser("~/.config/bws/env")


def _bws_cli_path() -> str:
    """Resolve BWS CLI path, preferring CARGO_HOME or ~/.cargo/bin."""
    cargo_home = os.environ.get("CARGO_HOME", os.path.expanduser("~/.cargo"))
    path = os.path.join(cargo_home, "bin", "bws")
    if os.path.exists(path):
        return path
    # Fallback: scan PATH
    for p in os.environ.get("PATH", "").split(":"):
        candidate = os.path.join(p, "bws")
        if os.path.exists(candidate):
            return candidate
    return "bws"  # let shell resolve


def _bws_env() -> dict[str, str]:
    """Build env dict for bws subprocess calls."""
    env = os.environ.copy()
    if os.path.exists(BWS_ENV_FILE):
        with open(BWS_ENV_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("export "):
                    line = line[7:]
                if "=" in line:
                    k, v = line.split("=", 1)
                    env[k] = v.strip('"').strip("'")
    return env


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_manifest(manifest_path: str) -> dict[str, Any]:
    """Load the secrets manifest YAML."""
    with open(manifest_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _bws_get_secret(secret_id: str) -> str:
    """Fetch a single secret value from BWS. Returns the raw value."""
    env = _bws_env()
    bws = _bws_cli_path()
    result = subprocess.run(
        [bws, "secret", "get", secret_id],
        capture_output=True, text=True, timeout=30,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"bws secret get failed (rc={result.returncode}): {result.stderr.strip()}")
    data = json.loads(result.stdout)
    return data["value"]


def _bws_list_secrets(project_id: str) -> list[dict[str, Any]]:
    """List all secrets in a BWS project."""
    env = _bws_env()
    bws = _bws_cli_path()
    result = subprocess.run(
        [bws, "secret", "list", project_id],
        capture_output=True, text=True, timeout=30,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"bws secret list failed (rc={result.returncode}): {result.stderr.strip()}")
    return json.loads(result.stdout)


def _fingerprint(value: str) -> str:
    """Return a short sha256 fingerprint of a value (value-free)."""
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def _read_env_keys(env_path: str) -> dict[str, str]:
    """Read key=value pairs from a .env file. Returns {KEY: value}."""
    result = {}
    if not os.path.exists(env_path):
        return result
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                result[key] = value
    return result


def _manifest_hash(manifest_path: str) -> str:
    """SHA256 of the manifest file."""
    with open(manifest_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _load_sidecar(sidecar_path: str) -> dict[str, Any] | None:
    """Load projection sidecar JSON, or None if missing."""
    if not os.path.exists(sidecar_path):
        return None
    with open(sidecar_path, encoding="utf-8") as f:
        return json.load(f)


def _load_profile_config(profile: str | None) -> dict[str, Any]:
    """Load Hermes config.yaml for a given profile."""
    root = get_default_hermes_root()
    if profile and profile != "default":
        config_path = os.path.join(root, "profiles", profile, "config.yaml")
    else:
        config_path = os.path.join(root, "config.yaml")
    if not os.path.exists(config_path):
        return {}
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# hermes secrets sync
# ---------------------------------------------------------------------------

def cmd_secrets_sync(args: argparse.Namespace) -> int:
    """Sync .env projections from BWS according to manifest."""
    manifest_path = args.manifest or DEFAULT_MANIFEST_PATH
    if not os.path.exists(manifest_path):
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    manifest = _load_manifest(manifest_path)
    target_name = args.target
    apply_mode = args.apply
    dry_run = args.dry_run

    target = None
    for t in manifest.get("targets", []):
        if t["name"] == target_name:
            target = t
            break
    if target is None:
        print(f"ERROR: target '{target_name}' not found", file=sys.stderr)
        return 1

    env_path = os.path.expanduser(target["path"])
    current_env = _read_env_keys(env_path)
    bws_project_id = manifest["bws_project"]["id"]

    try:
        bws_secrets = {s["key"]: s["id"] for s in _bws_list_secrets(bws_project_id)}
    except Exception as e:
        print(f"ERROR: BWS unavailable: {e}", file=sys.stderr)
        return 2

    projections: dict[str, str] = {}
    fingerprints: list[dict[str, Any]] = []
    exit_code = 0

    print(f"Target: {target['name']} -> {env_path}")
    mode = "APPLY" if apply_mode else "DRY-RUN" if dry_run else "CHECK"
    print(f"Mode: {mode}")

    for key_def in target.get("keys", []):
        source = key_def.get("source", {})
        if source.get("type") != "bws":
            continue

        canonical = key_def["canonical"]
        projection = key_def["projection"]
        bws_key = source.get("key", canonical)
        bws_id = bws_secrets.get(bws_key)

        if bws_id is None:
            # BWS_MISSING — uppercase enum
            fp_status = "BWS_MISSING"
            print(f"  {projection} ({canonical}): {fp_status}")
            fingerprints.append({
                "canonical": canonical, "projection": projection,
                "fingerprint": None, "status": fp_status,
            })
            exit_code = 1
            continue

        try:
            bws_value = _bws_get_secret(bws_id)
        except Exception as e:
            print(f"  {projection} ({canonical}): BWS_ERROR — {e}")
            fingerprints.append({
                "canonical": canonical, "projection": projection,
                "fingerprint": None, "status": "BWS_ERROR",
            })
            exit_code = 1
            continue

        bws_fp = _fingerprint(bws_value)
        local_fp = _fingerprint(current_env.get(projection, "")) if projection in current_env else None

        if local_fp == bws_fp:
            fp_status = "MATCH"
        elif local_fp is None:
            fp_status = "LOCAL_MISSING"
            exit_code = 1
        else:
            fp_status = "DRIFT"
            exit_code = 1
        print(f"  {projection} ({canonical}): {fp_status}")

        fingerprints.append({
            "canonical": canonical, "projection": projection,
            "fingerprint": bws_fp, "status": fp_status,
        })
        projections[projection] = bws_value

    managed_projections = set(projections.keys())
    unmanaged = {k: v for k, v in current_env.items() if k not in managed_projections}

    if apply_mode and exit_code == 0:
        mh = _manifest_hash(manifest_path)
        now = datetime.now(timezone.utc).isoformat()

        env_dir = os.path.dirname(env_path)
        fd, tmp_path = tempfile.mkstemp(dir=env_dir, prefix=".env.")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for proj_key, value in projections.items():
                    f.write(f"{proj_key}={value}\n")
                for k, v in unmanaged.items():
                    f.write(f"{k}={v}\n")
            os.chmod(tmp_path, 0o600)
            os.rename(tmp_path, env_path)

            sidecar_path = f"{env_path}.hermes-projection.json"
            sidecar = {
                "generated_by": "hermes secrets sync",
                "manifest_sha256": mh,
                "target": target_name,
                "generated_at": now,
                "fresh_until": None,
                "bws_project": manifest["bws_project"]["name"],
                "keys": fingerprints,
                "manual_edits_forbidden": True,
            }
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump(sidecar, f, indent=2)

            print(f"\nApplied {len(projections)} keys to {env_path}")
            print(f"Sidecar: {sidecar_path}")
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    elif dry_run:
        print(f"\nWould apply {len(projections)} keys to {env_path}")

    else:
        matched = sum(1 for fp in fingerprints if fp["status"] == "MATCH")
        drifted = sum(1 for fp in fingerprints if fp["status"] == "DRIFT")
        missing = sum(1 for fp in fingerprints if fp["status"] in ("BWS_MISSING", "LOCAL_MISSING"))
        print(f"\nResult: {matched} MATCH, {drifted} DRIFT, {missing} MISSING")

    return exit_code


# ---------------------------------------------------------------------------
# hermes secrets check (with fingerprint comparison)
# ---------------------------------------------------------------------------

def cmd_secrets_check(args: argparse.Namespace) -> int:
    """Check projection fingerprints against manifest AND sidecar vs live .env."""
    manifest_path = args.manifest or DEFAULT_MANIFEST_PATH
    if not os.path.exists(manifest_path):
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    manifest = _load_manifest(manifest_path)
    targets_to_check = [args.target] if args.target else [t["name"] for t in manifest.get("targets", [])]
    exit_code = 0

    for target_name in targets_to_check:
        target = next((t for t in manifest.get("targets", []) if t["name"] == target_name), None)
        if target is None:
            print(f"WARNING: target '{target_name}' not found")
            exit_code = max(exit_code, 1)
            continue

        env_path = os.path.expanduser(target["path"])
        if not os.path.exists(env_path):
            print(f"FAIL: .env not found: {env_path}")
            exit_code = 2
            continue

        current_env = _read_env_keys(env_path)
        sidecar_path = f"{env_path}.hermes-projection.json"
        sidecar = _load_sidecar(sidecar_path)

        if sidecar is None:
            print(f"{target_name}: WARNING — no projection sidecar")
            exit_code = max(exit_code, 1)
        else:
            mh = _manifest_hash(manifest_path)
            if sidecar.get("manifest_sha256") != mh:
                print(f"{target_name}: FAIL — manifest hash mismatch")
                exit_code = 2
            else:
                print(f"{target_name}: manifest hash OK")

            # Compare sidecar fingerprints with current .env values
            for key_entry in sidecar.get("keys", []):
                proj = key_entry["projection"]
                expected_fp = key_entry.get("fingerprint")
                expected_status = key_entry.get("status", "")

                if expected_status in ("BWS_MISSING", "BWS_ERROR"):
                    print(f"  {proj}: {expected_status} — cannot verify")
                    exit_code = max(exit_code, 1)
                    continue

                if proj not in current_env:
                    print(f"  {proj}: MISSING from .env")
                    exit_code = 2
                    continue

                actual_fp = _fingerprint(current_env[proj])
                if expected_fp and actual_fp != expected_fp:
                    print(f"  {proj}: DRIFT (sidecar_fp={expected_fp}, env_fp={actual_fp})")
                    exit_code = 2
                elif expected_fp:
                    print(f"  {proj}: MATCH")

        print(f"  {target_name}: {len(current_env)} keys present")

    if exit_code == 0:
        print("All checks passed.")
    return exit_code


# ---------------------------------------------------------------------------
# hermes secrets preflight (with fingerprint drift + break-glass)
# ---------------------------------------------------------------------------

def _read_break_glass(target_name: str) -> list[dict[str, Any]]:
    """Read active break-glass markers for a target."""
    marker_dir = os.path.join(get_default_hermes_root(), "break-glass")
    if not os.path.isdir(marker_dir):
        return []
    markers = []
    prefix = target_name.replace("/", "-")
    for fname in sorted(os.listdir(marker_dir)):
        if fname.startswith(prefix) and fname.endswith(".json"):
            with open(os.path.join(marker_dir, fname), encoding="utf-8") as f:
                markers.append(json.load(f))
    return markers


def cmd_secrets_preflight(args: argparse.Namespace) -> int:
    """Runtime preflight check before gateway start."""
    manifest_path = args.manifest or DEFAULT_MANIFEST_PATH
    if not os.path.exists(manifest_path):
        print(f"FATAL: manifest not found: {manifest_path}")
        return 2

    manifest = _load_manifest(manifest_path)
    profile = args.profile
    strict = args.strict

    target_name = f"profile:{profile}" if profile and profile != "default" else "root/default"
    target = next((t for t in manifest.get("targets", []) if t["name"] == target_name), None)
    if target is None:
        print(f"FATAL: target '{target_name}' not found")
        return 2

    env_path = os.path.expanduser(target["path"])
    if not os.path.exists(env_path):
        print(f"FATAL: .env not found: {env_path}")
        return 2

    current_env = _read_env_keys(env_path)
    sidecar_path = f"{env_path}.hermes-projection.json"
    sidecar = _load_sidecar(sidecar_path)
    mh = _manifest_hash(manifest_path)
    exit_code = 0

    # Check break-glass markers
    bg_markers = _read_break_glass(target_name)
    now = datetime.now(timezone.utc)
    active_bg = False
    for bg in bg_markers:
        expires_str = bg.get("expires_at", "")
        if expires_str:
            try:
                expires = datetime.fromisoformat(expires_str)
                if now < expires:
                    active_bg = True
                    print(f"WARNING: active break-glass for {target_name} (reason={bg.get('reason')}, expires={expires_str})")
            except ValueError:
                pass

    # Check sidecar
    if sidecar is None:
        if not active_bg:
            print(f"FATAL: no projection sidecar and no active break-glass")
            return 2
        else:
            print(f"WARNING: no sidecar, but active break-glass allows bypass")
    else:
        if sidecar.get("manifest_sha256") != mh:
            if not active_bg:
                print(f"FATAL: manifest hash mismatch")
                return 2
            else:
                print(f"WARNING: manifest hash mismatch, but active break-glass allows bypass")
        else:
            print(f"OK: sidecar fresh")

        # Check freshness TTL
        generated_at = sidecar.get("generated_at")
        if generated_at:
            try:
                gen_time = datetime.fromisoformat(generated_at)
                age = (now - gen_time).total_seconds()
                if age > 86400:
                    print(f"WARNING: projection is {age/3600:.1f}h old (TTL: 24h)")
                    if strict:
                        exit_code = max(exit_code, 1)
            except ValueError:
                pass

        # Compare fingerprints: sidecar vs current .env
        for key_entry in sidecar.get("keys", []):
            proj = key_entry["projection"]
            expected_fp = key_entry.get("fingerprint")
            expected_status = key_entry.get("status", "")

            if expected_status in ("BWS_MISSING", "BWS_ERROR"):
                continue  # cannot verify

            if proj not in current_env:
                print(f"FATAL: required key missing: {proj} ({key_entry.get('canonical', '?')})")
                exit_code = 2
                continue

            actual_fp = _fingerprint(current_env[proj])
            if expected_fp and actual_fp != expected_fp:
                print(f"FATAL: fingerprint drift: {proj}")
                exit_code = 2

    # Check required keys from manifest
    for key_def in target.get("keys", []):
        if key_def.get("required") and key_def.get("source", {}).get("type") == "bws":
            proj = key_def["projection"]
            if proj not in current_env:
                print(f"FATAL: required key missing: {proj} ({key_def['canonical']})")
                exit_code = 2

    # Check for undeclared secret-like keys
    for k, v in current_env.items():
        declared = any(d["projection"] == k for d in target.get("keys", []))
        if not declared and len(v) > 20 and any(c in k.upper() for c in ["KEY", "TOKEN", "SECRET", "PASSWORD"]):
            print(f"WARNING: undeclared secret-like key: {k}")
            if strict:
                exit_code = max(exit_code, 1)

    if exit_code == 0:
        print(f"OK: preflight passed for {target_name}")
    else:
        print(f"FAIL: preflight failed for {target_name} (exit={exit_code})")
    return exit_code


# ---------------------------------------------------------------------------
# hermes secrets install-systemd (portable)
# ---------------------------------------------------------------------------

def cmd_secrets_install_systemd(args: argparse.Namespace) -> int:
    """Generate systemd drop-in for ExecStartPre preflight."""
    manifest_path = args.manifest or DEFAULT_MANIFEST_PATH
    profile = args.profile or "default"
    strict = args.strict

    if profile == "arisu":
        unit_name = "hermes-gateway-arisu.service"
    else:
        unit_name = "hermes-gateway.service"

    # Portable systemd user dir
    xdg_config = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    override_dir = os.path.join(xdg_config, "systemd", "user", f"{unit_name}.d")
    override_path = os.path.join(override_dir, "preflight.conf")

    python_exe = sys.executable
    strict_flag = " --strict" if strict else ""
    preflight_cmd = (
        f"{python_exe} -m hermes_cli.main secrets preflight "
        f"--profile {profile} --manifest {manifest_path}{strict_flag}"
    )

    content = f"""[Service]
# Generated by: hermes secrets install-systemd
# Phase D — runtime preflight gate before gateway start
ExecStartPre={preflight_cmd}
"""

    if args.dry_run:
        print(f"Would create: {override_path}")
        print(content)
    else:
        os.makedirs(override_dir, exist_ok=True)
        with open(override_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Created: {override_path}")
        print("Run 'systemctl --user daemon-reload' to activate.")

    print(f"\nNote: preflight runs as ExecStartPre — if it fails, the gateway will NOT start.")
    return 0


# ---------------------------------------------------------------------------
# hermes secrets provider-check (real config validation)
# ---------------------------------------------------------------------------

def cmd_secrets_provider_check(args: argparse.Namespace) -> int:
    """Check provider/base_url/key tuple bindings against actual config."""
    manifest_path = args.manifest or DEFAULT_MANIFEST_PATH
    manifest = _load_manifest(manifest_path)
    exit_code = 0

    bindings = manifest.get("provider_bindings", [])
    if not bindings:
        print("No provider bindings declared in manifest.")
        return 0

    for binding in bindings:
        profile = binding["profile"]
        provider = binding["provider"]
        base_url = binding.get("base_url", "")
        cred_source = binding.get("credential_source")

        # Load actual profile config
        config = _load_profile_config(profile)
        model_cfg = config.get("model", {})
        actual_provider = model_cfg.get("provider", "")
        actual_base_url = model_cfg.get("base_url", "")

        print(f"\n{profile}:")
        print(f"  expected: provider={provider}, base_url={base_url}")
        print(f"  actual:   provider={actual_provider}, base_url={actual_base_url}")

        if actual_provider != provider:
            print(f"  FAIL: provider mismatch (expected={provider}, got={actual_provider})")
            exit_code = 1
        elif actual_base_url != base_url:
            print(f"  FAIL: base_url mismatch (expected={base_url}, got={actual_base_url})")
            exit_code = 1
        else:
            print(f"  OK: provider + base_url match")

        if cred_source:
            print(f"  credential: {cred_source}")
        else:
            print(f"  credential: (none declared)")

    # Check alias declarations
    for target in manifest.get("targets", []):
        for key_def in target.get("keys", []):
            if key_def.get("classification") == "secret-alias":
                pb = key_def.get("provider_binding", {})
                print(f"\n  alias: {key_def['projection']} -> {key_def.get('alias_for', '?')} "
                      f"(provider={pb.get('provider')}, base_url={pb.get('base_url')})")

    if exit_code == 0:
        print("\nProvider check passed.")
    return exit_code


# ---------------------------------------------------------------------------
# hermes secrets break-glass (TTL fixed)
# ---------------------------------------------------------------------------

def cmd_secrets_break_glass(args: argparse.Namespace) -> int:
    """Create, list, or revoke break-glass markers."""
    marker_dir = os.path.join(get_default_hermes_root(), "break-glass")
    os.makedirs(marker_dir, exist_ok=True)

    if args.revoke:
        target = args.target or "root/default"
        marker_path = os.path.join(marker_dir, f"{target.replace('/', '-')}.json")
        if os.path.exists(marker_path):
            os.unlink(marker_path)
            print(f"Revoked break-glass for {target}")
        else:
            print(f"No active break-glass for {target}")
        return 0

    if args.list:
        found = False
        for fname in sorted(os.listdir(marker_dir)):
            if fname.endswith(".json"):
                found = True
                with open(os.path.join(marker_dir, fname), encoding="utf-8") as f:
                    data = json.load(f)
                print(f"  {data['target']}: reason={data['reason']}, "
                      f"expires={data.get('expires_at', '?')}, actor={data.get('actor', '?')}")
        if not found:
            print("No active break-glass markers.")
        return 0

    # Create break-glass
    target = args.target or "root/default"
    reason = args.reason or "emergency manual edit"
    ttl = args.ttl or 3600

    now = datetime.now(timezone.utc)
    expires = now + timedelta(seconds=ttl)

    marker = {
        "target": target,
        "reason": reason,
        "actor": os.environ.get("USER", "hermes-cli"),
        "created_at": now.isoformat(),
        "expires_at": expires.isoformat(),
        "ttl_seconds": ttl,
        "scope_keys": args.keys.split(",") if args.keys else [],
        "reconciliation_required": True,
    }

    marker_path = os.path.join(marker_dir, f"{target.replace('/', '-')}.json")
    with open(marker_path, "w", encoding="utf-8") as f:
        json.dump(marker, f, indent=2)

    print(f"Break-glass created for {target}")
    print(f"  Reason: {reason}")
    print(f"  TTL: {ttl}s (expires: {expires.isoformat()})")
    print(f"  Marker: {marker_path}")
    print("WARNING: Remember to reconcile and revoke after use.")
    return 0


# ---------------------------------------------------------------------------
# hermes secrets retire
# ---------------------------------------------------------------------------

def cmd_secrets_retire(args: argparse.Namespace) -> int:
    """Mark a key as retired in the manifest."""
    manifest_path = args.manifest or DEFAULT_MANIFEST_PATH
    manifest = _load_manifest(manifest_path)

    for target in manifest.get("targets", []):
        if target["name"] == args.target:
            for key_def in target["keys"]:
                if key_def["canonical"] == args.key:
                    key_def["status"] = "retired"
                    print(f"Marked {args.key} as retired in {args.target}")
                    with open(manifest_path, "w", encoding="utf-8") as f:
                        yaml.dump(manifest, f, default_flow_style=False)
                    print(f"Manifest updated: {manifest_path}")
                    return 0
    print(f"Key '{args.key}' not found in target '{args.target}'")
    return 1


# ---------------------------------------------------------------------------
# Subparser registration
# ---------------------------------------------------------------------------

def _exit_with(command_func):
    """Wrap a subcommand so argparse dispatch propagates integer exit codes.

    ``hermes_cli.main`` invokes ``args.func(args)`` but does not currently use
    integer return values as process exit codes.  Runtime gates such as
    ``secrets check`` and ``secrets preflight`` must fail closed for shell,
    systemd ExecStartPre, and CI callers, so secrets subcommands convert their
    return codes to ``SystemExit`` at the command boundary.
    """

    def _wrapped(args: argparse.Namespace) -> None:
        result = command_func(args)
        if isinstance(result, int):
            raise SystemExit(result)

    return _wrapped


def register_secrets_subparsers(secrets_parser: argparse.ArgumentParser) -> None:
    """Register all subcommands under 'hermes secrets'."""
    sub = secrets_parser.add_subparsers(dest="secrets_command", required=True)

    # source/provider
    #
    # Source integrations configure where runtime secrets are fetched from.
    # They are deliberately nested below ``source`` so the top-level
    # ``sync/check/preflight`` commands remain the BWS manifest SSOT and
    # operators do not confuse a provider setup wizard with the policy gate.
    source_parser = sub.add_parser(
        "source",
        aliases=["provider"],
        help="Configure external secret sources such as Bitwarden",
    )
    source_sub = source_parser.add_subparsers(
        dest="secrets_source",
        required=True,
    )
    bitwarden_parser = source_sub.add_parser(
        "bitwarden",
        aliases=["bw"],
        help="Bitwarden Secrets Manager integration",
    )
    from hermes_cli import secrets_cli as _secrets_cli
    _secrets_cli.register_cli(bitwarden_parser)

    # sync
    sync_parser = sub.add_parser("sync", help="Sync .env from BWS according to manifest")
    sync_parser.add_argument("--target", required=True)
    sync_parser.add_argument("--apply", action="store_true")
    sync_parser.add_argument("--dry-run", action="store_true")
    sync_parser.add_argument("--manifest", help=f"Path to manifest (default: {DEFAULT_MANIFEST_PATH})")
    sync_parser.set_defaults(func=_exit_with(cmd_secrets_sync))

    # check
    check_parser = sub.add_parser("check", help="Validate fingerprints (sidecar vs live .env)")
    check_parser.add_argument("--target")
    check_parser.add_argument("--manifest", help=f"Path to manifest (default: {DEFAULT_MANIFEST_PATH})")
    check_parser.set_defaults(func=_exit_with(cmd_secrets_check))

    # preflight
    preflight_parser = sub.add_parser("preflight", help="Runtime gate before gateway start")
    preflight_parser.add_argument("--profile", default="default")
    preflight_parser.add_argument("--strict", action="store_true")
    preflight_parser.add_argument("--manifest", help=f"Path to manifest (default: {DEFAULT_MANIFEST_PATH})")
    preflight_parser.set_defaults(func=_exit_with(cmd_secrets_preflight))

    # install-systemd
    install_parser = sub.add_parser("install-systemd", help="Install ExecStartPre preflight into systemd")
    install_parser.add_argument("--profile", default="default")
    install_parser.add_argument("--strict", action="store_true")
    install_parser.add_argument("--dry-run", action="store_true")
    install_parser.add_argument("--manifest", help=f"Path to manifest (default: {DEFAULT_MANIFEST_PATH})")
    install_parser.set_defaults(func=_exit_with(cmd_secrets_install_systemd))

    # provider-check
    prov_parser = sub.add_parser("provider-check", help="Validate provider/base_url/key against live config")
    prov_parser.add_argument("--profile", default="arisu")
    prov_parser.add_argument("--manifest", help=f"Path to manifest (default: {DEFAULT_MANIFEST_PATH})")
    prov_parser.set_defaults(func=_exit_with(cmd_secrets_provider_check))

    # break-glass
    bg_parser = sub.add_parser("break-glass", help="Create/list/revoke break-glass markers")
    bg_parser.add_argument("--target")
    bg_parser.add_argument("--reason")
    bg_parser.add_argument("--ttl", type=int)
    bg_parser.add_argument("--keys")
    bg_parser.add_argument("--list", action="store_true")
    bg_parser.add_argument("--revoke", action="store_true")
    bg_parser.add_argument("--manifest", help=f"Path to manifest (default: {DEFAULT_MANIFEST_PATH})")
    bg_parser.set_defaults(func=_exit_with(cmd_secrets_break_glass))

    # retire
    retire_parser = sub.add_parser("retire", help="Mark a key as retired in manifest")
    retire_parser.add_argument("--target", required=True)
    retire_parser.add_argument("--key", required=True)
    retire_parser.add_argument("--manifest", help=f"Path to manifest (default: {DEFAULT_MANIFEST_PATH})")
    retire_parser.set_defaults(func=_exit_with(cmd_secrets_retire))
