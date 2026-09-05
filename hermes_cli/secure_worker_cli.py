"""CLI wiring for deterministic secure remote-worker controls."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from hermes_cli.secure_worker import (
    PackPolicy,
    SecurityBoundaryError,
    WorkerImageLock,
    audit_profile_boundary,
    build_context_pack,
    destroy_context_pack,
    render_local_safe_profile,
    render_ox_profile,
    verify_proposed_diff,
    verify_admission_receipt,
    write_admission_receipt,
)


def register_cli(parent: argparse.ArgumentParser) -> None:
    sub = parent.add_subparsers(dest="secure_worker_action")

    audit = sub.add_parser("audit", help="Audit a profile boundary without changing it")
    audit.add_argument("--config", required=True)
    audit.add_argument("--pack")
    audit.add_argument("--manifest")
    audit.add_argument("--policy")
    audit.add_argument("--attestation")
    audit.add_argument("--image-lock", required=True)
    audit.add_argument("--receipt")
    audit.set_defaults(func=cmd_audit)

    run = sub.add_parser("run", help="Re-audit and launch from an admission receipt")
    run.add_argument("--config", required=True)
    run.add_argument("--pack", required=True)
    run.add_argument("--manifest", required=True)
    run.add_argument("--policy", required=True)
    run.add_argument("--attestation", required=True)
    run.add_argument("--image-lock", required=True)
    run.add_argument("--receipt", required=True)
    run.add_argument("hermes_args", nargs=argparse.REMAINDER)
    run.set_defaults(func=cmd_run)

    pack = sub.add_parser("pack", help="Build a sanitized Git workspace")
    pack.add_argument("--source", required=True)
    pack.add_argument("--file", action="append", required=True, dest="files")
    pack.add_argument("--pack", required=True, dest="pack_root")
    pack.add_argument("--manifest", required=True)
    pack.add_argument("--policy", required=True)
    pack.set_defaults(func=cmd_pack)

    verify = sub.add_parser("verify", help="Verify a proposal diff against its manifest")
    verify.add_argument("--pack", required=True, dest="pack_root")
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--policy", required=True)
    verify.set_defaults(func=cmd_verify)

    render = sub.add_parser("profile-render", help="Render a fail-closed profile")
    render.add_argument("--kind", choices=("local-safe", "ox-sanitized"), required=True)
    render.add_argument("--cwd", required=True)
    render.add_argument("--output", required=True)
    render.add_argument("--model", default="hermes-qwen3-fast")
    render.add_argument("--worker-image", required=True)
    render.add_argument("--broker-python")
    render.add_argument("--staging-owner")
    render.add_argument("--staging-repo")
    render.set_defaults(func=cmd_profile_render)

    attest = sub.add_parser("attest", help="Record a short-lived Privacy Mode attestation")
    attest.add_argument("--output", required=True)
    attest.add_argument("--provider", default="nous", choices=("nous",))
    attest.add_argument("--ttl-minutes", type=int, default=60)
    attest.add_argument("--confirm-privacy-mode", action="store_true")
    attest.set_defaults(func=cmd_attest)

    destroy = sub.add_parser("destroy", help="Destroy a manifest-bound disposable pack")
    destroy.add_argument("--pack", required=True, dest="pack_root")
    destroy.add_argument("--manifest", required=True)
    destroy.add_argument("--receipt", required=True)
    destroy.add_argument("--quarantine", required=True)
    destroy.set_defaults(func=cmd_destroy)

    def _show_help(_args: argparse.Namespace) -> int:
        parent.print_help()
        return 1

    parent.set_defaults(func=_show_help)


def _load_yaml(path: str) -> dict[str, object]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SecurityBoundaryError("configuration must be a YAML mapping")
    return value


def _docker_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _audit_from_args(args: argparse.Namespace):
    if args.pack:
        pack = Path(args.pack).resolve()
        for label, raw in (
            ("runtime config", args.config),
            ("policy", args.policy),
            ("privacy attestation", args.attestation),
            ("worker image lock", args.image_lock),
        ):
            if raw:
                path = Path(raw).resolve()
                if path == pack or pack in path.parents or Path(raw).is_symlink():
                    raise SecurityBoundaryError(
                        f"{label} must remain outside the model-writable pack"
                    )
    config = _load_yaml(args.config)
    value = None
    if args.attestation:
        value = json.loads(Path(args.attestation).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise SecurityBoundaryError("attestation must be a JSON object")
    return config, audit_profile_boundary(
        config,
        pack_root=args.pack,
        manifest_path=args.manifest,
        policy=PackPolicy.from_json(args.policy) if args.policy else None,
        privacy_attestation=value,
        docker_available=_docker_available(),
        admitted_worker_image=WorkerImageLock.from_json(args.image_lock).image,
    )


def cmd_audit(args: argparse.Namespace) -> int:
    try:
        _config, report = _audit_from_args(args)
    except (OSError, ValueError, yaml.YAMLError, SecurityBoundaryError) as exc:
        print(f"DENY: {exc}")
        return 1
    if not report.allowed:
        for reason in report.reasons:
            print(f"DENY: {reason}")
        return 1
    try:
        if report.policy_sha256 and not args.receipt:
            raise SecurityBoundaryError("remote audit requires an external admission receipt")
        if args.receipt:
            write_admission_receipt(report, args.receipt, pack_root=args.pack)
    except (OSError, ValueError, SecurityBoundaryError) as exc:
        print(f"DENY: {exc}")
        return 1
    print("ALLOW: profile boundary preflight passed")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    denied_flags = {
        "--profile",
        "-p",
        "--provider",
        "--model",
        "-m",
        "--toolsets",
        "-t",
        "--yolo",
        "--ignore-user-config",
        "--safe-mode",
    }
    hermes_args = list(args.hermes_args)
    if hermes_args[:1] == ["--"]:
        hermes_args = hermes_args[1:]
    has_override = any(
        item in denied_flags
        or any(item.startswith(flag + "=") for flag in denied_flags if flag.startswith("--"))
        or item.startswith("-m")
        or item.startswith("-t")
        or item.startswith("-p")
        for item in hermes_args
    )
    if not hermes_args or has_override:
        print("DENY: run requires fixed-profile Hermes arguments without runtime overrides")
        return 1
    try:
        config, report = _audit_from_args(args)
        if not report.allowed:
            raise SecurityBoundaryError("effective runtime no longer passes admission")
        verify_admission_receipt(report, args.receipt, pack_root=args.pack)
        config_path = Path(args.config).resolve()
        pack = Path(args.pack).resolve()
        if config_path == pack or pack in config_path.parents or config_path.is_symlink():
            raise SecurityBoundaryError("runtime config must remain outside the model-writable pack")
        with tempfile.TemporaryDirectory(prefix="hermes-secure-worker-") as home:
            target = Path(home) / "config.yaml"
            target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            env = {key: value for key, value in os.environ.items() if not key.startswith("HERMES_")}
            for key in (
                "HERMES_STAGING_GITHUB_TOKEN",
                "HERMES_STAGING_GITHUB_OWNER",
                "HERMES_STAGING_GITHUB_REPO",
            ):
                if key in os.environ:
                    env[key] = os.environ[key]
            env["HERMES_HOME"] = home
            executable = str(Path(sys.executable).resolve())
            return subprocess.run(
                [
                    executable,
                    "-m",
                    "hermes_cli.main",
                    "--ignore-rules",
                    "chat",
                    *hermes_args,
                ],
                env=env,
                check=False,
            ).returncode
    except (OSError, ValueError, yaml.YAMLError, SecurityBoundaryError) as exc:
        print(f"DENY: {exc}")
        return 1


def cmd_pack(args: argparse.Namespace) -> int:
    try:
        policy = PackPolicy.from_json(args.policy)
        manifest = build_context_pack(
            args.source, args.files, args.pack_root, args.manifest, policy
        )
    except (OSError, ValueError, SecurityBoundaryError) as exc:
        print(f"DENY: {exc}")
        return 1
    print(f"created sanitized pack with {len(manifest.files)} file(s)")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    try:
        policy = PackPolicy.from_json(args.policy)
        receipt = verify_proposed_diff(args.pack_root, args.manifest, policy)
    except (OSError, ValueError, SecurityBoundaryError) as exc:
        print(f"DENY: {exc}")
        return 1
    print(json.dumps(asdict(receipt), sort_keys=True))
    return 0


def cmd_profile_render(args: argparse.Namespace) -> int:
    output = Path(args.output)
    if output.exists():
        print("DENY: profile destination already exists")
        return 1
    try:
        if args.kind == "local-safe":
            config = render_local_safe_profile(args.cwd, args.model, args.worker_image)
        else:
            if not args.broker_python or not args.staging_owner or not args.staging_repo:
                raise SecurityBoundaryError(
                    "ox-sanitized requires --broker-python, --staging-owner, and --staging-repo"
                )
            config = render_ox_profile(
                args.cwd,
                args.broker_python,
                args.staging_owner,
                args.staging_repo,
                args.worker_image,
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    except (OSError, SecurityBoundaryError) as exc:
        print(f"DENY: {exc}")
        return 1
    print(f"rendered {args.kind} profile at {output}")
    return 0


def cmd_attest(args: argparse.Namespace) -> int:
    output = Path(args.output)
    if output.exists():
        print("DENY: attestation destination already exists")
        return 1
    if not args.confirm_privacy_mode:
        print("DENY: --confirm-privacy-mode is required after checking the provider account")
        return 1
    if args.ttl_minutes < 1 or args.ttl_minutes > 120:
        print("DENY: attestation TTL must be between 1 and 120 minutes")
        return 1
    now = datetime.now(timezone.utc)
    row = {
        "schema": "hermes.secure-worker.privacy-attestation.v1",
        "provider": args.provider,
        "privacy_mode": True,
        "attested_at": now.isoformat(),
        "expires_at": (now + timedelta(minutes=args.ttl_minutes)).isoformat(),
    }
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        print(f"DENY: unable to write attestation: {exc}")
        return 1
    print(f"recorded {args.provider} Privacy Mode attestation")
    return 0


def cmd_destroy(args: argparse.Namespace) -> int:
    try:
        destroy_context_pack(
            args.pack_root,
            args.manifest,
            args.receipt,
            quarantine_root=args.quarantine,
        )
    except (OSError, ValueError, SecurityBoundaryError) as exc:
        print(f"DENY: {exc}")
        return 1
    print("destroyed manifest-bound secure-worker pack")
    return 0
