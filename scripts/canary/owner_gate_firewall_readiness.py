#!/usr/bin/env python3
"""Install and attest the exact iptables-nft owner-gate metadata boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


_RELEASE_ROOT = str(Path(__file__).resolve(strict=True).parents[2])
if _RELEASE_ROOT not in sys.path:
    sys.path.insert(0, _RELEASE_ROOT)

from scripts.canary import owner_gate_foundation as foundation


READINESS_SCHEMA = "muncho-owner-gate-metadata-firewall-readiness.v1"
EXECUTOR_UID = 29103
CHAINS = (
    "MUNCHO_OWNER_GATE_METADATA_A",
    "MUNCHO_OWNER_GATE_METADATA_B",
)
METADATA = "169.254.169.254/32"
IPTABLES = "/usr/sbin/iptables-nft"
IPTABLES_SAVE = "/usr/sbin/iptables-nft-save"
RULE_SOURCE_LINES = (
    f"-A OUTPUT -d {METADATA} -j @CHAIN@",
    "-A @CHAIN@ -m owner --uid-owner 0 -j RETURN",
    f"-A @CHAIN@ -p udp -m udp --dport 53 -m owner --uid-owner {EXECUTOR_UID} -j RETURN",
    f"-A @CHAIN@ -p tcp -m tcp --dport 53 -m owner --uid-owner {EXECUTOR_UID} -j RETURN",
    f"-A @CHAIN@ -p tcp -m tcp --dport 80 -m owner --uid-owner {EXECUTOR_UID} -j RETURN",
    "-A @CHAIN@ -j REJECT --reject-with icmp-admin-prohibited",
)


def expected_rules(chain: str) -> tuple[str, ...]:
    if chain not in CHAINS:
        raise OwnerGateFirewallError("owner_gate_firewall_chain_invalid")
    return tuple(line.replace("@CHAIN@", chain) for line in RULE_SOURCE_LINES)


def _rule_argv(chain: str) -> tuple[tuple[str, ...], ...]:
    return (
        ("-m", "owner", "--uid-owner", "0", "-j", "RETURN"),
        (
            "-p", "udp", "--dport", "53", "-m", "owner",
            "--uid-owner", str(EXECUTOR_UID), "-j", "RETURN",
        ),
        (
            "-p", "tcp", "--dport", "53", "-m", "owner",
            "--uid-owner", str(EXECUTOR_UID), "-j", "RETURN",
        ),
        (
            "-p", "tcp", "--dport", "80", "-m", "owner",
            "--uid-owner", str(EXECUTOR_UID), "-j", "RETURN",
        ),
        ("-j", "REJECT", "--reject-with", "icmp-admin-prohibited"),
    )


class OwnerGateFirewallError(RuntimeError):
    """Stable iptables-nft firewall readiness error."""


def _source_digest(path: Path, *, expected_uid: int = 0) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        state = os.fstat(descriptor)
        if (
            not stat.S_ISREG(state.st_mode)
            or state.st_uid != expected_uid
            or stat.S_IMODE(state.st_mode) & 0o022
        ):
            raise OwnerGateFirewallError("owner_gate_firewall_source_invalid")
        digest = hashlib.sha256()
        content = bytearray()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
            content.extend(chunk)
        try:
            lines = tuple(content.decode("ascii", errors="strict").splitlines())
        except UnicodeError as exc:
            raise OwnerGateFirewallError("owner_gate_firewall_source_invalid") from None
        if lines != RULE_SOURCE_LINES:
            raise OwnerGateFirewallError("owner_gate_firewall_source_invalid")
        return digest.hexdigest()
    except OSError as exc:
        raise OwnerGateFirewallError("owner_gate_firewall_source_invalid") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _default_status_runner(argv: Sequence[str]) -> int:
    try:
        return subprocess.run(
            tuple(argv),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=15,
            env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LC_ALL": "C.UTF-8"},
        ).returncode
    except (OSError, subprocess.SubprocessError) as exc:
        raise OwnerGateFirewallError("owner_gate_firewall_command_failed") from None


def install_firewall(
    *,
    rules_path: Path,
    runner: Callable[[Sequence[str]], int] = _default_status_runner,
    require_root: bool = True,
    expected_source_uid: int = 0,
) -> str:
    if require_root and os.geteuid() != 0:  # windows-footgun: ok — Debian root boundary
        raise OwnerGateFirewallError("owner_gate_firewall_root_required")
    digest = _source_digest(rules_path, expected_uid=expected_source_uid)
    active: list[str] = []
    for chain in CHAINS:
        chain_status = runner((IPTABLES, "-w", "5", "-S", chain))
        if chain_status == 1:
            # iptables-nft returns rc=2, not rc=1, when `-C` references a
            # jump target that does not exist.  A fresh Debian host therefore
            # has to establish chain existence before probing the OUTPUT jump.
            continue
        if chain_status != 0:
            raise OwnerGateFirewallError("owner_gate_firewall_command_failed")
        status = runner((
            IPTABLES, "-w", "5", "-C", "OUTPUT", "-d", METADATA,
            "-j", chain,
        ))
        if status == 0:
            active.append(chain)
        elif status != 1:
            raise OwnerGateFirewallError("owner_gate_firewall_command_failed")
    if len(active) > 1:
        # This is the safe residue of an interrupted generation switch.  Do not
        # mutate either chain without first attesting the live rules explicitly.
        raise OwnerGateFirewallError("owner_gate_firewall_multiple_active_generations")
    previous = active[0] if active else None
    chain = next(item for item in CHAINS if item != previous)

    create = runner((IPTABLES, "-w", "5", "-N", chain))
    if create not in {0, 1}:
        raise OwnerGateFirewallError("owner_gate_firewall_command_failed")
    if runner((IPTABLES, "-w", "5", "-F", chain)) != 0:
        raise OwnerGateFirewallError("owner_gate_firewall_command_failed")

    # Populate the inactive generation completely before it can receive any
    # traffic.  The final REJECT is therefore present before the OUTPUT jump.
    for rule in _rule_argv(chain):
        if runner((IPTABLES, "-w", "5", "-A", chain, *rule)) != 0:
            raise OwnerGateFirewallError("owner_gate_firewall_command_failed")
    if runner((
        IPTABLES, "-w", "5", "-I", "OUTPUT", "1", "-d", METADATA,
        "-j", chain,
    )) != 0:
        raise OwnerGateFirewallError("owner_gate_firewall_command_failed")

    # The new, fully populated generation is now first in OUTPUT.  Removing
    # the previous generation can fail without reopening metadata access.
    if previous is not None:
        for _ in range(16):
            removed = runner((
                IPTABLES, "-w", "5", "-D", "OUTPUT", "-d", METADATA,
                "-j", previous,
            ))
            if removed == 1:
                break
            if removed != 0:
                raise OwnerGateFirewallError("owner_gate_firewall_command_failed")
        else:
            raise OwnerGateFirewallError("owner_gate_firewall_duplicate_jump")
        if runner((IPTABLES, "-w", "5", "-F", previous)) != 0:
            raise OwnerGateFirewallError("owner_gate_firewall_command_failed")
        if runner((IPTABLES, "-w", "5", "-X", previous)) != 0:
            raise OwnerGateFirewallError("owner_gate_firewall_command_failed")
    return digest


def validate_live_ruleset(output: bytes | str) -> Mapping[str, Any]:
    try:
        text = output.decode("ascii", errors="strict") if isinstance(output, bytes) else output
    except UnicodeError as exc:
        raise OwnerGateFirewallError("owner_gate_firewall_ruleset_invalid") from None
    if not isinstance(text, str) or len(text) > 1024 * 1024:
        raise OwnerGateFirewallError("owner_gate_firewall_ruleset_invalid")
    lines = tuple(line.strip() for line in text.splitlines())
    output_rules = tuple(line for line in lines if line.startswith("-A OUTPUT "))
    chain_rules = {
        chain: tuple(line for line in lines if line.startswith(f"-A {chain} "))
        for chain in CHAINS
    }
    matches: list[str] = []
    for chain in CHAINS:
        managed = expected_rules(chain)
        jump = managed[0]
        other = next(item for item in CHAINS if item != chain)
        if (
            not output_rules
            or output_rules[0] != jump
            or output_rules.count(jump) != 1
            or chain_rules[chain] != managed[1:]
            or chain_rules[other]
        ):
            continue
        # The managed jump must be the first ordered OUTPUT rule.  No later
        # explicit metadata match or additional managed-chain jump is allowed:
        # both are ambiguous residue/bypass surfaces even though the managed
        # chain currently ends in REJECT.
        if any(
            "169.254.169.254" in rule
            or any(f"-j {candidate}" in rule for candidate in CHAINS)
            for rule in output_rules[1:]
        ):
            continue
        matches.append(chain)
    if len(matches) != 1:
        raise OwnerGateFirewallError("owner_gate_firewall_rules_invalid")
    chain = matches[0]
    relevant = expected_rules(chain)
    projection = {
        "backend": "iptables-nft",
        "chain": chain,
        "rules": list(relevant),
        "ordered_output_rules": list(output_rules),
    }
    return {**projection, "projection_sha256": foundation.sha256_json(projection)}


def _default_output_runner(argv: Sequence[str]) -> bytes:
    try:
        completed = subprocess.run(
            tuple(argv),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=15,
            env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LC_ALL": "C.UTF-8"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise OwnerGateFirewallError("owner_gate_firewall_query_failed") from None
    if completed.returncode != 0 or not completed.stdout or len(completed.stdout) > 1024 * 1024:
        raise OwnerGateFirewallError("owner_gate_firewall_query_failed")
    return completed.stdout


def _valid_boot_id(value: str) -> bool:
    parts = value.split("-")
    return [len(part) for part in parts] == [8, 4, 4, 4, 12] and all(
        character in "0123456789abcdef" for part in parts for character in part
    )


def collect_receipt(
    *,
    rules_path: Path,
    boot_id_path: Path = Path("/proc/sys/kernel/random/boot_id"),
    runner: Callable[[Sequence[str]], bytes] = _default_output_runner,
    now_unix: int | None = None,
    require_root: bool = True,
    expected_source_uid: int = 0,
) -> Mapping[str, Any]:
    if require_root and os.geteuid() != 0:  # windows-footgun: ok — Debian root boundary
        raise OwnerGateFirewallError("owner_gate_firewall_root_required")
    rules_sha256 = _source_digest(rules_path, expected_uid=expected_source_uid)
    projection = validate_live_ruleset(runner((IPTABLES_SAVE, "-t", "filter")))
    try:
        boot_id = boot_id_path.read_text(encoding="ascii").strip()
    except OSError as exc:
        raise OwnerGateFirewallError("owner_gate_firewall_boot_id_invalid") from None
    if not _valid_boot_id(boot_id):
        raise OwnerGateFirewallError("owner_gate_firewall_boot_id_invalid")
    observed = int(time.time()) if now_unix is None else now_unix
    if not isinstance(observed, int) or isinstance(observed, bool) or observed <= 0:
        raise OwnerGateFirewallError("owner_gate_firewall_time_invalid")
    unsigned = {
        "schema": READINESS_SCHEMA,
        "backend": "iptables-nft",
        "boot_id": boot_id,
        "rules_source_sha256": rules_sha256,
        "live_projection_sha256": projection["projection_sha256"],
        "executor_uid": EXECUTOR_UID,
        "root_admin_metadata_allowed": True,
        "other_unprivileged_uids_blocked": True,
        "web_uid_blocked": True,
        "authority_uid_blocked": True,
        "observed_at_unix": observed,
        "ready": True,
    }
    return {**unsigned, "receipt_sha256": foundation.sha256_json(unsigned)}


def write_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    if not path.is_absolute() or path.parent != Path("/run/muncho-owner-gate"):
        raise OwnerGateFirewallError("owner_gate_firewall_receipt_path_invalid")
    payload = foundation.canonical_json_bytes(receipt)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, flags, 0o440)
        os.fchown(descriptor, 0, EXECUTOR_UID)
        os.fchmod(descriptor, 0o440)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError as exc:
        raise OwnerGateFirewallError("owner_gate_firewall_receipt_write_failed") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    install = subparsers.add_parser("install")
    install.add_argument("--rules", type=Path, required=True)
    receipt = subparsers.add_parser("receipt")
    receipt.add_argument("--rules", type=Path, required=True)
    receipt.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.operation == "install":
        install_firewall(rules_path=arguments.rules)
        return 0
    value = collect_receipt(rules_path=arguments.rules)
    write_receipt(arguments.receipt, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
