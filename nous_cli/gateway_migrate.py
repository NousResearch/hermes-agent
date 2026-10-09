"""CLI presentation and orchestration for Gateway multiplex migration."""
from __future__ import annotations

import sys

from gateway import migration as gm


def _print(lines: list[str] | tuple[str, ...]) -> None:
    for line in lines:
        print(line)


def _service_label(service: tuple[str, bool]) -> str:
    kind, system = service
    if kind == "systemd":
        return f"systemd ({'system' if system else 'user'})"
    if kind == "s6":
        return "s6 slot"
    return "Windows scheduled task" if kind == "windows" else kind


def _remove_verb(service: tuple[str, bool]) -> str:
    return "park" if service[0] == "s6" else "uninstall"


def _signalled_gateways(plan: gm.MigrationPlan) -> list[str]:
    def _who(p: gm.ProfileGateway, suffix: str = "") -> str:
        where = f"pid {p.pid}" if p.pid else f"supervised by {p.service_label()}"
        return f"{p.name} ({where}{suffix})"

    signalled = [_who(p) for p in plan.standalone_secondaries if p.pid or p.services]
    default = plan.default
    if default.services or default.pid:
        signalled.append(_who(default, ", restarted onto the new flag"))
    return signalled


def format_plan(plan: gm.MigrationPlan, *, dry_run: bool) -> list[str]:
    head = "Migration plan (dry run — nothing changed)" if dry_run else "Migration plan"
    lines = [
        head,
        f"  default home: {plan.default_home}",
        "",
        "  profile      gateway pid   service",
    ]
    for p in plan.profiles:
        lines.append(f"  {p.name:<12} {str(p.pid or '-'):<13} {p.service_label()}")
    if plan.standalone_by_config:
        lines.append(
            "  Standalone by config (gateway.standalone: true), left alone: "
            + ", ".join(plan.standalone_by_config)
        )
        lines.append(
            "    (temporary compatibility shim; remove the key and re-run once the gaps it "
            "covers for you are fixed)"
        )
    lines.append("")
    if plan.already_multiplexed:
        suffix = f" (serving {', '.join(plan.live_served)})" if plan.live_served else " (flag on)"
        lines.append(f"  ✓ The default gateway is already multiplexing{suffix}.")
        return lines
    if plan.interrupted:
        lines.append(
            "  ↻ An earlier migration was interrupted before the default gateway came up "
            f"(flag on, no live multiplexer; manifest {plan.default_home / gm.MANIFEST_NAME}); "
            "this run resumes it."
        )
    elif plan.multiplex_flag_on and plan.standalone_secondaries:
        lines.append(
            "  ↻ Half-migrated host: the flag is on, but the profile(s) below still own a "
            "gateway. This run converges them."
        )

    steps: list[str] = []
    for p in plan.standalone_secondaries:
        service_work = " + ".join(
            f"{_remove_verb(service)} {_service_label(service)}" for service in p.services
        )
        what = " + ".join(
            part for part in (f"stop pid {p.pid}" if p.pid else "", service_work) if part
        )
        steps.append(f"  - {p.name}: {what}")
    if len(plan.profiles) < 2:
        return lines + _plan_tail(plan)
    if not steps:
        lines.append(
            "  No secondary profile runs its own gateway; the only step is turning the flag on:"
        )
    else:
        lines += ["  Steps:", *steps]
    lines.append(
        f"  - default: set gateway.multiplex_profiles: true in "
        f"{plan.default_home / 'config.yaml'}"
    )
    target = gm.planned_target_service(plan)
    lines.append(
        f"  - default: {'restart' if plan.default.has_gateway else 'start'} the gateway"
        + (f" via {target[0]}" if target else " (detached)")
        + f", verify it serves {len(plan.expected_served_names)} profiles"
    )
    lines.append(
        f"  - record the previous state in {plan.default_home / gm.MANIFEST_NAME} "
        "(used to undo a FAILED apply, and to resume this command after a crash)"
    )
    signalled = _signalled_gateways(plan)
    if signalled:
        lines += [
            "",
            "  ⚠ This SIGTERMs running gateway process(es): " + ", ".join(signalled) + ".",
            "    They drain in-flight turns and exit; their profiles are served by the host "
            "gateway afterwards.",
        ]
    return lines + _plan_tail(plan)


def _plan_tail(plan: gm.MigrationPlan) -> list[str]:
    lines: list[str] = []
    if plan.blockers:
        if plan.manifest is None:
            lines += ["", "  ✗ Blockers (fix these first, nothing will be changed):"]
        else:
            lines += [
                "",
                "  ⚠ Blockers found, but a migration is already in progress on this host "
                f"(manifest {plan.default_home / gm.MANIFEST_NAME}).",
                "    This run RESUMES it and converges anyway — refusing would strand the profiles whose",
                "    gateways an earlier attempt already removed. Fix these afterwards:",
            ]
        lines += [f"    • {blocker}" for blocker in plan.blockers]
    if plan.notices:
        lines += ["", "  Notices:"]
        lines += [f"    • {notice}" for notice in plan.notices]
    return lines


def format_update_warning(plan: gm.MigrationPlan, auto_blockers: list[str]) -> list[str]:
    return [
        "⚠ Your profiles each run their own gateway. A single multiplexed gateway is the recommended",
        "  setup, but this install cannot be migrated automatically yet:",
        *[f"    • {blocker}" for blocker in (*plan.blockers, *auto_blockers)],
        f"  After fixing the above, run:  {gm.MIGRATE_COMMAND}",
        "  (`hermes update` will migrate automatically once nothing blocks it.)",
    ]


def cmd_migrate(args) -> None:
    """`hermes gateway migrate [--multiplex] [--dry-run] [--yes]`."""
    reason = gm.host_migration_refusal()
    if reason:
        print(f"✗ {reason}")
        raise SystemExit(1)
    plan = gm.build_migration_plan()
    dry_run = getattr(args, "dry_run", False)
    _print(format_plan(plan, dry_run=dry_run))
    if dry_run or plan.already_multiplexed:
        return
    if plan.manifest is None and (plan.blocked or len(plan.profiles) < 2):
        raise SystemExit(1 if plan.blocked else 0)
    if not getattr(args, "yes", False) and sys.stdin.isatty():
        from nous_cli.legacy import prompt_yes_no

        if not prompt_yes_no("Apply this migration now?", True):
            print("Aborted; nothing changed.")
            return
    print()
    result = gm.apply_migration_result(plan)
    _print(result.lines)
    raise SystemExit(0 if result.succeeded else 1)


def maybe_auto_migrate_after_update() -> None:
    """CLI updater hook for deterministic post-update migration."""
    from gateway.migration_guards import auto_migration_blockers, auto_migration_opted_out

    if gm.host_migration_refusal() is not None or auto_migration_opted_out(gm.default_home()):
        return
    plan = gm.build_migration_plan()
    if plan.already_multiplexed or len(plan.profiles) < 2:
        return
    resuming = plan.manifest is not None
    if not plan.standalone_secondaries and not resuming:
        return
    print()
    blockers = auto_migration_blockers(plan)
    if not resuming and (plan.blocked or blockers):
        _print(format_update_warning(plan, blockers))
        return
    print("→ Migrating per-profile gateways onto one multiplexed default gateway...")
    _print(format_plan(plan, dry_run=False))
    result = gm.apply_migration_result(plan)
    _print(result.lines)
