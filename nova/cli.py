"""``python -m nova`` — operator commands for the platform layer.

Deliberately small. This is not a replacement for the runtime's own CLI, which stays
exactly where it is for engineering and debugging; it covers only the platform
operations that have no runtime equivalent: validating a tenant bundle, previewing what
applying it would do, applying it, and reporting what a runtime currently holds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from nova import __version__
from nova.apply import apply_bundle
from nova.audit import AuditLog, NullAuditLog
from nova.errors import NovaError
from nova.observability import configure
from nova.control.auth import PRINCIPALS_FILENAME, ROLES as AUTH_ROLES
from nova.deploy.aws import MODULE_PATH, TFVARS_FILENAME
from nova.runtime import available_runtimes, get_runtime
from nova.spec import load_bundle


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nova", description="NOVA platform operations for an agent runtime."
    )
    parser.add_argument("--version", action="version", version=f"nova {__version__}")
    runtimes = available_runtimes()
    parser.add_argument(
        "--runtime",
        default=runtimes[0] if runtimes else None,
        choices=runtimes or None,
        help=f"runtime adapter to target (available: {', '.join(runtimes)})",
    )
    parser.add_argument(
        "--home", type=Path, default=None, help="runtime home directory (default: auto-detected)"
    )
    parser.add_argument(
        "--log-level",
        default="",
        metavar="LEVEL",
        help="operational logging to stderr: debug, info, warning, error. Off unless set "
             "here or in NOVA_LOG_LEVEL — a CLI printing structured logs over its own "
             "output is hostile to whoever is running it",
    )
    parser.add_argument(
        "--log-format", choices=["json", "text"], default="",
        help="json for a log shipper (default), text for a human watching a terminal",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="load and fully validate a tenant bundle")
    validate.add_argument("bundle", type=Path)
    validate.add_argument("--json", action="store_true", help="print the resolved bundle as JSON")

    bundle_cmd = sub.add_parser(
        "bundle", help="package a tenant bundle for transport to a runtime host"
    )
    bundle_sub = bundle_cmd.add_subparsers(dest="bundle_command", required=True)

    bundle_package = bundle_sub.add_parser(
        "package", help="write a deterministic archive of a validated bundle"
    )
    bundle_package.add_argument("bundle", type=Path)
    bundle_package.add_argument(
        "--out", type=Path, required=True, help="archive to write (.tgz)"
    )
    bundle_package.add_argument("--json", action="store_true")

    bundle_unpack = bundle_sub.add_parser(
        "unpack", help="extract an archive onto a runtime host's bundle directory"
    )
    bundle_unpack.add_argument("archive", type=Path)
    bundle_unpack.add_argument("--into", type=Path, required=True)
    bundle_unpack.add_argument(
        "--replace", action="store_true",
        help=(
            "replace the destination's contents. Required when it is not empty: without "
            "it the archive would merge, leaving behind any file this bundle deleted"
        ),
    )
    bundle_unpack.add_argument("--json", action="store_true")

    plan = sub.add_parser("plan", help="show what applying a bundle would change")
    plan.add_argument("bundle", type=Path)

    apply_cmd = sub.add_parser("apply", help="make the runtime match a bundle")
    apply_cmd.add_argument("bundle", type=Path)
    apply_cmd.add_argument(
        "--include-disabled",
        action="store_true",
        help="also materialize agents whose spec sets enabled: false",
    )

    status = sub.add_parser("status", help="report what the runtime currently holds")
    status.add_argument("--json", action="store_true")

    knowledge = sub.add_parser("knowledge", help="manage the tenant's knowledge corpora")
    knowledge_sub = knowledge.add_subparsers(dest="knowledge_command", required=True)

    ingest_cmd = knowledge_sub.add_parser("ingest", help="index declared sources")
    ingest_cmd.add_argument("bundle", type=Path)
    ingest_cmd.add_argument(
        "--source",
        action="append",
        default=[],
        dest="sources",
        metavar="ID",
        help="only this source (repeatable); default: every declared source",
    )
    ingest_cmd.add_argument(
        "--force",
        action="store_true",
        help="re-chunk every document, including those whose content is unchanged",
    )
    ingest_cmd.add_argument(
        "--dry-run",
        action="store_true",
        help="walk, extract and chunk without writing the index",
    )

    knowledge_status = knowledge_sub.add_parser("status", help="what the index currently holds")
    knowledge_status.add_argument("bundle", type=Path)
    knowledge_status.add_argument("--json", action="store_true")

    knowledge_search = knowledge_sub.add_parser(
        "search", help="run a search exactly as an agent would"
    )
    knowledge_search.add_argument("bundle", type=Path)
    knowledge_search.add_argument("query")
    knowledge_search.add_argument(
        "--as-agent",
        default="",
        metavar="ID",
        help="scope to this agent's granted corpora — the check that matters before a rollout",
    )
    knowledge_search.add_argument("--source", action="append", default=[], dest="sources")
    knowledge_search.add_argument("--limit", type=int, default=5)

    doctor = sub.add_parser(
        "doctor", help="what each agent still needs before it can run"
    )
    doctor.add_argument("bundle", type=Path)
    doctor.add_argument("--json", action="store_true")

    objective = sub.add_parser("objective", help="plan, submit and follow business objectives")
    objective_sub = objective.add_subparsers(dest="objective_command", required=True)

    obj_list = objective_sub.add_parser("list", help="declared objectives and how they route")
    obj_list.add_argument("bundle", type=Path)

    obj_plan = objective_sub.add_parser(
        "plan", help="route an objective and show the plan — writes nothing"
    )
    obj_plan.add_argument("bundle", type=Path)
    obj_plan.add_argument("objective")

    obj_submit = objective_sub.add_parser("submit", help="place an objective's steps on the board")
    obj_submit.add_argument("bundle", type=Path)
    obj_submit.add_argument("objective")
    obj_submit.add_argument(
        "--dry-run", action="store_true", help="show what would be created, write nothing"
    )

    obj_status = objective_sub.add_parser("status", help="an objective's state, read from the runtime")
    obj_status.add_argument("bundle", type=Path)
    obj_status.add_argument("objective", nargs="?", default="")
    obj_status.add_argument("--json", action="store_true")

    serve_cmd = sub.add_parser("serve", help="run the read-only Control API and dashboard")
    serve_cmd.add_argument("bundle", type=Path)
    serve_cmd.add_argument("--host", default="127.0.0.1", help="default: loopback only")
    serve_cmd.add_argument("--port", type=int, default=8787)
    serve_cmd.add_argument(
        "--principals",
        type=Path,
        default=None,
        metavar="FILE",
        help="who may call the control plane (default: <home>/control-principals.yaml)",
    )
    serve_cmd.add_argument("--tls-cert", default="", metavar="FILE", help="TLS certificate")
    serve_cmd.add_argument("--tls-key", default="", metavar="FILE", help="TLS private key")
    serve_cmd.add_argument(
        "--behind-tls-proxy",
        action="store_true",
        help="TLS terminates at a proxy in front of this process; required to bind a "
             "non-loopback interface without --tls-cert",
    )

    backup_cmd = sub.add_parser("backup", help="archive the state NOVA owns")
    backup_cmd.add_argument("bundle", type=Path)
    backup_cmd.add_argument("destination", type=Path, help="path for the .tar.gz")
    backup_cmd.add_argument(
        "--include-secrets", action="store_true",
        help="also archive each agent's .env. The archive then CONTAINS CREDENTIALS and "
             "must be handled as one",
    )

    restore_cmd = sub.add_parser("restore", help="restore a NOVA backup into this home")
    restore_cmd.add_argument("archive", type=Path)
    restore_cmd.add_argument("--dry-run", action="store_true")

    reconcile_cmd = sub.add_parser(
        "reconcile", help="diagnose what an interrupted run left unfinished"
    )
    reconcile_cmd.add_argument("bundle", type=Path)
    reconcile_cmd.add_argument("--json", action="store_true")

    audit_cmd = sub.add_parser("audit", help="inspect and protect the audit log")
    audit_sub = audit_cmd.add_subparsers(dest="audit_command", required=True)

    audit_seal = audit_sub.add_parser(
        "seal", help="fingerprint the audit log so later tampering is detectable"
    )
    audit_seal.add_argument(
        "destination", type=Path,
        help="where to write the seal — keep it somewhere the runtime user cannot write, "
             "or it proves nothing",
    )
    audit_verify = audit_sub.add_parser("verify", help="check the log against a seal")
    audit_verify.add_argument("seal", type=Path)
    audit_verify.add_argument("--json", action="store_true")

    audit_status = audit_sub.add_parser("status", help="size, segments and retention")

    channels_cmd = sub.add_parser("channels", help="the customer's connected channels")
    channels_sub = channels_cmd.add_subparsers(dest="channels_command", required=True)

    ch_list = channels_sub.add_parser("list", help="what is connected and who it reaches")
    ch_list.add_argument("bundle", type=Path)
    ch_list.add_argument("--json", action="store_true")

    ch_plan = channels_sub.add_parser("plan", help="what connecting would change, writing nothing")
    ch_plan.add_argument("bundle", type=Path)
    ch_plan.add_argument("--json", action="store_true")

    ch_apply = channels_sub.add_parser("apply", help="make the runtime deliver declared channels")
    ch_apply.add_argument("bundle", type=Path)
    ch_apply.add_argument("--actor", default="", help="who is connecting (default: $USER)")

    ch_providers = channels_sub.add_parser("providers", help="channels NOVA offers, and the evidence")
    ch_providers.add_argument("--json", action="store_true")

    work_cmd = sub.add_parser("work", help="act on work already on the board")
    work_sub = work_cmd.add_subparsers(dest="work_command", required=True)

    for name, helptext in (
        ("release", "let a queued or held item through to run"),
        ("resume", "return a held item to whatever phase it was in"),
    ):
        parser_ = work_sub.add_parser(name, help=helptext)
        parser_.add_argument("bundle", type=Path)
        parser_.add_argument("task_id")
        parser_.add_argument("--reason", default="", help="recorded with the decision")
        parser_.add_argument("--actor", default="", help="who is deciding (default: $USER)")

    work_reject = work_sub.add_parser(
        "reject", help="send an item awaiting review back, with a reason"
    )
    work_reject.add_argument("bundle", type=Path)
    work_reject.add_argument("task_id")
    work_reject.add_argument("--reason", required=True, help="what the worker must change")
    work_reject.add_argument("--actor", default="")

    work_note = work_sub.add_parser("note", help="attach a note a worker will read")
    work_note.add_argument("bundle", type=Path)
    work_note.add_argument("task_id")
    work_note.add_argument("note")
    work_note.add_argument("--actor", default="")

    deploy_cmd = sub.add_parser(
        "deploy", help="render infrastructure input from a bundle (never applies it)"
    )
    deploy_sub = deploy_cmd.add_subparsers(dest="deploy_command", required=True)

    deploy_render = deploy_sub.add_parser(
        "render",
        help="write Terraform variables for the tenant's declared infrastructure",
    )
    deploy_render.add_argument("bundle", type=Path)
    deploy_render.add_argument(
        "--out", type=Path, default=None, metavar="FILE",
        help=f"where to write (default: {MODULE_PATH}/{TFVARS_FILENAME} beside the module)",
    )
    deploy_render.add_argument(
        "--module", type=Path, default=None, metavar="DIR",
        help=f"the Terraform root module (default: ./{MODULE_PATH})",
    )
    deploy_render.add_argument("--json", action="store_true", help="print instead of writing")

    deploy_show = deploy_sub.add_parser(
        "show", help="what the agents would be able to reach, read from the bundle"
    )
    deploy_show.add_argument("bundle", type=Path)
    deploy_show.add_argument("--json", action="store_true")

    token = sub.add_parser("token", help="manage control-plane access")
    token_sub = token.add_subparsers(dest="token_command", required=True)
    token_new = token_sub.add_parser("new", help="mint a token and print the entry to add")
    token_new.add_argument("name", help="who this token is for, as it will appear in logs")
    token_new.add_argument(
        "--role", choices=list(AUTH_ROLES), default="viewer",
        help="viewer: operational state. admin: also policy, decisions and cost",
    )
    token_list = token_sub.add_parser("list", help="who may call the control plane")
    token_list.add_argument("--principals", type=Path, default=None, metavar="FILE")

    return parser


def _report(report, *, verb: str) -> None:
    print(report.summary())
    for label, ids in (("created", report.created), ("changed", report.changed)):
        for agent_id in ids:
            print(f"  {label:9} {agent_id}")
    for agent_id in report.skipped:
        print(f"  {'skipped':9} {agent_id} (disabled in its spec)")
    for warning in report.warnings:
        print(f"  warning:  {warning}")
    if report.dry_run:
        print(f"\nNothing was written. Re-run `nova {verb}` without --dry-run to apply.")


def _backup(args) -> int:
    """``nova backup`` / ``nova restore`` — move the state NOVA owns, not the secrets."""
    from nova.backup import create, restore

    runtime = get_runtime(args.runtime, home=args.home)
    home = runtime.state_location

    if args.command == "restore":
        report = restore(args.archive, home, dry_run=args.dry_run)
        print(report.summary())
        if report.manifest.includes_secrets:
            print("  this archive contained credentials; they have been restored 0600")
        for name in report.skipped:
            print(f"  skipped (unsafe path): {name}")
        for agent, names in sorted(report.missing_env.items()):
            print(
                f"  {agent}: still needs {', '.join(names)} — add them to "
                f"{home / 'profiles' / agent / '.env'}"
            )
        if args.dry_run:
            print("\nNothing was written. Re-run without --dry-run to restore.")
        return 1 if report.missing_env and not args.dry_run else 0

    bundle = load_bundle(args.bundle)
    required = {}
    for spec in bundle.agents:
        row = runtime.deployment_readiness(spec, bundle.deployment)
        if row.get("required"):
            required[spec.id] = list(row["required"])

    if args.include_secrets:
        print(
            "WARNING: --include-secrets means this archive CONTAINS CREDENTIALS.\n"
            "         It is written 0600; treat every copy of it as a secret."
        )
    manifest = create(
        home,
        args.destination,
        tenant_id=bundle.tenant_id,
        agents=[spec.id for spec in bundle.agents],
        required_env=required,
        include_secrets=args.include_secrets,
        never_archive=runtime.never_archive(),
    )
    print(f"backed up {manifest.files} file(s) to {args.destination}")
    if not manifest.includes_secrets:
        names = sorted({n for names in required.values() for n in names})
        print(
            f"  credentials are NOT included. After restoring, re-provision: "
            f"{', '.join(names) or '(none needed)'}"
        )
    print("  not included: the work board, sessions, memories — the runtime's and the customer's")
    return 0


def _reconcile(args) -> int:
    """``nova reconcile`` — what an interrupted run left, and whether it matters."""
    from nova.reconcile import reconcile

    bundle = load_bundle(args.bundle)
    runtime = get_runtime(args.runtime, home=args.home, tenant_id=bundle.tenant_id)
    audit = AuditLog.for_home(
        runtime.state_location, tenant_id=bundle.tenant_id, actor="nova-cli"
    )
    result = reconcile(audit, bundle, runtime)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0 if not result.actionable else 1

    print(result.summary())
    for finding in result.findings:
        mark = "NEEDS APPLY" if finding.needs_action else finding.state
        print(f"\n  [{mark}] {finding.kind} / {finding.subject or '(none)'}")
        print(f"           started {finding.started_at}  correlation {finding.correlation_id}")
        print(f"           {finding.detail}")
    for warning in result.warnings:
        print(f"  warning: {warning}")
    return 1 if result.actionable else 0


def _audit(args) -> int:
    """``nova audit ...`` — keep the governance record finite and its edits visible."""
    from nova.audit import AuditLog
    from nova.audit.integrity import (
        DEFAULT_KEEP, DEFAULT_MAX_BYTES, read_seal, seal, segments, verify, write_seal,
    )

    runtime = get_runtime(args.runtime, home=args.home)
    log_path = AuditLog.for_home(runtime.state_location, tenant_id="").path

    if args.audit_command == "status":
        found = segments(log_path)
        if not found:
            print(f"no audit log at {log_path}")
            return 0
        total = sum(p.stat().st_size for p in found)
        print(f"{log_path}")
        for segment in found:
            print(f"  {segment.name:24} {segment.stat().st_size / 1024:10.1f} KiB")
        print(f"\n  {len(found)} segment(s), {total / 1024 / 1024:.1f} MiB total")
        print(
            f"  rotates past {DEFAULT_MAX_BYTES // 1024 // 1024} MiB, keeping "
            f"{DEFAULT_KEEP} segments — so retention is bounded and OLD EVENTS ARE "
            "DELETED.\n  A deployment with a retention obligation must ship events "
            "off-host before they rotate."
        )
        return 0

    if args.audit_command == "seal":
        if not segments(log_path):
            print(f"no audit log at {log_path}", file=sys.stderr)
            return 1
        written = write_seal(seal(log_path), args.destination)
        document = read_seal(written)
        print(f"sealed {document.lines} event(s) across {len(document.segments)} segment(s)")
        print(f"  -> {written}")
        print(
            "\nThis seal is only worth where you keep it. Beside the log it proves little: "
            "anyone who can rewrite one can rewrite the other. Move it to a host the "
            "runtime user cannot reach."
        )
        return 0

    # verify
    result = verify(log_path, read_seal(args.seal))
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0 if result.ok else 1
    if result.ok:
        print(f"audit log matches the seal ({result.checked} segment(s) checked)")
        if result.appended:
            print(f"  {result.appended} event(s) appended since — expected on a running system")
        for note in result.findings:
            print(f"  note: {note}")
        return 0
    print(f"AUDIT LOG DOES NOT MATCH THE SEAL ({result.checked} segment(s) checked)")
    for finding in result.findings:
        print(f"  {finding}")
    return 1


def _token(args) -> int:
    """``nova token ...`` — mint and inspect control-plane credentials.

    The token is printed **once** and never stored: NOVA writes the digest into the
    principals file and keeps nothing it could later leak. That is the same rule the
    deployment seam follows for model credentials, applied to NOVA's own front door — and
    it means a stolen principals file is an inconvenience rather than an incident.
    """
    from nova.control.auth import PrincipalStore, hash_token, new_token

    runtime = get_runtime(args.runtime, home=args.home)

    if args.token_command == "list":
        path = args.principals or (runtime.state_location / PRINCIPALS_FILENAME)
        store = PrincipalStore.load(path)
        if not store.configured:
            print(f"no principals configured at {path}")
            return 0
        print(f"{path}")
        for entry in store.describe():
            print(f"  {entry['name']:24} {entry['role']}")
        return 0

    secret = new_token()
    path = runtime.state_location / PRINCIPALS_FILENAME
    print(f"token for {args.name} ({args.role}) — shown once, not stored by NOVA:\n")
    print(f"  {secret}\n")
    print(f"Add this entry to {path}:\n")
    print("principals:")
    print(f"  - name: {args.name}")
    print(f"    role: {args.role}")
    print(f"    token_sha256: {hash_token(secret)}")
    print(
        "\nCallers send it as:  Authorization: Bearer <token>\n"
        "Revoke by deleting the entry; no other principal is affected."
    )
    return 0


def _bundle(args) -> int:
    """``nova bundle`` — the transport step, made reproducible.

    Packaging validates first, so an archive never carries declarations that do not parse;
    the archive is rooted at the bundle's contents, so there is no ``--strip-components``
    to forget and no tenant-named directory to strand beside the real one.
    """
    import json as _json

    from nova import package as _package

    if args.bundle_command == "package":
        report = _package.package(args.bundle, args.out)
        if args.json:
            print(_json.dumps(report, indent=2, sort_keys=True))
            return 0
        print(f"{report['tenant_id']}: {report['files']} file(s) -> {report['archive']}")
        print(f"  bundle digest:  {report['bundle_digest']}")
        print(f"  archive sha256: {report['archive_sha256']}")
        print("\nOn the host, extract with `nova bundle unpack <archive> --into "
              "/var/lib/nova/bundle`, then `nova apply /var/lib/nova/bundle`.")
        return 0

    report = _package.unpack(args.archive, args.into, replace=args.replace)
    if args.json:
        print(_json.dumps(report, indent=2, sort_keys=True))
        return 0
    verb = "replaced" if report["replaced"] else "extracted"
    print(f"{verb} {report['files']} file(s) into {report['destination']}")
    # Named, not counted. These are declarations the tenant deleted, and an operator
    # reading a deployment log should be able to see which ones went.
    for name in report["removed"]:
        print(f"  removed (not in this bundle): {name}")
    return 0


def _doctor(args) -> int:
    """``nova doctor`` — deployment readiness, reported and never fixed.

    NOVA writes the name of every credential and none of the values, so "materialized" and
    "able to run" are different states. Exits non-zero when any agent cannot run, so a
    deployment pipeline fails here rather than at the first task.
    """
    bundle = load_bundle(args.bundle)
    runtime = get_runtime(args.runtime, home=args.home, tenant_id=bundle.tenant_id)
    rows = [runtime.deployment_readiness(spec, bundle.deployment) for spec in bundle.agents]

    if args.json:
        print(json.dumps({"tenant_id": bundle.tenant_id, "agents": rows}, indent=2))
        return 0 if all(row.get("ready", True) for row in rows) else 1

    print(f"{bundle.tenant_id}: {len(rows)} agent(s)")
    for spec, row in zip(bundle.agents, rows):
        resolved = row.get("provider") or {}
        mark = "ok " if row.get("ready", True) else "NOT READY"
        print(f"\n  [{mark}] {spec.id}")
        print(f"           provider: {resolved.get('provider') or '(runtime default)'}"
              f"   model: {resolved.get('model') or '(runtime default)'}")
        for name in row.get("required", []):
            where = row.get("resolved_from", {}).get(name)
            print(f"           {name}: {where or 'MISSING'}")
        if not row.get("ready", True):
            print(f"           -> add the missing name(s) to {row.get('env_file')}")
        shared = row.get("host_wide") or []
        if shared and len(rows) > 1:
            # Only worth saying on a multi-agent host: with one agent there is nothing to
            # be isolated from, and a warning that is always true is one nobody reads.
            print(
                f"           note: {', '.join(shared)} resolves from the host environment, "
                "so every agent here shares it"
            )
        for warning in row.get("warnings", []):
            print(f"           warning: {warning}")

    unready = [row for row in rows if not row.get("ready", True)]
    if unready:
        print(
            f"\n{len(unready)} agent(s) cannot run yet. NOVA never writes credentials — "
            "those files are yours and survive every apply."
        )
        return 1
    print("\nevery agent can resolve its credentials")
    return 0


def _objective(args) -> int:
    """``nova objective ...`` — route, submit and follow declared business processes."""
    from nova.supervisor import collect, plan_order, route_objective, submit_objective

    bundle = load_bundle(args.bundle)
    runtime = get_runtime(args.runtime, home=args.home, tenant_id=bundle.tenant_id)

    if not bundle.objectives:
        print(f"{bundle.tenant_id}: no objectives declared (objectives/ is absent or empty)")
        return 0

    def find(objective_id: str):
        for spec in bundle.objectives:
            if spec.id == objective_id:
                return spec
        known = ", ".join(spec.id for spec in bundle.objectives)
        raise NovaError(f"no objective {objective_id!r}; declared: {known}")

    if args.objective_command == "list":
        for spec in bundle.objectives:
            decision = route_objective(spec, bundle.agents)
            state = "ok" if decision.allowed else f"REFUSED ({len(decision.refusals)})"
            flag = "" if spec.enabled else "  (disabled)"
            print(f"  {spec.id:26} {len(spec.steps)} step(s)  owner={spec.owner:18} {state}{flag}")
        return 0

    if args.objective_command == "plan":
        spec = find(args.objective)
        decision = route_objective(spec, bundle.agents)
        print(f"{spec.title}  ({spec.id})")
        print(f"owner: {spec.owner}")
        if spec.acceptance:
            print(f"done when: {spec.acceptance}")
        print()
        # Printed in execution order rather than declaration order: the question a reader
        # has in front of a plan is what runs when, and steps with no dependencies at the
        # top of the list is the answer.
        for step in plan_order(spec):
            routing = next(r for r in decision.steps if r.step_id == step.id)
            mark = "ok " if routing.allowed else "REFUSED"
            waits = f"  after {', '.join(step.depends_on)}" if step.depends_on else "  (starts immediately)"
            print(f"  [{mark}] {step.id:22} -> {step.assignee:18}{waits}")
            if not routing.allowed:
                print(f"           {routing.detail}")
        for warning in decision.warnings:
            print(f"\n  warning: {warning}")
        print()
        print(decision.explain().splitlines()[0])
        return 0 if decision.allowed else 1

    if args.objective_command == "submit":
        spec = find(args.objective)
        audit = (
            NullAuditLog(tenant_id=bundle.tenant_id)
            if args.dry_run
            else AuditLog.for_home(
                runtime.state_location, tenant_id=bundle.tenant_id, actor="nova-cli"
            )
        )
        report = submit_objective(
            spec,
            bundle.agents,
            runtime,
            audit=audit,
            tenant_id=bundle.tenant_id,
            dry_run=args.dry_run,
        )
        print(report.summary())
        for warning in report.warnings:
            print(f"  warning: {warning}")
        if report.refused:
            print("\nNothing was submitted. Fix the routing above, or widen the owner's "
                  "delegation.may_assign_to, then re-run.")
            return 1
        if report.result is not None:
            for item in report.result.items:
                # A dry run has created nothing, and a row that says "created" next to a
                # missing task id invites exactly the wrong reading.
                if args.dry_run:
                    mark = "would add"
                else:
                    mark = "created" if item.created else "existing"
                print(f"  {mark:9} {item.key:44} {item.task_id or '-'}  {item.state or '-'}")
        if args.dry_run:
            print("\nNothing was written. Re-run without --dry-run to submit.")
        return 0

    # status
    selected = [find(args.objective)] if args.objective else list(bundle.objectives)
    tasks = runtime.list_tasks(limit=1000)
    reports = [collect(spec, runtime, tasks=tasks) for spec in selected]

    if args.json:
        print(json.dumps([report.to_dict() for report in reports], indent=2))
        return 0

    for report in reports:
        print(report.summary())
        for step in report.steps:
            state = step.state if step.submitted else "not submitted"
            flag = "  !" if step.needs_attention else ""
            print(f"    {step.step_id:22} {state:14} {step.assignee:18}{flag}")
            if step.last_error:
                print(f"      last error: {step.last_error}")
        for warning in report.warnings:
            print(f"    warning: {warning}")
    return 0


def _knowledge(args) -> int:
    """``nova knowledge ...`` — ingest, inspect and rehearse retrieval."""
    from nova.knowledge import KnowledgeIndex, ingest

    bundle = load_bundle(args.bundle)
    runtime = get_runtime(args.runtime, home=args.home, tenant_id=bundle.tenant_id)
    index_path = runtime.knowledge_index_path

    if args.knowledge_command == "ingest":
        if not bundle.knowledge.sources:
            print(f"{bundle.tenant_id}: no knowledge sources declared (knowledge.yaml is absent)")
            return 0
        audit = (
            NullAuditLog(tenant_id=bundle.tenant_id)
            if args.dry_run
            else AuditLog.for_home(
                runtime.state_location, tenant_id=bundle.tenant_id, actor="nova-cli"
            )
        )
        report = ingest(
            bundle.knowledge,
            index_path,
            extractor=runtime,
            source_ids=args.sources or None,
            audit=audit,
            force=args.force,
            dry_run=args.dry_run,
        )
        print(report.summary())
        for source in report.sources:
            for path, reason in source.skipped:
                print(f"  skipped  {source.source_id}/{path}: {reason}")
            for path in source.removed:
                print(f"  removed  {source.source_id}/{path} (no longer on disk)")
        if args.dry_run:
            print("\nNothing was written. Re-run without --dry-run to index.")
        else:
            print(f"\nindex: {index_path}")
        return 0

    if not index_path.is_file():
        print(
            f"no knowledge index at {index_path}. Run `nova knowledge ingest "
            f"{args.bundle}` first.",
            file=sys.stderr,
        )
        return 1

    with KnowledgeIndex.open(index_path, create=False) as index:
        if args.knowledge_command == "status":
            stats = index.stats()
            if args.json:
                print(json.dumps({"index": str(index_path), "sources": stats}, indent=2))
                return 0
            print(f"index: {index_path}")
            declared = {source.id for source in bundle.knowledge.sources}
            for source_id in sorted(set(stats) | declared):
                row = stats.get(source_id)
                if row is None:
                    print(f"  {source_id:22} declared, never ingested")
                    continue
                extra = "" if source_id in declared else "   (no longer declared)"
                print(
                    f"  {source_id:22} {row['documents']:5} docs  {row['chunks']:6} chunks  "
                    f"{row['bytes'] / 1024:8.1f} KiB{extra}"
                )
            return 0

        # search
        scope = list(args.sources)
        if args.as_agent:
            spec = bundle.agent(args.as_agent)
            granted = list(spec.knowledge.sources)
            # Intersect rather than union: --as-agent is a rehearsal of that agent's real
            # scope, so it must never be able to reach past it.
            scope = [s for s in scope if s in granted] if scope else granted
            if not scope:
                print(f"{args.as_agent} is granted no knowledge sources; nothing to search.")
                return 0
        elif not scope:
            scope = [source.id for source in bundle.knowledge.sources]

        hits = index.search(args.query, source_ids=scope, limit=args.limit)
        print(f"searching {', '.join(scope)} for {args.query!r}")
        if not hits:
            print("no passages matched")
            return 0
        for position, hit in enumerate(hits, start=1):
            print(f"\n[{position}] {hit.doc_title or hit.doc_path} — {hit.citation}")
            print(f"    score {hit.score:.4g}   source {hit.source_id}")
            for line in hit.snippet.splitlines():
                print(f"    {line}")
    return 0


def _channels(args) -> int:
    """``nova channels`` — the customer's channels, without a browser."""
    import getpass

    from nova.audit import new_correlation_id
    from nova.channels.derive import plan_derivations
    from nova.channels.providers import catalogue as _catalogue

    PROVIDERS = _catalogue()

    if args.channels_command == "providers":
        if args.json:
            print(json.dumps([p.to_dict() for p in PROVIDERS], indent=2))
            return 0
        print(f"{'PROVIDER':12} {'TRANSPORT':10} {'EVIDENCE':16} CREDENTIALS")
        for provider in PROVIDERS:
            env = ", ".join(provider.required_env) or "(none)"
            print(f"  {provider.id:10} {provider.transport.value:10} "
                  f"{provider.verification.value:16} {env}")
        print(
            "\nEvidence is how the claim was established. Everything reads 'source_read' "
            "until somebody connects a real provider — a tick that means 'a plugin exists' "
            "is the tick a customer signs a contract on."
        )
        return 0

    bundle = load_bundle(args.bundle)
    runtime = get_runtime(args.runtime, home=args.home, tenant_id=bundle.tenant_id)

    derivations = plan_derivations(bundle)

    if args.channels_command == "list":
        rows = runtime.channel_readiness(bundle.channels, derivations)
        if args.json:
            print(json.dumps(rows, indent=2))
            return 0
        if not bundle.channels:
            print(f"{bundle.tenant_id}: no channels declared — the workforce is reachable "
                  "only through NOVA itself")
            return 0
        by_id = {row["id"]: row for row in rows}
        print(f"{bundle.tenant_id}: connected channels")
        for channel in bundle.channels:
            row = by_id.get(channel.id, {})
            state = "connected" if row.get("ready") else "needs credentials"
            if not channel.enabled:
                state = "disabled"
            print(f"\n  {channel.display_name or channel.id}  ({channel.provider})  —  {state}")
            print(f"    may reach: {', '.join(channel.allowed_agents)}")
            if channel.approval.declared:
                print(f"    needs a human for: {', '.join(channel.approval.required_for)}")
            for route in channel.routes:
                where = route.conversation or route.workspace or "everything else"
                print(f"    {where:28} -> {route.agent}")
            for agent, missing in sorted((row.get("missing_by_agent") or {}).items()):
                if missing:
                    print(f"    {agent} still needs: {', '.join(missing)}")
        return 0

    if args.channels_command == "plan":
        result = runtime.apply_channels(
            bundle.channels, audit=NullAuditLog(), correlation_id=new_correlation_id(),
            derivations=derivations, dry_run=True,
        )
        if args.json:
            print(json.dumps(result, indent=2))
            return 0
        print(f"{bundle.tenant_id}: {len(result['routes'])} route(s) would be compiled")
        for route in result["routes"]:
            where = route.get("chat_id") or route.get("guild_id") or "(catch-all)"
            print(f"  {route['platform']:12} {where:24} -> {route['profile']}")
        print(f"  agents served: {', '.join(result['served_agents']) or '(none)'}")
        for entry in derivations:
            print(
                f"  approval:  {entry.base_agent} on {entry.channel_id} additionally "
                f"escalates {', '.join(entry.added_approvals)} (as {entry.id})"
            )
        for warning in result["warnings"]:
            print(f"  warning:  {warning}")
        print("\nNothing was written. Re-run `nova channels apply` to apply.")
        return 0

    actor = args.actor or (getpass.getuser() if hasattr(getpass, "getuser") else "")
    if not actor:
        print("error: could not determine who is connecting; pass --actor", file=sys.stderr)
        return 1
    audit = AuditLog.for_home(runtime.state_location, tenant_id=bundle.tenant_id, actor=actor)
    result = runtime.apply_channels(
        bundle.channels, audit=audit, correlation_id=new_correlation_id(),
        derivations=derivations,
    )
    print(f"connected {len(bundle.channels)} channel(s), "
          f"{len(result['routes'])} route(s)   (by {actor})")
    for warning in result["warnings"]:
        print(f"  warning:  {warning}")
    print(f"  agents served: {', '.join(result['served_agents']) or '(none)'}")
    print("\nThe gateway reads this at start. Restart it to pick up the change.")
    return 0


#: How the CLI names a decision the operator asked for.
_WORK_ACTIONS = {"release": "release", "resume": "resume", "reject": "reject", "note": "annotate"}


def _work(args) -> int:
    """``nova work`` — the same decisions the control plane offers, without a browser.

    Both paths go through ``runtime.decide_work``, so the audit record is identical and
    there is no "the CLI does it differently" to discover during an incident.
    """
    import getpass

    from nova.audit import new_correlation_id

    bundle = load_bundle(args.bundle)
    runtime = get_runtime(args.runtime, home=args.home, tenant_id=bundle.tenant_id)

    if not runtime.capabilities.work_decisions:
        print(f"error: runtime {runtime.name!r} cannot act on work items", file=sys.stderr)
        return 1

    # Refused rather than defaulted to something anonymous: a decision recorded against
    # "someone" is the exact failure identity exists to prevent.
    actor = args.actor or (getpass.getuser() if hasattr(getpass, "getuser") else "")
    if not actor:
        print("error: could not determine who is deciding; pass --actor", file=sys.stderr)
        return 1

    audit = AuditLog.for_home(runtime.state_location, tenant_id=bundle.tenant_id, actor=actor)
    decision = runtime.decide_work(
        args.task_id,
        _WORK_ACTIONS[args.work_command],
        actor=actor,
        audit=audit,
        correlation_id=new_correlation_id(),
        reason=getattr(args, "reason", "") or "",
        note=getattr(args, "note", "") or "",
    )
    if decision.applied:
        where = f" -> {decision.resulting_status}" if decision.resulting_status else ""
        print(f"{decision.action} {decision.task_id}{where}   (by {actor})")
        return 0
    print(f"{decision.task_id}: {decision.reason}", file=sys.stderr)
    return 1


def _deploy(args) -> int:
    """``nova deploy`` — render, and refuse to apply.

    NOVA writes a file and stops. Running Terraform is the customer's DevOps team using
    their own credentials, which is not a limitation to work around: an installer that
    holds credentials able to create IAM roles in a customer account is the thing their
    security review exists to prevent, and we would rather not be able to.
    """
    from nova.deploy.aws import render_tfvars, write_tfvars

    bundle = load_bundle(args.bundle)
    deployment = bundle.deployment

    if args.deploy_command == "show":
        rows = [
            {
                "id": i.id,
                "description": i.description,
                "agents": list(i.agents),
                "grants": [
                    {"actions": list(s.actions), "resources": list(s.resources)}
                    for s in i.statements
                ],
            }
            for i in deployment.integrations
        ]
        if args.json:
            print(json.dumps({"tenant": bundle.tenant_id, "integrations": rows}, indent=2))
            return 0
        print(f"{bundle.tenant_id}: what the agents can reach outside the runtime")
        if not rows:
            print("  nothing. No integrations are declared, so no customer system is "
                  "reachable from an agent.")
            return 0
        for row in rows:
            who = ", ".join(row["agents"]) or "(not attributed to an agent)"
            print(f"\n  {row['id']}  —  {row['description'] or 'no description'}")
            print(f"    for: {who}")
            for grant in row["grants"]:
                for action in grant["actions"]:
                    print(f"    {action}")
                for resource in grant["resources"]:
                    print(f"      on {resource}")
        return 0

    payload = render_tfvars(
        tenant_id=bundle.tenant_id,
        infrastructure=deployment.infrastructure,
        integrations=deployment.integrations,
        known_agents=[spec.id for spec in bundle.agents],
    )

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    module = args.module or Path(MODULE_PATH)
    destination = args.out or module / TFVARS_FILENAME
    write_tfvars(destination, payload)

    print(f"wrote {destination}")
    print(f"  tenant       {bundle.tenant_id}")
    print(f"  integrations {len(deployment.integrations)}")
    missing = [
        name for name, value in (
            ("region", deployment.infrastructure.region),
            ("vpc_id", deployment.infrastructure.vpc_id),
            ("subnet_id", deployment.infrastructure.subnet_id),
            ("image_uri", deployment.infrastructure.image_uri),
        ) if not value
    ]
    if missing:
        print(
            f"  still needed: {', '.join(missing)} — supply on the Terraform command line, "
            f"or add an infrastructure block to deployment.yaml"
        )
    print(
        f"\nNOVA does not apply this. Your DevOps team runs, with their own credentials:\n"
        f"  terraform -chdir={module} init\n"
        f"  terraform -chdir={module} plan"
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure(level=args.log_level, fmt=args.log_format)

    try:
        if args.command == "validate":
            bundle = load_bundle(args.bundle)
            if args.json:
                print(json.dumps(bundle.to_dict(), indent=2))
            else:
                print(f"{bundle.tenant_id}: {len(bundle.agents)} agent(s) valid")
                for spec in bundle.agents:
                    state = "" if spec.enabled else "  (disabled)"
                    print(f"  {spec.id:22} {bundle.identity.display_name_for(spec.id, spec.name)}{state}")
                print(f"digest: {bundle.digest()}")
            return 0

        if args.command == "status":
            runtime = get_runtime(args.runtime, home=args.home)
            agents = runtime.list_agents()
            if args.json:
                print(
                    json.dumps(
                        {
                            "runtime": runtime.describe(),
                            "agents": [
                                {
                                    "id": a.agent_id,
                                    "managed_by_nova": a.managed_by_nova,
                                    "digest": a.digest,
                                }
                                for a in agents
                            ],
                        },
                        indent=2,
                    )
                )
                return 0
            print(f"runtime: {runtime.name}   state: {runtime.state_location}")
            if not agents:
                print("no agents materialized")
                return 0
            for agent in agents:
                owner = "nova" if agent.managed_by_nova else "not nova-managed"
                print(f"  {agent.agent_id:22} {owner}")
            return 0

        if args.command == "knowledge":
            return _knowledge(args)

        if args.command == "objective":
            return _objective(args)

        if args.command == "bundle":
            return _bundle(args)

        if args.command == "doctor":
            return _doctor(args)

        if args.command == "channels":
            return _channels(args)

        if args.command == "work":
            return _work(args)

        if args.command == "deploy":
            return _deploy(args)

        if args.command == "token":
            return _token(args)

        if args.command == "audit":
            return _audit(args)

        if args.command == "reconcile":
            return _reconcile(args)

        if args.command in ("backup", "restore"):
            return _backup(args)

        if args.command == "serve":
            from nova.control import ControlAPI, serve

            bundle = load_bundle(args.bundle)
            runtime = get_runtime(args.runtime, home=args.home, tenant_id=bundle.tenant_id)
            api = ControlAPI(
                bundle,
                runtime,
                audit=AuditLog.for_home(
                    runtime.state_location, tenant_id=bundle.tenant_id, actor="nova-control"
                ),
            )
            from nova.control.auth import PrincipalStore

            principals_path = args.principals or (
                runtime.state_location / PRINCIPALS_FILENAME
            )
            principals = PrincipalStore.load(principals_path)
            if principals.configured:
                who = ", ".join(
                    f"{entry['name']}({entry['role']})" for entry in principals.describe()
                )
                print(f"principals: {who}")
            else:
                print(
                    f"no principals file at {principals_path} — serving loopback only, "
                    "every local caller is admin. Add one with `nova token new`."
                )
            serve(
                api,
                host=args.host,
                port=args.port,
                principals=principals,
                tls_certfile=args.tls_cert,
                tls_keyfile=args.tls_key,
                behind_tls_proxy=args.behind_tls_proxy,
                ready=lambda url: print(f"{bundle.identity.product_name} control plane: {url}"),
            )
            return 0

        # plan / apply both load a bundle and drive the same code path.
        bundle = load_bundle(args.bundle)
        runtime = get_runtime(args.runtime, home=args.home, tenant_id=bundle.tenant_id)
        dry_run = args.command == "plan"
        audit = (
            NullAuditLog(tenant_id=bundle.tenant_id)
            if dry_run
            else AuditLog.for_home(
                runtime.state_location, tenant_id=bundle.tenant_id, actor="nova-cli"
            )
        )
        report = apply_bundle(
            bundle,
            runtime,
            audit=audit,
            dry_run=dry_run,
            include_disabled=getattr(args, "include_disabled", False),
        )
        _report(report, verb="apply")
        if not dry_run:
            print(f"\naudit: {audit.path}")
            open_intents = audit.open_intents()
            if open_intents:
                # A count told an operator something was wrong and nothing about what.
                print(
                    f"WARNING: {len(open_intents)} unfinished change(s) in the audit log "
                    "from an interrupted run.\n"
                    f"         Run `nova reconcile {args.bundle}` to see which of them "
                    "still need anything."
                )
        return 0

    except NovaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
