"""CLI helpers for the Torben Signal COO operator."""

from __future__ import annotations

import json
import re
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import yaml

from hermes_constants import get_hermes_home

from .action_ledger import HANDLE_RE, ActionLedger, parse_time
from .auth_policy import evaluate_runtime_auth
from .calendar_audit import render_calendar_alignment_audit
from .email_audit import (
    collect_gmail_inbox_audit,
    render_inbox_audit_report,
    write_json_artifact as write_email_json_artifact,
)
from .email_hygiene import apply_hygiene_action
from .google_auth import (
    account_for_alias,
    build_auth_url,
    check_account,
    exchange_auth_code,
    load_google_accounts,
)
from .google_evidence import collect_google_ea_evidence, write_json_artifact
from .gtm_radar_adapter import (
    DEFAULT_MAGNUS_RADAR_PATH,
    build_torben_gtm_radar_adapter,
    load_magnus_gtm_radar,
    write_gtm_radar_adapter_artifacts,
)
from .gtm_reply_router import route_gtm_radar_reply
from .morning_brief import render_morning_brief_text
from .operator import TorbenOperator
from .runtime_secrets import validate_runtime_env_template
from .relationship_learning import apply_relationship_learning_answer


def _default_ledger_path() -> Path:
    return get_hermes_home() / "state" / "torben-action-ledger.json"


def _default_runtime_env_path() -> Path:
    return get_hermes_home() / "runtime.env.op"


def _default_google_accounts_path() -> Path:
    return get_hermes_home() / "config" / "google_accounts.yaml"


def _default_google_evidence_path() -> Path:
    return get_hermes_home() / "state" / "torben-google-ea-evidence-latest.json"


def _default_calendar_audit_path() -> Path:
    return get_hermes_home() / "state" / "torben-calendar-alignment-audit-latest.json"


def _default_calendar_audit_brief_path() -> Path:
    return get_hermes_home() / "state" / "torben-calendar-alignment-audit-latest.txt"


def _default_morning_brief_path() -> Path:
    return get_hermes_home() / "state" / "torben-morning-brief-latest.txt"


def _default_inbox_audit_path() -> Path:
    return get_hermes_home() / "state" / "torben-inbox-audit-latest.json"


def _default_inbox_audit_report_path() -> Path:
    return get_hermes_home() / "state" / "torben-inbox-audit-latest.txt"


def _default_gtm_radar_state_path() -> Path:
    return get_hermes_home() / "state" / "torben-gtm-radar-adapter-state.json"


def _default_gtm_radar_output_path() -> Path:
    return get_hermes_home() / "state" / "torben-gtm-radar-latest.json"


def _default_gtm_radar_text_path() -> Path:
    return get_hermes_home() / "state" / "torben-gtm-radar-latest.txt"


def _default_gtm_content_package_dir() -> Path:
    return get_hermes_home() / "state" / "gtm-content-packages"


def _default_relationship_context_path() -> Path:
    return get_hermes_home() / "config" / "relationship_context.yaml"


def _read_json_file(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def _json_print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _load_torben_config() -> dict[str, Any]:
    config_path = get_hermes_home() / "config.yaml"
    if not config_path.exists():
        return {}
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return {}
    return data


def _cmd_ea_brief(args: Namespace) -> int:
    evidence = _read_json_file(args.evidence)
    operator = TorbenOperator(ledger_path=args.ledger or _default_ledger_path())
    brief = operator.generate_ea_brief(evidence, now=parse_time(getattr(args, "now", None)))
    if args.json:
        _json_print(brief.to_dict())
    else:
        print(brief.text, end="")
    return 0


def _cmd_operating_brief(args: Namespace) -> int:
    evidence = _read_json_file(args.evidence)
    operator = TorbenOperator(ledger_path=args.ledger or _default_ledger_path())
    brief = operator.generate_operating_brief(evidence, now=parse_time(getattr(args, "now", None)))
    if args.json:
        _json_print(brief.to_dict())
    else:
        print(brief.text, end="")
    return 0


def _cmd_resolve_reply(args: Namespace) -> int:
    reply_text = " ".join(args.reply).strip()
    operator = TorbenOperator(ledger_path=args.ledger or _default_ledger_path())
    resolution = operator.resolve_reply(reply_text)
    apply_result = None
    if (
        resolution.status == "resolved"
        and resolution.record is not None
        and _reply_is_approval(reply_text)
        and resolution.record.executor_state.get("mutation_type") == "gmail_hygiene"
        and "approve_hygiene_apply" in resolution.record.allowed_next_actions
    ):
        apply_result = apply_hygiene_action(
            ledger=operator.ledger,
            config_path=_default_google_accounts_path(),
            handle=resolution.record.handle,
            approved_by="signal-reply",
        )
    learn_result = None
    if (
        resolution.status == "resolved"
        and resolution.record is not None
        and resolution.record.executor_state.get("mutation_type") == "relationship_learning"
    ):
        answer = _reply_without_handle(reply_text)
        if answer:
            learn_result = apply_relationship_learning_answer(
                ledger=operator.ledger,
                relationship_context_path=_default_relationship_context_path(),
                handle=resolution.record.handle,
                answer=answer,
                approved_by="signal-reply",
            )
    if args.json:
        payload = resolution.to_dict()
        if apply_result is not None:
            payload["applied_action"] = apply_result
            refreshed = operator.ledger.get(resolution.record.handle)
            if refreshed is not None:
                payload["record"] = refreshed.to_dict()
        if learn_result is not None:
            payload["learned_contact"] = learn_result
            refreshed = operator.ledger.get(resolution.record.handle)
            if refreshed is not None:
                payload["record"] = refreshed.to_dict()
        _json_print(payload)
    else:
        if apply_result is not None:
            status = "applied" if not apply_result.get("errors") else "blocked"
            print(
                f"{status}: {resolution.record.handle} - "
                f"{apply_result.get('operation')} "
                f"({apply_result.get('external_mutations', 0)} Gmail mutation(s))"
            )
        elif learn_result is not None:
            if learn_result.get("discarded"):
                print(f"discarded: {resolution.record.handle} - relationship learning skipped")
            else:
                contact = learn_result.get("learned_contact") or {}
                print(f"learned: {resolution.record.handle} - {contact.get('name') or 'contact'}")
        elif resolution.record:
            print(f"{resolution.status}: {resolution.record.handle} - {resolution.record.summary}")
        elif resolution.candidates:
            print(f"{resolution.status}: {resolution.reason}")
            for candidate in resolution.candidates:
                print(f"- {candidate.handle}: {candidate.summary}")
        else:
            print(f"{resolution.status}: {resolution.reason or 'no matching action'}")
    return 1 if apply_result is not None and apply_result.get("errors") else 0


def _reply_without_handle(reply_text: str) -> str:
    answer = HANDLE_RE.sub("", reply_text).strip()
    answer = re.sub(r"^(approve|approved|apply|learn|answer|for|re:|regarding)\b[:\s-]*", "", answer, flags=re.I).strip()
    return answer


def _reply_is_approval(reply_text: str) -> bool:
    normalized = f" {reply_text.strip().lower()} "
    approval_terms = (
        " approve ",
        " approved ",
        " approve_hygiene_apply ",
        " go ahead ",
        " do it ",
        " apply ",
    )
    return any(term in normalized for term in approval_terms)


def _cmd_learn_contact(args: Namespace) -> int:
    result = apply_relationship_learning_answer(
        ledger=ActionLedger(args.ledger or _default_ledger_path()),
        relationship_context_path=args.relationship_context or _default_relationship_context_path(),
        handle=args.handle,
        answer=" ".join(args.answer).strip(),
        approved_by=args.approved_by,
    )
    if args.json:
        _json_print(result)
    else:
        if result.get("discarded"):
            print(f"discarded: {args.handle} - relationship learning skipped")
        else:
            contact = result.get("learned_contact") or {}
            print(f"learned: {args.handle} - {contact.get('name') or 'contact'}")
            print(f"Store: {result.get('learned_contacts_path')}")
    return 0


def _cmd_scopes(args: Namespace) -> int:
    scopes = {
        "operator": {
            "name": "Torben",
            "interface": "Signal",
            "role": "single user-facing COO operator",
        },
        "scopes": [
            {
                "scope": "ea",
                "worker": "EA",
                "model_lane": "openai-codex",
                "autonomy": "draft_stage_then_explicit_approval",
                "connectors": ["google-calendar", "gmail", "local-action-ledger"],
                "first_jobs": [
                    "daily brief",
                    "email triage",
                    "calendar prep",
                    "reminder and family follow-up",
                    "open-loop memory",
                ],
            },
            {
                "scope": "gtm",
                "worker": "GTM / Magnus",
                "model_lane": "xai-oauth",
                "autonomy": "draft_posts_then_explicit_approval",
                "connectors": ["x", "linkedin", "browser-automation", "engagement-ledger"],
                "first_jobs": [
                    "research thought-leadership angles",
                    "draft X and LinkedIn posts",
                    "stage image direction",
                    "learn from engagement",
                ],
            },
            {
                "scope": "finance",
                "worker": "Finance",
                "model_lane": "openai-codex",
                "autonomy": "live_trading_only_after_auth_policy_and_signal_approval",
                "connectors": ["robinhood-agentic-mcp", "gemini", "monarch-money-mcp"],
                "first_jobs": [
                    "market signal review",
                    "hard-limited options and margin trade review",
                    "position monitoring",
                    "Monarch cashflow and savings analysis",
                ],
            },
        ],
    }
    if args.json:
        _json_print(scopes)
    else:
        print("Torben: one Signal-facing COO operator")
        for scope in scopes["scopes"]:
            print(
                f"- {scope['scope']}: {scope['worker']} via {scope['model_lane']} "
                f"({scope['autonomy']})"
            )
    return 0


def _cmd_google_accounts(args: Namespace) -> int:
    accounts = load_google_accounts(args.config or _default_google_accounts_path())
    payload = {
        "accounts": [
            {
                "alias": account.alias,
                "email": account.email,
                "role": account.role,
                "enabled": account.enabled,
                "scopes": list(account.scopes),
                "token_path": str(account.token_path),
                "client_secret_path": str(account.client_secret_path),
            }
            for account in accounts.values()
        ]
    }
    if args.json:
        _json_print(payload)
    else:
        for account in payload["accounts"]:
            enabled = "enabled" if account["enabled"] else "disabled"
            print(f"{account['alias']}: {account['email']} ({account['role']}, {enabled})")
    return 0


def _cmd_google_auth_url(args: Namespace) -> int:
    account = account_for_alias(args.config or _default_google_accounts_path(), args.account)
    result = build_auth_url(account)
    if args.json:
        _json_print(result.to_dict())
    else:
        print(f"Account: {result.account} <{result.email}>")
        print("Open this URL on the device you are using for SSH:")
        print(result.url)
        print()
        print("After Google redirects to localhost:1, paste the full final URL into:")
        print(
            f"  hermes torben google-auth-code --account {result.account} "
            "'<PASTED_REDIRECT_URL_OR_CODE>'"
        )
    return 0


def _cmd_google_auth_code(args: Namespace) -> int:
    account = account_for_alias(args.config or _default_google_accounts_path(), args.account)
    result = exchange_auth_code(account, args.code_or_url)
    if args.json:
        _json_print(result.to_dict())
    else:
        print(f"OK: authenticated {result.account} <{result.email}>")
        print(f"Token path: {result.token_path}")
    return 0


def _cmd_google_auth_check(args: Namespace) -> int:
    config = args.config or _default_google_accounts_path()
    if args.account:
        accounts = [account_for_alias(config, args.account)]
    else:
        accounts = list(load_google_accounts(config).values())
    results = [check_account(account) for account in accounts]
    if args.json:
        _json_print({"accounts": [result.to_dict() for result in results]})
    else:
        for result in results:
            detail = f": {result.reason}" if result.reason else ""
            print(f"{result.status}: {result.account} <{result.email}>{detail}")
    return 0 if all(result.status.startswith("authenticated") for result in results) else 1


def _cmd_google_ea_evidence(args: Namespace) -> int:
    now = parse_time(getattr(args, "now", None))
    max_calendar_block_candidates = getattr(args, "max_calendar_block_candidates", 3)
    if max_calendar_block_candidates <= 0:
        max_calendar_block_candidates = None
    payload = collect_google_ea_evidence(
        config_path=args.config or _default_google_accounts_path(),
        now=now,
        days=args.days,
        max_calendar_events=args.max_calendar_events,
        max_email_messages=args.max_email_messages,
        max_calendar_block_candidates=max_calendar_block_candidates,
        include_secondary_calendars=bool(getattr(args, "include_secondary_calendars", False)),
    )
    output_path = write_json_artifact(payload, args.output or _default_google_evidence_path())
    if args.json:
        _json_print({"output_path": str(output_path), **payload})
    else:
        diagnostics = payload.get("source_diagnostics", {}).get("google", {})
        audit = diagnostics.get("audit", {})
        print(f"Evidence path: {output_path}")
        print(f"Accounts: {len(diagnostics.get('accounts') or [])}")
        print(f"Calendar events: {diagnostics.get('calendar_events_collected', 0)}")
        print(f"Email messages: {diagnostics.get('email_messages_collected', 0)}")
        print(f"Calendar block candidates: {diagnostics.get('calendar_block_candidates', 0)}")
        print(f"Google read API calls: {audit.get('google_read_api_calls', 0)}")
        print(f"Google write API calls: {audit.get('google_write_api_calls', 0)}")
        print(f"External mutations: {audit.get('external_mutations', 0)}")
    return 0


def _cmd_calendar_audit(args: Namespace) -> int:
    now = parse_time(getattr(args, "now", None))
    max_block_candidates = None if args.max_block_candidates <= 0 else args.max_block_candidates
    payload = collect_google_ea_evidence(
        config_path=args.config or _default_google_accounts_path(),
        now=now,
        days=args.days,
        max_calendar_events=args.max_calendar_events,
        max_email_messages=0,
        max_calendar_block_candidates=max_block_candidates,
        include_secondary_calendars=not args.primary_only,
    )
    output_path = write_json_artifact(payload, args.output or _default_calendar_audit_path())
    report = render_calendar_alignment_audit(payload, max_items=args.max_items)
    brief_output = Path(args.brief_output or _default_calendar_audit_brief_path())
    brief_output.parent.mkdir(parents=True, exist_ok=True)
    brief_output.write_text(report, encoding="utf-8")
    if args.json:
        _json_print({"output_path": str(output_path), "brief_output": str(brief_output), **payload})
    else:
        print(report, end="")
        print(f"\nJSON: {output_path}")
        print(f"Text: {brief_output}")
    return 0


def _cmd_morning_brief(args: Namespace) -> int:
    evidence = _read_json_file(args.evidence)
    ea = evidence.get("ea") if isinstance(evidence.get("ea"), dict) else evidence
    morning = ea.get("morning_brief") if isinstance(ea, dict) else None
    if not isinstance(morning, dict):
        raise ValueError("Evidence does not contain ea.morning_brief")
    text = render_morning_brief_text(morning)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    else:
        output_path = None
    if args.json:
        _json_print({"output_path": str(output_path) if output_path else None, "morning_brief": morning, "text": text})
    else:
        print(text, end="")
        if output_path:
            print(f"\nText: {output_path}")
    return 0


def _cmd_gtm_radar(args: Namespace) -> int:
    radar = load_magnus_gtm_radar(args.radar_path or DEFAULT_MAGNUS_RADAR_PATH)
    ledger = ActionLedger(args.ledger or _default_ledger_path())
    payload = build_torben_gtm_radar_adapter(
        radar,
        ledger=ledger,
        state_path=args.state or _default_gtm_radar_state_path(),
        max_items=args.max_items,
        now=parse_time(getattr(args, "now", None)),
        mark_delivered=not args.preview,
        stage_actions=not args.preview,
    )
    if args.output or args.text_output:
        write_gtm_radar_adapter_artifacts(
            payload,
            json_path=args.output or _default_gtm_radar_output_path(),
            text_path=args.text_output or _default_gtm_radar_text_path(),
        )
    if args.json:
        _json_print(payload)
    else:
        print(str(payload.get("text") or ""), end="")
    return 0


def _cmd_gtm_reply(args: Namespace) -> int:
    reply_text = " ".join(args.reply).strip()
    ledger = ActionLedger(args.ledger or _default_ledger_path())
    result = route_gtm_radar_reply(
        ledger=ledger,
        reply_text=reply_text,
        output_dir=args.output_dir or _default_gtm_content_package_dir(),
        now=parse_time(getattr(args, "now", None)),
        approved_by="cli-replay",
    )
    if args.json:
        _json_print(result.to_dict())
    else:
        print(result.text or f"{result.status}: {result.reason or 'not handled'}", end="")
        if result.text and not result.text.endswith("\n"):
            print()
    return 0 if result.handled else 1


def _cmd_inbox_audit(args: Namespace) -> int:
    now = parse_time(getattr(args, "now", None))
    payload = collect_gmail_inbox_audit(
        config_path=args.config or _default_google_accounts_path(),
        relationship_context_path=args.relationship_context or _default_relationship_context_path(),
        days=args.days,
        max_messages_per_account=args.max_messages_per_account,
        max_body_fetches_per_account=args.max_body_fetches_per_account,
        fetch_workers=args.fetch_workers,
        now=now,
    )
    output_path = write_email_json_artifact(payload, args.output or _default_inbox_audit_path())
    report = render_inbox_audit_report(
        payload,
        max_sources=args.max_sources,
        max_messages=args.max_messages,
    )
    report_output = Path(args.report_output or _default_inbox_audit_report_path())
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(report, encoding="utf-8")
    if args.json:
        _json_print({"output_path": str(output_path), "report_output": str(report_output), **payload})
    else:
        print(report, end="")
        print(f"\nJSON: {output_path}")
        print(f"Text: {report_output}")
    return 0


def _cmd_secrets_check(args: Namespace) -> int:
    report = validate_runtime_env_template(
        args.env_file or _default_runtime_env_path(),
        required_keys=getattr(args, "required_key", None) or [],
    )
    if args.json:
        _json_print(report.to_dict())
    else:
        print("valid:", str(report.valid).lower())
        if report.missing_required:
            print("missing_required:", ", ".join(report.missing_required))
        if report.plaintext_secret_keys:
            print("plaintext_secret_keys:", ", ".join(report.plaintext_secret_keys))
        if report.invalid_op_refs:
            print("invalid_op_refs:", ", ".join(report.invalid_op_refs))
    return 0 if report.valid else 1


def _cmd_auth_check(args: Namespace) -> int:
    config = _load_torben_config()
    report = evaluate_runtime_auth(
        config,
        optional_env_file=args.env_file or _default_runtime_env_path(),
    )
    if args.json:
        _json_print(report.to_dict())
    else:
        print("valid:", str(report.valid).lower())
        print("strategy:", report.strategy)
        print("default_provider:", report.default_provider)
        print("gtm_provider:", report.gtm_provider)
        print("finance_execution:", report.finance_execution)
        print("onepassword_bootstrap:", report.onepassword_bootstrap)
        if report.warnings:
            print("warnings:", "; ".join(report.warnings))
    return 0 if report.valid else 1


def _cmd_route(args: Namespace) -> int:
    config = _load_torben_config()
    routing = ((config.get("torben") or {}).get("model_routing") or {})
    if args.json:
        _json_print(routing)
        return 0
    if not routing:
        print("No torben.model_routing block found.")
        return 1
    default = routing.get("default") or {}
    gtm = routing.get("gtm") or {}
    print(
        "default:",
        f"{default.get('provider', 'unknown')} / {default.get('model', 'unknown')}",
        f"({default.get('scope', 'unknown')})",
    )
    print(
        "gtm:",
        f"{gtm.get('provider', 'unknown')} / {gtm.get('model', 'unknown')}",
        f"({gtm.get('scope', 'unknown')})",
    )
    return 0


def torben_command(args: Namespace) -> int:
    action = getattr(args, "torben_action", None)
    if action == "ea-brief":
        return _cmd_ea_brief(args)
    if action == "operating-brief":
        return _cmd_operating_brief(args)
    if action == "resolve-reply":
        return _cmd_resolve_reply(args)
    if action == "learn-contact":
        return _cmd_learn_contact(args)
    if action == "scopes":
        return _cmd_scopes(args)
    if action == "google-accounts":
        return _cmd_google_accounts(args)
    if action == "google-auth-url":
        return _cmd_google_auth_url(args)
    if action == "google-auth-code":
        return _cmd_google_auth_code(args)
    if action == "google-auth-check":
        return _cmd_google_auth_check(args)
    if action == "google-ea-evidence":
        return _cmd_google_ea_evidence(args)
    if action == "calendar-audit":
        return _cmd_calendar_audit(args)
    if action == "morning-brief":
        return _cmd_morning_brief(args)
    if action == "gtm-radar":
        return _cmd_gtm_radar(args)
    if action == "gtm-reply":
        return _cmd_gtm_reply(args)
    if action == "inbox-audit":
        return _cmd_inbox_audit(args)
    if action == "secrets-check":
        return _cmd_secrets_check(args)
    if action == "auth-check":
        return _cmd_auth_check(args)
    if action == "route":
        return _cmd_route(args)
    print(
        "usage: hermes torben {ea-brief,operating-brief,resolve-reply,learn-contact,scopes,google-ea-evidence,calendar-audit,morning-brief,gtm-radar,gtm-reply,inbox-audit,secrets-check,auth-check,route}",
        file=sys.stderr,
    )
    return 2
