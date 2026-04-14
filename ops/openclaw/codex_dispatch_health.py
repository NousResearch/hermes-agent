#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path('/home/yeqiuqiu/clawd-architect/ops/openclaw')
REPO_ROOT = ROOT.parent.parent
SCRIPTS_DIR = REPO_ROOT / 'scripts'

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from session_topology_routing_policy_contract import (
        load_routing_policy,
        routing_policy_codex_quota_exhaustion_reason_prefixes,
        routing_policy_codex_quota_routine_dispatch_lane_disable_statuses,
        routing_policy_telegram_direct_worker_target_disallowed_lane_tokens,
    )
except Exception:
    load_routing_policy = None
    routing_policy_codex_quota_exhaustion_reason_prefixes = None
    routing_policy_codex_quota_routine_dispatch_lane_disable_statuses = None
    routing_policy_telegram_direct_worker_target_disallowed_lane_tokens = None

RECONCILE = ROOT / 'codex_lane_reconcile.py'
AUDIT = ROOT / 'codex_worker_pool_audit.py'
USAGE = ROOT / 'codex_usage_report.py'
MANIFEST = ROOT / 'codex_worker_pool_manifest.json'
LANE_MAP = ROOT / 'codex_lane_account_map.json'
DEFAULT_ROUTING_POLICY = REPO_ROOT / 'docs' / 'ops' / 'session_topology_routing_policy_v1.json'
DEFAULT_ROUTING_POLICY_SCHEMA = REPO_ROOT / 'docs' / 'ops' / 'schemas' / 'session_topology_routing_policy.schema.json'

SEVERITY_HEALTHY = 'healthy'
SEVERITY_PROBATIONARY = 'probationary'
SEVERITY_QUARANTINED = 'quarantined'
RECOVERED_USAGE_LIMIT_PROBATION_HOURS = 4.0

DEFAULT_ROUTINE_DISPATCH_LANE_DISABLE_STATUSES = [
    SEVERITY_PROBATIONARY,
    SEVERITY_QUARANTINED,
]
DEFAULT_QUOTA_EXHAUSTION_REASON_PREFIXES = [
    'quota_exhausted',
    'quota_exhausted_additional',
    'runtime_bodyless_usage_limit',
]
DEFAULT_TELEGRAM_DIRECT_WORKER_TARGET_DISALLOWED_LANE_TOKENS = [
    'cockpit',
    'direct',
    'inbox',
    'main',
    'main_session',
    'telegram_direct',
]

AUTH_DRIFT_REASON_HINTS = {
    'lane_missing_from_audit',
    'agent_config_missing',
    'configured_model_missing',
    'model_mismatch',
    'desired_profile_missing',
    'order_not_pinned_to_desired_profile',
    'binding_ambiguity',
    'desired_email_mismatch',
}

RUNTIME_QUOTA_REASON_HINTS = {
    'runtime_bodyless_usage_limit',
    'runtime_usage_limit_recovered',
}


class DispatchHealthError(RuntimeError):
    def __init__(self, code: str, message: str, *, command: list[str] | None = None):
        super().__init__(message)
        self.code = str(code or 'dispatch_health_error')
        self.message = str(message or '').strip() or self.code
        self.command = [str(part) for part in (command or []) if str(part).strip()]


def emit_failure_payload(*, code: str, message: str, command: list[str] | None = None) -> dict[str, Any]:
    payload = {
        'status': 'error',
        'error_code': str(code or 'dispatch_health_error'),
        'message': str(message or '').strip() or str(code or 'dispatch_health_error'),
    }
    if command:
        payload['command'] = [str(part) for part in command]
    return payload


def run_json(cmd: list[str]):
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr = str(proc.stderr or '').strip()
        stdout = str(proc.stdout or '').strip()
        detail = stderr or stdout or 'command_failed'
        raise DispatchHealthError(
            'runner_command_failed',
            f'command failed ({proc.returncode}): {" ".join(cmd)}: {detail[:240]}',
            command=cmd,
        )

    stdout = str(proc.stdout or '')
    if not stdout.strip():
        raise DispatchHealthError(
            'runner_stdout_empty',
            f'empty stdout from {" ".join(cmd)}',
            command=cmd,
        )

    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise DispatchHealthError(
            'runner_stdout_not_json',
            f'non-json output from {" ".join(cmd)}: {exc.msg}',
            command=cmd,
        ) from exc


def parse_iso(ts: str | None):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace('Z', '+00:00'))
    except Exception:
        return None


def format_dt(dt: datetime | None):
    if not isinstance(dt, datetime):
        return None
    return dt.astimezone(timezone.utc).isoformat()


def gather_expected_agents(manifest: dict, include_orchestrator: bool):
    agents = []
    if include_orchestrator:
        orch = manifest.get('orchestrator', {})
        if orch.get('agent'):
            agents.append(orch['agent'])
    for worker in manifest.get('workers', []):
        if worker.get('agent'):
            agents.append(worker['agent'])
    return agents


def collect_allow_shared_account_by_agent(lane_map_obj: dict):
    out = {}
    for lane in lane_map_obj.get('lanes', []) or []:
        if not isinstance(lane, dict):
            continue
        agent = lane.get('agent')
        if not agent:
            continue
        out[agent] = bool(lane.get('allowSharedAccount'))
    return out


def classify_shared_accounts(account_to_agents: dict[str, list[str]], allow_shared_account_by_agent: dict[str, bool]):
    shared = {acc: sorted(agents) for acc, agents in account_to_agents.items() if len(agents) > 1}
    allowlisted = {}
    blocked = {}
    for account_id, agents in shared.items():
        if all(allow_shared_account_by_agent.get(agent, False) for agent in agents):
            allowlisted[account_id] = agents
        else:
            blocked[account_id] = agents
    return shared, allowlisted, blocked


def dedupe_keep_order(values: list[str]):
    seen = set()
    out = []
    for value in values:
        token = str(value or '').strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _normalize_lower_token_list(values: list[str]):
    out = []
    seen = set()
    for value in values:
        token = str(value or '').strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def load_codex_quota_routing_policy(*, policy_path: Path, policy_schema_path: Path):
    lane_disable_statuses = list(DEFAULT_ROUTINE_DISPATCH_LANE_DISABLE_STATUSES)
    quota_exhaustion_reason_prefixes = list(DEFAULT_QUOTA_EXHAUSTION_REASON_PREFIXES)
    disallowed_lane_tokens = list(DEFAULT_TELEGRAM_DIRECT_WORKER_TARGET_DISALLOWED_LANE_TOKENS)
    policy_id = None
    load_status = 'fallback_default'
    load_reason = None
    load_details: dict[str, Any] = {}

    if load_routing_policy is None:
        load_reason = 'routing_policy_contract_import_unavailable'
    else:
        ok, reason, details, policy_doc = load_routing_policy(policy_path, policy_schema_path)
        if ok and isinstance(policy_doc, Mapping):
            if routing_policy_codex_quota_routine_dispatch_lane_disable_statuses is not None:
                loaded_statuses = routing_policy_codex_quota_routine_dispatch_lane_disable_statuses(policy_doc)
                lane_disable_statuses = _normalize_lower_token_list(sorted(loaded_statuses))
            if routing_policy_codex_quota_exhaustion_reason_prefixes is not None:
                loaded_prefixes = routing_policy_codex_quota_exhaustion_reason_prefixes(policy_doc)
                lane_prefixes = [str(token or '').strip().lower() for token in loaded_prefixes]
                quota_exhaustion_reason_prefixes = _normalize_lower_token_list(lane_prefixes)
            if routing_policy_telegram_direct_worker_target_disallowed_lane_tokens is not None:
                loaded_disallowed_tokens = routing_policy_telegram_direct_worker_target_disallowed_lane_tokens(policy_doc)
                disallowed_lane_tokens = _normalize_lower_token_list(sorted(loaded_disallowed_tokens))

            if not lane_disable_statuses:
                lane_disable_statuses = list(DEFAULT_ROUTINE_DISPATCH_LANE_DISABLE_STATUSES)
            if not quota_exhaustion_reason_prefixes:
                quota_exhaustion_reason_prefixes = list(DEFAULT_QUOTA_EXHAUSTION_REASON_PREFIXES)
            if not disallowed_lane_tokens:
                disallowed_lane_tokens = list(DEFAULT_TELEGRAM_DIRECT_WORKER_TARGET_DISALLOWED_LANE_TOKENS)

            policy_id = str(policy_doc.get('policy_id') or '').strip() or None
            load_status = 'ok'
            load_reason = None
            load_details = dict(details or {})
        else:
            load_reason = str(reason or 'routing_policy_invalid')
            load_details = dict(details or {})

    return {
        'policyId': policy_id,
        'policyPath': str(policy_path),
        'policySchemaPath': str(policy_schema_path),
        'loadStatus': load_status,
        'loadReason': load_reason,
        'loadDetails': load_details,
        'laneDisableStatuses': lane_disable_statuses,
        'quotaExhaustionReasonPrefixes': quota_exhaustion_reason_prefixes,
        'telegramDirectWorkerTargetDisallowedLaneTokens': disallowed_lane_tokens,
    }


def add_action(actions: list[dict], *, action_id: str, title: str, command: str, reason: str, priority: int):
    if any(row.get('id') == action_id for row in actions):
        return
    actions.append(
        {
            'id': action_id,
            'title': title,
            'reason': reason,
            'command': command,
            'priority': int(priority),
        }
    )


def build_recommended_actions(*, agent: str, status: str, quarantine_reasons: list[str], probation_reasons: list[str], lookback_hours: float):
    actions: list[dict] = []
    all_reasons = dedupe_keep_order(list(quarantine_reasons) + list(probation_reasons))

    has_auth_drift = any(
        (reason in AUTH_DRIFT_REASON_HINTS) or reason.startswith('reconcile:')
        for reason in all_reasons
    )
    has_quota_block = any(reason.startswith('quota_exhausted') for reason in all_reasons) or any(
        reason in RUNTIME_QUOTA_REASON_HINTS
        for reason in all_reasons
    )
    has_usage_probe_error = any(reason.startswith('usage_probe_error:') for reason in all_reasons)

    if has_auth_drift:
        add_action(
            actions,
            action_id='lane_reconcile',
            title='Reconcile lane auth/profile binding',
            reason='auth_or_profile_drift_detected',
            command=(
                f'python3 {RECONCILE} --dry-run && '
                f'python3 {RECONCILE} && '
                f'python3 {AUDIT} --lane-map {LANE_MAP}'
            ),
            priority=10,
        )

    if any(reason == 'shared_account_binding' for reason in all_reasons):
        add_action(
            actions,
            action_id='shared_account_resolution',
            title='Resolve shared-account binding policy',
            reason='shared_account_binding',
            command=(
                f'python3 {ROOT / "codex_lane_set_account.py"} --agent {agent} --email <dedicated-lane-email>'
            ),
            priority=20,
        )

    if has_usage_probe_error:
        add_action(
            actions,
            action_id='retry_usage_probe',
            title='Retry usage probe for the lane',
            reason='usage_probe_error',
            command=f'python3 {USAGE} --json --agent {agent}',
            priority=30,
        )

    if has_quota_block:
        add_action(
            actions,
            action_id='quota_cooldown_gate',
            title='Keep lane in cooldown until quota gate clears',
            reason='quota_or_runtime_usage_limit',
            command=(
                f'python3 {USAGE} --json --agent {agent} && '
                f'python3 {ROOT / "codex_dispatch_health.py"} '
                f'--agent {agent} --lookback-hours {max(0.0, float(lookback_hours))} '
                f'--require-healthy-agent {agent}'
            ),
            priority=40,
        )

    if status != SEVERITY_HEALTHY:
        add_action(
            actions,
            action_id='dispatch_recheck',
            title='Re-run target-scoped dispatch gate before reuse',
            reason='non_healthy_lane_status',
            command=(
                f'python3 {ROOT / "codex_preflight_audit.py"} --agent {agent} '
                f'--lookback-hours {max(0.0, float(lookback_hours))}'
            ),
            priority=90,
        )

    actions.sort(key=lambda row: (int(row.get('priority') or 0), str(row.get('id') or '')))
    return actions


def usage_windows_exhausted(usage: dict):
    exhausted = []
    if not isinstance(usage, dict):
        return exhausted
    for win in usage.get('windows', []) or []:
        used = win.get('usedPercent')
        try:
            used_val = float(used)
        except Exception:
            continue
        if used_val >= 99.9:
            exhausted.append(
                {
                    'label': win.get('label'),
                    'usedPercent': used_val,
                    'resetAt': win.get('resetAt'),
                }
            )
    return exhausted


def _safe_reason_token(value: str | None):
    raw = str(value or '').strip().lower()
    if not raw:
        return 'unknown_limit'
    return ''.join(ch if ch.isalnum() else '_' for ch in raw).strip('_') or 'unknown_limit'


def _rate_limit_windows_exhausted(rate_limit: dict):
    exhausted = []
    if not isinstance(rate_limit, dict):
        return exhausted

    if rate_limit.get('limit_reached') is True:
        exhausted.append(
            {
                'label': 'limit_reached',
                'usedPercent': 100.0,
                'resetAt': None,
            }
        )

    for key in ('primary_window', 'secondary_window'):
        window = rate_limit.get(key)
        if not isinstance(window, dict):
            continue
        try:
            used_val = float(window.get('used_percent'))
        except Exception:
            continue
        if used_val >= 99.9:
            exhausted.append(
                {
                    'label': key,
                    'usedPercent': used_val,
                    'resetAt': window.get('reset_at'),
                }
            )
    return exhausted


def additional_rate_limits_exhausted(usage: dict):
    exhausted = []
    if not isinstance(usage, dict):
        return exhausted

    raw = usage.get('raw')
    if not isinstance(raw, dict):
        return exhausted
    additional = raw.get('additional_rate_limits')
    if not isinstance(additional, list):
        return exhausted

    for row in additional:
        if not isinstance(row, dict):
            continue
        limit_name = row.get('limit_name') or row.get('metered_feature') or 'unknown_limit'
        metered_feature = row.get('metered_feature')
        exhausted_windows = _rate_limit_windows_exhausted(row.get('rate_limit') or {})
        if not exhausted_windows:
            continue
        exhausted.append(
            {
                'limitName': limit_name,
                'limitToken': _safe_reason_token(limit_name),
                'meteredFeature': metered_feature,
                'windows': exhausted_windows,
            }
        )
    return exhausted


def scan_lane_runtime(agent: str, cutoff: datetime):
    session_dir = Path(f'/home/yeqiuqiu/.openclaw/agents/{agent}/sessions')
    out = {
        'sessionDir': str(session_dir),
        'lookbackCutoff': format_dt(cutoff),
        'recentBodylessUsageLimitErrors': 0,
        'recentAnyErrors': 0,
        'latestBodylessUsageLimitErrorAt': None,
        'latestSuccessAt': None,
        'latestMessageAt': None,
        'sampleErrors': [],
    }
    if not session_dir.exists():
        return out

    latest_bodyless = None
    latest_success = None
    latest_any = None

    for path in sorted(session_dir.glob('*.jsonl')):
        try:
            lines = path.read_text().splitlines()
        except Exception:
            continue

        for line in lines:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get('type') != 'message':
                continue
            msg = obj.get('message') or {}
            if msg.get('role') != 'assistant':
                continue

            ts_ms = msg.get('timestamp')
            if isinstance(ts_ms, (int, float)):
                at = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            else:
                at = parse_iso(obj.get('timestamp'))
            if not at:
                continue
            if at < cutoff:
                continue

            if latest_any is None or at > latest_any:
                latest_any = at

            stop = str(msg.get('stopReason') or '').strip().lower()
            content = msg.get('content')
            err = str(msg.get('errorMessage') or '')
            err_l = err.lower()

            if stop == 'error':
                out['recentAnyErrors'] += 1
                is_usage_limit = ('usage limit' in err_l) or ('quota' in err_l) or ('429' in err_l)
                is_bodyless = isinstance(content, list) and len(content) == 0
                if is_usage_limit and is_bodyless:
                    out['recentBodylessUsageLimitErrors'] += 1
                    if latest_bodyless is None or at > latest_bodyless:
                        latest_bodyless = at
                    if len(out['sampleErrors']) < 3:
                        out['sampleErrors'].append(
                            {
                                'at': format_dt(at),
                                'session': path.name,
                                'errorMessage': err[:240],
                            }
                        )
                continue

            has_content = isinstance(content, list) and len(content) > 0
            if has_content:
                if latest_success is None or at > latest_success:
                    latest_success = at

    out['latestBodylessUsageLimitErrorAt'] = format_dt(latest_bodyless)
    out['latestSuccessAt'] = format_dt(latest_success)
    out['latestMessageAt'] = format_dt(latest_any)
    return out


def recovered_usage_limit_probation_active(
    *,
    now: datetime,
    latest_error: datetime | None,
    latest_success: datetime | None,
    probation_hours: float = RECOVERED_USAGE_LIMIT_PROBATION_HOURS,
) -> bool:
    if latest_error is None or latest_success is None:
        return False
    if latest_success <= latest_error:
        return False
    probation_window = timedelta(hours=max(probation_hours, 0.0))
    return (now - latest_success) <= probation_window


def main():
    ap = argparse.ArgumentParser(description='Produce machine-checkable Codex lane dispatch health posture.')
    ap.add_argument('--manifest', default=str(MANIFEST))
    ap.add_argument('--lane-map', default=str(LANE_MAP))
    ap.add_argument('--routing-policy', default=str(DEFAULT_ROUTING_POLICY))
    ap.add_argument('--routing-policy-schema', default=str(DEFAULT_ROUTING_POLICY_SCHEMA))
    ap.add_argument('--lookback-hours', type=float, default=24.0)
    ap.add_argument('--include-orchestrator', action='store_true', help='Include orchestrator lane in posture output')
    ap.add_argument('--agent', action='append', help='Restrict output to one or more agents')
    ap.add_argument('--require-healthy-agent', action='append', help='Fail if any listed agent is not healthy')
    ap.add_argument('--json-output', help='Write full posture JSON to file')
    ap.add_argument('--strict', action='store_true', help='Fail if any lane is not healthy')
    args = ap.parse_args()

    try:
        routing_policy_path_raw = Path(args.routing_policy).expanduser()
        routing_policy_path = routing_policy_path_raw if routing_policy_path_raw.is_absolute() else (REPO_ROOT / routing_policy_path_raw).resolve()
        routing_policy_schema_raw = Path(args.routing_policy_schema).expanduser()
        routing_policy_schema_path = (
            routing_policy_schema_raw if routing_policy_schema_raw.is_absolute() else (REPO_ROOT / routing_policy_schema_raw).resolve()
        )

        codex_quota_routing_policy = load_codex_quota_routing_policy(
            policy_path=routing_policy_path,
            policy_schema_path=routing_policy_schema_path,
        )
        routine_dispatch_lane_disable_statuses = _normalize_lower_token_list(
            list(codex_quota_routing_policy.get('laneDisableStatuses') or [])
        )
        if not routine_dispatch_lane_disable_statuses:
            routine_dispatch_lane_disable_statuses = list(DEFAULT_ROUTINE_DISPATCH_LANE_DISABLE_STATUSES)

        quota_exhaustion_reason_prefixes = _normalize_lower_token_list(
            list(codex_quota_routing_policy.get('quotaExhaustionReasonPrefixes') or [])
        )
        if not quota_exhaustion_reason_prefixes:
            quota_exhaustion_reason_prefixes = list(DEFAULT_QUOTA_EXHAUSTION_REASON_PREFIXES)

        manifest = json.loads(Path(args.manifest).read_text())
        lane_map_obj = json.loads(Path(args.lane_map).read_text())
        expected_agents = gather_expected_agents(manifest, include_orchestrator=args.include_orchestrator)

        if args.agent:
            allowed = set(args.agent)
            expected_agents = [agent for agent in expected_agents if agent in allowed]

        expected_email_by_agent = {
            lane.get('agent'): lane.get('email')
            for lane in lane_map_obj.get('lanes', [])
            if lane.get('agent')
        }
        allow_shared_account_by_agent = collect_allow_shared_account_by_agent(lane_map_obj)

        reconcile = run_json([sys.executable, str(RECONCILE), '--dry-run'])
        audit = run_json([sys.executable, str(AUDIT), '--lane-map', args.lane_map])
        usage = run_json([sys.executable, str(USAGE), '--json'])

        reconcile_by_agent = {row.get('agent'): row for row in reconcile.get('results', []) if row.get('agent')}
        audit_by_agent = {row.get('agent'): row for row in audit.get('lanes', []) if row.get('agent')}
        usage_by_agent = {row.get('agent'): row for row in usage.get('lanes', []) if row.get('agent')}

        account_to_agents = defaultdict(list)
        for agent in expected_agents:
            lane = audit_by_agent.get(agent) or {}
            desired = lane.get('desiredAccount') or {}
            acc = desired.get('accountId')
            if acc:
                account_to_agents[acc].append(agent)

        shared_accounts, allowlisted_shared_accounts, blocked_shared_accounts = classify_shared_accounts(
            account_to_agents,
            allow_shared_account_by_agent,
        )

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=max(args.lookback_hours, 0.0))
        lanes = []

        for agent in expected_agents:
            reconcile_row = reconcile_by_agent.get(agent, {})
            lane = audit_by_agent.get(agent, {})
            usage_lane = usage_by_agent.get(agent, {})
            runtime = scan_lane_runtime(agent, cutoff=cutoff)
            expected_email = (expected_email_by_agent.get(agent) or '').strip().lower()

            quarantine_reasons = []
            probation_reasons = []

            if not lane:
                quarantine_reasons.append('lane_missing_from_audit')
                effective_ambiguity_reasons = []
                shared_account_allowlisted = False
            else:
                if reconcile_row and reconcile_row.get('status') != 'ok':
                    quarantine_reasons.append(f"reconcile:{reconcile_row.get('status')}")
                config_drift_reasons = list(lane.get('configDriftReasons') or [])
                if 'agent_missing_in_active_config' in config_drift_reasons:
                    quarantine_reasons.append('agent_config_missing')
                elif 'configured_model_missing' in config_drift_reasons:
                    quarantine_reasons.append('configured_model_missing')
                elif 'configured_model_mismatch' in config_drift_reasons or lane.get('modelMatches') is False:
                    quarantine_reasons.append('model_mismatch')
                if lane.get('desiredProfilePresent') is not True:
                    quarantine_reasons.append('desired_profile_missing')
                if lane.get('orderPinnedToDesired') is not True:
                    quarantine_reasons.append('order_not_pinned_to_desired_profile')
                desired = lane.get('desiredAccount') or {}
                desired_email = (desired.get('email') or '').strip().lower()
                if expected_email and desired_email and desired_email != expected_email:
                    quarantine_reasons.append('desired_email_mismatch')

                desired_account_id = desired.get('accountId')
                shared_account_allowlisted = bool(
                    desired_account_id
                    and desired_account_id in allowlisted_shared_accounts
                    and allow_shared_account_by_agent.get(agent, False)
                )
                effective_ambiguity_reasons = [
                    reason
                    for reason in (lane.get('ambiguityReasons') or [])
                    if not (reason == 'shared_desired_account' and shared_account_allowlisted)
                ]
                if lane.get('bindingAmbiguous') is True and effective_ambiguity_reasons:
                    quarantine_reasons.append('binding_ambiguity')

                if desired_account_id and desired_account_id in blocked_shared_accounts:
                    quarantine_reasons.append('shared_account_binding')

            usage_obj = usage_lane.get('usage') if isinstance(usage_lane, dict) else None
            usage_error = usage_obj.get('error') if isinstance(usage_obj, dict) else None
            exhausted = usage_windows_exhausted(usage_obj if isinstance(usage_obj, dict) else {})
            additional_exhausted = additional_rate_limits_exhausted(usage_obj if isinstance(usage_obj, dict) else {})
            if exhausted:
                labels = ','.join([str(item.get('label') or '?') for item in exhausted])
                quarantine_reasons.append(f'quota_exhausted:{labels}')
            if additional_exhausted:
                for item in additional_exhausted:
                    quarantine_reasons.append(f"quota_exhausted_additional:{item.get('limitToken') or 'unknown_limit'}")
            elif usage_error:
                probation_reasons.append(f'usage_probe_error:{usage_error}')

            bodyless_errors = int(runtime.get('recentBodylessUsageLimitErrors') or 0)
            latest_error = parse_iso(runtime.get('latestBodylessUsageLimitErrorAt'))
            latest_success = parse_iso(runtime.get('latestSuccessAt'))
            if bodyless_errors > 0:
                if latest_success and latest_error and latest_success > latest_error:
                    if recovered_usage_limit_probation_active(
                        now=now,
                        latest_error=latest_error,
                        latest_success=latest_success,
                    ):
                        probation_reasons.append('runtime_usage_limit_recovered')
                else:
                    quarantine_reasons.append('runtime_bodyless_usage_limit')

            if quarantine_reasons:
                status = SEVERITY_QUARANTINED
            elif probation_reasons:
                status = SEVERITY_PROBATIONARY
            else:
                status = SEVERITY_HEALTHY

            quarantine_reasons = dedupe_keep_order(quarantine_reasons)
            probation_reasons = dedupe_keep_order(probation_reasons)
            recommended_actions = build_recommended_actions(
                agent=agent,
                status=status,
                quarantine_reasons=quarantine_reasons,
                probation_reasons=probation_reasons,
                lookback_hours=args.lookback_hours,
            )

            lanes.append(
                {
                    'agent': agent,
                    'status': status,
                    'quarantineReasons': quarantine_reasons,
                    'probationReasons': probation_reasons,
                    'recommendedActions': recommended_actions,
                    'expectedEmail': expected_email_by_agent.get(agent),
                    'desiredAccount': (lane or {}).get('desiredAccount'),
                    'allowSharedAccount': allow_shared_account_by_agent.get(agent, False),
                    'sharedAccountAllowlisted': shared_account_allowlisted,
                    'reconcileStatus': reconcile_row.get('status'),
                    'ready': lane.get('ready'),
                    'bindingAmbiguous': lane.get('bindingAmbiguous'),
                    'ambiguityReasons': lane.get('ambiguityReasons') if isinstance(lane, dict) else None,
                    'effectiveAmbiguityReasons': effective_ambiguity_reasons,
                    'configEntryPresent': lane.get('configEntryPresent') if isinstance(lane, dict) else None,
                    'configDriftReasons': lane.get('configDriftReasons') if isinstance(lane, dict) else None,
                    'usage': usage_obj,
                    'additionalRateLimitExhausted': additional_exhausted,
                    'runtime': runtime,
                }
            )

        counts = {
            SEVERITY_HEALTHY: sum(1 for lane in lanes if lane['status'] == SEVERITY_HEALTHY),
            SEVERITY_PROBATIONARY: sum(1 for lane in lanes if lane['status'] == SEVERITY_PROBATIONARY),
            SEVERITY_QUARANTINED: sum(1 for lane in lanes if lane['status'] == SEVERITY_QUARANTINED),
        }

        routine_dispatch_disabled_lanes = []
        for lane in lanes:
            lane_status = str(lane.get('status') or '').strip().lower()
            if lane_status not in routine_dispatch_lane_disable_statuses:
                continue

            quarantine_reasons = dedupe_keep_order(list(lane.get('quarantineReasons') or []))
            probation_reasons = dedupe_keep_order(list(lane.get('probationReasons') or []))
            disable_reasons = dedupe_keep_order(quarantine_reasons + probation_reasons)

            quota_exhausted_signal = any(
                any(str(reason or '').startswith(prefix) for prefix in quota_exhaustion_reason_prefixes)
                for reason in disable_reasons
            )

            routine_dispatch_disabled_lanes.append(
                {
                    'agent': lane.get('agent'),
                    'status': lane_status,
                    'disableReasons': disable_reasons,
                    'quotaExhaustedSignal': quota_exhausted_signal,
                }
            )

        require_healthy = set(args.require_healthy_agent or [])
        failed_required = [lane['agent'] for lane in lanes if lane['agent'] in require_healthy and lane['status'] != SEVERITY_HEALTHY]

        payload = {
            'generatedAt': datetime.now(timezone.utc).isoformat(),
            'lookbackHours': args.lookback_hours,
            'manifestPath': str(Path(args.manifest)),
            'laneMapPath': str(Path(args.lane_map)),
            'sharedAccountBindings': shared_accounts,
            'sharedAccountBindingsAllowlisted': allowlisted_shared_accounts,
            'sharedAccountBindingsBlocked': blocked_shared_accounts,
            'statusCounts': counts,
            'routineDispatchLaneDisablePolicy': codex_quota_routing_policy,
            'routineDispatchDisabledLaneCount': len(routine_dispatch_disabled_lanes),
            'routineDispatchDisabledLanes': routine_dispatch_disabled_lanes,
            'lanes': lanes,
        }

        if args.json_output:
            out_path = Path(args.json_output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, indent=2) + '\n')
            payload['jsonOutputPath'] = str(out_path)

        print(json.dumps(payload, indent=2))

        if failed_required:
            raise SystemExit(3)
        if args.strict and (counts[SEVERITY_PROBATIONARY] > 0 or counts[SEVERITY_QUARANTINED] > 0):
            raise SystemExit(2)

    except DispatchHealthError as exc:
        print(json.dumps(emit_failure_payload(code=exc.code, message=exc.message, command=exc.command), indent=2))
        raise SystemExit(2)
    except Exception as exc:
        print(
            json.dumps(
                emit_failure_payload(
                    code='dispatch_health_unhandled_error',
                    message=f'{exc.__class__.__name__}: {str(exc)[:240]}',
                ),
                indent=2,
            )
        )
        raise SystemExit(2)


if __name__ == '__main__':
    main()
