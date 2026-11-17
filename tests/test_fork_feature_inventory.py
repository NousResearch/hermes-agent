"""Fork feature inventory — every EverAfter-specific capability must stay wired.

This is a merge guard, not a unit test. The 2026-08-24 upstream sync showed the
real risk of a big merge is not conflicts (those are visible) but fork code that
merges cleanly and stops running because upstream moved the call site out from
under it. Two live examples that shipped clean and were caught only by running
things: ``backfill_session_scope`` lost the line that executed its writer, and
``hermes_cli/main.py`` kept an inline ``tools`` subparser after upstream
extracted it — leaving a ``NameError: tools_sub`` that broke every CLI parse.

Each entry asserts a *call site*, not just that a symbol exists somewhere. When
one fails after a merge, the feature is silently disabled — find where upstream
moved the anchor and re-apply, do not delete the assertion.

See docs/superpowers/plans/2026-08-24-upstream-sync.md.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


# (feature, file, needle, what the needle proves)
WIRING = [
    # ── RBAC: the three enforcement points ────────────────────────────────
    ("RBAC policy", "gateway/tool_access.py", "FLOOR_TOOLSETS", "floor toolsets defined"),
    ("RBAC policy", "gateway/tool_access.py", "BUILTIN_ROLES", "built-in roles defined"),
    ("RBAC policy", "gateway/tool_access.py", "def _effective_grant", "channel_roles union"),
    ("RBAC point A", "gateway/authz_mixin.py", "policy_for_source", "message gate in authz mixin"),
    ("RBAC point A", "gateway/authz_mixin.py", "_rbac_policy.is_authorized", "authorization decision wired"),
    ("RBAC point B", "gateway/run.py", "return filter_enabled_toolsets(", "toolset filter in resolver"),
    ("RBAC point C", "model_tools.py", "denial_for_current_tool", "execution backstop"),
    # ── RBAC admin surfaces ───────────────────────────────────────────────
    ("hermes users", "hermes_cli/users.py", "def register_users_subcommands", "registrar"),
    ("hermes users", "hermes_cli/main.py", "register_users_subcommands(subparsers)", "registered in main"),
    ("hermes tools rbac", "hermes_cli/tools_list.py", "def register_tools_rbac_subcommand", "registrar"),
    ("hermes tools rbac", "hermes_cli/subcommands/tools.py", "register_tools_rbac_subcommand(tools_sub)", "hooked into parser"),
    # ── Fork-only toolsets ────────────────────────────────────────────────
    ("notion_api", "tools/notion_api_tool.py", 'toolset="notion"', "own toolset"),
    ("jira_api", "tools/jira_api_tool.py", "_ALLOWED_METHODS", "read-only guard"),
    ("jira_api_write", "tools/jira_api_tool.py", "_WRITE_ALLOWLIST", "curated write allowlist, own toolset"),
    ("slack_post_thread", "tools/slack_post_thread_tool.py", 'toolset="slack_post"', "non-floor toolset"),
    ("webflow_asset_upload", "tools/webflow_asset_tool.py", "is_protected_data_path", "credential-read guard"),
    ("ownership tool", "tools/ownership_tool.py", 'toolset="ownership"', "floor toolset"),
    ("pdf_pages", "tools/pdf_pages_tool.py", 'toolset="pdf_pages"', "own toolset, fixed argv"),
    ("allow_bots_channels", "plugins/platforms/slack/adapter.py", "def _slack_allow_bots_channels", "per-channel app admission"),
    ("allow_bots_channels", "gateway/authz_mixin.py", "_slack_allow_bots_channels(source)", "gate admits listed channels; RBAC outranks bypass"),
    # ── Cost attribution ──────────────────────────────────────────────────
    ("hermes costs", "agent/cost_attribution.py", "def attribute_sessions", "lineage query"),
    ("hermes costs", "agent/cost_attribution.py", "def reprice", "reprice pass"),
    ("hermes costs", "hermes_cli/subcommands/costs.py", "def build_costs_parser", "parser"),
    ("hermes costs", "hermes_cli/main.py", "build_costs_parser(subparsers, cmd_costs=cmd_costs)", "registered in main"),
    ("pricing overrides", "agent/usage_pricing.py", "pricing_override_for(model_name, route)", "override consulted first"),
    ("relevance gate accounting", "gateway/run.py", "def _relevance_gate_accounting_token", "classifier usage booked to the channel session"),
    ("relevance gate accounting", "gateway/run.py", 'task="relevance_gate"', "aux task tag"),
    # ── Cost budget + alerts ──────────────────────────────────────────────
    ("cost budget", "agent/cost_budget.py", "def preflight_cron_job", "skip decision + overrun alert"),
    ("cost budget", "agent/cost_budget.py", "def watch", "per-tick threshold/run/unpriced alerts"),
    ("cost budget", "cron/scheduler.py", "_cost_budget.preflight_cron_job(job, adapters=adapters, loop=loop)", "pre-run skip in the shared run body"),
    ("cost budget", "cron/scheduler.py", 'mark_job_skipped(job["id"], "skipped_budget"', "skip recorded without consuming a run"),
    ("cost budget", "cron/jobs.py", "def mark_job_skipped", "skip keeps streak/repeat, re-arms one-shots"),
    ("cost budget", "tools/cronjob_tools.py", 'updates["discretionary"] = True', "set-only discretionary flag"),
    ("cost budget", "cron/scheduler.py", "_cost_budget.watch(adapters=adapters, loop=loop)", "tick watches this profile's budget"),
    ("cost budget", "cron/executions.py", "note: Optional[str] = None", "ledger records the skip on a completed row"),
    ("cost_budget_status", "tools/cost_budget_tool.py", 'toolset="cost_budget"', "own non-floor toolset"),
    # ── Drive per-user access check ───────────────────────────────────────
    ("Drive access check", "plugins/google_drive_sa/access.py", "def require_access", "gate"),
    ("Drive access check", "plugins/google_drive_sa/access.py", "def filter_listing", "listing filter"),
    ("Drive folder_access", "plugins/google_drive_sa/access.py", "def folder_grant", "operator folder grants for unreadable ACLs"),
    ("Drive access check", "plugins/google_drive_sa/tools.py", "access.require_access(file_id, access.READER)", "read gated"),
    ("Drive access check", "plugins/google_drive_sa/tools.py", "access.filter_listing(", "list gated"),
    ("Drive access check", "plugins/google_drive_sa/sheets_tools.py", "access.require_access(sid, access.WRITER)", "sheet writes gated"),
    ("Drive access check", "plugins/google_drive_sa/docs_tools.py", "access.require_access(doc_id, access.WRITER)", "doc writes gated"),
    ("Drive access check", "plugins/google_drive_sa/identity.py", "def resolve_email", "email resolver"),
    ("Drive access check", "cron/tool_approval_context.py", "def get_cron_job_id", "cron owner path"),
    ("Drive access check", "cron/scheduler.py", "job_id=job.get(\"id\")", "scheduler passes job id"),
    ("Config bridge", "gateway/config.py", 'bridged["user_emails"]', "user_emails -> extra"),
    # ── Multi-user session isolation ──────────────────────────────────────
    ("Session visibility", "hermes_state.py", "def build_visibility_where", "SQL scope fragment"),
    ("Session visibility", "hermes_state.py", "def session_row_visible", "row-level gate"),
    ("Session scope backfill", "hermes_state.py", "self._execute_write(_do)", "backfill actually writes"),
    ("search_messages gate", "hermes_state_search.py", "build_visibility_where(scope", "SQL gate applied"),
    ("session_search gate", "tools/session_search_tool.py", "resolve_search_scope()", "scope resolved once"),
    ("session_search title gate", "tools/session_search_tool.py", "if not session_row_visible(session_meta, scope)", "title path gated"),
    ("reconcile via async boundary", "gateway/run.py", "await self.async_session_store.reconcile_db_scope()", "does not block the loop"),
    # ── Cross-user data protection ────────────────────────────────────────
    ("Protected paths", "agent/file_safety.py", "def is_protected_data_path", "matcher"),
    ("Protected paths", "tools/file_tools.py", "is_protected_data_path", "enforced in file tools"),
    ("Data-access audit", "agent/data_access_audit.py", "def record_access", "audit writer"),
    ("Data-access audit", "tools/terminal_tool.py", "record_command_access", "terminal logs access"),
    # ── Automation ownership ──────────────────────────────────────────────
    ("Ownership gate", "agent/automation_ownership.py", "def check_edit", "soft edit gate"),
    ("Ownership gate", "agent/automation_ownership.py", "def _require_owner_or_admin", "shared owner check"),
    ("Ownership: skills", "tools/skill_manager_tool.py", "_CONFIRM_OWNER", "confirm token wired"),
    ("Ownership: cron", "tools/cronjob_tools.py", "register_creator", "creator registered"),
    ("Ownership: files", "tools/file_tools.py", "_automation_ownership_check", "write/patch gated"),
    ("hermes own", "hermes_cli/own.py", "def run_own", "CLI entrypoint"),
    ("Ownership guidance", "agent/system_prompt.py", "AUTOMATION_OWNERSHIP_GUIDANCE", "in stable segment"),
    # ── Cron RBAC ceiling ─────────────────────────────────────────────────
    ("Cron ceiling", "cron/rbac_ceiling.py", "def apply_cron_toolset_ceiling", "ceiling fn"),
    ("Cron ceiling", "cron/scheduler.py", "enabled_toolsets=_cron_enabled_toolsets_with_ceiling(job, _cfg)", "wired at AIAgent build"),
    ("Cron create gate", "tools/cronjob_tools.py", "_rbac_creation_error", "create/update gate"),
    ("Cron unattended ack", "tools/cronjob_tools.py", "_unattended_ack_error", "ack gate"),
    ("Cron capability preflight", "cron/capability_preflight.py", "def evaluate_job_capabilities", "effective-vs-required evaluation"),
    ("Cron capability preflight", "tools/cronjob_tools.py", "_capability_preflight(", "create/update rejects unmet requirements"),
    ("Cron capability preflight", "cron/scheduler.py", '("capabilities", lambda: _preflight_check_capabilities(job, cfg))', "runtime blocked_config check"),
    ("Cron capability preflight", "tools/ownership_tool.py", "_cron_capabilities_after_transfer", "transfer re-evaluates under new owner"),
    ("Cron failure visibility", "cron/scheduler.py", "def _agent_reported_failure", "[FAILED] marker fails the run"),
    ("Cron failure visibility", "cron/scheduler.py", "def _alert_owner_of_undelivered_failure", "local-only failures DM the owner"),
    ("Cron no-shell policy", "cron/scheduler.py", 'disabled += ["terminal", "code_execution"]', "cron agents never get a shell"),
    ("Cron post_script", "cron/scheduler.py", "def _run_job_post_script", "deterministic apply step"),
    ("Cron post_script", "tools/cronjob_tools.py", '"post_script": {', "schema field"),
    ("Cron script imports", "cron/capability_preflight.py", "def script_import_problems", "gateway-interpreter import check"),
    ("Cron script imports", "cron/scheduler.py", "_import_reason = _preflight_check_script_imports(job)", "checked before any script runs"),
    ("Cron transfer gate", "tools/ownership_tool.py", "confirm_capability_loss", "transfer refused on capability loss"),
    ("slack_post_thread root post", "tools/slack_post_thread_tool.py", "def _get_permalink", "root posts + permalink receipt"),
    ("Cron approval context", "cron/tool_approval_context.py", "def set_cron_tool_context", "owner grant export"),
    # ── Quiet channels / silent completion ────────────────────────────────
    ("Quiet channels", "gateway/run.py", "def _is_quiet_channel", "resolver"),
    ("Quiet channels", "plugins/platforms/slack/adapter.py", "def _slack_quiet_channels", "adapter half"),
    ("Quiet channels", "plugins/platforms/slack/adapter.py", "in self._slack_quiet_channels()", "reactions suppressed"),
    ("Silent completion", "gateway/run.py", "agent._silent_completion_ok = _is_quiet_channel", "set per turn"),
    ("turn_end", "agent/conversation_loop.py", "_called_terminal_turn_end", "loop honours turn_end"),
    ("Silent empty net", "agent/conversation_loop.py", "_should_accept_silent_empty", "empty-turn safety net"),
    ("Codex silent finish", "agent/conversation_loop.py", "_codex_incomplete_exhausted_result", "quiet codex path"),
    ("slack_react", "tools/slack_react_tool.py", "HERMES_SESSION_MESSAGE_ID", "targets triggering message"),
    ("slack_react targeting", "plugins/platforms/slack/adapter.py", "message_id=ts,", "source carries the ts"),
    # ── Relevance pre-gate ────────────────────────────────────────────────
    ("Relevance gate", "gateway/run.py", "_relevance_gate_should_skip", "invoked in dispatch"),
    ("Relevance bypass", "gateway/platforms/base.py", "directly_addressed", "MessageEvent field"),
    ("Relevance bypass", "plugins/platforms/slack/adapter.py", "directly_addressed=bool(is_one_to_one_dm or is_mentioned)", "adapter sets it"),
    # ── Model overrides ───────────────────────────────────────────────────
    ("Channel models", "gateway/run.py", "def _apply_channel_model_override", "per-channel"),
    ("Skill models", "gateway/run.py", "_apply_skill_model_override", "per-skill"),
    ("Cron skill models", "cron/scheduler.py", "def _effective_job_model_fields", "job/skill precedence"),
    ("Cron delivery format", "cron/scheduler.py", 'delivery_content = f"**{task_name}**\\n\\n{content}"', "bold-name wrapper, no job_id/footer"),
    ("Delegation models", "tools/delegate_tool.py", "_override_runtime_cache", "per-task override"),
    # ── Per-tool approval gate ────────────────────────────────────────────
    ("Approval gate", "tools/approval.py", "def tool_requires_approval", "require_for_tools glob"),
    ("Approval gate", "tools/approval.py", "def check_tool_approval", "entrypoint"),
    # ── Display / memory ──────────────────────────────────────────────────
    ("Home-path collapse", "agent/redact.py", "def collapse_home_path", "helper"),
    ("Home-path collapse", "agent/display.py", "return collapse_home_path(preview) if preview else preview", "applied at boundary"),
    ("Slack author anchor", "plugins/platforms/slack/adapter.py", "def _anchor_message_author", "helper"),
    ("Slack author anchor", "plugins/platforms/slack/adapter.py", "text = _anchor_message_author(text, user_name, is_dm)", "call site"),
    # ── Agentic outbound messaging (fork override of an upstream removal) ──
    ("send_message registered", "tools/send_message_tool.py", 'registry.register(', "registration restored"),
    ("send_message registered", "tools/send_message_tool.py", 'toolset="messaging"', "under the messaging toolset"),
    ("messaging toolset", "toolsets.py", '"messaging": {', "toolset restored"),
    ("Memory global-only", "tools/memory_tool.py", "def memory_schema_for", "global-only schema"),
    ("Holographic scopes", "plugins/memory/holographic/__init__.py", "def _bundle_for_current_scope", "per-scope stores"),
    # ── Config plumbing ───────────────────────────────────────────────────
    ("Config bridge", "gateway/config.py", 'bridged["user_roles"]', "user_roles -> extra"),
    ("Config bridge", "gateway/config.py", 'bridged["user_names"]', "user_names -> extra"),
    ("Config bridge", "gateway/config.py", 'bridged["quiet_channels"]', "quiet_channels -> extra"),
    ("Config defaults", "hermes_cli/config_defaults.py", '"quiet_channels"', "fork slack defaults"),
    ("Slack yaml hook", "plugins/platforms/slack/adapter.py", "SLACK_HOME_CHANNEL_PROMPT", "home_channel_prompt bridged"),
    # ── Bot-authored thread roots must not become a permanent wake zone ────
    ("Bot-thread wake opt-in", "plugins/platforms/slack/adapter.py", "def _slack_wake_in_bot_authored_threads", "helper"),
    ("Bot-thread wake opt-in", "plugins/platforms/slack/adapter.py", "and self._slack_wake_in_bot_authored_threads()", "gates check 4"),
    ("Bot-thread wake opt-in", "plugins/platforms/slack/adapter.py", "SLACK_WAKE_IN_BOT_AUTHORED_THREADS", "yaml -> env bridge"),
    ("Bot-thread wake opt-in", "hermes_cli/config_defaults.py", '"wake_in_bot_authored_threads": False', "defaults to pre-sync behaviour"),
]


@pytest.mark.parametrize(
    "feature,relpath,needle,proves",
    WIRING,
    ids=[f"{f}:{Path(p).name}:{i}" for i, (f, p, _, _) in enumerate(WIRING)],
)
def test_fork_feature_is_wired(feature, relpath, needle, proves):
    path = ROOT / relpath
    assert path.exists(), (
        f"{feature}: {relpath} is missing. Upstream may have moved or deleted it — "
        f"find the new home and re-apply the fork change."
    )
    text = path.read_text(encoding="utf-8", errors="replace")
    assert needle in text, (
        f"{feature}: {relpath} no longer contains the call site that proves "
        f"'{proves}'.\n"
        f"  expected: {needle}\n"
        f"This usually means an upstream merge relocated the anchor and the fork "
        f"behaviour is now silently disabled. Re-wire it; do not delete this check."
    )


def test_fork_toolsets_are_registered():
    """The fork's integrations plus ownership must exist as their own
    toolsets, so RBAC can gate each independently."""
    import toolsets

    all_toolsets = set(toolsets.get_all_toolsets())
    for name in ("notion", "jira", "jira_write", "slack_post", "webflow_assets",
                 "ownership", "slack", "video_frames", "pdf_pages", "cost_budget"):
        assert name in all_toolsets, (
            f"toolset '{name}' is not registered — RBAC cannot gate it, and any "
            f"role granting it becomes a no-op."
        )


def test_floor_toolsets_reach_valid_role_users_only():
    """Floors are granted at the enforcement surface, never to roleless users."""
    from gateway.tool_access import FLOOR_TOOLSETS, policy_from_extra

    assert {"clarify", "todo", "slack", "ownership"} <= set(FLOOR_TOOLSETS)

    policy = policy_from_extra({"user_roles": {"U_OP": "operator"}})
    assert policy.enabled
    for floor in ("ownership", "slack", "todo", "clarify"):
        assert policy.can_use_tool("U_OP", floor), f"floor {floor} denied to a valid role"
        assert not policy.can_use_tool("U_NOBODY", floor), (
            f"floor {floor} leaked to a roleless user — deny-until-assigned is broken"
        )


def test_agent_can_still_choose_to_send_messages():
    """The bot must be able to DM a user by agentic choice.

    Upstream removed the ``messaging`` toolset and the ``send_message``
    registration on the view that an agent should not decide to send messages on
    its own (see the NOTE in tools/send_message_tool.py). The fork's automations
    depend on exactly that — a cron worker reporting back to whoever asked, an
    ownership hand-off notice, a triage bot following up.

    This is the silent kind of regression: the module keeps importing, every
    direct caller (cron delivery, ``hermes send``, ``automation_ownership.
    _send_dm``) keeps working, and the suite stays green — while the model can no
    longer choose to send. So assert the model-facing wiring, not the import.
    """
    import toolsets

    all_toolsets = toolsets.get_all_toolsets()
    assert "messaging" in all_toolsets, (
        "the 'messaging' toolset is gone — the agent can no longer choose to "
        "send a message. An upstream sync most likely removed it again; restore "
        "the entry in toolsets.py."
    )

    entry = toolsets.TOOLSETS["messaging"] if hasattr(toolsets, "TOOLSETS") else None
    if entry is not None:
        assert "send_message" in entry.get("tools", []), (
            "'messaging' no longer contains send_message"
        )


def test_operator_role_still_excludes_terminal():
    """The operator role must never reach a host shell: `printenv` / `cat
    ~/.hermes/.env` would hand over every credential on the box."""
    from gateway.tool_access import policy_from_extra

    policy = policy_from_extra({"user_roles": {"U_OP": "operator"}})
    assert not policy.can_use_tool("U_OP", "terminal")
    assert not policy.can_use_tool("U_OP", "code_execution")
