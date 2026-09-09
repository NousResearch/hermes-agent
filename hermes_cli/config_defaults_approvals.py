"""Default ``approvals`` block for DEFAULT_CONFIG (see config_defaults.py).

Pure-data leaf module — must not import from hermes_cli.config. Comments are the user-facing
docs of config.yaml.
"""

# Approvals for dangerous commands.
# mode: manual (always prompt) | smart (aux LLM auto-approves low-risk) | off (= --yolo)
# cron_mode / single_query_mode / unattended_mode: deny | approve — what to do when a
#   cron job, a -q session (HERMES_INTERACTIVE=1 but nobody to answer), or an unattended
#   platform (webhook, msgraph_webhook, api_server; no /approve channel) hits one.
#   deny blocks instantly so the agent finds another way instead of waiting out the
#   timeout and failing closed.
# timeout: seconds before an unanswered prompt fails closed (CLI and gateway). 60s
#   proved too tight for Telegram/Discord push notifications, hence 300.
APPROVALS_DEFAULTS = {
    # single_query_mode — what to do when a single-query (-q) session hits a dangerous command. -q runs
    # export HERMES_INTERACTIVE=1 (for interactive sudo prompts) but have NO user waiting to answer
    # approval prompts — an unanswered prompt just waits the full timeout then fails closed, so the
    # agent is forced to work around the block (often via execute_code). This setting makes that intent
    # explicit: deny    — block the command and let the agent find another way (default, safe; mirrors
    # cron_mode deny) approve — auto-approve all dangerous commands in single-query mode These surfaces
    # bind a session platform like chat gateways do, but have no send_exec_approval and no /approve
    # channel — a pending approval there just blocks for the full timeout with nobody to answer (#37284,
    # #87509): deny    — block the command instantly and let the agent find another way (default, safe;
    # mirrors cron_mode deny) approve — auto-approve all dangerous commands on unattended platforms
    # Shared by the CLI prompt and gateway/messaging waits. Messaging approvals arrive as a push
    # notification the user may not see immediately — 60s proved too tight on Telegram/Discord (the
    # prompt expired before the user reached their phone), so the default is 300.
    "mode": "smart",
    "timeout": 300,
    "cron_mode": "deny",
    "single_query_mode": "deny",
    "unattended_mode": "deny",
    # Extra rules appended to the smart-approval guardian's SYSTEM prompt, e.g. "Always ESCALATE
    # commands touching /etc".
    "smart_policy": "",
    # After this many consecutive guardian DENYs in a session, the deny message escalates to a
    # hard-stop (report to user / ask for /approve). Approval resets; 0 off.
    "denial_breaker_threshold": 3,
    # Case-insensitive fnmatch globs against terminal commands; a match blocks even under --yolo
    # / mode=off. Quote in YAML when starting with * or containing {}/!/: e.g. "git push
    # --force*".
    "deny": [],
    # Operator rules that FORCE the approval prompt for commands no built-in pattern flags
    # (restarting your own gateway, kubectl against prod). Same glob matching as `deny`. A
    # string entry is reviewed by a human every time (no smart approval, no "Always"); a dict
    # entry {pattern, description?, review: human|smart} may hand the first look to the smart
    # guardian instead. See the security guide.
    "command_approval_required": [],
    # /reload-mcp confirms before rebuilding the MCP tool set (it invalidates the prompt cache,
    # so the next message re-sends full input). "Always Approve" → false.
    "mcp_reload_confirm": True,
    # /clear, /new, /reset, /undo confirm before discarding state (Approve Once / Always Approve
    # / Cancel via tools.slash_confirm; native buttons on Telegram/ Discord/Slack). "Always
    # Approve" → false. HERMES_TUI_NO_CONFIRM=1 skips the TUI modal.
    "destructive_slash_confirm": True,
}
