"""Risk classification and typed-confirmation policy for Hermes actions.

This module is intentionally small and dependency-light so it can be shared by
the read-only control inventory, CLI approval gates, and tests without pulling
in the live tool registry or mutating runtime state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


class RiskClass:
    READ_ONLY = "read_only"
    LOCAL_WRITE = "local_write"
    PRIVATE_DATA_ACCESS = "private_data_access"
    CREDENTIAL_SENSITIVE = "credential_sensitive"
    EXTERNAL_SIDE_EFFECT = "external_side_effect"
    DESTRUCTIVE = "destructive"
    FINANCIAL_OR_ACCOUNT_ACTION = "financial_or_account_action"
    UNKNOWN_RESTRICTED = "unknown_restricted"


@dataclass(frozen=True)
class RiskDecision:
    risk_class: str
    risk_tier: str
    approval_policy: str
    requires_typed_confirmation: bool = False
    default_restricted: bool = False
    reason: str = ""


READ_ONLY_TOOLS = frozenset({
    "browser_back",
    "browser_console",
    "browser_get_images",
    "browser_snapshot",
    "browser_vision",
    "clarify",
    "feishu_doc_read",
    "feishu_drive_list_comment_replies",
    "feishu_drive_list_comments",
    "ha_get_state",
    "ha_list_entities",
    "ha_list_services",
    "read_file",
    "search_files",
    "session_search",
    "skill_view",
    "skills_list",
    "video_analyze",
    "vision_analyze",
    "web_extract",
    "web_search",
    "x_search",
    "yb_query_group_info",
    "yb_query_group_members",
    "yb_search_sticker",
})

LOCAL_WRITE_TOOLS = frozenset({
    "delegate_task",
    "image_generate",
    "kanban_block",
    "kanban_comment",
    "kanban_complete",
    "kanban_create",
    "kanban_heartbeat",
    "kanban_link",
    "kanban_unblock",
    "patch",
    "text_to_speech",
    "video_generate",
    "write_file",
})

PRIVATE_DATA_TOOLS = frozenset({
    "memory",
})

CREDENTIAL_SENSITIVE_TOOLS = frozenset({
    "browser_cdp",
    "browser_click",
    "browser_dialog",
    "browser_navigate",
    "browser_press",
    "browser_scroll",
    "browser_type",
    "computer_use",
    "cronjob",
    "discord_admin",
    "execute_code",
    "process",
    "skill_manage",
    "terminal",
})

EXTERNAL_SIDE_EFFECT_TOOLS = frozenset({
    "discord",
    "feishu_drive_add_comment",
    "feishu_drive_reply_comment",
    "ha_call_service",
    "send_message",
    "yb_send_dm",
    "yb_send_sticker",
})

DESTRUCTIVE_TOOLS = frozenset({
    "delete_file",
    "file_delete",
})


def _decision(
    risk_class: str,
    risk_tier: str,
    approval_policy: str,
    *,
    reason: str,
    default_restricted: bool = False,
) -> RiskDecision:
    return RiskDecision(
        risk_class=risk_class,
        risk_tier=risk_tier,
        approval_policy=approval_policy,
        requires_typed_confirmation=approval_policy == "typed_confirm",
        default_restricted=default_restricted,
        reason=reason,
    )


def _read_only(reason: str) -> RiskDecision:
    return _decision(RiskClass.READ_ONLY, "R0", "allow", reason=reason)


def _local_write(reason: str) -> RiskDecision:
    return _decision(RiskClass.LOCAL_WRITE, "R2", "confirm", reason=reason)


def _private_data(reason: str, *, typed: bool = False) -> RiskDecision:
    return _decision(
        RiskClass.PRIVATE_DATA_ACCESS,
        "R4" if typed else "R3",
        "typed_confirm" if typed else "confirm",
        reason=reason,
    )


def _credential_sensitive(reason: str) -> RiskDecision:
    return _decision(
        RiskClass.CREDENTIAL_SENSITIVE,
        "R4",
        "typed_confirm",
        reason=reason,
    )


def _external_side_effect(reason: str) -> RiskDecision:
    return _decision(
        RiskClass.EXTERNAL_SIDE_EFFECT,
        "R4",
        "typed_confirm",
        reason=reason,
    )


def _destructive(reason: str) -> RiskDecision:
    return _decision(
        RiskClass.DESTRUCTIVE,
        "R4",
        "typed_confirm",
        reason=reason,
    )


def _financial_or_account(reason: str) -> RiskDecision:
    return _decision(
        RiskClass.FINANCIAL_OR_ACCOUNT_ACTION,
        "R5",
        "deny",
        reason=reason,
    )


def _unknown_restricted(reason: str) -> RiskDecision:
    return _decision(
        RiskClass.UNKNOWN_RESTRICTED,
        "R4",
        "typed_confirm",
        reason=reason,
        default_restricted=True,
    )


FINANCIAL_OR_ACCOUNT_RE = re.compile(
    r"\b("
    r"buy|sell|trade|transfer|withdraw|deposit|purchase|payment|refund|"
    r"delete[-_ ]?account|close[-_ ]?account|change[-_ ]?owner|"
    r"aws\s+(?:iam|organizations|account)|appstore|testflight"
    r")\b",
    re.IGNORECASE,
)

REMOTE_SHELL_RE = re.compile(
    r"\b(curl|wget)\b[^\n]*(?:\|\s*(?:ba)?sh\b|"
    r"\|\s*zsh\b|<\s*<\s*\(|\bsh\s+<\s*\()",
    re.IGNORECASE,
)

CREDENTIAL_WRITE_RE = re.compile(
    r"("
    r"\b(API_KEY|TOKEN|SECRET|PASSWORD|PRIVATE_KEY|AUTH|CREDENTIAL)\s*=|"
    r"(--api-key|--token|--password|--secret)\b|"
    r"(\.env(?:\.[^\s/]+)?|config\.yaml|authorized_keys|\.ssh/|"
    r"\.netrc|\.npmrc|\.pypirc|keychain|docker\s+login|gh\s+auth|aws\s+configure)"
    r")",
    re.IGNORECASE,
)

CREDENTIAL_DESTINATION_RE = re.compile(
    r"(?:>>?|\btee\b[^\n]*|\bcp\b[^\n]*|\bmv\b[^\n]*|\binstall\b[^\n]*)\s+"
    r"['\"]?(?:~|\$HOME|\$\{HOME\}|/Users/[^/\s]+)?/?"
    r"(?:(?:\.hermes|\.config|\.aws|\.ssh|\.docker|\.kube|\.gnupg|"
    r"\.openai|\.anthropic|\.gemini|\.xai|\.npm|\.pypirc|\.netrc)"
    r"(?:/[^;\n|&\s'\"]*)?|"
    r"[^;\n|&\s'\"]*(?:token|secret|credential|password|private[_-]?key|"
    r"api[_-]?key|provider[-_]?token|auth)[^;\n|&\s'\"]*)",
    re.IGNORECASE,
)

SECRET_LITERAL_RE = re.compile(
    r"(?:sk-[A-Za-z0-9_\-]{12,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"gh[pousr]_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9\-]{10,}|"
    r"AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----)",
    re.IGNORECASE,
)

EXTERNAL_SIDE_EFFECT_RE = re.compile(
    r"\b("
    r"send|email|post|publish|deploy|upload|release|"
    r"npm\s+publish|twine\s+upload|gh\s+release|git\s+push|"
    r"slack|discord|telegram|webhook"
    r")\b",
    re.IGNORECASE,
)

DESTRUCTIVE_RE = re.compile(
    r"("
    r"\bgit\s+reset\s+--hard\b|"
    r"\bgit\s+clean\s+-[^\s]*f|"
    r"\bgit\s+branch\s+-D\b|"
    r"\brm\s+-[^\s]*r|"
    r"\bfind\b[^\n]*-(?:delete|exec(?:dir)?\s+(?:/\S*/)?rm)\b|"
    r"\bDROP\s+(?:TABLE|DATABASE)\b|"
    r"\bTRUNCATE\s+(?:TABLE\s+)?\w+|"
    r"\bDELETE\s+FROM\b(?![^\n]*\bWHERE\b)|"
    r"\bhermes\s+gateway\s+(?:stop|restart)\b|"
    r"\bhermes\s+update\b"
    r")",
    re.IGNORECASE,
)

TMP_RECURSIVE_DELETE_RE = re.compile(
    r"\brm\s+-[^\s]*r[^\n]*(?:/tmp/|/tmp\b|/private/tmp/|/private/tmp\b)",
    re.IGNORECASE,
)

LOCAL_WRITE_RE = re.compile(
    r"("
    r"\b(write|patch|edit|save|create|append|overwrite)\b|"
    r"\b(cp|mv|install|tee|sed\s+-[^\s]*i)\b|"
    r">>?"
    r")",
    re.IGNORECASE,
)

READ_ONLY_RE = re.compile(
    r"\b(read|list|show|status|health|search|inspect|view|describe|"
    r"cat|ls|grep|rg|find\s+\S+\s+-print|git\s+status|git\s+diff)\b",
    re.IGNORECASE,
)

PRIVATE_MUTATION_RE = re.compile(
    r"\b(add|append|create|delete|forget|mutate|remove|reset|save|store|update|write)\b",
    re.IGNORECASE,
)


def classify_command(command: str, *, description: str | None = None) -> RiskDecision:
    """Classify a shell-ish command or shortcut without executing it."""
    text = f"{command or ''} {description or ''}".strip()
    if not text:
        return _read_only("empty command text")
    if FINANCIAL_OR_ACCOUNT_RE.search(text):
        return _financial_or_account("financial or account-management operation")
    if REMOTE_SHELL_RE.search(text):
        return _external_side_effect("remote content execution through shell")
    if CREDENTIAL_WRITE_RE.search(text) or CREDENTIAL_DESTINATION_RE.search(text):
        return _credential_sensitive("credential or secret material handling")
    if SECRET_LITERAL_RE.search(text) and LOCAL_WRITE_RE.search(text):
        return _credential_sensitive("secret-shaped value written to local destination")
    if EXTERNAL_SIDE_EFFECT_RE.search(text):
        return _external_side_effect("external side effect or publication")
    if TMP_RECURSIVE_DELETE_RE.search(text):
        return _local_write("temporary-directory recursive delete")
    if DESTRUCTIVE_RE.search(text):
        return _destructive("destructive local operation")
    if LOCAL_WRITE_RE.search(text):
        return _local_write("local write operation")
    if READ_ONLY_RE.search(text):
        return _read_only("read-only inspection")
    return _decision(RiskClass.READ_ONLY, "R1", "allow", reason="unclassified low-risk command")


def classify_tool_action(
    tool_name: str,
    *,
    action: str | None = None,
    description: str | None = None,
    toolset: str | None = None,
) -> RiskDecision:
    """Classify a registered tool or plugin action by explicit safe defaults."""
    name = (tool_name or "").strip()
    normalized = name.lower()
    text = " ".join(part for part in (name, action or "", description or "", toolset or "") if part).strip()

    if normalized in READ_ONLY_TOOLS:
        if normalized == "session_search":
            return _private_data("read-only private session search", typed=False)
        return _read_only("explicit read-only tool mapping")
    if normalized in LOCAL_WRITE_TOOLS:
        if DESTRUCTIVE_RE.search(text):
            return _destructive("mapped local-write tool with destructive action text")
        return _local_write("explicit local-write tool mapping")
    if normalized in PRIVATE_DATA_TOOLS:
        return _private_data("private persistent memory access", typed=True)
    if normalized in CREDENTIAL_SENSITIVE_TOOLS:
        return _credential_sensitive("credential-sensitive or host-control tool")
    if normalized in EXTERNAL_SIDE_EFFECT_TOOLS:
        return _external_side_effect("external side-effect tool")
    if normalized in DESTRUCTIVE_TOOLS:
        return _destructive("explicit destructive tool mapping")

    # Fallback for plugin/shortcut names that carry obvious risk words.
    command_decision = classify_command(text)
    if command_decision.risk_class not in {RiskClass.READ_ONLY, RiskClass.LOCAL_WRITE}:
        return command_decision
    if PRIVATE_MUTATION_RE.search(text) and "memory" in text.lower():
        return _private_data("unmapped memory mutation-capable action", typed=True)
    if READ_ONLY_RE.search(text):
        return command_decision
    return _unknown_restricted("unmapped tool defaults to restricted policy")


def typed_confirmation_phrase(decision: RiskDecision) -> str:
    """Return a stable exact phrase that does not include secret-bearing text."""
    label = decision.risk_class.replace("_", " ").upper()
    return f"CONFIRM {label}"


def validate_typed_confirmation(raw_value: object, expected_phrase: str) -> bool:
    """Require byte-for-byte exact user intent; whitespace and aliases fail."""
    if raw_value is None:
        return False
    return str(raw_value) == expected_phrase
