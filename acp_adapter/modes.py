"""ACP's edit modes, one table for the in-process server and the gateway path.

mode id -> (edit approval policy, display name, description). On the gateway path the policy is the
session's own on the owner (``session.mutate edit_policy``), enforced where the edit runs
(``gateway/session_local_editor.py``); sensitive paths always ask.
"""

DEFAULT_MODE = "default"
DEFAULT_EDIT_APPROVAL_POLICY = "ask"
EDIT_APPROVAL_POLICY_CONFIG_ID = "edit_approval_policy"
MODES: dict[str, tuple[str, str, str]] = {
    "default": ("ask", "Default", "Ask before edits."),
    "accept_edits": ("workspace_session", "Accept Edits",
                     "Auto-allow workspace and temp-dir edits; still asks for sensitive paths."),
    "dont_ask": ("session", "Don't Ask", "Auto-allow file edits for this session except sensitive paths."),
}
MODE_TO_EDIT_APPROVAL_POLICY = {mode: spec[0] for mode, spec in MODES.items()}
EDIT_APPROVAL_POLICY_TO_MODE = {spec[0]: mode for mode, spec in MODES.items()}
