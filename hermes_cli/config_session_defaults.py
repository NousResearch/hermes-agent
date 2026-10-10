"""Defaults for session-level user preferences."""

SESSION_DEFAULTS = {
    # Per-terminal `hermes -c`: each CLI session writes a breadcrumb under
    # $HERMES_HOME/terminal-sessions/<terminal-id>, so bare -c/--continue resumes THIS
    # terminal's session (tmux/kitty/wezterm pane, tty). false = resume globally most-recent.
    "terminal_continue": True,
    # Optional sub-contexts inside one session. The main response classifies
    # each turn without another model call; historical all-message context stays
    # the default unless explicitly enabled.
    "topic_segmentation": {"enabled": False},
}
