"""Pure-data factory for display configuration defaults."""


def _display_defaults():
    """Display settings shared by the CLI, TUI and messaging surfaces."""
    return {
        "compact": False,
        "personality": "",
        "resume_display": "full",
        # Recap tuning for /resume and startup resume.
        "resume_exchanges": 10,            # max user+assistant pairs to show
        "resume_max_user_chars": 300,      # truncate user message text
        "resume_max_assistant_chars": 200, # truncate non-last assistant text
        "resume_max_assistant_lines": 3,   # truncate non-last assistant lines
        # Skip tool-call-only assistant entries in the recap so it isn't dominated by `[2 tool
        # calls: ...]` lines; False shows them inline.
        "resume_skip_tool_only": True,
        "busy_input_mode": "interrupt",  # interrupt | queue | steer
        # steer mode: false hides only the "Steered into current run" bubble; steering itself still
        # happens.
        "busy_steer_ack_enabled": True,
        # Classic CLI multiline beyond Alt+Enter: Ctrl+J newline, trailing backslash+Enter
        # continues, Shift+Enter reported distinctly. False restores the c-j submit fallback for
        # POSIX PTYs whose plain Enter arrives as LF.
        "cli_multiline_shortcuts": True,
        # `hermes`/`hermes chat` launches "cli" (prompt_toolkit REPL) | "tui" (Ink); --cli, --tui, HERMES_TUI=1 win.
        # cli_layout = the REPL's look: "modern" (glyph rows, status under the composer) | "classic" (boxed panels).
        "interface": "cli", "cli_layout": "modern",
        # Native TUI uses the terminal's primary buffer and scrollback instead of the custom
        # alternate-screen viewport. Flags win: `--native` / `--tui-native` and `--cli`.
        "tui_native": False,
        # `hermes --tui` auto-resumes the most recent human-facing session (like `hermes -c`).
        # HERMES_TUI_RESUME=<id> always wins.
        "tui_auto_resume_recent": False,
        # Desktop reopens the last chat/page on cold start (also in Settings → Appearance).
        "resume_last_session": True,
        # One-time TUI hint ("subagents working · /agents to watch live") on first delegation.
        "tui_agents_nudge": True,
        "bell_on_complete": False,
        "tab_activity_indicator": False,
        "tab_title_idle": "{title}",
        "tab_title_busy": "{spinner} Hermes",
        "tab_title_done": "{title}",
        "tab_title_spinner_frames": [
            "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏",
        ],
        "bell_on_prompt": False,   # bell when a blocking prompt opens (clarify/approval/sudo)
        # Stream reasoning live before the response; otherwise thinking models show only a spinner
        # for tens of seconds.
        "show_reasoning": True,
        # Post-response "Reasoning" recap collapses to 10 lines; true prints it all (live streaming
        # is always full).
        "reasoning_full": False,
        # Background self-improvement notices in chat: "off" (review still runs) | "on" (generic "💾
        # Memory updated") | "verbose" (content preview). Per-platform via
        # display.platforms.<platform>.memory_notifications.
        "memory_notifications": "on",
        # Gateway notices when a terminal(background=true) process finishes: "concise" (one line;
        # failures append an output tail) | "all" (running updates + final raw output) | "result"
        # (final raw only) | "error" (raw only on non-zero exit) | "off".
        "background_process_notifications": "concise",
        "streaming": False,
        "timestamps": False,      # message timestamps (CLI labels, TUI rows, desktop transcript)
        "timestamp_format": "%H:%M",  # strftime format, e.g. "%b-%d %H:%M"
        "final_response_markdown": "strip",  # render | strip | raw
        # Preserve recent classic-CLI output across Ctrl+L, /redraw and resize clears; disable if an
        # emulator misbehaves with replayed scrollback.
        "persistent_output": True,
        "persistent_output_max_lines": 200,
        # Also clear terminal scrollback on classic-CLI full redraw/resize recovery; enable when a
        # terminal/tmux stack stamps stale prompt chrome into scrollback.
        "cli_rebuild_scrollback_on_redraw": False,
        # Print a one-line summary of resolved modal prompts (approval/clarify) to scrollback.
        "persist_prompts": True,
        "inline_diffs": True,     # inline diff previews for write_file/patch/skill_manage
        # Append a one-line advisory to the final response when a write_file/patch failed this turn
        # and was never superseded by a successful write to the same path (catches "half the
        # parallel patches failed, model claims success").
        "file_mutation_verifier": True,
        # Nous credits status-bar notices (usage bands, grant-spent, depleted/restored). False mutes
        # them; balance data and /usage keep working.
        "credits_notices": True,
        # Append a one-line explanation when a turn ends with no usable reply (empty after retries,
        # truncated stream, pending tool result, iteration/budget limit) instead of the bare
        # "(empty)" sentinel.
        "turn_completion_explainer": True,
        "show_cost": False,       # $ cost in the status bar
        "battery": False,         # battery read-out first in status bar; no-op w/o battery
        # Focus view (/focus): display-only. Pins tool_progress to "off", reports per-turn
        # hidden-line count, pins a "focus" status segment. focus_saved_tool_progress holds the mode
        # /focus off restores. Never affects what the model sees (focus_view.py).
        "focus_view": False,
        "focus_saved_tool_progress": "all",
        "skin": "default",
        # UI language for static messages (approval prompts, some gateway slash replies); not agent
        # responses/logs/tool outputs. en, zh, ja, de, es, fr, tr, uk; unknown → en.
        "language": "en",
        # TUI busy indicator: kaomoji | emoji | unicode (braille) | ascii. `/indicator <style>`.
        "tui_status_indicator": "kaomoji",
        # Seconds between idle prompt_toolkit redraws in the classic CLI; keeps wall-clock
        # status-bar read-outs ticking and the bottom chrome from going stale. 0 disables it if it
        # fights terminal auto-scroll in non-fullscreen mode.
        # See #45592.
        "cli_refresh_interval": 1.0,
        # Vi/vim keybindings in the CLI input composer (config-only, no slash command).
        # Off by default, preserving prompt_toolkit's standard emacs bindings.
        "vim_mode": False,
        "user_message_preview": {  # CLI: submitted user-message lines echoed to scrollback
            "first_lines": 2,
            "last_lines": 2,
        },
        # Gateway: natural mid-turn assistant status messages. Desktop: keep mid-turn narration
        # between tool calls instead of collapsing to the final message.
        "interim_assistant_messages": True,
        # Engine warning/failure notifications stay visible unless an operator opts in.
        # Does not suppress task results, manual commands, or existing logs.
        "suppress_warning_notifications": False,
        # Codex Responses commentary channel: true delivers completed commentary as mid-turn interim
        # updates; false routes it to reasoning (visible only with show_reasoning).
        "show_commentary": True,
        "tool_progress_command": False,  # enable /verbose command in messaging gateway
        # display.tool_progress_overrides is deprecated (use display.platforms); a user-set value is
        # still honored at runtime and folded into platforms by migration.
        "tool_preview_length": 0,  # max chars for tool call previews (0 = no limit)
        # Human-phrased status labels for built-in tools ("Reading <file>") in CLI spinner and
        # gateway/desktop tool-progress; custom/plugin/MCP tools use the raw preview.
        "friendly_tool_labels": True,
        # CLI-only post-turn line: "⋯ 12.4s · edited 2 files +18 -3 · read 4 files · ran 3
        # commands". Never in quiet/non-interactive or gateway surfaces (own footer).
        "turn_summary": True,
        # CLI-only: cumulative turn output tokens on the live spinner ("· ↓ 1.2k tok").
        "spinner_token_flow": True,
        # Gateway tool-progress grouping where edits are supported: "accumulate" edits one bubble |
        # "separate" one message per tool (noisier). Needs tool_progress enabled. Per-platform:
        # display.platforms.<platform>.tool_progress_grouping.
        "tool_progress_grouping": "accumulate",
        # Custom long-running status phrases. Defaults: gateway/assets/status_phrases.yaml.
        # `path`/`paths` = HERMES_HOME-relative YAML files/dirs (or conventional status_phrases.yaml
        # / status_phrases/*.yaml). Keys: status, generic. mode: "append" (default) | "replace".
        # Per-platform: display.platforms.<platform>.status_phrases.
        "status_phrases": {},
        # Reasoning summary rendering: "code" (💭 fenced block) | "blockquote" ("> ") | "subtext"
        # ("-# " Discord small grey text; Discord's default). Per-platform via
        # display.platforms.<platform>.reasoning_style.
        "reasoning_style": "code",
        # Auto-delete EphemeralReply system notices ("✨ New session started!", …) after N seconds
        # where deletion is supported (Telegram; others ignore). Agent responses are never touched.
        # 0 = disabled.
        "ephemeral_system_ttl": 0,
        # Per-platform display/streaming overrides; unset keys fall through to the global. Telegram
        # has smooth native draft streaming (on); Discord/Slack only edit-based streaming, which
        # flickers (off). Gap-fillers only: explicit user values win, and the global
        # streaming.enabled master switch still gates everything.
        "platforms": {
            "telegram": {"streaming": True},
            "discord": {"streaming": False},
            "slack": {"streaming": False},
            # WeCom native streaming (msgtype "stream" via aibot_respond_msg).
            "wecom": {"streaming": True},
        },
        # Gateway runtime footer on the FINAL message, e.g. `model · 68% · ~/projects/hermes`.
        # Per-platform: display.platforms.<platform>.runtime_footer.
        "runtime_footer": {
            "enabled": False,
            # order shown; drop any to hide. Opt-in extras: latency, served_model (alias → the
            # deployment a routing proxy reported / Hermes' fallback route).
            "fields": ["model", "context_pct", "cwd"],
        },
        # CLI/TUI status bar fields. Non-empty = only listed fields show (built-in order kept,
        # config controls visibility not ordering); empty = default set. Available: model,
        # context_detail, context_pct, cache_hit, latency, tps, compressions, bg_tasks,
        # bg_processes, bg_subagents, goal, git_branch (⎇ current branch, opt-in only), duration,
        # prompt_elapsed, idle_since, focus, yolo,
        # stash, battery, title, total_tokens (session Σ, opt-in only). Narrow terminals still drop
        # context_detail/prompt_elapsed/idle_since.
        "status_bar": {
            "fields": [],
        },
        "copy_shortcut": "auto",  # "auto" (platform default) | ctrl_c | ctrl_shift_c | disabled
        # Petdex animated mascot (github.com/crafter-station/petdex): cosmetic sprite across
        # CLI/TUI/desktop, managed with `hermes pets`. No effect on prompt caching.
        "pet": {
            "enabled": False,
            "slug": "",   # active pet slug in get_hermes_home()/pets/; empty → first installed
            # auto (detect kitty/iTerm2/sixel, else unicode half-blocks) | kitty | iterm | sixel |
            # unicode | off
            "render_mode": "auto",
            # Size scalar relative to native 192×208 frames, shared by desktop canvas and CLI/TUI
            # column width. Half-block fallback clamps to a legibility floor.
            "scale": 0.33,
            "unicode_cols": 0,  # Hard override for terminal column width; 0 = derive from scale.
        },
    }
