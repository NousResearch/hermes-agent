"""CLI presentation -- spinner, kawaii faces, tool preview formatting.

Pure display functions and classes with no AIAgent dependency.
Used by AIAgent._execute_tool_calls for CLI feedback.
"""

import json
import os
import random
import threading
import time

# ANSI escape codes for coloring tool failure indicators
_RED = "\033[31m"
_RESET = "\033[0m"


# =========================================================================
# Tool preview (one-line summary of a tool call's primary argument)
# =========================================================================

def build_tool_preview(tool_name: str, args: dict, max_len: int = 40) -> str:
    """Build a short preview of a tool call's primary argument for display."""
    primary_args = {
        "terminal": "command", "web_search": "query", "web_extract": "urls",
        "read_file": "path", "write_file": "path", "patch": "path",
        "search_files": "pattern", "browser_navigate": "url",
        "browser_click": "ref", "browser_type": "text",
        "image_generate": "prompt", "text_to_speech": "text",
        "vision_analyze": "question", "mixture_of_agents": "user_prompt",
        "skill_view": "name", "skills_list": "category",
        "schedule_cronjob": "name",
    }

    if tool_name == "process":
        action = args.get("action", "")
        sid = args.get("session_id", "")
        data = args.get("data", "")
        timeout_val = args.get("timeout")
        parts = [action]
        if sid:
            parts.append(sid[:16])
        if data:
            parts.append(f'"{data[:20]}"')
        if timeout_val and action == "wait":
            parts.append(f"{timeout_val}s")
        return " ".join(parts) if parts else None

    if tool_name == "todo":
        todos_arg = args.get("todos")
        merge = args.get("merge", False)
        if todos_arg is None:
            return "reading task list"
        elif merge:
            return f"updating {len(todos_arg)} task(s)"
        else:
            return f"planning {len(todos_arg)} task(s)"

    if tool_name == "session_search":
        query = args.get("query", "")
        return f"recall: \"{query[:25]}{'...' if len(query) > 25 else ''}\""

    if tool_name == "memory":
        action = args.get("action", "")
        target = args.get("target", "")
        if action == "add":
            content = args.get("content", "")
            return f"+{target}: \"{content[:25]}{'...' if len(content) > 25 else ''}\""
        elif action == "replace":
            return f"~{target}: \"{args.get('old_text', '')[:20]}\""
        elif action == "remove":
            return f"-{target}: \"{args.get('old_text', '')[:20]}\""
        return action

    if tool_name == "send_message":
        target = args.get("target", "?")
        msg = args.get("message", "")
        if len(msg) > 20:
            msg = msg[:17] + "..."
        return f"to {target}: \"{msg}\""

    if tool_name.startswith("rl_"):
        rl_previews = {
            "rl_list_environments": "listing envs",
            "rl_select_environment": args.get("name", ""),
            "rl_get_current_config": "reading config",
            "rl_edit_config": f"{args.get('field', '')}={args.get('value', '')}",
            "rl_start_training": "starting",
            "rl_check_status": args.get("run_id", "")[:16],
            "rl_stop_training": f"stopping {args.get('run_id', '')[:16]}",
            "rl_get_results": args.get("run_id", "")[:16],
            "rl_list_runs": "listing runs",
            "rl_test_inference": f"{args.get('num_steps', 3)} steps",
        }
        return rl_previews.get(tool_name)

    key = primary_args.get(tool_name)
    if not key:
        for fallback_key in ("query", "text", "command", "path", "name", "prompt"):
            if fallback_key in args:
                key = fallback_key
                break

    if not key or key not in args:
        return None

    value = args[key]
    if isinstance(value, list):
        value = value[0] if value else ""

    preview = str(value).strip()
    if not preview:
        return None
    if len(preview) > max_len:
        preview = preview[:max_len - 3] + "..."
    return preview


# =========================================================================
# KawaiiSpinner
# =========================================================================

class KawaiiSpinner:
    """Animated spinner with kawaii faces for CLI feedback during tool execution."""

    SPINNERS = {
        'dots': ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'],
        'bounce': ['⠁', '⠂', '⠄', '⡀', '⢀', '⠠', '⠐', '⠈'],
        'grow': ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█', '▇', '▆', '▅', '▄', '▃', '▂'],
        'arrows': ['←', '↖', '↑', '↗', '→', '↘', '↓', '↙'],
        'star': ['✶', '✷', '✸', '✹', '✺', '✹', '✸', '✷'],
        'moon': ['🌑', '🌒', '🌓', '🌔', '🌕', '🌖', '🌗', '🌘'],
        'pulse': ['◜', '◠', '◝', '◞', '◡', '◟'],
        'brain': ['🧠', '💭', '💡', '✨', '💫', '🌟', '💡', '💭'],
        'sparkle': ['⁺', '˚', '*', '✧', '✦', '✧', '*', '˚'],
    }

    KAWAII_WAITING = [
        "(｡◕‿◕｡)", "(◕‿◕✿)", "٩(◕‿◕｡)۶", "(✿◠‿◠)", "( ˘▽˘)っ",
        "♪(´ε` )", "(◕ᴗ◕✿)", "ヾ(＾∇＾)", "(≧◡≦)", "(★ω★)",
    ]

    KAWAII_THINKING = [
        "(｡•́︿•̀｡)", "(◔_◔)", "(¬‿¬)", "( •_•)>⌐■-■", "(⌐■_■)",
        "(´･_･`)", "◉_◉", "(°ロ°)", "( ˘⌣˘)♡", "ヽ(>∀<☆)☆",
        "٩(๑❛ᴗ❛๑)۶", "(⊙_⊙)", "(¬_¬)", "( ͡° ͜ʖ ͡°)", "ಠ_ಠ",
    ]

    THINKING_VERBS = [
        "pondering", "contemplating", "musing", "cogitating", "ruminating",
        "deliberating", "mulling", "reflecting", "processing", "reasoning",
        "analyzing", "computing", "synthesizing", "formulating", "brainstorming",
    ]

    def __init__(self, message: str = "", spinner_type: str = 'dots'):
        self.message = message
        self.spinner_frames = self.SPINNERS.get(spinner_type, self.SPINNERS['dots'])
        self.running = False
        self.thread = None
        self.frame_idx = 0
        self.start_time = None
        self.last_line_len = 0

    def _animate(self):
        while self.running:
            if os.getenv("HERMES_SPINNER_PAUSE"):
                time.sleep(0.1)
                continue
            frame = self.spinner_frames[self.frame_idx % len(self.spinner_frames)]
            elapsed = time.time() - self.start_time
            line = f"  {frame} {self.message} ({elapsed:.1f}s)"
            clear = '\r' + ' ' * self.last_line_len + '\r'
            print(clear + line, end='', flush=True)
            self.last_line_len = len(line)
            self.frame_idx += 1
            time.sleep(0.12)

    def start(self):
        if self.running:
            return
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._animate, daemon=True)
        self.thread.start()

    def update_text(self, new_message: str):
        self.message = new_message

    def stop(self, final_message: str = None):
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)
        print('\r' + ' ' * (self.last_line_len + 5) + '\r', end='', flush=True)
        if final_message:
            print(f"  {final_message}", flush=True)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


# =========================================================================
# Kawaii face arrays (used by AIAgent._execute_tool_calls for spinner text)
# =========================================================================

KAWAII_SEARCH = [
    "♪(´ε` )", "(｡◕‿◕｡)", "ヾ(＾∇＾)", "(◕ᴗ◕✿)", "( ˘▽˘)っ",
    "٩(◕‿◕｡)۶", "(✿◠‿◠)", "♪～(´ε｀ )", "(ノ´ヮ`)ノ*:・゚✧", "＼(◎o◎)／",
]
KAWAII_READ = [
    "φ(゜▽゜*)♪", "( ˘▽˘)っ", "(⌐■_■)", "٩(｡•́‿•̀｡)۶", "(◕‿◕✿)",
    "ヾ(＠⌒ー⌒＠)ノ", "(✧ω✧)", "♪(๑ᴖ◡ᴖ๑)♪", "(≧◡≦)", "( ´ ▽ ` )ノ",
]
KAWAII_TERMINAL = [
    "ヽ(>∀<☆)ノ", "(ノ°∀°)ノ", "٩(^ᴗ^)۶", "ヾ(⌐■_■)ノ♪", "(•̀ᴗ•́)و",
    "┗(＾0＾)┓", "(｀・ω・´)", "＼(￣▽￣)／", "(ง •̀_•́)ง", "ヽ(´▽`)/",
]
KAWAII_BROWSER = [
    "(ノ°∀°)ノ", "(☞゚ヮ゚)☞", "( ͡° ͜ʖ ͡°)", "┌( ಠ_ಠ)┘", "(⊙_⊙)？",
    "ヾ(•ω•`)o", "(￣ω￣)", "( ˇωˇ )", "(ᵔᴥᵔ)", "＼(◎o◎)／",
]
KAWAII_CREATE = [
    "✧*。٩(ˊᗜˋ*)و✧", "(ﾉ◕ヮ◕)ﾉ*:・ﾟ✧", "ヽ(>∀<☆)ノ", "٩(♡ε♡)۶", "(◕‿◕)♡",
    "✿◕ ‿ ◕✿", "(*≧▽≦)", "ヾ(＾-＾)ノ", "(☆▽☆)", "°˖✧◝(⁰▿⁰)◜✧˖°",
]
KAWAII_SKILL = [
    "ヾ(＠⌒ー⌒＠)ノ", "(๑˃ᴗ˂)ﻭ", "٩(◕‿◕｡)۶", "(✿╹◡╹)", "ヽ(・∀・)ノ",
    "(ノ´ヮ`)ノ*:・ﾟ✧", "♪(๑ᴖ◡ᴖ๑)♪", "(◠‿◠)", "٩(ˊᗜˋ*)و", "(＾▽＾)",
    "ヾ(＾∇＾)", "(★ω★)/", "٩(｡•́‿•̀｡)۶", "(◕ᴗ◕✿)", "＼(◎o◎)／",
    "(✧ω✧)", "ヽ(>∀<☆)ノ", "( ˘▽˘)っ", "(≧◡≦) ♡", "ヾ(￣▽￣)",
]
KAWAII_THINK = [
    "(っ°Д°;)っ", "(；′⌒`)", "(・_・ヾ", "( ´_ゝ`)", "(￣ヘ￣)",
    "(。-`ω´-)", "( ˘︹˘ )", "(¬_¬)", "ヽ(ー_ー )ノ", "(；一_一)",
]
KAWAII_GENERIC = [
    "♪(´ε` )", "(◕‿◕✿)", "ヾ(＾∇＾)", "٩(◕‿◕｡)۶", "(✿◠‿◠)",
    "(ノ´ヮ`)ノ*:・ﾟ✧", "ヽ(>∀<☆)ノ", "(☆▽☆)", "( ˘▽˘)っ", "(≧◡≦)",
]


# =========================================================================
# Cute tool message (completion line that replaces the spinner)
# =========================================================================

def _detect_tool_failure(tool_name: str, result: str | None) -> tuple[bool, str]:
    """Inspect a tool result string for signs of failure.

    Returns ``(is_failure, suffix)`` where *suffix* is an informational tag
    like ``" [exit 1]"`` for terminal failures, or ``" [error]"`` for generic
    failures.  On success, returns ``(False, "")``.
    """
    if result is None:
        return False, ""

    if tool_name == "terminal":
        try:
            data = json.loads(result)
            exit_code = data.get("exit_code")
            if exit_code is not None and exit_code != 0:
                return True, f" [exit {exit_code}]"
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
        return False, ""

    # Generic heuristic for non-terminal tools
    lower = result[:500].lower()
    if '"error"' in lower or '"failed"' in lower or result.startswith("Error"):
        return True, " [error]"

    return False, ""


def get_cute_tool_message(
    tool_name: str, args: dict, duration: float, result: str | None = None,
) -> str:
    """Generate a formatted tool completion line for CLI quiet mode.

    Format: ``| {emoji} {verb:9} {detail}  {duration}``

    When *result* is provided the line is checked for failure indicators.
    Failed tool calls get a red prefix and an informational suffix.
    """
    dur = f"{duration:.1f}s"
    is_failure, failure_suffix = _detect_tool_failure(tool_name, result)

    def _trunc(s, n=40):
        s = str(s)
        return (s[:n-3] + "...") if len(s) > n else s

    def _path(p, n=35):
        p = str(p)
        return ("..." + p[-(n-3):]) if len(p) > n else p

    def _wrap(line: str) -> str:
        """Apply red coloring and failure suffix when the tool failed."""
        if not is_failure:
            return line
        return f"{_RED}{line}{failure_suffix}{_RESET}"

    if tool_name == "web_search":
        return _wrap(f"┊ 🔍 search    {_trunc(args.get('query', ''), 42)}  {dur}")
    if tool_name == "web_extract":
        urls = args.get("urls", [])
        if urls:
            url = urls[0] if isinstance(urls, list) else str(urls)
            domain = url.replace("https://", "").replace("http://", "").split("/")[0]
            extra = f" +{len(urls)-1}" if len(urls) > 1 else ""
            return _wrap(f"┊ 📄 fetch     {_trunc(domain, 35)}{extra}  {dur}")
        return _wrap(f"┊ 📄 fetch     pages  {dur}")
    if tool_name == "web_crawl":
        url = args.get("url", "")
        domain = url.replace("https://", "").replace("http://", "").split("/")[0]
        return _wrap(f"┊ 🕸️  crawl     {_trunc(domain, 35)}  {dur}")
    if tool_name == "terminal":
        return _wrap(f"┊ 💻 $         {_trunc(args.get('command', ''), 42)}  {dur}")
    if tool_name == "process":
        action = args.get("action", "?")
        sid = args.get("session_id", "")[:12]
        labels = {"list": "ls processes", "poll": f"poll {sid}", "log": f"log {sid}",
                  "wait": f"wait {sid}", "kill": f"kill {sid}", "write": f"write {sid}", "submit": f"submit {sid}"}
        return _wrap(f"┊ ⚙️  proc      {labels.get(action, f'{action} {sid}')}  {dur}")
    if tool_name == "read_file":
        return _wrap(f"┊ 📖 read      {_path(args.get('path', ''))}  {dur}")
    if tool_name == "write_file":
        return _wrap(f"┊ ✍️  write     {_path(args.get('path', ''))}  {dur}")
    if tool_name == "patch":
        return _wrap(f"┊ 🔧 patch     {_path(args.get('path', ''))}  {dur}")
    if tool_name == "search_files":
        pattern = _trunc(args.get("pattern", ""), 35)
        target = args.get("target", "content")
        verb = "find" if target == "files" else "grep"
        return _wrap(f"┊ 🔎 {verb:9} {pattern}  {dur}")
    if tool_name == "browser_navigate":
        url = args.get("url", "")
        domain = url.replace("https://", "").replace("http://", "").split("/")[0]
        return _wrap(f"┊ 🌐 navigate  {_trunc(domain, 35)}  {dur}")
    if tool_name == "browser_snapshot":
        mode = "full" if args.get("full") else "compact"
        return _wrap(f"┊ 📸 snapshot  {mode}  {dur}")
    if tool_name == "browser_click":
        return _wrap(f"┊ 👆 click     {args.get('ref', '?')}  {dur}")
    if tool_name == "browser_type":
        return _wrap(f"┊ ⌨️  type      \"{_trunc(args.get('text', ''), 30)}\"  {dur}")
    if tool_name == "browser_scroll":
        d = args.get("direction", "down")
        arrow = {"down": "↓", "up": "↑", "right": "→", "left": "←"}.get(d, "↓")
        return _wrap(f"┊ {arrow}  scroll    {d}  {dur}")
    if tool_name == "browser_back":
        return _wrap(f"┊ ◀️  back      {dur}")
    if tool_name == "browser_press":
        return _wrap(f"┊ ⌨️  press     {args.get('key', '?')}  {dur}")
    if tool_name == "browser_close":
        return _wrap(f"┊ 🚪 close     browser  {dur}")
    if tool_name == "browser_get_images":
        return _wrap(f"┊ 🖼️  images    extracting  {dur}")
    if tool_name == "browser_vision":
        return _wrap(f"┊ 👁️  vision    analyzing page  {dur}")
    if tool_name == "todo":
        todos_arg = args.get("todos")
        merge = args.get("merge", False)
        if todos_arg is None:
            return _wrap(f"┊ 📋 plan      reading tasks  {dur}")
        elif merge:
            return _wrap(f"┊ 📋 plan      update {len(todos_arg)} task(s)  {dur}")
        else:
            return _wrap(f"┊ 📋 plan      {len(todos_arg)} task(s)  {dur}")
    if tool_name == "session_search":
        return _wrap(f"┊ 🔍 recall    \"{_trunc(args.get('query', ''), 35)}\"  {dur}")
    if tool_name == "memory":
        action = args.get("action", "?")
        target = args.get("target", "")
        if action == "add":
            return _wrap(f"┊ 🧠 memory    +{target}: \"{_trunc(args.get('content', ''), 30)}\"  {dur}")
        elif action == "replace":
            return _wrap(f"┊ 🧠 memory    ~{target}: \"{_trunc(args.get('old_text', ''), 20)}\"  {dur}")
        elif action == "remove":
            return _wrap(f"┊ 🧠 memory    -{target}: \"{_trunc(args.get('old_text', ''), 20)}\"  {dur}")
        return _wrap(f"┊ 🧠 memory    {action}  {dur}")
    if tool_name == "skills_list":
        return _wrap(f"┊ 📚 skills    list {args.get('category', 'all')}  {dur}")
    if tool_name == "skill_view":
        return _wrap(f"┊ 📚 skill     {_trunc(args.get('name', ''), 30)}  {dur}")
    if tool_name == "image_generate":
        return _wrap(f"┊ 🎨 create    {_trunc(args.get('prompt', ''), 35)}  {dur}")
    if tool_name == "text_to_speech":
        return _wrap(f"┊ 🔊 speak     {_trunc(args.get('text', ''), 30)}  {dur}")
    if tool_name == "vision_analyze":
        return _wrap(f"┊ 👁️  vision    {_trunc(args.get('question', ''), 30)}  {dur}")
    if tool_name == "mixture_of_agents":
        return _wrap(f"┊ 🧠 reason    {_trunc(args.get('user_prompt', ''), 30)}  {dur}")
    if tool_name == "send_message":
        return _wrap(f"┊ 📨 send      {args.get('target', '?')}: \"{_trunc(args.get('message', ''), 25)}\"  {dur}")
    if tool_name == "schedule_cronjob":
        return _wrap(f"┊ ⏰ schedule  {_trunc(args.get('name', args.get('prompt', 'task')), 30)}  {dur}")
    if tool_name == "list_cronjobs":
        return _wrap(f"┊ ⏰ jobs      listing  {dur}")
    if tool_name == "remove_cronjob":
        return _wrap(f"┊ ⏰ remove    job {args.get('job_id', '?')}  {dur}")
    if tool_name.startswith("rl_"):
        rl = {
            "rl_list_environments": "list envs", "rl_select_environment": f"select {args.get('name', '')}",
            "rl_get_current_config": "get config", "rl_edit_config": f"set {args.get('field', '?')}",
            "rl_start_training": "start training", "rl_check_status": f"status {args.get('run_id', '?')[:12]}",
            "rl_stop_training": f"stop {args.get('run_id', '?')[:12]}", "rl_get_results": f"results {args.get('run_id', '?')[:12]}",
            "rl_list_runs": "list runs", "rl_test_inference": "test inference",
        }
        return _wrap(f"┊ 🧪 rl        {rl.get(tool_name, tool_name.replace('rl_', ''))}  {dur}")
    if tool_name == "execute_code":
        code = args.get("code", "")
        first_line = code.strip().split("\n")[0] if code.strip() else ""
        return _wrap(f"┊ 🐍 exec      {_trunc(first_line, 35)}  {dur}")
    if tool_name == "delegate_task":
        tasks = args.get("tasks")
        if tasks and isinstance(tasks, list):
            return _wrap(f"┊ 🔀 delegate  {len(tasks)} parallel tasks  {dur}")
        return _wrap(f"┊ 🔀 delegate  {_trunc(args.get('goal', ''), 35)}  {dur}")

    preview = build_tool_preview(tool_name, args) or ""
    return _wrap(f"┊ ⚡ {tool_name[:9]:9} {_trunc(preview, 35)}  {dur}")
