"""Viewer payload limits; the full tool input/output stays in canonical history."""
import json

MAX_TOOL_EVENT_CHARS = 64 * 1024


def bounded_args(args):
    encoded = json.dumps(args, ensure_ascii=True, default=str)
    if len(encoded) <= MAX_TOOL_EVENT_CHARS:
        return args
    # Retain ordinary scalar paths/commands for editor locations even when a content field is large.
    result = {}
    remaining = MAX_TOOL_EVENT_CHARS // 2
    for key, value in args.items():
        if remaining <= 0 or len(result) >= 64:
            break
        name = str(key)[:256]
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=True, default=str)
        text = text[:min(4096, remaining)]
        result[name] = text
        remaining -= len(json.dumps([name, text], ensure_ascii=True))
    return result


def _structured_cwd(result):
    """The ``cwd`` a terminal result carries at top level (present only when the command changed directory),
    whether the result arrived parsed or as its JSON text; None for anything else."""
    data = result
    if isinstance(result, str):
        try:
            data = json.loads(result)
        except ValueError:
            return None
    cwd = data.get("cwd") if isinstance(data, dict) else None
    return cwd if isinstance(cwd, str) and 0 < len(cwd) <= 4096 else None


def bounded_result(result, is_error):
    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=True, default=str)
    if len(text) <= MAX_TOOL_EVENT_CHARS:
        return text
    # ACP reads errors from the result as well as the event flag; keep valid JSON after truncation.
    bounded = {'error' if is_error else 'output': text[:MAX_TOOL_EVENT_CHARS // 2], 'truncated': True}
    # The one structured field a viewer acts on survives the cut: the TUI's OSC 7 cwd report reads it,
    # and a long-output command that also changed directory must still move the terminal.
    cwd = _structured_cwd(result)
    if cwd is not None:
        bounded['cwd'] = cwd
    return json.dumps(bounded)
