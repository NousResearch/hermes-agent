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


def bounded_result(result, is_error):
    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=True, default=str)
    if len(text) <= MAX_TOOL_EVENT_CHARS:
        return text
    # ACP reads errors from the result as well as the event flag; keep valid JSON after truncation.
    return json.dumps({'error' if is_error else 'output': text[:MAX_TOOL_EVENT_CHARS // 2], 'truncated': True})
