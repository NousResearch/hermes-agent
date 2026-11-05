"""Read-only TODO carrier projection shared by REST and gateway history."""
import re
from typing import Any


def strip_todo_snapshot(content: Any) -> Any:
    from tools.todo_tool import TODO_INJECTION_HEADER
    if isinstance(content, str):
        match = re.search(r'(?:^|\n)[ \t]*' + re.escape(TODO_INJECTION_HEADER) + r'\r?\n- [\s\S]+$', content)
        return content[:match.start()].rstrip() if match else content
    if isinstance(content, list):
        cleaned = []
        for part in content:
            projected = strip_todo_snapshot(part)
            if projected == '' or (isinstance(projected, dict) and projected.get('text') == ''):
                continue
            cleaned.append(projected)
        return cleaned
    if isinstance(content, dict) and content.get('type') in (None, 'text', 'input_text', 'output_text') and isinstance(content.get('text'), str):
        return {**content, 'text': strip_todo_snapshot(content['text'])}
    return content


def project_todo_message(message: dict) -> dict:
    if message.get('role') != 'user':
        return message
    original = message.get('content')
    content = strip_todo_snapshot(original)
    if content == original:
        return message
    if content == '' or content == [] or (isinstance(content, dict) and content.get('text') == ''):
        return {**message, 'display_kind': 'hidden'}
    return {**message, 'content': content}
