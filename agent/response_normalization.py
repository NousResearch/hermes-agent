import json


def normalize_assistant_content(assistant_message):
    """Normalize non-string assistant content in place for downstream consumers."""
    if assistant_message.content is not None and not isinstance(assistant_message.content, str):
        raw = assistant_message.content
        if isinstance(raw, dict):
            assistant_message.content = raw.get("text", "") or raw.get("content", "") or json.dumps(raw)
        elif isinstance(raw, list):
            parts = []
            for part in raw:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif isinstance(part, dict) and "text" in part:
                    parts.append(str(part["text"]))
            assistant_message.content = "\n".join(parts)
        else:
            assistant_message.content = str(raw)
