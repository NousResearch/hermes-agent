"""Private-to-durable projection for a protected turn's model-authored messages."""
from typing import Any


def seal_protected_messages(messages: list[dict[str, Any]], final_response: Any, *, boundary: int = 0) -> None:
    """Keep prior turns unchanged and expose only the approved current-turn final narrative."""
    boundary = boundary if isinstance(boundary, int) and 0 <= boundary <= len(messages) else len(messages)
    current = messages[boundary:]
    final_index = next((i for i in range(len(current) - 1, -1, -1)
                        if current[i].get("role") == "assistant" and not current[i].get("tool_calls")
                        and not current[i].get("_pre_delivery_repair_synthetic")), None)
    sealed = []
    for index, row in enumerate(current):
        if row.get("_pre_delivery_repair_synthetic"):
            continue
        if row.get("role") == "assistant":
            if row.get("tool_calls"):
                row = {key: value for key, value in row.items()
                       if key not in {"reasoning", "reasoning_content", "api_content"}}
                row["content"] = ""
            elif index == final_index and final_response:
                row = {key: value for key, value in row.items()
                       if key not in {"reasoning", "reasoning_content", "api_content"}}
                row["content"] = final_response
            else:
                continue
        sealed.append(row)
    messages[boundary:] = sealed
