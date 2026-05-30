"""
Relay Send Tool — 向其他 agent 发送消息。

通过 relay 平台适配器发送消息到指定 agent。
"""

import json
import os
import time
from tools.registry import registry


def relay_send(target: str, content: str, task_id: str = None) -> str:
    """Send a message to another agent via the relay platform.

    Args:
        target: Target agent name (e.g. "zhizhiruo" for 周芷若)
        content: Message content to send

    Returns:
        JSON string with success status
    """
    import requests

    sidecar_url = os.getenv("RELAY_SIDECAR_URL", "http://127.0.0.1:8767")
    relay_port = os.getenv("RELAY_PORT", "8766")

    # Method 1: POST to relay adapter (if gateway is running)
    relay_url = f"http://127.0.0.1:{relay_port}/message"
    payload = {
        "from": "zhangwuji",
        "to": target,
        "content": content,
        "id": f"msg_{int(time.time())}",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    try:
        resp = requests.post(relay_url, json=payload, timeout=10)
        if resp.status_code == 200:
            return json.dumps({
                "success": True,
                "target": target,
                "method": "relay",
                "message_id": payload["id"],
            })
    except requests.ConnectionError:
        pass  # Relay not running, try sidecar directly
    except Exception:
        pass

    # Method 2: POST directly to sidecar (fallback)
    sidecar_msg_url = f"{sidecar_url}/message"
    try:
        resp = requests.post(sidecar_msg_url, json=payload, timeout=10)
        if resp.status_code == 200:
            return json.dumps({
                "success": True,
                "target": target,
                "method": "sidecar",
                "message_id": payload["id"],
            })
        return json.dumps({
            "success": False,
            "error": f"Sidecar returned {resp.status_code}",
        })
    except requests.ConnectionError:
        return json.dumps({
            "success": False,
            "error": "Neither relay adapter nor sidecar is reachable",
        })
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


registry.register(
    name="relay_send",
    toolset="relay",
    schema={
        "name": "relay_send",
        "description": "Send a message to another agent (e.g. 周芷若/zhizhiruo) via the relay platform. Use this for real-time inter-agent communication.",
        "parameters": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Target agent name, e.g. 'zhizhiruo' for 周芷若",
                },
                "content": {
                    "type": "string",
                    "description": "Message content to send",
                },
            },
            "required": ["target", "content"],
        },
    },
    handler=lambda args, **kw: relay_send(
        target=args.get("target", ""),
        content=args.get("content", ""),
        task_id=kw.get("task_id"),
    ),
)
