"""Gotify notification plugin for Hermes Agent.

Sends high-value, actionable notifications through a Gotify server.
Messages are rendered as **markdown** in the Gotify client by default
(via the ``extras.client::display.contentType`` field).

Configuration (in ``~/.hermes/.env``):

    GOTIFY_URL          Base URL of the Gotify server (e.g. http://gotify.local)
    GOTIFY_APP_TOKEN    Application token from Gotify's "Apps" page
    GOTIFY_CONTENT_TYPE  Optional. Content type for message rendering.
                         Default: ``text/markdown``. Set to ``text/plain``
                         to disable markdown rendering.
"""

import json
import os
import urllib.parse
import urllib.request


def register(ctx):
    schema = {
        "name": "gotify_send",
        "description": (
            "Send a short, high-value notification to the operator through Gotify. "
            "Messages are rendered as markdown in the Gotify client. "
            "Use only for: completed long-running tasks with concrete results, "
            "failed or blocked tasks, human approval requests, "
            "auth/API key/quota/rate-limit errors, cost or budget anomalies, "
            "security-relevant findings, failed CI/CD/backup/scheduled jobs, "
            "daily/weekly summaries, or artifacts ready for review. "
            "Do NOT use for routine chatter, every tool call, normal responses, "
            "verbose logs, intermediate thoughts, trivial successes, "
            "duplicate messages, or unverified results."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short notification title, max 120 chars. Format: [Hermes:<instance>] <event>"
                },
                "message": {
                    "type": "string",
                    "description": (
                        "Notification body, rendered as markdown. Include concrete "
                        "result, status, impact, and next action. Use **bold**, "
                        "[links](url), `code`, and lists for clarity."
                    )
                },
                "priority": {
                    "type": "integer",
                    "description": "Gotify priority: 2=info, 5=important, 8=urgent, 10=critical.",
                    "default": 5
                }
            },
            "required": ["message"]
        }
    }

    def handle_gotify_send(params, **kwargs):
        del kwargs

        base_url = os.environ.get("GOTIFY_URL", "").rstrip("/")
        token = os.environ.get("GOTIFY_APP_TOKEN", "")

        if not base_url or not token:
            return json.dumps({
                "success": False,
                "error": "Missing GOTIFY_URL or GOTIFY_APP_TOKEN environment variable."
            })

        content_type = os.environ.get("GOTIFY_CONTENT_TYPE", "text/markdown")

        title = str(params.get("title") or "Hermes").strip()[:120]
        message = str(params["message"]).strip()
        priority = int(params.get("priority", 5))

        payload = {
            "title": title,
            "message": message[:4000],
            "priority": max(1, min(10, priority)),
            "extras": {
                "client::display": {
                    "contentType": content_type
                }
            }
        }

        url = f"{base_url}/message?token={urllib.parse.quote(token, safe='')}"
        data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = resp.read().decode("utf-8", "replace")
                return json.dumps({
                    "success": 200 <= resp.status < 300,
                    "status": resp.status,
                    "response": body
                })
        except Exception as exc:
            return json.dumps({
                "success": False,
                "error": str(exc)
            })

    ctx.register_tool(
        name="gotify_send",
        toolset="gotify",
        schema=schema,
        handler=handle_gotify_send,
        description="Send important notifications to Gotify with markdown rendering.",
    )
