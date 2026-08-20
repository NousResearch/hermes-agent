# Approval polling recipe

Full implementation of the push-approval loop described in the push approval section of SKILL.md. Use it when you are writing code that must survive a disabled key, not when you are making a one-off call.

```python
import time
import requests

PROXY = "https://proxy.shieldnode.app"
KEY = "shieldnode_..."  # virtual key, load from $SHIELDNODE_<SERVICE>_KEY

def call_with_approval(path, *, method="GET", json=None, agent="Hermes",
                       minutes=15, max_wait_s=300, poll_s=30):
    """
    Call the proxy. If the key is disabled and the user has the mobile app,
    notify them and poll until they approve, decline, or time out.
    """
    headers = {
        "X-Api-Key": KEY,
        "X-Agent-Name": agent,                  # appears in the user's notification
        "X-Approval-Duration": f"{minutes}m",   # honored on 403, ignored on 200
    }

    elapsed = 0
    while True:
        r = requests.request(method, f"{PROXY}{path}", headers=headers, json=json)

        if r.status_code < 400:
            return r  # proxy let it through

        if r.status_code == 403:
            body = r.json()
            err = body.get("error")

            if err == "approval_required":
                # Tell the user once, then poll silently.
                if elapsed == 0:
                    print(f"[ShieldNode] Approval pending on your phone "
                          f"({body['requested_minutes']} min requested).")
                if elapsed >= min(max_wait_s, body.get("timeout_seconds", 300)):
                    raise TimeoutError("ShieldNode approval timed out.")
                wait = body.get("poll_interval_seconds", poll_s)
                time.sleep(wait)
                elapsed += wait
                continue

            if err == "approval_denied":
                raise PermissionError("User declined access on ShieldNode mobile.")

            if err == "key_disabled":
                raise PermissionError(
                    "Key is disabled and no mobile device is registered. "
                    "The user needs to re-enable it in the dashboard."
                )

        # Anything else (401, 429, 5xx, path/method restrictions) surfaces as-is.
        r.raise_for_status()


# A consequential action on a classic API, which is what this flow is for:
# the user gets a push, approves once, and the send goes through.
data = call_with_approval(
    "/emails",
    method="POST",
    json={
        "from": "reports@example.com",
        "to": ["owner@example.com"],
        "subject": "Weekly report",
        "html": "<p>Numbers attached.</p>",
    },
    agent="Hermes",
    minutes=5,
).json()
```

## Why each part matters

- **Read `poll_interval_seconds` and `timeout_seconds` from the body**, do not hardcode. The server tunes them.
- **Print once, at `elapsed == 0`.** Repeating the message every poll is the single most annoying agent behaviour in this flow.
- **Separate the three 403 errors.** `approval_denied` is a decision, `key_disabled` is a missing app, a timeout is silence. Reporting all three as "access denied" hides what the user needs to do.
- **Let non-403 errors raise.** A 429 or a 500 is not an approval problem, and looping on it wastes the user's quota.

## Choosing a duration

| Workload | Suggested |
|---|---|
| Chat, completion, inference (OpenAI, Anthropic, Mistral) | 30 min |
| Long batch, training, video or audio generation | 2 h |
| One-shot lookups (geocoding, currency, weather, CMS reads) | 15 min |
| Unattended 24/7 cron | Use a scheduled window instead (see the scheduled windows section of SKILL.md), or tell the user to leave the key always-on |

If the user gives you a standing instruction ("for OpenAI ask 30 min by default"), encode it as the header on your first call to that service rather than asking again each time.
