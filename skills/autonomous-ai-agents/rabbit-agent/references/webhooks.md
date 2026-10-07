# Webhook Subscriptions

Create dynamic webhook subscriptions so external services (GitHub, GitLab, Stripe, CI/CD, IoT sensors, monitoring tools) can trigger Rabbit agent runs by POSTing events to a URL.

## Setup (Required First)

The webhook platform must be enabled before subscriptions can be created. Check with:
```bash
rabbit webhook list
```

If it says "Webhook platform is not enabled", set it up:

### Option 1: Setup wizard
```bash
rabbit gateway setup
```
Follow the prompts to enable webhooks, set the port, and set a global HMAC secret.

### Option 2: Manual config
Add to `~/.rabbit/config.yaml`:
```yaml
platforms:
  webhook:
    enabled: true
    extra:
      port: 8644
      secret: "your-webhook-secret-here"
```

Omitting `host` uses the dual-stack default and listens on both IPv4 and IPv6.
Set a specific address only when you intentionally want to restrict the bind.

### Option 3: Environment variables
Add to `${RABBIT_HOME:-~/.rabbit}/.env`:
```bash
WEBHOOK_ENABLED=true
WEBHOOK_PORT=8644
WEBHOOK_SECRET=your-webhook-secret-here
```

After configuration, start (or restart) the gateway:
```bash
rabbit gateway run
# Or if using systemd:
systemctl --user restart rabbit-gateway
```

Verify it's running:
```bash
curl http://localhost:8644/health
```

## Commands

All management is via the `rabbit webhook` CLI command:

### Create a subscription
```bash
rabbit webhook subscribe <name> \
  --prompt "Prompt template with {payload.fields}" \
  --events "event1,event2" \
  --description "What this does" \
  --skills "skill1,skill2" \
  --deliver telegram \
  --deliver-chat-id "12345" \
  --secret "optional-custom-secret"
```

Returns the webhook URL and HMAC secret. The user configures their service to POST to that URL.

### Filter or transform payloads before the agent runs

Two mechanisms narrow broad event streams (e.g. Todoist/GitHub fire on every update) so only relevant payloads wake the agent:

- **Declarative `filters`** (config.yaml routes only): list of conditions on payload fields, event type, or headers — operators `equals`, `not_equals`, `contains`, `exists`, `missing`, `in`, `in_file`, `regex`, with `all`/`any`/`not` grouping. Non-matching events are ignored with HTTP 200.
- **Route scripts** (`--script` on subscribe, or `script:` on a config route): a script under `~/.rabbit/scripts/` receives the payload as JSON on stdin. JSON stdout replaces the payload before prompt templating; empty stdout, `[SILENT]`, or a nonzero exit ignores the webhook. `.sh`/`.bash` run with bash, everything else with Python. Scripts cannot live outside `~/.rabbit/scripts/` (path traversal is blocked).

```bash
rabbit webhook subscribe todoist-rabbit \
  --prompt "Task changed: {payload.content}" \
  --script "todoist-rabbit-label.py" \
  --deliver telegram --deliver-chat-id "12345"
```

Full filter syntax: https://github.com/seven0070/Rabbit-/tree/main/website/docs/user-guide/messaging/webhooks#payload-filters

### List subscriptions
```bash
rabbit webhook list
```

### Remove a subscription
```bash
rabbit webhook remove <name>
```

### Test a subscription
```bash
rabbit webhook test <name>
rabbit webhook test <name> --payload '{"key": "value"}'
```

## Prompt Templates

Prompts support `{dot.notation}` for accessing nested payload fields:

- `{issue.title}` — GitHub issue title
- `{pull_request.user.login}` — PR author
- `{data.object.amount}` — Stripe payment amount
- `{sensor.temperature}` — IoT sensor reading

If no prompt is specified, the full JSON payload is dumped into the agent prompt.

## Common Patterns

### GitHub: new issues
```bash
rabbit webhook subscribe github-issues \
  --events "issues" \
  --prompt "New GitHub issue #{issue.number}: {issue.title}\n\nAction: {action}\nAuthor: {issue.user.login}\nBody:\n{issue.body}\n\nPlease triage this issue." \
  --deliver telegram \
  --deliver-chat-id "-100123456789"
```

Then in GitHub repo Settings → Webhooks → Add webhook:
- Payload URL: the returned webhook_url
- Content type: application/json
- Secret: the returned secret
- Events: "Issues"

### GitHub: PR reviews
```bash
rabbit webhook subscribe github-prs \
  --events "pull_request" \
  --prompt "PR #{pull_request.number} {action}: {pull_request.title}\nBy: {pull_request.user.login}\nBranch: {pull_request.head.ref}\n\n{pull_request.body}" \
  --skills "github-code-review" \
  --deliver github_comment
```

### Stripe: payment events
```bash
rabbit webhook subscribe stripe-payments \
  --events "payment_intent.succeeded,payment_intent.payment_failed" \
  --prompt "Payment {data.object.status}: {data.object.amount} cents from {data.object.receipt_email}" \
  --deliver telegram \
  --deliver-chat-id "-100123456789"
```

### CI/CD: build notifications
```bash
rabbit webhook subscribe ci-builds \
  --events "pipeline" \
  --prompt "Build {object_attributes.status} on {project.name} branch {object_attributes.ref}\nCommit: {commit.message}" \
  --deliver discord \
  --deliver-chat-id "1234567890"
```

### Generic monitoring alert
```bash
rabbit webhook subscribe alerts \
  --prompt "Alert: {alert.name}\nSeverity: {alert.severity}\nMessage: {alert.message}\n\nPlease investigate and suggest remediation." \
  --deliver origin
```

### Direct delivery (no agent, zero LLM cost)

For use cases where you just want to push a notification through to a user's chat — no reasoning, no agent loop — add `--deliver-only`. The rendered `--prompt` template becomes the literal message body and is dispatched directly to the target adapter.

Use this for:
- External service push notifications (Supabase/Firebase webhooks → Telegram)
- Monitoring alerts that should forward verbatim
- Inter-agent pings where one agent is telling another agent's user something
- Any webhook where an LLM round trip would be wasted effort

```bash
rabbit webhook subscribe antenna-matches \
  --deliver telegram \
  --deliver-chat-id "123456789" \
  --deliver-only \
  --prompt "🎉 New match: {match.user_name} matched with you!" \
  --description "Antenna match notifications"
```

The POST returns `200 OK` on successful delivery, `502` on target failure — so upstream services can retry intelligently. HMAC auth, rate limits, and idempotency still apply.

Requires `--deliver` to be a real target (telegram, discord, slack, github_comment, etc.) — `--deliver log` is rejected because log-only direct delivery is pointless.

### Let the user reply to a delivery

Each event runs in its own session, so by default the agent in the target chat has no record of what a route delivered ("what are you referring to?"). Add `--mirror-to-session` and each delivered message is also written into that chat's session as `[Webhook delivery: <route>]` + the text. Off by default: the text enters the conversation as if the user had sent it (on `--deliver-only` routes it is the raw payload), so only use it for sources the user trusts. Skipped when the chat has no session yet.

## Security

- Each subscription gets an auto-generated HMAC-SHA256 secret (or provide your own with `--secret`)
- The webhook adapter validates signatures on every incoming POST
- Static routes from config.yaml cannot be overwritten by dynamic subscriptions
- Subscriptions persist to `~/.rabbit/webhook_subscriptions.json`

## How It Works

1. `rabbit webhook subscribe` writes to `~/.rabbit/webhook_subscriptions.json`
2. The webhook adapter hot-reloads this file on each incoming request (mtime-gated, negligible overhead)
3. When a POST arrives matching a route, the adapter formats the prompt and triggers an agent run
4. The agent's response is delivered to the configured target (Telegram, Discord, GitHub comment, etc.)

## Troubleshooting

If webhooks aren't working:

1. **Is the gateway running?** Check with `systemctl --user status rabbit-gateway` or `ps aux | grep gateway`
2. **Is the webhook server listening?** `curl http://localhost:8644/health` should return `{"status": "ok"}`
3. **Check gateway logs:** `grep webhook ~/.rabbit/logs/gateway.log | tail -20`
4. **Signature mismatch?** Verify the secret in your service matches the one from `rabbit webhook list`. GitHub sends `X-Hub-Signature-256`, GitLab sends `X-Gitlab-Token`.
5. **Firewall/NAT?** The webhook URL must be reachable from the service. For local development, use a tunnel (ngrok, cloudflared).
6. **Wrong event type?** Check `--events` filter matches what the service sends. Use `rabbit webhook test <name>` to verify the route works.
