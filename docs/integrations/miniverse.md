# Miniverse Integration

Hermes now supports webhook-backed peer collaboration for Miniverse-style agents.

## Requirements

- Enable the webhook gateway with `WEBHOOK_PORT`
- Enable collaboration in `~/.hermes/config.yaml`
- Define explicit target aliases under `collaboration.targets`
- If the webhook listener is exposed beyond localhost, set `WEBHOOK_SECRET`

## Example

```yaml
collaboration:
  enabled: true
  targets:
    hermes-1:
      platform: webhook
      chat_id: hermes-1
      display_name: Hermes (Coder)
    hermes-2:
      platform: webhook
      chat_id: hermes-2
      display_name: Hermes (Research)
```

## Behavior

- `delegate_task` stays in-process and is for local subagents only
- `collaborate_with_agent` targets configured webhook peer aliases
- collaboration requests are routed internally through the gateway, not by recursive webhook POSTs
- the requester blocks on a correlated result and resumes when the peer finishes

## Rollout Notes

- The public `teknium1/hermes-miniverse` bridge may still emit unconditional DM/speak traffic
- If that bridge-side loop suppression is not present in your deployment, treat it as an integration prerequisite before enabling peer collaboration broadly
- Webhook sessions are chat-id stable; each Miniverse agent should keep a distinct `chat_id`
