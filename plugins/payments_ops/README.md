# Payments Ops Slack Surface

This plugin adds the small-screen companion workflow for the dashboard payments board.

What it does:

- Registers Slack Block Kit action handlers for `needs_review`, `ready_to_pay`, `paid`, and `ignored`
- Opens a compact Slack modal from the mobile inbox message so status changes are one tap away
- Adds `/payments-due` and `/payments-ready` slash commands for quick queue summaries
- Best-effort refreshes the paired Slack canvas after a status change when canvas channel env vars are configured

Operator scripts:

- Render a canvas spec:
  - `python3 scripts/payments-slack-surface.py canvas-spec`
- Publish the canvas:
  - `python3 scripts/payments-slack-surface.py publish-canvas`
- Post the compact mobile inbox:
  - `python3 scripts/payments-slack-surface.py post-mobile-inbox --channel <channel-or-dm-id>`

Relevant env vars:

- `SLACK_BOT_TOKEN`
- `HERMES_PAYMENTS_DASHBOARD_URL`
- `PAYMENTS_SLACK_CANVAS_CHANNEL_KEY`
- `PAYMENTS_SLACK_CANVAS_CHANNEL_NAME`
