# Hermes Primary Host Model

Hermes can follow the same steady-state pattern as `openclaw-home`:

- `Primary MacBook` is the production owner
- `iPhone / M5 MacBook` are client devices
- this current laptop does not need to stay open

The important difference from Slack / Discord is that LINE requires a public HTTPS webhook.

## Steady state

```text
LINE user
  -> LINE Messaging API
  -> fixed HTTPS webhook
  -> cloudflared named tunnel
  -> Hermes gateway on Primary Mac
  -> model API

iPhone / M5 MacBook
  -> Tailscale
  -> Primary Mac for maintenance only
```

## What Tailscale is for

Tailscale is not the LINE webhook path.

Tailscale is used for:

- remote terminal access to the Primary Mac
- remote browser/admin access from M5 or iPhone
- operating the machine when you are away

LINE webhook delivery itself still needs:

- a public HTTPS hostname
- recommended: Cloudflare named tunnel with a fixed hostname

## Primary host responsibilities

On the Primary Mac, keep these always-on:

1. `hermes gateway` launchd service
2. `cloudflared` named tunnel launchd service
3. `Tailscale`
4. no-sleep / lid-close-safe configuration

## One-time setup on the Primary Mac

### 1. Install Hermes and configure LINE env

Copy your working `~/.hermes` profile or re-run:

```bash
cd ~/.hermes/hermes-agent
source venv/bin/activate
hermes gateway setup
```

Confirm these are present in `~/.hermes/.env`:

- `LINE_ENABLED=true`
- `LINE_CHANNEL_ACCESS_TOKEN=...`
- `LINE_CHANNEL_SECRET=...`
- `LINE_WEBHOOK_PORT=8646`
- `LINE_WEBHOOK_PATH=/line/webhook`

### 2. Install Hermes gateway as launchd

```bash
cd ~/.hermes/hermes-agent
source venv/bin/activate
hermes gateway install
hermes gateway start
```

### 3. Create a fixed Cloudflare tunnel

First login:

```bash
cloudflared tunnel login
```

Then create the named tunnel:

```bash
cd ~/.hermes/hermes-agent
./scripts/setup_line_named_tunnel.sh hermes-line line-hermes.example.com
```

Then start the tunnel service:

```bash
launchctl load ~/Library/LaunchAgents/com.soichiyo.cloudflared.hermes-line.plist
```

### 4. Update LINE Developers webhook URL

Set:

```text
https://line-hermes.example.com/line/webhook
```

## Daily operations

Use:

```bash
cd ~/.hermes/hermes-agent
./scripts/hermes-line-primary.sh status
./scripts/hermes-line-primary.sh restart
./scripts/hermes-line-primary.sh logs
```

## Reliability checklist

Like `openclaw-home`, treat the Primary Mac as an always-on host:

- Tailscale auto-start enabled
- Hermes gateway launchd service installed
- cloudflared launch agent installed
- power connected
- lid-close does not suspend the machine
- reboot recovery tested once

## Human work still required

These cannot be completed automatically:

1. `cloudflared tunnel login`
2. picking a hostname on a Cloudflare-managed domain
3. updating the LINE Developers webhook URL
4. macOS sleep / Amphetamine settings on the Primary Mac
