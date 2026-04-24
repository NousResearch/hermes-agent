# LINE webhook fixed URL with Cloudflare named tunnel

`trycloudflare.com` is temporary. For a stable LINE webhook URL, use a named tunnel.

## One-time human steps

1. Log in to Cloudflare:

```bash
cloudflared tunnel login
```

2. Pick a hostname under a Cloudflare-managed zone, for example:

```text
line-hermes.example.com
```

## Automated setup

Run:

```bash
cd /Users/soichiyo/.hermes/hermes-agent
./scripts/setup_line_named_tunnel.sh hermes-line line-hermes.example.com
```

This script will:

- create or reuse the named tunnel
- route DNS to it
- write `~/.cloudflared/config.yml`
- write a launchd plist at `~/Library/LaunchAgents/com.soichiyo.cloudflared.hermes-line.plist`

## Start the tunnel as a user service

```bash
launchctl unload ~/Library/LaunchAgents/com.soichiyo.cloudflared.hermes-line.plist 2>/dev/null || true
launchctl load ~/Library/LaunchAgents/com.soichiyo.cloudflared.hermes-line.plist
```

## Make Hermes gateway persistent too

```bash
cd /Users/soichiyo/.hermes/hermes-agent
source venv/bin/activate
hermes gateway install
hermes gateway start
```

## LINE Developers webhook URL

Use:

```text
https://line-hermes.example.com/line/webhook
```
