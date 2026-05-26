<p align="center">
  <img src="assets/banner.png" alt="Hermes Agent" width="100%">
</p>

# Inkbox-powered Hermes Agent ☤

<p align="center">
  <a href="https://github.com/NousResearch/hermes-agent"><img src="https://img.shields.io/badge/Upstream-NousResearch%2Fhermes--agent-blueviolet?style=for-the-badge" alt="Upstream"></a>
  <a href="https://inkbox.ai"><img src="https://img.shields.io/badge/Powered%20by-Inkbox-FFD700?style=for-the-badge" alt="Inkbox"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT"></a>
</p>

A fork of [Nous Research's Hermes Agent](https://github.com/NousResearch/hermes-agent) wired up to [Inkbox](https://inkbox.ai). The agent runtime is the same — the difference is every install ships with a **real email address, real phone number, and tunnel runtime** out of the box.

## Quick Install

One command. Installs deps, then walks you through Inkbox self-signup (no account prerequisite), provisions email + phone, and installs the gateway as a background service.

**Linux / macOS / WSL2 / Termux:**

```bash
curl -fsSL https://raw.githubusercontent.com/inkbox-ai/hermes-agent/inkbox/scripts/install.sh | bash
```

**Windows (early beta):**

```powershell
irm https://raw.githubusercontent.com/inkbox-ai/hermes-agent/inkbox/scripts/install.ps1 | iex
```

**By the end your agent has:**

- A real email address (`yourname@inkboxmail.com`) it can send and receive from
- A real phone number that handles inbound SMS and voice
- A persistent identity + contact list that survives across sessions
- A background service that auto-starts on boot, with HMAC-verified webhooks

## What this fork adds

| Capability | Upstream Hermes | This fork |
| --- | --- | --- |
| Chat platforms (Telegram, Slack, Discord, …) | 🟢 | 🟢 |
| Real email address | 🟡 DIY (SMTP) | 🟢 |
| Real phone (SMS + voice) | 🟡 DIY (Twilio + SIP) | 🟢 |
| Public webhook tunnel | 🟡 DIY (ngrok / server) | 🟢 |
| Persistent contact list | 🟡 DIY (agent memory) | 🟢 |

Specifically:

- The `inkbox` SDK and `aiohttp` are **core dependencies** (no `[inkbox]` extra to remember)
- `hermes gateway setup` runs Inkbox self-signup inline, mints the API key, writes it to `~/.hermes/.env`
- Tunnel runtime + gateway adapter built in — the agent receives webhooks without you hosting a server
- Gateway service installer wires the runtime into systemd / launchd / Scheduled Task
- HMAC signature verification on inbound webhooks + tunnel traffic

If you want **just the Hermes runtime** and prefer to bring your own SMTP / Twilio / SIP plumbing, use upstream at [`NousResearch/hermes-agent`](https://github.com/NousResearch/hermes-agent).

## Synced daily with upstream

We try to sync from [`NousResearch/hermes-agent`](https://github.com/NousResearch/hermes-agent) daily.

## Hermes itself

Everything beyond the Inkbox wiring — the CLI, supported models, skills, memory, tools, terminal backends, scheduled jobs, RL training, contributing — is unchanged from upstream:

- **Repo:** [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent)
- **Docs:** [hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs/)
- **Discord:** [NousResearch](https://discord.gg/NousResearch)

## License

MIT — same as upstream. See [LICENSE](LICENSE).
