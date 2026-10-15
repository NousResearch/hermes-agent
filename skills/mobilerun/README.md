# Mobilerun Skill for Hermes Agent

Give [Hermes Agent](https://github.com/NousResearch/hermes-agent) native control of Android and iOS devices via the [Mobilerun](https://mobilerun.ai) API.

Tap, swipe, type, take screenshots, read screen state, run autonomous AI agent tasks, manage devices, apps, proxies, eSIM, GPS, credentials, and webhooks.

## Install

```bash
hermes skills install https://raw.githubusercontent.com/Ramtx/mobilerun_h/main/SKILL.md
```

Or copy the `SKILL.md` and `references/` folder into your Hermes skills directory:

```bash
cp -r . ~/.hermes/skills/autonomous-ai-agents/mobilerun/
```

## Prerequisites

- [Hermes Agent](https://github.com/NousResearch/hermes-agent) installed
- Mobilerun API key (`dr_sk_...`) from [cloud.mobilerun.ai/api-keys](https://cloud.mobilerun.ai/api-keys)
- A connected device (personal phone via [Portal APK](https://github.com/droidrun/mobilerun-portal), or cloud-hosted)

## Files

| File | Purpose |
|------|---------|
| `SKILL.md` | Main skill — runbook with Quick Reference, Decision tree, Procedure, Pitfalls, Verification |
| `references/api.md` | Full API endpoint reference |
| `references/setup.md` | Auth, Portal setup, device types, billing |
| `references/troubleshooting.md` | 10 common issues with fixes |

## Links

- [Mobilerun](https://mobilerun.ai) — the platform
- [Mobilerun Docs](https://docs.mobilerun.ai) — full documentation
- [Droidrun Portal](https://github.com/droidrun/mobilerun-portal) — Android app for connecting personal devices
- [Discord](https://discord.gg/kc2JYQfX2c) — community
