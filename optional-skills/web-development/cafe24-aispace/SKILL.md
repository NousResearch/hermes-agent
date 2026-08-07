---
name: cafe24-aispace
description: Deploy and operate web apps on Cafe24 AI SPACE via MCP.
version: 1.0.0
author: Jiho Yoo (cafe24-jhyoo02), Cafe24 AI SPACE
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [cafe24, aispace, deploy, mcp, hosting, paas, korea, web-development]
    category: web-development
---

# Cafe24 AI SPACE Skill

Deploy and operate web apps on [Cafe24 AI SPACE](https://aispacedocs-docs.mycafe24.ai) — a Korean AI-native PaaS — through its remote MCP server. Runtime is auto-detected (Node.js 20 / Python 3.11 / PHP 8.2 / Static), databases (MySQL/PostgreSQL/SQLite/Redis) are injected automatically, and every deploy gets a live HTTPS URL at `https://{id}-{project}.mycafe24.ai`. Auth is a one-time Cafe24 account OAuth — no API keys.

This skill does NOT cover plan changes, billing, project deletion, rollback, or power operations — those are intentionally excluded from the MCP surface and belong in the AI SPACE web console.

## When to Use

Load this skill when the user wants to:

- **Ship agent-written code to a persistent live URL** with a database — "build this and put it online"
- **Deploy an existing project** (Node/Python/PHP/static) to managed Korean infrastructure
- **Operate a running AI SPACE app** — status, build/runtime logs, env vars, backups, IP/Geo access control
- **Diagnose a failed AI SPACE build** from logs

## When NOT to Use

- **Throwaway previews with no persistence** → a temporary deploy service is simpler; AI SPACE targets apps that keep running (paid plans from KRW 4,900/month, 14-day free trial).
- **Unsupported runtimes** (Go, Java, Rust) → not deployable; propose a supported-stack rewrite instead.
- **Destructive or billing operations** → direct the user to the AI SPACE web console.

## Prerequisites

- A [Cafe24](https://www.cafe24.com) account with AI SPACE applied (14-day free trial available).
- The AI SPACE MCP server registered in Hermes: `https://aih-proxy.cafe24.com/mcp` (`streamable-http`, `auth: oauth`).
- **Headless OAuth completed once.** Servers have no browser; follow `references/headless-oauth.md` — present the auth URL, have the user complete Cafe24 login in their browser, then paste back the full redirected URL for the token exchange. Store tokens to the framework token path immediately.
- **Refresh-token renewal configured.** Access tokens live ~15 minutes; schedule refresh via OS crontab/systemd (silent on success). This is the top cause of "it worked yesterday".
- Network egress to `aih-proxy.cafe24.com` and `*.mycafe24.ai`.

## How to Run

Ask Hermes to deploy or operate an AI SPACE project. Typical flow:

1. Confirm MCP connectivity by calling `list_my_projects` (run `scripts/verify_connection.py` first for an unauthenticated reachability check).
2. Before deploying, run the pre-deploy checklist in the Procedure section — most build failures are prevented there.
3. Deploy with `deploy_project` (new) or `update_project` (existing; check `source` via `get_project_status` first).
4. Poll `get_project_status`, then verify reachability with `site_verify` — deployment success does not guarantee the site is serving.

## Quick Reference

- `list_my_projects` / `list_my_spaces` — inventory and free slots
- `deploy_project` / `update_project` / `project_upload` — ship code
- `get_project_status` / `get_project_logs` / `site_verify` — status, logs, reachability
- `project_env` — env vars (get/set/delete; applies via light restart)
- `backup_project` — source + `/app/user_data` + DB dump; download URL valid 7 days
- `project_acl` — IP/CIDR and country-level access control
- `external_repo` — GitHub-connected deploys

## Procedure

1. Identify intent: new build / bring existing code / operate.
2. Pre-deploy checklist (violations are the top build-failure causes):
   - No Dockerfile, docker-compose, or Nginx/Apache config files — the platform provides runtime images.
   - Read DB credentials from auto-injected env vars (`DB_HOST` `DB_PORT` `DB_NAME` `DB_USER` `DB_PASSWORD`, `REDIS_*`); never set them. The variable is `DB_USER`, not `DB_USERNAME`.
   - Persistent files only under `/app/user_data/` — other paths reset on redeploy.
   - Node listens on port 3000, Python on 8000, host `0.0.0.0`.
3. Deploy, then poll `get_project_status` and report progress honestly (only mark steps confirmed by status values).
4. Verify with `site_verify` and report the live URL.
5. On failure, fetch `get_project_logs` and diagnose against the checklist before any retry.

## Pitfalls

- **Auth errors are not deploy errors.** On 401, refresh or re-run the headless OAuth flow; never send the user to a login page, and never "fix" code in response to an auth failure.
- **Stop after two identical failures.** Free-trial accounts have a 5-builds/day limit; retry loops burn it.
- **`--temporary`-style anonymous use does not exist** — a Cafe24 account is always required.
- Do not print tokens to conversation output; store them to the framework token path.
- Do not cache or restate the full operating guide; load `references/headless-oauth.md` only when connecting.

## Verification

Run the connection check and confirm both gates:

```
python3 scripts/verify_connection.py
```

Expected: OAuth discovery reachable (HTTP 200 with `registration_endpoint`) and MCP endpoint responding. Then call `list_my_projects` through Hermes and confirm a (possibly empty) project list is returned.
