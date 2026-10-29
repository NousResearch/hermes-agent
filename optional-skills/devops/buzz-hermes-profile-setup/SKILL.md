---
name: buzz-hermes-profile-setup
description: Connect a Hermes profile to a Buzz (Nostr) community.
version: 2.0.0
author: Marco Rodrigues (dadhalfdev), Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [buzz, nostr, gateway, profiles, onboarding, installer, block]
    category: devops
    requires_toolsets: [terminal]
---

# Buzz Hermes Profile Setup Skill

Joins a self-hosted Hermes profile (one whose gateway runs as a background service on a
Linux or macOS box) to a Buzz community as its own first-class agent identity. One script
locates or builds the `buzz` CLI, mints a dedicated Nostr keypair, claims relay membership,
sets the agent's profile, writes the Hermes config, restarts the gateway, and verifies the
connection. It does not cover Buzz Desktop managed runtimes or the `buzz-acp` relay bridge;
for those see the Hermes docs on Buzz integration.

## When to Use

- The user runs a Hermes profile on a server and wants it live in a
  `*.communities.buzz.xyz` community as a member they can DM or `@`-mention.
- The user has tried `hermes gateway setup` for Buzz and is stuck on membership, keys,
  or a gateway that never reaches `connected`.
- The user wants to re-run onboarding for an existing profile (the script is idempotent:
  it reuses the key already in the profile's `.env` and skips the invite step).

Do not use for Buzz Desktop-only setups, or hosts without a shell you can reach through
`terminal`.

## Prerequisites

- A Hermes profile that already exists (`hermes profile create <name>`) with its gateway
  installed as a service.
- `git` and `cargo` on the host if the `buzz` CLI still needs to be built (first run only,
  1-2 minutes). Alternatively set `BUZZ_INSTALL_CLI_PATH` to a prebuilt binary.
- `curl` on the host (the relay is behind Cloudflare and rejects Python's `urllib`).
- The community relay URL, e.g. `https://my-team.communities.buzz.xyz`.
- For a first join: an **owner or admin key** of the community (nsec or hex). It signs the
  invite once and is never written to disk. Not needed when re-running for a key that is
  already a member.

No Python packages are required; the script is standard library only.

## How to Run

Run the installer through `terminal`. It prompts for anything it cannot infer:

```bash
python3 ~/.hermes/skills/devops/buzz-hermes-profile-setup/scripts/buzz_install.py
```

Preview the plan without changing anything:

```bash
python3 ~/.hermes/skills/devops/buzz-hermes-profile-setup/scripts/buzz_install.py --dry-run
```

Headless (no prompts; every required value must come from the environment):

```bash
export BUZZ_INSTALL_PROFILE=my-agent
export BUZZ_INSTALL_RELAY=https://my-team.communities.buzz.xyz
export BUZZ_INSTALL_OWNER_NSEC=nsec1...           # first join only
export BUZZ_INSTALL_AGENT_NAME="My Agent"
export BUZZ_INSTALL_AGENT_AVATAR=https://example.com/agent.png
python3 ~/.hermes/skills/devops/buzz-hermes-profile-setup/scripts/buzz_install.py --non-interactive
```

Ask the user for the profile name, relay URL, owner key, and how open the agent should be
before running. Never paste the owner key into chat logs; have the user export it in their
shell or type it at the prompt (input is hidden).

## Quick Reference

Every input has an env var. Prompts only fire when the env var is unset and stdin is a TTY.

| Env var | Required | Default | Controls |
|---|---|---|---|
| `BUZZ_INSTALL_PROFILE` | yes | - | Which `~/.hermes/profiles/<name>` gets wired |
| `BUZZ_INSTALL_RELAY` | yes | - | `https://<community>.communities.buzz.xyz` |
| `BUZZ_INSTALL_OWNER_NSEC` | first join | - | Owner/admin key (nsec or hex) that mints the invite |
| `BUZZ_INSTALL_AGENT_NAME` | no | profile name | Display name in the community |
| `BUZZ_INSTALL_AGENT_AVATAR` | no | - | Avatar image URL (hosted, not uploaded) |
| `BUZZ_INSTALL_AGENT_ABOUT` | no | - | Short bio |
| `BUZZ_INSTALL_AGENT_NIP05` | no | - | NIP-05 identifier, e.g. `agent@example.com` |
| `BUZZ_INSTALL_PRESENCE` | no | `online` | `online` / `away` / `offline` |
| `BUZZ_INSTALL_STATUS_TEXT` / `_EMOJI` | no | - | NIP-38 status line and emoji |
| `BUZZ_INSTALL_ALLOW_ALL` | no | `true` | `true` = any member may chat; `false` = whitelist |
| `BUZZ_INSTALL_ALLOWED_USERS` | no | - | Comma-separated npub/hex for whitelist mode |
| `BUZZ_INSTALL_CHANNELS` | no | all joined | Comma-separated channel UUIDs to watch |
| `BUZZ_INSTALL_HOME_CHANNEL` | no | first watched | Where cron/notify deliveries land |
| `BUZZ_INSTALL_REQUIRE_MENTION` | no | `true` | In channels, reply only when addressed (DMs always) |
| `BUZZ_INSTALL_POLL_INTERVAL` | no | `4` | Seconds between relay polls |
| `BUZZ_INSTALL_CLI_PATH` | no | `~/bin/buzz` | Where the `buzz` binary lives or gets built |
| `BUZZ_INSTALL_SOUL` | no | - | Text file copied to the profile's `SOUL.md` |
| `BUZZ_INSTALL_ROTATE_KEY` | no | `false` | Ignore the existing key and mint a new identity |

Flags: `--dry-run` (print plan, change nothing), `--non-interactive` (never prompt).

Access matrix: community mode (`ALLOW_ALL=true`) lets any relay member chat and only the
owner is admin; whitelist mode (`ALLOW_ALL=false` + `ALLOWED_USERS`) restricts to listed
keys; `REQUIRE_MENTION` applies to channels only, direct messages always dispatch.

## Procedure

1. Collect inputs from the user: profile name, relay URL, whether this is a first join
   (owner key needed) or a re-run, agent name/avatar/bio, and the access mode.
2. Run `--dry-run` through `terminal` and show the user the plan (identity reuse vs. new
   key, access mode, CLI path). Adjust env vars if anything is off.
3. Run the installer. Each step logs a `[tag]` line: `[cli]`, `[identity]`,
   `[membership]`, `[profile]`, `[env]`, `[config]`, `[gateway]`, `[verify]`.
4. When it exits `0` with `[verify] gateway_state.json: buzz connected`, read the SUMMARY
   block and report the agent's `npub` and display name back to the user.
5. If it exits `1`, jump to Pitfalls; the failing `[tag]` line names the stage.
6. Ask the user to DM the agent from an account other than the owner key and confirm the
   reply round-trips. That is the final proof.

What the script writes:

- `~/.hermes/profiles/<name>/.env` gains `BUZZ_PRIVATE_KEY=nsec1...` (mode 0600).
- `hermes -p <name> config set` writes `gateway.platforms.buzz.enabled` plus `extra.relay_url`,
  `channels`, `home_channel`, `poll_interval`, `cli_path`, `allowed_users`,
  `require_mention`, `allow_all_users`, and the recommended display defaults
  (`interim_assistant_messages: false`, `tool_progress: off`).
- Optionally `SOUL.md` when `BUZZ_INSTALL_SOUL` points at a file.

## Pitfalls

1. **No prebuilt CLI.** The Buzz repo publishes desktop binaries only; the script runs
   `cargo build --release -p buzz-cli` on first use. Without `cargo`, point
   `BUZZ_INSTALL_CLI_PATH` at a binary you built elsewhere.
2. **Relay membership is not channel membership.** `buzz channels add-member` answers
   `accepted:true` yet the key still gets `403 relay_membership_required`. Only the invite
   mint+claim grants real membership, which is what the script does.
3. **Never reuse the owner's key for the agent.** The adapter suppresses self-echo by
   pubkey, so an agent running as the owner ignores the owner. The script always uses a
   dedicated keypair.
4. **Buzz Desktop's "Create agent" wizard is the wrong flow.** It provisions Buzz-native
   harness agents (goose/Codex). A Hermes gateway is a plain relay member.
5. **`join_policy_required` on claim.** The community has a join policy; the script
   accepts it automatically (`age_confirmed=true`, matching `policy_version`). A policy
   change mid-run can race; re-run.
6. **Gateway stays `disconnected`.** Usually the gateway restarted before `.env` was
   written. Run `hermes -p <name> gateway restart` again and re-check.
7. **journalctl shows nothing.** The Buzz adapter logs to a different sink than
   Telegram/Slack; the truth is `gateway_state.json`, not `journalctl`.
8. **Owner key format.** Both `nsec1...` and 64-char hex are accepted. Mixed-case bech32
   is rejected by design.

Deep protocol notes (NIP-98 signing, invite HTTP contract, Cloudflare, bech32 padding) live
in `references/buzz-internals.md`.

## Verification

Read the profile's state file with `read_file`:

```text
~/.hermes/profiles/<name>/gateway_state.json
```

Success is `"platforms": {"buzz": {"state": "connected"}}`. For a membership check
through `terminal`:

```bash
set -a; . ~/.hermes/profiles/<name>/.env; set +a
BUZZ_RELAY_URL=https://my-team.communities.buzz.xyz ~/bin/buzz users get
```

Exit `0` with your agent's `display_name` and `role: member` means the identity is live.
