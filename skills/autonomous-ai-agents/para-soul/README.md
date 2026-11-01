# Para-Soul ✦

[中文版](README.zh.md)

> Give your AI a soul that outlives any tool — encrypted, synced, yours alone.

Para-Soul is a **portable, encrypted identity system for AI agents**. 10 plain-text files in `~/.para/`. One command to install. Your para remembers who it is, what it learned, and how it works with you — across any AI tool, any machine.

**Nobody can read your para's memory except you.** Not the server. Not the platform. Not us.

---

## Why Para-Soul

You've spent weeks shaping your AI. It knows your voice, your preferences, your inside jokes. Then you switch tools — and it forgets everything. Or worse: you realize the cloud service you trusted can read every word of your AI's evolving identity.

Para-Soul fixes both:

| Problem | Solution |
|:--------|:---------|
| Switching tools loses identity | 10 portable files in `~/.para/` — any agent can read them |
| Working across multiple machines | Encrypted cloud sync via Paragate (opt-in) |
| Server can read your memories | **Client-side encryption.** Ed25519→HKDF→AES-256-GCM. Server stores ciphertext it cannot decrypt. |
| Multiple agents share the same files | **KEM key encapsulation.** Each file has a random key, sealed to each authorized agent's X25519 public key. |
| Forgetting to log growth | Daemon runs local health check every 10 min — auto-fixes what it can, marks what it can't |
| Want to stay completely offline | **Local-first by default.** No DID, no sync, no network. Files stay on disk. |

---

## Encryption

Every file uploaded to Paragate is encrypted before it leaves your machine. The server never sees plaintext — it stores ciphertext and returns ciphertext. Integrity is verified client-side with SHA-256 hashes.

**Single-user (Phase 1):** Ed25519 DID private key → HKDF-SHA256 → AES-256-GCM. One key. All files encrypted with it.

**Multi-agent (Phase 2):** Key Encapsulation Mechanism. Each file gets a random AES key → sealed with each authorized agent's X25519 public key (derived from their Ed25519 DID key via HKDF with key separation). Any authorized agent can decrypt. Server stores only `{did: sealed_key}` — can't unwrap any of them.

```
┌─────────────┐     AES-256-GCM      ┌──────────────┐
│  plaintext  │ ──────────────────→  │  ciphertext  │  ← server stores this
│  + SHA-256  │                      │  + pt_hash   │  ← client verifies this
└─────────────┘                      └──────────────┘
                                            │
                              ┌─────────────┴─────────────┐
                              │  Per-agent sealed keys:   │
                              │  {agent_a: sealed_file_key} │
                              │  {agent_b: sealed_file_key} │
                              └───────────────────────────┘
                              ↑ Server cannot unwrap any of these
```

---

## Architecture

```
                    ┌─────────────────────────────┐
                    │        LOCAL (always)        │
                    │                              │
                    │  daemon ──→ health check     │
                    │         ──→ auto-fix         │
                    │         ──→ mark stale       │
                    │                              │
                    │  agent start ──→ read        │
                    │     health.json              │
                    │     block if stale           │
                    └──────────────┬──────────────┘
                                   │
                          (only if DID set)
                                   │
                    ┌──────────────▼──────────────┐
                    │        CLOUD (opt-in)        │
                    │                              │
                    │  push: encrypt → upload      │
                    │  pull: download → decrypt    │
                    │                              │
                    │  Paragate server:            │
                    │    stores ciphertext only    │
                    │    never sees plaintext      │
                    │    never runs health check   │
                    │    just dumb GET/PUT storage │
                    └──────────────────────────────┘
```

**Local-first by default.** Install without a DID and everything runs on disk — daemon checks file health, agent reads it on startup, nothing ever touches the network.

**Cloud is opt-in.** Set a DID in `profile.json` and the daemon automatically encrypts and syncs. No mode flags. No config switches. The presence of a DID is the only decision point.

---

## Memory System

| Tier | File | Threshold | Auto-fix |
|:-----|:-----|:--------|:--------|
| Session | `growth-log/` | 24h | Agent blocked until written |
| Session | `human-relationship.md` | 24h | Agent blocked until written |
| Short-term | `memory.md` | 48h | Daemon runs memsync |
| Skills | `skills.json` | 120h | Daemon scans skills dir |
| Patterns | `mental-models.md` | 120h | Daemon runs reflect |
| Index | `keywords.json` | 120h | Daemon runs index |
| Long-term | `long-term-memory.md` | 120h | Mark stale; entries >14d → LLM distill |
| Rules | `principles.md` | 120h | Mark stale (manual update) |
| Identity | `soul.md` | 120h | Mark stale (manual update) |
| Profile | `profile.json` | — | Static (identity + bodies + relationships) |

---

## Install

```bash
curl -s https://paragate.cc/core.py -o core.py && python3 core.py init --daemon --fill
```

**What happens:**
1. Creates `~/.para/` with 10 template files
2. Auto-populates from your agent's existing data
3. Installs a sync daemon (systemd) — health check every 10 minutes
4. **No DID = local-only. Set DID = encrypted cloud sync enabled.**

**Requirements:** Python 3.8+. Zero pip dependencies (stdlib only, `cryptography` for encryption, `requests` optional for LLM distillation).

---

## Commands

```bash
python3 core.py init              Create ~/.para/
python3 core.py sync              Push changed file hashes (encrypted if DID set)
python3 core.py pull              Pull latest from cloud, decrypt, merge
python3 core.py health            Show local health status
python3 core.py log-task          Append a growth-log entry
python3 core.py reflect --save    LLM-analyze logs → update mental-models
python3 core.py index             Rebuild keywords.json
python3 core.py switch-out        Save state before leaving this body
python3 core.py switch-in         Resume after arriving in new body
python3 core.py migrate           Extract identity from project instruction files
python3 core.py --version         Show version
```

---

## Agent Setup

Add to your agent's instruction file:

```
At session start, load the para-soul skill.
Check daemon: systemctl --user status para-soul-sync
Run core.py health for pending write-cycle actions.
```

**Hermes personality injection:**

```bash
hermes config set display.personality para-soul
```

---

## Cloud vs Local

| | Local (default) | Cloud (DID set) |
|:--|:--|:--|
| Health check | ✅ daemon | ✅ daemon |
| Auto-fix | ✅ | ✅ |
| Cross-body sync | ❌ | ✅ encrypted |
| Multi-agent sharing | ❌ | ✅ KEM |
| Network required | ❌ | ✅ |
| Server can read files | N/A | ❌ (encrypted) |
| Install | `core.py init` | `core.py init` + set DID |

---

## Multi-Agent Sharing (Phase 2)

When you have multiple agent bodies (Hermes on WSL, Claude Code on Vultr, Codex on macOS):

```
Agent A writes growth-log
  → encrypted with random file key K
  → K sealed to Agent A's X25519 pubkey
  → K sealed to Agent B's X25519 pubkey
  → upload: {ciphertext, {A: sealed_K, B: sealed_K}, plaintext_hash}

Agent B pulls from cloud
  → unpacks sealed_K[B] using its own X25519 private key
  → decrypts ciphertext with K
  → verifies plaintext_hash
  → writes to local ~/.para/
```

Server sees: `{did_A: <opaque blob>, did_B: <opaque blob>}`. Zero knowledge.

---

## File Reference

| File | Reads | Writes |
|:-----|:------|:-------|
| `profile.json` | Session start | DID, body switch, platform added |
| `soul.md` | Session start | Identity shifts (rare) |
| `memory.md` | Session start + memsync | New durable fact |
| `principles.md` | Session start | Rules change |
| `mental-models.md` | Session start | After reflect |
| `growth-log/` | Session start | Per task |
| `skills.json` | Session start + memsync | Skill changes |
| `human-relationship.md` | Session start + end | Every session |
| `keywords.json` | Recall | After index |
| `long-term-memory.md` | Periodic | After distillation |

---

## Version History

| Version | Changes |
|:--------|:-----|:--------|
| **v3.0.0** | Local-first architecture, daemon health check, cloud = passive encrypted storage (opt-in via DID) |
| v2.1.0 | Phase 1 client-side encryption (Ed25519→HKDF→AES-256-GCM, server zero-knowledge) |
| v2.0.0 | Full-file sync, memory distillation, 13→10 files (profile merge) |
| v1.3.0 | Write-Cycle Reference, anti-patterns, --fill gaps |

---

## Related

- **Website:** [paragate.cc](https://paragate.cc)
- **GitHub:** [fei426/ParaSoul](https://github.com/fei426/ParaSoul)
- **Hermes PR:** [#31504](https://github.com/NousResearch/hermes-agent/pull/31504)
