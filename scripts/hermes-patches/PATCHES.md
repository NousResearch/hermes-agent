# Hermes Runtime Patches

Patches applied via cherry-pick from upstream (NousResearch/hermes-agent). 
Each entry records the upstream commit, our local commit, and what it does.

DO NOT run `hermes update` — it wipes these patches via git pull/reset.
Use `git cherry-pick <sha>` for future upstream integration.

## Active Patches

### P152 — Session state survives gateway restarts
- **Cherry-picked:** 2026-05-16
- **Upstream:** `e0e7397c` fix(session): persist auto-reset state across gateway restarts
- **Local:** `0a67a1a21`
- **Files:** gateway/session.py, gateway/run.py
- **Why:** Session auto-reset state persists across gateway restarts (MOL-576).

### P153 — Skip OpenViking upload symlinks in memory
- **Cherry-picked:** 2026-05-16
- **Upstream:** `63991bbd` fix(memory): skip OpenViking upload symlinks
- **Local:** `ebf5a3a88`
- **Files:** plugins/memory/tiered store + upload
- **Why:** Prevents memory provider from following symlinks in upload dirs.

### P154 — Silence memory provider teardown output
- **Cherry-picked:** 2026-05-16
- **Upstream:** `55ba02be` fix(background-review): silence memory provider teardown output leak
- **Local:** `90964fc77`
- **Files:** run_agent.py
- **Why:** Suppresses noisy memory provider shutdown output during background review.
- **Conflict:** run_agent.py — resolved by accepting incoming (tool whitelist + provider shutdown).

### P155 — Show context compaction status
- **Cherry-picked:** 2026-05-16
- **Upstream:** `00ad3d3c` fix: show context compaction status
- **Local:** (auto-merged into 90964fc77 sequence)
- **Files:** run_agent.py
- **Why:** Visibility into when context compaction fires — we can now see it happening.

### P156 — Compression model context-length detection with custom providers
- **Cherry-picked:** 2026-05-16
- **Upstream:** `7becb19e` fix(auxiliary): forward custom_providers to compression model context-length detection
- **Local:** `2b646ed20`
- **Files:** agent/auxiliary_client.py
- **Why:** Compression model works correctly with custom providers (our DeepSeek setup).

### P157 — Keep image results from poisoning text-only sessions
- **Cherry-picked:** 2026-05-16
- **Upstream:** `a28add19` fix(agent): keep image tool results from poisoning text-only sessions
- **Local:** `f2cf44134`
- **Files:** run_agent.py
- **Why:** Prevents image tool results from silently consuming context in text-only model sessions.
- **Conflict:** run_agent.py — resolved by accepting incoming (new image-rejection error patterns).

### P158 — Docs: media impact on session context
- **Cherry-picked:** 2026-05-16
- **Upstream:** `1dd33988` docs: clarify media impact on session context
- **Local:** `707d40b59`
- **Files:** website/docs/user-guide/sessions.md
- **Why:** Documents how media attachments affect context budget.

## Unreachable (need fetch)

These commits are on upstream main but not in our local object store (after last fetch cutoff):

- `627f8a5f` security: sanitize tool error strings before injecting into model context (May 16)
- `585d6b64` fix(gateway): merge rapid TEXT follow-ups during active sessions (May 16)

To pick these, we'd need to fetch upstream into a non-runtime temp clone, then cherry-pick from there.
