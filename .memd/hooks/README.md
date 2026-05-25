> **Generated file.** These scripts are synced from `.memd/hooks/` by `scripts/sync-integration-hooks.sh`.
> Edit the source at `.memd/hooks/` and re-run the script. Do not edit files in this directory directly.

# memd Hook Kit

These scripts are the default agent loop integration for `memd`.

Use them when a client wants:

- a bundle-backed wake-up surface before work starts
- a stable live-capture path while task state changes
- durable spill at a compaction boundary
- a single stable path into the memory manager

For per-project bootstrap, use:

```bash
memd setup --output .memd --project <project> --namespace <namespace> --agent <agent>
```

Check bundle health with:

```bash
memd status --output .memd
```

Resume the default memory snapshot from the bundle:

```bash
memd resume --output .memd
```

Refresh the startup wake-up surface and write it into the bundle:

```bash
memd wake --output .memd --intent current_task --write
```

Force a manual refresh of the same bootstrap path in an existing session:

```bash
memd refresh --output .memd
```

That also refreshes:

- `.memd/mem.md`
- `.memd/wake.md`
- `.memd/events.md`
- `.memd/agents/CLAUDE_IMPORTS.md`

For Codex, that wake path is the pre-turn read step in the harness pack flow.
It pulls compiled memory first, then refreshes the visible wakeup files after a
successful backend read.

Persist a memory into the same bundle lane:

```bash
memd remember --output .memd --kind decision --content "Store the outcome worth keeping."
```

Emit a shared handoff and refresh the same markdown memory files with shared
lane/source information:

```bash
memd handoff --output .memd
```

Agent-specific bundle entrypoints are generated under `.memd/agents/`:

- `codex.sh`
- `claude-code.sh`
- `agent-zero.sh`
- `openclaw.sh`
- `opencode.sh`
- `hermes.sh`

For Claude Code, import `.memd/agents/CLAUDE_IMPORTS.md` from project
`CLAUDE.md` and verify it with `/memory`. That bridge loads only
`.memd/wake.md` by default; deeper recall stays explicit.

The same bundle also writes `.memd/COMMANDS.md`, and you can inspect the
catalog at any time with:

```bash
memd commands --output .memd --summary
```

OpenClaw is the second harness pack after Codex and uses the same shared hook
kit, but its primary flow is context + spill instead of wake + capture.

Hermes is the adoption-focused harness pack after OpenClaw and uses the same
shared hook kit, but its primary flow is onboarding-friendly wake + capture +
spill with cloud-first reach and self-host later.

Agent Zero is the zero-friction harness pack after Hermes and uses the same
shared hook kit, but its primary flow is fast resume + durable remember +
clean handoff + spill for fresh sessions.

OpenCode is the shared-lane harness pack after Agent Zero and uses the same
shared hook kit, but its primary flow is resume + remember + handoff + spill
for explicit continuity clients.

## Environment

Set:

- `MEMD_BASE_URL` - defaults to the bundle's exported value; if no bundle env is loaded it falls back to the shared Tailscale endpoint for the hosted deployment
- `MEMD_PROJECT` - required for context fetches
- `MEMD_NAMESPACE` - optional namespace lane inside the project
- `MEMD_AGENT` - required for context fetches
- `MEMD_ROUTE` - defaults to `auto`
- `MEMD_INTENT` - defaults to `current_task`
- `MEMD_WORKSPACE` - optional shared workspace lane
- `MEMD_VISIBILITY` - optional `private|workspace|public`
- `MEMD_LIMIT` - defaults to `8`
- `MEMD_MAX_CHARS` - defaults to `280`
- `MEMD_RAG_URL` - optional; bundle backend config can supply this when present

## Context Hook

```bash
./memd-context.sh
```

This now calls `memd resume --prompt` under the bundle defaults and defaults the
intent to `current_task`. It now routes through `memd wake --write` so the same
startup call both renders the live wake-up view and refreshes the generated
memory files in the bundle.

For Codex bundles, the wake path also refreshes `.memd/wake.md`,
`.memd/mem.md`, and the Codex agent copies after a successful backend
read. Cached local bundle markdown is only trusted after the current session
has already completed one live wake. A brand-new session must not silently
bootstrap from stale cache.

The installed `memd-hook-context` shim now routes through this script, so the
default installed hook path also gets the richer wake-up surface.

## Capture Hook

```bash
printf 'changed auth flow: keep optimistic UI disabled for now\n' | ./memd-capture.sh
```

This routes through `memd hook capture --stdin --summary` under the active
bundle defaults and writes an episodic live-memory update back into the hosted
backend. Use it whenever task state changes and you want the live backend to
stay ahead of transcript loss.

For Codex bundles, a successful capture also refreshes the local wake/memory
files so the visible bundle stays in sync. If capture or recall fails, the
script keeps the existing local bundle truth and preserves the turn result
instead of overwriting it with partial state.

If captured line starts with typed prefix like `decision:`, `preference:`,
`constraint:`, `fact:`, `runbook:`, `procedural:`, or `status:`, `memd hook
capture` now auto-promotes durable memory even without explicit
`--promote-kind`.

When auto-promotion fires, `memd` also auto-tags durable memory from content:
- kind tag like `decision` or `preference`
- `correction` when superseding stale memory
- `design-memory` for UX/UI/design preferences
- `product-direction` for memory-loop/startup-surface style product truth

If a captured event is durable truth instead of transient task state, promote it
in the same call:

```bash
printf 'decision: keep wake as the universal startup surface\n' | memd hook capture --output .memd --stdin --promote-kind decision --promote-tag 10-star --promote-tag product-direction
```

That records both the live episodic update and a durable typed project memory.

If the new durable memory corrects a stale belief, supersede the stale memory in
the same call:

```bash
printf 'corrected fact: hosted backend health does not prove usable agent memory\n' | memd hook capture --output .memd --stdin --promote-kind fact --promote-tag correction --promote-supersede <stale-memory-uuid>
```

Or let `memd` find likely stale targets first:

```bash
printf 'corrected fact: hosted backend health does not prove usable agent memory\n' | memd hook capture --output .memd --stdin --promote-kind fact --promote-supersede-query "hosted backend health"
```

For `corrected fact:` / `corrected decision:` / `corrected preference:` /
`corrected constraint:` / `correction:` payloads, `memd` now infers that
supersede query automatically when no explicit supersede target is provided.

## Stop Save Hook

```bash
./memd-stop-save.sh
```

This is the parity hook against MemPalace's periodic save checkpoint. It reads
the session transcript metadata from stdin and blocks every `MEMD_SAVE_INTERVAL`
user messages. The block reason forces the agent to persist state into `memd`
before ending the turn instead of relying on manual memory discipline.

Use it for harnesses that support a `Stop` hook.

## PreCompact Save Hook

```bash
./memd-precompact-save.sh
```

This hook always blocks right before context compaction and tells the agent to
checkpoint, write durable truth, and spill any available compaction packet
before compaction proceeds.

Use it for harnesses that support a `PreCompact` hook.

## PostCompact Restore Hook

```bash
./memd-postcompact-restore.sh
```

Companion to `memd-precompact-save.sh`. Runs AFTER compaction completes and
BEFORE any `PreToolUse` hook fires, so the post-compaction turn inherits the
prior session's file-interaction ledger. The hook invokes
`memd hook restore --session-id <SID> --output <BUNDLE_ROOT>`, which copies
the newest sealed ledger back into `file_interactions.json` and appends an
ndjson restore record to `<BUNDLE_ROOT>/logs/ledger-restore.ndjson`.

Feature flag: `MEMD_A4_LEDGER_SURVIVAL` (default `0` during dogfood). When
`0`, the hook exits 0 immediately — zero overhead, zero risk. Flip to `1`
after the 7-day dogfood window shows zero breach lines under normal use.

Normative contract: [`docs/contracts/hook-handoff.md`](../../docs/contracts/hook-handoff.md).

Use it for harnesses that support a `PostCompact` hook (Claude Code, Codex).

## Install on Unix

```bash
./install.sh
```

Optional:

- `MEMD_BIN=/path/to/memd ./install.sh`

## Spill Hook

```bash
./memd-spill.sh --stdin --apply < compaction.json
```

## Example Hook Wiring

For Codex-compatible hook runners:

```json
{
  "Stop": [{
    "type": "command",
    "command": "/absolute/path/to/memd-hook-stop-save",
    "timeout": 30
  }],
  "PreCompact": [{
    "type": "command",
    "command": "/absolute/path/to/memd-hook-precompact-save",
    "timeout": 30
  }]
}
```

## Install on Windows

```powershell
./install.ps1
```

Optional:

- `$env:MEMD_BIN = "C:\\path\\to\\memd.exe"`
