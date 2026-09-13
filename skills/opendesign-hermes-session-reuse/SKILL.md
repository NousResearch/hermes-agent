---
name: od-hermes-session-reuse
author: ox-alpha (design profile)
description: Use when OD chat spawns a Hermes session per message.
version: 1.1.0
license: MIT
metadata:
  hermes:
    tags: [opendesign, acp, session-reuse, resume, envelope, transcript, entropy]
    category: integration
---

# OD ↔ Hermes Session Reuse (one project = one session)

## Overview

OpenDesign (OD) drives Hermes Agent over ACP. The stock integration has 8 defects that compound into unusable session entropy: a new session per message, template-named titles, history replay spam, and envelopes burying the user's actual words. This skill is the complete diagnosis + repair playbook. Every change is reversible.

**Core mental model**: OD wraps every user turn in a "charter envelope" (an English instruction pack: project context + workspace + rules) with the user's real words at the tail. The tail marker comes in two variants: `## user` and `# User request`.

## When to Use

- The OD sidebar spawns a new session per message, titled `# Instructions (read first)` / `Set up OpenDesign ask mode` etc.
- The user asks "why can't I see my own words in Hermes?"
- Resume replays hours-old replies into the OD window before answering
- `hermes sessions list` ACP count balloons
- The user wants "one OD project window = one Hermes session"
- Historical fragments need consolidation into per-project ledgers

## The 8-Defect Map (diagnosis order = repair order)

| # | Defect | Symptom | Fix location |
|---|--------|---------|--------------|
| ② | Session reuse not enabled for Hermes | `session/new` every turn | OD: `hermesAgentDef` + `resumesSessionViaAcpLoad: true` |
| ③ | Capture reads only OpenCode's `openCodeSessionId` | `agent_sessions` table stays 0 rows | OD: fallback to standard `result.sessionId` |
| ④ | `session/load` missing `mcpServers` | OD shows `json-rpc id 2: Invalid params` | OD: add `mcpServers: mcpServers ?? []` |
| ⑤ | Load response lacks `sessionId` | OD: `invalid session/new response` (JSON contains `currentHermesSessionId` = resume already worked) | Hermes: `acp/schema.py` add field + echo |
| ⑥ | Full envelope persisted | titles/transcripts/search show English boilerplate | Hermes: `_extract_user_voice()` keeps only the tail |
| ⑦ | Resume replays all history | OD re-renders old replies every message | Hermes: skip `_replay_session_history` for OD clients |
| ⑥b | Envelope tail has two marker variants | extraction silently no-ops | regex `#{1,4}\s*(?:user|User request)\s*\r?\n` covers both |

②③④ live on the OD side; ⑤⑥⑦ on the Hermes side.

## Key Paths

```
OD runtime payload (patch HERE — not the install dir!):
  %APPDATA%\Open Design\launcher\channels\stable\namespaces\
    release-stable-win\versions\<VERSION>\payload\resources\app\prebundled\daemon\chunks\
  - chunk-4IN7PJG2.mjs   → hermesAgentDef          (defect ②)
  - chunk-EER2BOWK.mjs   → ACP client transport    (defects ③④)
  - chunk-N7AF5FHP.mjs   → image size mapping      (bonus, unrelated)

⚠ Copies under the OD install directory (D:\...\prebundled\daemon\) are used
  only by the CLI/MCP bridge — patching them changes nothing for chat.
  Prove the runtime path from the daemon-sidecar process command line.

Hermes side:
- <hermes-agent>/venv/Lib/site-packages/acp/schema.py   → LoadSessionResponse (⑤)
- <hermes-agent>/acp_adapter/server.py                  → load/resume/prompt  (⑤⑥⑦)
- <hermes-agent>/acp_adapter/session.py                 → SessionState fields (⑦)
```

## Repair Steps (each: backup → edit → node --check / py_compile → restart & verify)

### OD side (②③④) — full OD restart required

1. **Prove the runtime path** from the daemon-sidecar process command line.
2. **Backup**: `chunk-X.mjs → chunk-X.mjs.bak-<date>`.
3. **②** in `hermesAgentDef`, after `buildArgs`: `resumesSessionViaAcpLoad: true,`
4. **③** `durableSessionId = typeof result.openCodeSessionId === "string" ? result.openCodeSessionId : (typeof result.sessionId === "string" ? result.sessionId : null);`
5. **④** `writeRpc(nextId, "session/load", { sessionId: resumeSessionId, cwd: effectiveCwd, mcpServers: mcpServers ?? [] }, "session/load");`
6. **Restart**: close the OD GUI gracefully (all OD processes exit with it), relaunch via PowerShell `Start-Process`.
   - ⚠️ Never launch OD from a bash background job — the stdout pipe dies with the terminal and the Electron main process crashes with EPIPE.

### Hermes side (⑤⑥⑦) — live on the next message, no restart

5. **⑤** `schema.py` `LoadSessionResponse`: `session_id: Annotated[Optional[str], Field(alias="sessionId")] = None`; echo it in `load_session`'s return.
6. **⑥** add to `server.py` and use at the `run_conversation` call site:

```python
def _extract_user_voice(text: str) -> str:
    import re as _re
    if not text:
        return text
    m = _re.search(r"(?:^|\n)#{1,4}\s*(?:user|User request)\s*\r?\n(?:\r?\n)?", text)
    if m:
        tail = text[m.end():].strip()
        if tail:
            return tail
    return text

persist_user_message=_extract_user_voice(user_text) or "[Image attachment]",
```

- ⚠️ `import re` inside the function (or verify the module header) — a missing import kills the turn with `name 're' is not defined`.
- ⚠️ The model still receives the full envelope (`user_content` untouched); only the archive is cleaned.

7. **⑦** `session.py` `SessionState`: `skip_history_replay: bool = False`. In `server.py` `initialize`: `self._od_client_name = client_name`. In `load_session`/`resume_session` after `update_cwd`: `state.skip_history_replay = "open-design" in client_name.lower()`. Wrap both `_replay_session_history(state)` calls in `if not getattr(state, "skip_history_replay", False):` (indent the except bodies one level deeper).

### Verification per defect

| Defect | Acceptance |
|--------|-----------|
| ②③ | Two messages in one OD window → `hermes sessions list` grows by exactly 1; OD `agent_sessions` gains a row |
| ④⑤ | No more Invalid params / invalid session/new response |
| ⑥ | Desktop transcript shows the user's words; auto-title = topic of the actual prompt |
| ⑦ | No replay spam; direct answer to the new message |
| Final | One OD project window maps to one session long-term; answers recall hours-old context = true resume |

## Fragment Consolidation (history cleanup)

Hand off to the sibling skill `od-fragment-consolidator` (and `scripts/consolidate.py`): envelope-to-voice translation, per-project merge, ledger renaming, OD mapping pre-fill, and interactive cleanup of the emptied fragments.

## Pitfalls (all hit in production)

| Pitfall | Consequence | Prevention |
|---------|-------------|------------|
| Patching the install directory | No effect; a wasted test cycle | Prove the daemon-sidecar command line first |
| Launching OD from a bash background job | EPIPE crash dialog | PowerShell `Start-Process` |
| Regex covering only `## user` | Silent failure; half the envelopes stay | Regression-test with a REAL envelope from the DB |
| Helper uses `re` without import | Runtime `name 're' is not defined` | Import inside the function |
| Smoke tests injecting dependencies manually | Masks real import errors | Verify via real import: `from acp_adapter.server import X` |
| Restoring from a stale backup mid-fix | Wipes already-applied patches | Keep backups in sync with the latest good state |
| Killing the daemon process | It's the ancestor of live sessions and changes ports on restart | Close the OD GUI instead; daemon exits with it |
| OD auto-update (new version dir) | Disk patches lost | Re-apply ②③④; Hermes update/reinstall wipes ⑤⑥⑦ |

## Related

- Image size mapping (16:9 → 2752x1536 in `chunk-N7AF5FHP.mjs`) is an independent neighbor patch.
- Archive sweeps: `hermes sessions archive --title <template-name> --dry-run` first, always.
- Architecture: OD→Hermes syncs in real time; Hermes→OD has no text channel (product architecture). Knowledge flows through the shared brain (memory + session search on the same HERMES_HOME).
