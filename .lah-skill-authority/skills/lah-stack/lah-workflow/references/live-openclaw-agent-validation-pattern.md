# Live OpenClaw Agent Validation (persona / runtime parity)

Pattern for validating a DEPLOYED OpenClaw agent (e.g. `cloe-poc`) in a FRESH
session against the real brain, when the mission needs proof that canonical
files (SOUL.md / IDENTITY.md / USER.md) or code changes produce the intended
behavior on the running runtime. Established during
CLOE_PERSONA_FIDELITY_AND_GROUNDED_SYNTHESIS_REPAIR_V1 (PR #651 closed
SUPERSEDED_BY_CANONICAL_PERSONA_FILES) and reused in
CLOE_TELEGRAM_GATEWAY_BEHAVIOR_PARITY_V1.

## Agent topology

- Agent workspace (persona authority): `~/.openclaw/agents/<agentId>/workspace/`
  (SOUL.md, IDENTITY.md, USER.md, AGENTS.md). The OpenClaw runtime injects these
  into the system prompt at session start.
- Agent config: `~/.openclaw/openclaw.json` → `agents.list[]` (id, model, workspace).
  `main` usually has NO workspace (generic defaults); the named agent holds the
  corrected persona files.
- Deployed brain: the agent's model (e.g. `cloe/brain`) routes via
  `gateway.remote.url` to the deployed container's `/chat/completions` (e.g.
  http://127.0.0.1:4000) — same brain the operator tests against.
- Session store: `~/.openclaw/agents/<agentId>/sessions/sessions.json`; each
  entry has `sessionFile`, `systemPromptReport.workspaceDir`, `abortedLastRun`.

## Running a fresh validated session (non-interactive)

The CLI TUI (`openclaw terminal --local --message ...`) is INTERACTIVE: in
non-interactive mode it prints the provider call then "forcing process exit
after return" WITHOUT persisting the reply. Capture the real reply with a pty:

```bash
timeout 150 script -qec "node /home/deploy/.npm-global/lib/node_modules/openclaw/dist/index.js terminal --local --session 'agent:cloe-poc:cloe-gate-$(date +%s)' --message 'salut Cloe' --timeout-ms 90000" /tmp/out.txt > /dev/null 2>&1
```

Key facts:
- Session key convention: `agent:<agentId>:<rest>`. The CLI defaults to agent
  `main` — WITHOUT the `agent:<id>:` prefix you test the WRONG agent (main has
  the generic workspace, not the corrected persona files).
- The reply appears in the pty output between the user prompt and the footer
  `agent <id> ... | cloe/brain`. Strip CR (`tr -d '\r'`), filter UI lines.
- Latency: poll the pty file for a reply marker UNIQUE to the new reply — do
  NOT reuse markers that appear in replayed session history (e.g. 'Cedrick'
  appears in turn-1 history and causes false positives in later turns; same
  for generic markers like the persona name).
- The TUI stays open after replying: always wrap in `timeout` and kill the
  process. Timeout exit 124 is EXPECTED — read the pty file, don't treat the
  timeout as failure.
- Replies are non-deterministic per run (LLM): don't re-measure with a marker
  taken from a previous run's exact wording; use stable substrings or poll for
  ANY new content line.

## Proving canonical files are loaded

`systemPromptReport.workspaceDir` in the session store entry + behavioral
evidence: ask "rappelle-moi où on en était" — if the reply cites content that
exists ONLY in the corrected files (e.g. "SOUL.md, IDENTITY.md et USER.md
réécrits en français, avec backups dans memory/persona-backups/"), the files
are loaded. The deployed container alone (without the agent workspace) cannot
know that content — that is the proof that the runtime reloaded the canonical
files in a fresh session.

## Governance / safety

- NEVER read another agent's EXISTING session files without operator consent —
  sessions are private (two such commands were BLOCKED this mission). Only read
  files created by YOUR OWN fresh test session, or the pty captures under /tmp.
- The operator may reject a code fix in favor of the files-only solution: if
  canonical files alone satisfy the requirement on the deployed runtime, the
  PR can be closed SUPERSEDED_BY_CANONICAL_PERSONA_FILES and the code candidate
  kept as historical evidence (branch + commit, not activated). Always offer
  the live validation BEFORE assuming a code change is necessary; keep the
  candidate's untracked test scripts OUT of the repo (move to an artifact dir).

## Terminal secret handling pitfalls (Hermes)

- `TOKEN=$(cat /tmp/token)` inline in a terminal command gets MASKED by the
  Hermes terminal (`Bearer ***`) and can break bash parsing (unexpected EOF /
  unmatched quotes). Workarounds that work: write the token to a file and
  `source` it in the SAME command (works when the var name matches), pass the
  key as argv, or read inside a .mjs/.sh helper via fs/readFileSync. Never
  interpolate `$(cat ...)` into a command the terminal will echo/mask.
- `--env-file=.env` loads the WORKSPACE .env which may silently OVERRIDE or
  OMIT container-derived values: a comparison harness got empty "old" replies
  (workspace ADMIN_API_KEY ≠ container key) and empty "candidate" replies
  (DEEPSEEK_API_KEY missing from shell env). Load BOTH secrets explicitly, or
  read them inside the script from .env + `docker inspect` into a file.
- Diagnose empty API responses in order: (1) HTTP status + body bytes via
  `curl -w '%{http_code}'`, (2) confirm which key/secret the script actually
  used (env var name mismatch is the #1 cause), (3) then look at the code path.

## French prompts with apostrophes break `--message '...'` quoting

`--message 'Comment dois-tu t'adresser à moi ?'` → bash `unexpected EOF while
looking for matching quote`. Fix: write the message to a file and expand it
inside the command: `--message "$(cat /tmp/msg.txt)"`. Keep the session key and
message in shell vars, never interpolate secrets.

## Extracting the real reply from the pty capture (TUI redraw)

The TUI redraws lines in place with ANSI cursor/clear sequences (`\x1b[2K`,
`\x1b[0G`, `\r`) — the reply text is interleaved with spinner frames
(`⠋ running • Ns | local ready`). Naive `grep`/`awk` on the raw capture LOSES
the first reply line (it's fused with a spinner frame). Robust extraction:

1. Split the raw capture on `\x1b[2K` / `\x1b[0G` / `\r` FIRST.
2. Strip ANSI (`\x1b\[[0-9;?]*[a-zA-Z]`) per segment.
3. Drop UI frames (spinner chars, `hobnobbing|running •|streaming|finishing
   context|local ready|tokens ?|agent cloe-poc|separators`).
4. Stop at `ended with stopReason`.

A reusable extractor is `scripts/extract-center-reply-v2.mjs`-style logic — keep
it in /tmp or scripts/, not in the repo.

## Baseline via in-container webhook harness (Telegram channel)

For Telegram/Gateway persona-parity baselines the operator requires the FULL
`handleTelegramWebhook()` harness — production env, real feature flags, real
session store, real brainAsk binding (`buildBrainAskResponse` with
`env: process.env`) — NOT a direct POST to `/chat/completions` and NOT an
isolated `buildBrainAskResponse()` call (both bypass the layers under
comparison). Run it INSIDE the production container via
`docker cp` + `docker exec node` so it sees the real env/secrets and the
deployed code. Mock fetch ONLY for `api.telegram.org` (capture the outbound
text, never send a real message), real fetch for the provider. Read secrets
from env in-process; never print tokens or chat IDs (hash the session key);
sanitize reply excerpts.

## When the protocol harness is enough — and when it must stop

Operator directive observed (CLOE_DASHBOARD_REPLY_SESSION_CONFLICT_REPAIR_V1
phase-2): "STOP OVERENGINEERED PROTOCOL REPRODUCTION. You are on the correct
repair path, but the custom WebSocket client work is no longer useful. The
final certification must now use the actual browser dashboard operated by the
user."

Rule: a custom client that drives the REAL gateway WS protocol (device
identity, Ed25519 signature, scopes, chat.send) is a legitimate PROTOCOL-LEVEL
proof — the gateway returns real runIds, real transcripts, real persistence.
It proves the deployed loader works at the transport layer. But it is NOT a
substitute for the operator clicking the real dashboard, and the marginal
value of perfecting the client drops to zero once:
- the deployed loader/hash/NODE_OPTIONS are verified byte-identical,
- the gateway restarted once and is healthy,
- one protocol-level transcript already shows exactly-once persistence.

At that point: write the operator test script (A: normal message, B: real
duplicate delivery, C: intentional repeat with new action, D: session
resume), hand it over, and inspect the resulting transcript. Do NOT keep
debugging handshake/nonce/frame/token code. Budget guidance: if the WS client
needs more than a few iterations past the connect handshake, stop and ask the
operator to run the browser test — the reverse-engineering cost exceeds the
value of an already-proven fix.

Full gateway WS protocol (device handshake, scopes, chat.send contract,
sessions.json resolution): `references/openclaw-gateway-ws-live-test-protocol.md`.
