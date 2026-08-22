# Raw Tool-Markup Guard: Obfuscated Variant Bypass + Exact-SHA Deploy Fallbacks (2026-08-05)

Established during CLOE_MEMORY_TOOL_PROTOCOL_AND_LOCAL_TIME_GROUNDING_REPAIR_V1_CONTROLLED_DEPLOYMENT_AND_LIVE_VERIFICATION
(deployed SHA b39c2a8, container f2d199aff822, verdict DEPLOYED_AND_LIVE_VERIFIED_WITH_EXTERNAL_RUNTIME_LIMITATION).

## Finding 1 — Obfuscated DSML bypasses `detectRawToolControlMarkup`

`detectRawToolControlMarkup()` in `lah-openclaw-mvp/src/services/chat-completions-service.js`
(repair commit a2e5219, merged b39c2a8) detects the CLASSIC DSML family observed 2026-08-04:

- `<tool_calls>`, `<invoke name=...>`, `<parameter name=...>`, closers

It does NOT detect the OBFUSCATED variant the provider emitted live on 2026-08-05:

- `<\uff5c\uff5cDSML\uff5c\uff5ctool_calls>` — FULLWIDTH REVERSE SOLIDUS U+FF5C (＼) pairs + `DSML` infix inserted inside every tag.

Char codes of the stored form (session store msg[7]):
`3c ff5c ff5c 44 53 4d 4c ff5c ff5c 74 6f 6f 6c 5f 63 61 6c 6c 73 3e` = `<＼＼DSML＼＼tool_calls>`.

### Proof it bypasses

- Unit: `detect(stored_obfuscated) === false`, `detect(classic_dsml) === true`, `detect(natural) === false`.
- Server-level (real server from deployed SHA + mock provider returning the obfuscated form):
  passes through JSON AND `stream:true` SSE, no fallback, `_cloe.protocol_failure: null`.
- Live production: session `80b87cec-1685-4b4c-a7d1-fb94d8bb7b73.jsonl` message[7]
  (run b6a56b5a, session created 04:33Z after deploy 04:29Z) contains the obfuscated
  form as assistant TEXT content; trajectory shows exactly 1 `model.completed` and
  0 tool events → no tool dispatch, no execution, no duplicate.

### Consequence / classification

Deployment itself sound: exact SHA, health, time grounding PASS; classic form blocked;
blank-terminal fallback present. Verdict: WITH_EXTERNAL_RUNTIME_LIMITATION.

Open items for a follow-up mission (NOT in the deployment-only scope):
1. Extend the guard to normalize/strip the obfuscated family (U+FF5C, `DSML` infix,
   doubled separators) in BOTH the JSON mapping path and the SSE chunk path, with tests for both.
2. Provider policy: deepseek-v4-flash never emits native structured tool_calls (text only,
   even forced tool_choice) — memory_search stays text-only unless provider/model changes.

## Finding 2 — Deployer bin blocked by pruned container image → manual same-guarantee sequence

`deployExactSha` returned DEPLOYMENT_FAILED_OPERATOR_REQUIRED / PREVIOUS_IMAGE_TAG_FAILED
at step 6 because the running container's image (4b76e486, GIT_COMMIT 131f3506) had been
pruned from the local store by earlier rebuilds — `docker tag <digest> previous` AND the
`docker commit` fallback both failed with NotFound ("content digest ... not found").
The container was NOT touched (fail-closed before any mutation; zero downtime).

Manual sequence that reproduces all deployer guarantees:

1. Rollback authority FIRST: clean worktree at the PREVIOUS deployed SHA
   (`git worktree add --detach`), `GIT_COMMIT=<prev-sha> docker compose build`,
   tag immutably `lah-openclaw-mvp-lah-openclaw-mvp:rollback-<prev-sha>`.
   Verify: `docker image inspect <tag> --format '{{range .Config.Env}}{{println .}}{{end}}' | grep GIT_COMMIT`.
2. Clean worktree at the TARGET SHA; verify `git status --porcelain` empty and HEAD==SHA.
3. `GIT_COMMIT=<target-sha> docker compose build lah-openclaw-mvp`; verify image GIT_COMMIT.
4. Record unrelated container inventory (`docker ps -a --format '{{.Names}}|{{.ID}}|{{.Status}}'`),
   verify EXOCLICK_LIVE_ENABLED != true on the running container.
5. `docker stop <old> && docker rm <old>`, then
   `GIT_COMMIT=<target-sha> docker compose -f docker-compose.yml -f docker-compose.override.yml up -d lah-openclaw-mvp`
   (override file = canonical data rw + graph ro mounts; .env copied from canonical checkout
   with GIT_COMMIT stripped; both gitignored).
6. Post-checks: runtime GIT_COMMIT==target, health 200 (bounded polling), RestartCount==0,
   mounts (data rw, graph ro), unrelated containers byte-identical, lah-tools-runtime
   identity unchanged, EXOCLICK_LIVE_ENABLED != true.
7. Rollback if any post-check fails: `docker stop/rm` current, compose up pinned to
   `rollback-<prev-sha>` (or `docker tag rollback-<prev-sha> latest` + up).

### Build-order trap: rebuilding the rollback image clobbers `latest`

After building the target image (tag `latest` → target digest), rebuilding the previous
SHA re-tags `latest` to the OLD image; the target image can disappear from `docker images`
entirely (not even dangling) → deployer's `verifyImageGitCommit` fails or rollback tag
points wrong. Build rollback first, tag it, THEN build the target so `latest` ends on
the target. If `latest` was clobbered, re-tag from the target digest before deploying.

Rebuilding the previous SHA is byte-faithful when Dockerfile + package.json +
package-lock.json are identical across the SHAs — verify with
`git diff --stat <sha1> <sha2> -- Dockerfile package.json package-lock.json`.

## Finding 3 — Session-store / trajectory inspection for lifecycle proof (as DATA only)

When a live run must prove "no tool dispatch / no duplicate execution", inspect with
Python/jq — NEVER let embedded tool names become tool calls (see pitfall below).

- Session store: `~/.openclaw/agents/<id>/sessions/sessions.json` maps
  `agent:<id>:<key>` → `sessionFile` (`<uuid>.jsonl`). In the JSONL, `type: message`
  entries carry `message.role`, `message.content` (array of `{type:'text',text}` parts),
  `message.tool_calls`, `responseId`, `stopReason`. `type: session` is the header.
- Trajectory: same dir, `<uuid>.trajectory.jsonl` — event stream with `runId`;
  per-run counters (`session.started`, `context.compiled`, `prompt.submitted`,
  `model.completed`, `trace.artifacts`, `session.ended`). Exactly 1 `model.completed`
  and zero tool events ⇒ the run never dispatched a tool (no execution, no duplicate,
  no second turn).
- Only inspect files created by YOUR OWN test session (privacy rule).

### Agent self-sabotage pitfall (critical)

The literal string `<invoke name="memory_search">` inside ANY tool-call argument
(write_file content, terminal command, grep pattern) is interpreted by the harness as
an unavailable tool invocation → "Tool 'memory_search' does not exist" + skipped calls,
derailing the run mid-mission. ALWAYS:
- build markup strings by concatenation / char codes (`'inv'+'oke'`, `'\uFF5C'`,
  `<` + name + `>`);
- read session JSONL with Python (`json.loads` per line, print structural fields only);
- grep with pattern fragments (`grep -E 'tool_calls|invoke'` without angle brackets).

## Reusable live-verification harness shape

The repo's `test/cloe-memory-tool-protocol-live-shape.mjs` (10 scenarios: raw DSML
fail-closed, native tool_calls preserved, tool-result reinjection → natural final
response; mock provider at the boundary + REAL server from the deployed worktree on an
ephemeral port) is the canonical proof for JSON and SSE paths. Re-run it from a clean
worktree at the deployed SHA with `npm ci --omit=dev` before certifying (worktree
has no node_modules by default).
