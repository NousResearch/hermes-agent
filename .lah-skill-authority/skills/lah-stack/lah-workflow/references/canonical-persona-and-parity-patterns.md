# Canonical Persona & Channel-Parity Patterns (CLOE missions 2026-08-01)

Session-derived patterns from CLOE_SOUL_RESTORE / CLOE_PERSONA_FIDELITY /
CLOE_TELEGRAM_GATEWAY_BEHAVIOR_PARITY missions. Class-level lessons for any
mission touching the CLOE runtime persona, Telegram/Gateway parity, or
live-runtime comparison harnesses.

## 1. Canonical persona files can SUPERSEDE a code fix

- The operator corrected SOUL.md / IDENTITY.md / USER.md directly in the agent
  workspace (`~/.openclaw/agents/<agent>/workspace/`). The production runtime
  (302ed75) then restored Chloé's personality WITHOUT any code merge.
- The code candidate (PR #651) was closed as
  `SUPERSEDED_BY_CANONICAL_PERSONA_FILES` — a legitimate mission outcome.
- **Before merging a persona code fix, test the canonical files alone** on a
  fresh production session (new session key, real provider). If the files
  restore the behavior, the code fix adds risk without benefit.
- Human gate: the operator judges personality ("Est-ce que je reconnais
  Chloé ?"), not the test suite. Present full untruncated old-vs-candidate
  responses with per-dimension notes (prénom, tutoiement, humour, énergie,
  naturel, continuité, initiative). Never auto-verdict. Wait for the exact
  operator phrase before any deployment.

## 2. Terminal secret masking breaks `$(cat secretfile)` in bash

Symptom: `syntax error near unexpected token ')'` when the terminal masks a
secret; or node subprocesses see an empty env (source does not propagate
secrets). Fixes that work:
- Pass the key as an explicit argv: `node script.mjs "$KEY"` (argv works).
- Read the key in-process from `.env` via a `readDotEnv(name)` helper —
  values used in-process, never printed.
- Do NOT rely on `--env-file=.env` for a comparison harness if the workspace
  `.env` defines a DIFFERENT ADMIN key than the container — it silently
  swaps the auth and live calls return empty bodies.

## 3. Comparison harness pitfalls (old runtime vs candidate)

- `tools` + `tool_choice:'auto'` on every prompt → real provider calls the
  tool → `message.content` null → `""` recorded. Fix: social prompts get NO
  tools; factual prompts get `tool_choice:'none'`.
- Reproduce the REAL client shape: the gateway sends the stable-block persona
  as the FIRST system message. Omitting it makes the candidate answer with
  "vous" and generic phrasing — a harness artifact, not a code regression.
- Read the live deployed GIT_COMMIT from the container env (not a stale
  snapshot) and override the pilot corpus so "which commit is active" answers
  match the actual runtime.

## 4. Worktree residual dirt from another PR

A mission worktree created from a base that predates a merged PR (e.g. 0458509
vs PR#650) can carry dirty files whose imports don't resolve on the base
(`ERR_MODULE_NOT_FOUND` in grouped `node --test` runs). Classify
pre-existing-vs-regression:
```bash
git stash push -m "residual" && node --test ... ; git stash pop
```
If the suite passes with the stash applied, the failure is residual worktree
state, not the mission commits. Resolve stash-pop conflicts with
`git checkout --ours <path>` when merged main is authoritative, then
`git stash drop`.

## 5. `git merge-tree` false positive on "CONFLICT"

`git merge-tree $(git merge-base A B) A B | grep CONFLICT` matches the SQL
keyword in `ON CONFLICT(stable_id) DO UPDATE`. Grep for `^<<<<<<<` /
`^=======` / `^>>>>>>>` or `changed in both` instead.

## 6. CLOE runtime topology (for live tests)

- Agent workspace: `~/.openclaw/agents/cloe-poc/workspace/` (SOUL/IDENTITY/USER).
- Production brain: container `lah-openclaw-mvp` → POST
  `http://127.0.0.1:4000/chat/completions`, model `cloe/brain`.
- OpenClaw gateway: port 18789. Fresh session key format
  `agent:<agentId>:<key>` targets a specific agent.
- Capturing CLI replies: `openclaw terminal --local --session
  'agent:cloe-poc:<key>' --message '...'` runs non-interactively and forces
  exit before persisting — wrap in a pty (`script -qec`) and poll the capture
  file for reply markers to measure latency and extract the real answer.

## 7. Telegram/Gateway parity architecture (TELEGRAM_PARITY)

- Single canonical conversation service + contract; Gateway and Telegram both
  route through it (flags `CLOE_TELEGRAM_CANONICAL_PIPELINE_ENABLED`,
  `CLOE_TELEGRAM_CANONICAL_COMMANDS_ENABLED` — OFF keeps legacy pipeline).
- Shared: persona authority (`src/brain/cloe-persona.js`), error authority
  (`cloe-error-authority.js`, 7 codes with cognitive/render/delivery classes),
  brain router (provider/model parity), session store singleton (cross-channel
  memory), semantic renderer (lossless chunking vs old `slice(0,3500)`).
- Pattern to reuse for any channel-parity mission: contract → service →
  shared routers → parity suite → observability → shadow compare → cleanup.

## 8. JS regex `\b` fails after accented chars (intent-detection trap)

Symptom: a social/greeting pattern like `/^(t'?es toujours là)\b/i` NEVER
matches `"t'es toujours là ?"` even though the same pattern without `\b`
matches. Root cause: `à` (U+00E0) is NOT a `\w` word character in JS regex
(only ASCII `[A-Za-z0-9_]` are), so there is no `\b` boundary between `à` and
the following space.

Fix: end accent-final patterns with `(\b|\s|$)` instead of a bare `\b`:

```js
/^(t'?es toujours là|tu es toujours là|toujours en vie)(\b|\s|$)/i
```

Verify both forms in the REPL before committing (`re1` no-boundary vs `re2`
with-boundary); the failure is silent — the intent just never classifies as
social, so retrieval/tools fire on casual smalltalk.

## 9. `gh pr edit` can fail silently → verify body via REST API

Symptom: `gh pr edit <PR> --body-file x.md` exits 0 and prints only a benign
GraphQL warning (`Projects (classic) is being deprecated` /
`repository.pullRequest.projectCards`), but the PR body is NOT updated —
`gh pr view --json body` still returns the OLD text.

Fix / verification:
- After any `gh pr edit`, verify via the REST API, not `gh pr view`:
  `gh api repos/<owner>/<repo>/pulls/<N> --jq '.body'` and grep for a unique
  new string (e.g. the merge SHA or "126/126").
- If the body is stale, PATCH directly:
  `gh api -X PATCH repos/<owner>/<repo>/pulls/<N> -f body=@/tmp/body.md`
  (works even when `gh pr edit`'s GraphQL path is blocked by the projectCards
  deprecation query).
- The GraphQL `projectCards` warning is a deprecation notice, NOT an
  authorization error — do not treat it as "permission denied".

## 10. Post-merge "no deployment" verification (operator gate)

After an admin-bypass merge, the operator needs PROOF production is untouched.
Capture all four BEFORE declaring success:
- `docker inspect <container> --format '{{.Image}}'` — image SHA must be the
  rollback image, not a new build;
- `GIT_COMMIT` from container env (`docker inspect ... | grep ^GIT_COMMIT=`)
  — must equal the frozen production SHA, not the merged branch SHA;
- `RestartCount` — must stay 0 (a merge must not restart the container);
- `docker images | grep <name>` — no NEW image created by the merge.
Also note `docker ps` uptime ("Up N hours") as corroborating evidence. When
the operator defers deployment to a separate mission, record it explicitly in
the continuity JSON (`production.unchanged: true`, `deployment_deferred`).
