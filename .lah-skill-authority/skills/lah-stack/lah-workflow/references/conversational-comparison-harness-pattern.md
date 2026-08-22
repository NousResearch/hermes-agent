# Conversational Comparison Harness Pattern (persona / soul-restore / fidelity missions)

Established during CLOE_PERSONA_FIDELITY_AND_GROUNDED_SYNTHESIS_REPAIR_V1 and
CLOE_CONVERSATIONAL_SOUL_RESTORE_WITH_GROUNDED_FAST_RETRIEVAL_V1.

## When to use

Any mission where the operator must judge Chloé's (or any conversational
assistant's) restored personality side-by-side against the old runtime:
- persona fidelity restoration
- grounded-synthesis correction
- "does this still sound like my assistant" gates

## Harness shape (proven)

- OLD side = live production HTTP (`/chat/completions` on the deployed
  container) with the container's `ADMIN_API_KEY`.
- CANDIDATE side = local `buildNativeChatCompletions` (the REAL entry point)
  with the real provider (same DeepSeek model as production), an in-memory
  knowledge stack seeded like production (`seedPilotCorpus` + registry + hot
  cache + indexer).
- The candidate MUST receive the stable-block persona as the first system
  message — the gateway client in production always prepends
  `getStableBlock().systemPrompt`. Omitting it makes non-social paths answer
  in generic support-speak ("vous", "Comment puis-je vous aider").
- Secret handling: read `.env` in-process (`readDotEnv(name)` reading
  `join(process.cwd(), '.env')`) — do NOT rely on `source file && node` or
  `--env-file` propagating into the Hermes terminal shell; values are used
  in-process only, never printed. Passing the key as argv[2] also works but
  risks /proc exposure; prefer the env read.

## Pitfalls (each cost repair iterations)

1. **`tool_choice:'auto'` + tools on every prompt → empty candidate answers.**
   The real LLM calls the tool instead of writing text → `message.content`
   null → `candidate:""` in the artifact (P08/P10/P14/P20). HARNESS defect,
   not a runtime regression. Fix: social prompts (greeting/casual/simple) get
   NO tools; factual prompts get `tool_choice:'none'`. The comparison
   measures the conversational answer, not tool execution.

2. **Persona system prompt missing → "vous/votre" leaks.** Harness sent only
   `{role:'user'}`. Fix: prepend the stable-block persona. Verify with a
   scan: zero "vous|votre" across all responses before presenting.

3. **Stale static seed as runtime truth.** `pilot-corpus` carries a static
   `DEPLOYMENT_COMPLETED` (3ac856d from 2026-07-30); the candidate then
   answers "the server runs 3ac856d" although 302ed75 is live. Fix: after
   seeding, override with the LIVE commit
   (`CLOE_LIVE_GIT_COMMIT` env or `docker inspect <container>` GIT_COMMIT)
   via `indexer.applyEvent('DEPLOYMENT_COMPLETED', { commit: liveCommit, ... })`.

4. **JS `\b` fails after accented characters.** `\b` between an accented
   char (`à`, `é` — non-word in JS regex) and a following space does NOT
   exist. `/(t'es toujours là)\b/i` never matches "t'es toujours là ?".
   Fix: `(\b|\s|$)` instead of trailing `\b`. Test with the module's own
   predicate (`isSocialQuery`) — inline `node -e` regex tests can disagree.

5. **French apostrophes double-escape in written test files.**
   `'on fait quoi aujourd\'hui ?'` written via write_file can land as a
   literal `\\'` → SyntaxError. Fix: double-quoted strings for French
   prompts with apostrophes; `node --check` after writing; NEVER blanket
   `str.replace("\\'", "'")` — it un-escapes legitimate escapes and breaks
   the file (rewrite the file cleanly instead).

6. **Social fast path must classify more than greetings.** Extend
   `isSocialQuery` beyond "salut": ça va / tu vas bien / merci / bonne nuit /
   on continue / j'en ai marre / j'ai une idée / t'es toujours là /
   qu'est-ce que t'en penses / on fait quoi / on reprend. Rule: short
   message (< ~80 chars), no project-fact keywords (campaign, commit, deploy,
   postback, offer, cap, zone, stats), else it is NOT pure social and may
   need retrieval.

## Human gate protocol (operator-mandated)

- Present FULL untruncated responses (600-char slices are REJECTED).
- Per prompt: exact prompt, OLD response, CANDIDATE response, latency, LLM
  calls, retrieval calls, tool calls.
- Include a comment grid: prénom / tutoiement / humour / énergie / naturel /
  continuité / initiative.
- The operator's explicit phrase is the gate (e.g. "Chloé a retrouvé sa
  personnalité et sa qualité d'origine." / "Est-ce que je reconnais Chloé ?").
- Merging (admin bypass when GHA dead) is SEPARATE from deploying; BOTH wait
  on the human verdict. Never present an automated score as the verdict.

## Correctness markers that satisfied the operator

- 0 empty responses, 0 "vous/votre" across all prompts.
- Social prompts: zero retrieval, zero tool, one LLM call, persona directive.
- Persona style reproduced, not templated (reference style may appear in
  phrasing, not as a verbatim copy every time).
- Factuality: no invented caps/status; deployment commit coherent with the
  live runtime; entity separation (OurDream never contaminated by C99).
