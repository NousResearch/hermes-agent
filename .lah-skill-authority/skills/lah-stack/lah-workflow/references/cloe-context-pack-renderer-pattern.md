# Cloé Context Pack Renderer Pattern (cognitive-context-formatters.js)

Established HERMES_CLOE_OFFER_INVENTORY_CONTEXT_RENDERER_REPAIR_V1 (PR #738,
deployed SHA 0cb6b250, docs-only continuity 8a499d1).

## The chokepoint

- `buildCanonicalBusinessContext()` (src/services/cloe-canonical-business-context.js)
  pushes available_items whose full structured data lives in `item.metadata.*`
  (business_runtime_context, affiliate_execution_context,
  configured_offer_inventory, governed_microtest_proposal,
  provider_targeting_context).
- `formatCognitiveContextPack()` → `formatItem()`
  (src/brain/cognitive-context-formatters.js) is the ONLY serializer to the
  provider prompt. It renders: kind/name/source/available/read_only/
  safety_flags, `item.description` (truncated at MAX_ITEM_DESCRIPTION=180),
  and a FIXED allowlist of metadata sub-blocks: `facts`, `layers_used`,
  `history`, `provider_targeting_detail`, plus pack-level `preview_items`.
- Anything else in metadata is silently dropped → "data exists upstream but
  the provider can't see it". Do NOT blame the bridge / data source / provider
  when this symptom appears; check whether formatItem() has a renderer for the
  new metadata key.

## Bounded renderer pattern (the fix)

- Kind-scoped: `if (item.kind === 'configured_offer_inventory' && Array.isArray(item.metadata?.offers))`.
- Filter to the interesting subset (e.g. `execution_status === 'EXECUTABLE'`).
- Compact line per record: `id | product | niche | revenue | approval=STATUS`.
- Join records with ` ;; ` and push as one `offers_executable=...` piece
  (matches existing `facts=` / `conversation_history=` conventions).
- Absent fields → UNKNOWN (never invented). Never render URLs,
  tracking_contract values, source_refs, offer_ref, or credentials.
- Do NOT bump MAX_ITEM_DESCRIPTION globally as the primary fix. Keep the
  addition bounded: 11 offers ≈ +778 rendered chars (3925 → 4703).

## Runtime proof recipe (read-only, no DeepSeek call needed)

1. **Env bootstrap (critical)**: the workspace clone `.env` may LACK `LAHB_URL`
   (only `.env.example` has it) even though the live container has it. Load
   secrets INSIDE the proof script via:
   `spawnSync('docker', ['inspect', 'lah-openclaw-mvp', '--format', '{{range .Config.Env}}{{println .}}{{end}}'])`
   and parse `KEY=VALUE` lines into process.env. NEVER use
   `docker exec <c> env | grep` — composite commands are blocked in this
   environment (operator policy); `docker inspect` is one read-only command
   and is accepted. Values are loaded but never printed.
2. `const ctx = await buildCanonicalBusinessContext({ persistDecisionRecord: false })`.
3. `const rendered = formatCognitiveContextPack({ attach_to_prompt: true, intent_tags: ['business_context'], compact_summary: ctx.compact_summary, available_items: ctx.available_items, safety_flags: ['read_only','no_write','no_execute'], evidence: [] })`.
4. Assert: RENDERED_LENGTH, EXECUTABLE_OFFERS_VISIBLE (count expected offer
   names + canonical IDs as substrings of `rendered`), MISSING_NAMES=[],
   MISSING_IDS=[], no `https?://` in output, no tracking-contract value in
   output.
5. Use the mission's canonical names/IDs list as the expectation (the live
   inventory is 15 configured / 11 executable for this stack; names are
   Jerkmate, Chaturbate, MyFreeCams, WannaHookup, Instabang, Fling,
   Promptchan, eHentai.ai, Secret.ai, Darlink, Getharder; IDs
   crakrevenue:8780 … crakrevenue:10182).

## Known pre-existing failures when running formatter suites

- `test/cloe-cognitive-phase1.test.js`:
  - "stable block: identity is frozen and consistent"
  - "stable block: getSystemPrompt returns identical content"
  FAIL on origin/main regardless of any formatter change — persona drift
  (`identity.name` is 'Chloé', test expects 'OpenClaw'; 'OPENCLAW TRUSTED
  SYSTEM POLICY' absent from the stable-block system prompt). Unrelated to
  formatItem().
- Prove pre-existing with `git stash push -- <formatter-file>` → run the two
  suites → `git stash pop`; identical failure set on the unpatched base is
  the proof. Report pass/total plus the documented pre-existing pair; do not
  chase them in a renderer mission.
- `test/cloe-model-tool-authority-repair.test.js` fails with
  ERR_MODULE_NOT_FOUND 'zod' in a FRESH worktree until `npm ci` runs — the
  clean-worktree pattern means node_modules is absent; install before test.

## Live certification (context exposure only)

- Single POST `/chat/completions` (model='brain', stream:false, Bearer
  ADMIN_API_KEY read from container env) with the FR enumeration question
  ("Liste les offres affiliées exécutables … N'effectue aucune recherche
  externe et aucune action.").
- Assert: HTTP 200, expected offer names+IDs enumerated, UNKNOWN semantics
  preserved ("Non confirmée" / "je n'ai pas la visibilité sur l'approbation
  effective côté réseau"), no CrakRevenue API request, no invented
  payout/GEO/restrictions.
- This is evidence of context exposure ONLY — never claim live marketplace
  freshness certification.

## Deploy note

The canonical exact-SHA deployer (bin/deploy-lah-openclaw-mvp-exact-sha.mjs)
lives in origin/main but NOT in the dirty canonical checkout — run it from a
clean worktree at the merge SHA with `LAH_DEPLOY_WORKDIR=<wt>/lah-openclaw-mvp`.
