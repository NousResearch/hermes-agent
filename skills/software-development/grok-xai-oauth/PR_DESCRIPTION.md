# Add `grok-xai-oauth` skill for the official xAI Grok integration

## Summary
New bundled skill under `software-development/` that helps users get the most out of the recent xAI + Hermes OAuth integration (announced https://x.ai/news/grok-hermes).

Users with SuperGrok or X Premium+ can now drive Hermes with Grok 4.3 + native X search (with citations) + Imagine images/video + TTS using a simple browser OAuth — no `XAI_API_KEY` to manage.

## Why this skill
- The integration is brand new (May 2026). A high-quality, discoverable guide + recipes accelerates adoption for both Hermes users *and* xAI.
- Plays directly to Grok's current strengths (X search, strong reasoning, visual generation) while teaching good Hermes patterns (subagents, mixing providers, memory, cron).
- Includes a tiny helper script for status checks.
- Cross-references existing strong skills (`subagent-driven-development`, `hermes-agent-skill-authoring`, `writing-plans`).
- Also documents synergy with xAI's Grok Build CLI (the environment this skill was authored in).

## Changes
- `skills/software-development/grok-xai-oauth/SKILL.md` (full practical guide + workflows + pitfalls + verification + one-shot recipes)
- `skills/software-development/grok-xai-oauth/scripts/check-grok-oauth.sh` (small status helper, uses `${HERMES_SKILL_DIR}` substitution)
- Minor: added the new skill to `related_skills` in `hermes-agent-skill-authoring`
- Expanded Best-of-N section with concrete subagent dispatch example directly ported from Grok Build's best-of-n skill patterns.
- As a demonstration of the documented "Grok Build Synergy", also improved the author's grok-concierge (lib/hermes.ts) to better detect the xAI OAuth + new skill (proper fs imports, grokOAuthActive flag).

## Testing done (by author)
- Official Hermes validators (_validate_frontmatter, _validate_content_size) pass cleanly.
- Skill appears in `hermes skills list` (as local + enabled).
- Helper script runs and correctly surfaces auth state + Hermes version.
- Follows structure and tone of peer skills in `software-development/` (subagent-driven-development etc.).
- Authored, reviewed, and polished inside xAI Grok Build using its subagents, review loops, and skills system. Iterated on branch in the Hermes source clone.
- Also committed a small related improvement in the user's grok-concierge bridge project.

## How to test the skill
1. `hermes model` → select the xAI Grok OAuth provider (log in via browser if prompted).
2. Load / mention the skill naturally or via skills commands.
3. Try the one-shot recipes in the SKILL.md.
4. Run the helper: `bash ${HERMES_SKILL_DIR}/scripts/check-grok-oauth.sh`

## Contribution notes
- Placed in `software-development/` (appropriate for inference provider usage + agentic workflows).
- macOS + Linux focused in platforms for now (author's primary; easy to expand).
- Intentionally actionable and example-heavy so new users of the integration succeed immediately.
- Author is happy to iterate on review feedback.

This is a small but high-signal way to help both the Hermes project and the xAI Grok integration.

---

Authored with heavy use of Grok Build (the very tool from xAI) + the user's existing Hermes + grok-concierge setup. Feedback sent via Grok Build `/feedback` as well.
