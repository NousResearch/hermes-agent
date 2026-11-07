# Prompt patterns, decision tables & worked examples

## The variant decision table (verify terms quarterly)
| Situation | Use | License note |
|---|---|---|
| Just need commercial images, no infra | Hosted API ([pro]/[flex]) | License + provenance handled |
| Web-grounded/current content in the image | [max] via API | API terms |
| Free + commercial + local | [klein] 4B / FLUX.1 [schnell] | Apache 2.0 |
| Research/tinkering/fine-tune experiments | [dev] locally | Non-commercial for services; outputs OK |
| Client work on self-hosted weights | Paid BFL tier | Agency tier: 3 clients incl., then per-client |

## The scene-prompt pattern (not tag soup)
"[Subject with concrete detail], [action], in [setting]. [Lighting + time]. [Mood]. Shot on [camera/lens or
style]. Background: [detail]. Brand color #HEX on [element]. Sign text reads: 'EXACT TEXT'."
Long is fine (32k context) — specific beats clever. One change per iteration.

## The consistency workflow (campaign sets)
1. Generate/choose the canonical character + product refs (consented/original).
2. Multi-reference every campaign image with the same refs; vary only the scene prompt.
3. Fix near-misses with in-context edits, not re-rolls.
4. Side-by-side drift review before the set ships.

## The edit-instruction pattern
One targeted change per instruction: "Change [element] to [new state]; keep everything else identical." Chain
edits; re-check global consistency every ~3 edits. Text edits: "Replace the sign text with 'X'" + character-level
verify.

## Worked example A — "Blunt indie founder" (license catch + hex)
"almost self-hosted flux.2 dev for client thumbnails. read the license first: outputs fine, running it as a
client service — that's a paid tier, and they can spot-check that you're filtering. switched to the api. also
stopped typing 'our green' — it's #2E5E4E now, and it finally IS our green." — the outputs-vs-service line, hex
precision.

## Worked example B — "Warm bookkeeping studio" (consistency + edit)
"We made a little illustrated bookkeeper character — our own creation, no real face borrowed — and now she
appears in every seasonal post from the same three reference images. When the spring image was perfect except
the calendar said March twice, we didn't re-roll — one edit instruction fixed the calendar and kept everything
else. One small step: save your reference set once, and every future image starts consistent." — refs-first,
edit-not-reroll, original character.

## Never
Ship the first roll unreviewed · self-host [dev] for client/commercial services without the paid tier · skip
the filter/manual-review obligation on [dev] deployments · use a real person's likeness (real or AI lookalike)
without permission · clone a competitor's ad/trade dress · strip provenance metadata · trust rendered text
without character-level verification · invent stats for an "infographic" (route data viz to
infographic-and-data-viz) · claim WoopSocial generates images.
