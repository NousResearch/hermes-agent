/**
 * Reasoning-summary models (OpenAI's gpt-5.x family, and anything relaying the
 * Responses API onto the OpenAI chat wire) emit one delta per *completed*
 * summary part, each opening with a bold markdown heading:
 *
 *   **Investigating likely culprit PRs**
 *   **Inspecting message schema**
 *
 * The Responses API delimits those parts with `summary_index`; the chat wire
 * carries no such field, so concatenated deltas glue into
 * `...PRs****Inspecting...` — a `****` run markdown reads as neither a bold
 * close nor a bold open, leaving one unbroken, unspaced, half-bold paragraph.
 * The AI SDK hit the same bug (vercel/ai#6742).
 *
 * The backend now inserts the break as the deltas arrive. This repairs the text
 * we display: reasoning persisted before that fix, and any provider still
 * gluing its parts. Idempotent — a break already present is left alone.
 */

// A heading butting straight onto the previous part, in the two shapes the
// wire produces:
//   1. heading-onto-heading — `**One****Two**`, a bare `****` run.
//   2. prose-onto-heading   — `interaction!**Two**`.
// Emphasis that legitimately follows whitespace is left alone, and a heading
// must close on its own line to count as a summary part.
const GLUED_HEADING_RUN = /(?<!\*)\*{4}(?!\*)/g
const GLUED_AFTER_PROSE = /(?<=[^\s*])(\*\*(?=[^\s*])[^\n]*?\*\*)/g

export function separateGluedReasoningBlocks(text: string): string {
  return text.replace(GLUED_HEADING_RUN, '**\n\n**').replace(GLUED_AFTER_PROSE, '\n\n$1')
}
