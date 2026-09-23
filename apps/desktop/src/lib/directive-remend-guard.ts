/**
 * DIRECTIVE REMEND GUARD — keeps a finished transcript directive out of the
 * incomplete-markdown repair pass.
 *
 * `tailBoundedRemend` closes markdown constructs the model is still typing:
 * a dangling `*`, `_`, `` ` ``, `~~` or `[` in the LAST block gets a synthetic
 * closer appended so a half-streamed sentence doesn't flicker as raw syntax.
 *
 * A transcript directive is the pathological input for that repair. It is a
 * whole paragraph of `::name{key="value"}` whose VALUES are natural language:
 * a prompt like `Dọn các worktree wt-* đã merge` carries one unpaired `*`, so
 * the repair appends `*` AFTER the closing brace:
 *
 *     ::followup{p1="Dọn các worktree wt-* đã merge"}*
 *
 * Core's parser (`parseTranscriptDirective`) requires the directive to be the
 * ENTIRE paragraph — the trailing `*` breaks the match, the plugin never
 * claims the paragraph, and the panel silently degrades to raw text. The same
 * happens for a lone `_`, a single backtick, `~~` and `[`; each is ordinary
 * prose inside a quoted prompt.
 *
 * The repair is only correct for text the model is mid-way through. A
 * directive that already ends in `}` is COMPLETE, so this trims the window
 * back to the text before it: everything above still gets repaired, the
 * directive is passed through byte-for-byte, and a directive that is still
 * streaming (no closing brace yet) is left to the normal repair path.
 */

/** A complete directive paragraph: `::name` or `::name{...}` and nothing else
 *  on the line. Mirrors DIRECTIVE_RE in lib/transcript-directives.ts — kept
 *  deliberately narrow so ordinary prose starting with `::` is untouched. */
const COMPLETE_DIRECTIVE_LINE_RE = /^::[a-z][a-z0-9-]{0,63}(?:\{[^{}]{0,1024}\})?$/

/** Split point: where the trailing complete-directive paragraph begins, or
 *  -1 when the text does not end in one. Trailing whitespace is allowed (a
 *  streamed message often ends with a newline). */
export function completeDirectiveTailStart(text: string): number {
  const trimmedEnd = text.replace(/\s+$/, '')

  if (!trimmedEnd.endsWith('}') && !/::[a-z][a-z0-9-]*$/.test(trimmedEnd)) {
    return -1
  }

  const lineStart = trimmedEnd.lastIndexOf('\n') + 1
  const line = trimmedEnd.slice(lineStart)

  if (!COMPLETE_DIRECTIVE_LINE_RE.test(line)) {
    return -1
  }

  return lineStart
}

/**
 * Run `repair` on everything EXCEPT a trailing complete directive paragraph.
 * With no such paragraph the text is repaired exactly as before, so this is a
 * no-op for every message that carries no directive.
 */
export function remendPreservingTrailingDirective(text: string, repair: (input: string) => string): string {
  const start = completeDirectiveTailStart(text)

  if (start < 0) {
    return repair(text)
  }

  const head = text.slice(0, start)
  const tail = text.slice(start)

  // A directive alone in the message: nothing left to repair.
  if (!head) {
    return tail
  }

  // Repair the head WITHOUT the blank line that separates it from the
  // directive, then put the separator back. Handing the separator to the
  // repair would park the synthetic closer after it (`text\n\n**`), which
  // renders as a stray literal on its own line instead of closing the span
  // it belongs to.
  const separator = /\s*$/.exec(head)?.[0] ?? ''
  const body = head.slice(0, head.length - separator.length)

  return repair(body) + separator + tail
}
