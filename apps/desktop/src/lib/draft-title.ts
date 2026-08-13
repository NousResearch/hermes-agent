import { SLASH_COMMAND_RE } from './chat-runtime'

/** Matches `agent/title_generator.py`'s MAX_DERIVED_TITLE_CHARS, so a draft
 *  doesn't visibly reflow the moment the backend's derived title replaces it. */
const MAX_DRAFT_TITLE_CHARS = 48

/**
 * Name a draft after what the user has typed into it.
 *
 * The client-side twin of the backend's `derive_title`: first meaningful line,
 * whitespace collapsed, cut on a word boundary. It runs before any session
 * exists, so it can't reach the real titler — a draft has no persisted row and
 * no opening message yet, which is exactly what `apply_instant_title` needs.
 *
 * Empty when there's nothing worth naming, so the caller keeps "New session"
 * rather than showing a title that says less than the placeholder.
 */
export function deriveDraftTitle(text: string): string {
  const line =
    text
      .split('\n')
      .find(candidate => candidate.trim())
      ?.trim() ?? ''

  if (!line) {
    return ''
  }

  // A bare `/skin` names the draft after the command rather than the work, the
  // failure the backend titler summarizes away. Title from the argument instead;
  // with no argument there is no intent yet, so the placeholder stands.
  const body = (SLASH_COMMAND_RE.test(line) ? line.replace(/^\/\S+\s*/, '') : line).split(/\s+/).join(' ')

  if (body.length <= MAX_DRAFT_TITLE_CHARS) {
    return body
  }

  // Cut on a word boundary, unless that would throw away more than half of it.
  const cut = body.slice(0, MAX_DRAFT_TITLE_CHARS)
  const space = cut.lastIndexOf(' ')
  const kept = space > MAX_DRAFT_TITLE_CHARS / 2 ? cut.slice(0, space) : cut

  return `${kept.replace(/[\s,.;:—-]+$/, '')}…`
}
