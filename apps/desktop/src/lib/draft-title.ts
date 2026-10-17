import { SLASH_COMMAND_RE } from '@hermes/shared'

/** Matches `agent/title_generator.py`'s MAX_DERIVED_TITLE_CHARS, so a draft
 *  doesn't visibly reflow the moment the backend's derived title replaces it. */
const MAX_DRAFT_TITLE_CHARS = 48

// Mirrors `agent/title_generator.py`'s fence + inline-Markdown rules so the
// draft label and the backend's derived title agree on the same text.
const FENCE_LINE_RE = /^\s*(?:`{3,}|~{3,})\s*[\w+.-]*\s*$/
const MD_IMAGE_RE = /!\[([^\]]*)\]\([^)]*\)/g
const MD_LINK_RE = /\[([^\]]+)\]\([^)]*\)/g
const MD_CODE_RE = /`+([^`]*)`+/g
// A run of the same delimiter wrapping a word; a delimiter beside whitespace
// (`a * b`) is not emphasis, and an underscore inside a word (`snake_case`)
// never is (CommonMark intraword rule).
const MD_STAR_EMPHASIS_RE = /(\*{1,3}|~~)(?![\s*~])(.+?)(?<![\s*~])\1/g
const MD_UNDERSCORE_EMPHASIS_RE = /(?<!\w)(_{1,3})(?![\s_])(.+?)(?<![\s_])\1(?!\w)/g
const MD_LINE_PREFIX_RE = /^(?:#{1,6}\s+|>\s*|(?:[-*+]|\d{1,3}[.)])\s+(?:\[[ xX]\]\s+)?)+/

/**
 * Plain text of one Markdown line: image → its alt text, link → its label,
 * code → its content, emphasis unwrapped, leading heading/quote/list markers
 * dropped. The client-side twin of the backend's `strip_inline_markdown`.
 */
export function stripInlineMarkdown(line: string): string {
  let text = line
    .replace(MD_LINE_PREFIX_RE, '')
    .replace(MD_IMAGE_RE, '$1')
    .replace(MD_LINK_RE, '$1')
    .replace(MD_CODE_RE, '$1')

  let previous: null | string = null

  while (previous !== text) {
    // nested emphasis (***bold italic***, **_both_**) unwraps one layer per pass
    previous = text
    text = text.replace(MD_STAR_EMPHASIS_RE, '$2').replace(MD_UNDERSCORE_EMPHASIS_RE, '$2')
  }

  return text.split(/\s+/).filter(Boolean).join(' ')
}

/**
 * Name a draft after what the user has typed into it.
 *
 * The client-side twin of the backend's `derive_title`: first line carrying
 * real prose (never a code-fence delimiter or a line that is only markup),
 * Markdown syntax removed, whitespace collapsed, cut on a word boundary. It
 * runs before any session exists, so it can't reach the real titler — a draft
 * has no persisted row and no opening message yet, which is exactly what
 * `apply_instant_title` needs.
 *
 * Empty when there's nothing worth naming, so the caller keeps "New session"
 * rather than showing a title that says less than the placeholder.
 */
export function deriveDraftTitle(text: string): string {
  let body = ''

  for (const candidate of text.split('\n')) {
    const line = candidate.trim()

    if (!line || FENCE_LINE_RE.test(line)) {
      continue
    }

    // A bare `/skin` names the draft after the command rather than the work, the
    // failure the backend titler summarizes away. Title from the argument instead;
    // with no argument there is no intent yet, so the placeholder stands.
    if (SLASH_COMMAND_RE.test(line)) {
      body = stripInlineMarkdown(line.replace(/^\/\S+\s*/, ''))

      break
    }

    body = stripInlineMarkdown(line)

    if (body) {
      break
    }
  }

  if (!body) {
    return ''
  }

  if (body.length <= MAX_DRAFT_TITLE_CHARS) {
    return body
  }

  // Cut on a word boundary, unless that would throw away more than half of it.
  const cut = body.slice(0, MAX_DRAFT_TITLE_CHARS)
  const space = cut.lastIndexOf(' ')
  const kept = space > MAX_DRAFT_TITLE_CHARS / 2 ? cut.slice(0, space) : cut

  return `${kept.replace(/[\s,.;:—-]+$/, '')}…`
}
