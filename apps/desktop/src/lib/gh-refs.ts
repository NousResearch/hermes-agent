/**
 * Pure helpers for bare GitHub issue/PR references (`#123`) in assistant prose.
 *
 * Mirrors `session-refs.ts`: a sync linkifier for the hot preprocess path, kept
 * free of React/store/API imports. Whether a ref CAN resolve is decided later —
 * the rewrite emits `#gh-ref/<n>` fragment hrefs (same convention as `#preview/`
 * and `#session/`: no custom scheme, survives markdown sanitization) that
 * `GhRefLink` resolves against the session's repo at render time.
 */

// A `#` immediately followed by digits is never a markdown ATX heading (headings
// need whitespace after the run of `#`), so digits can be linked unambiguously.
// The lookbehind excludes word chars (footnote syntax `[^1]`, identifiers like
// `abc#12`), markdown link syntax (`](#46)`), URL fragments (a bare-URL
// autolink has already consumed its `#...` tail by the time this runs), and
// other `#`s (`##46`). Max 7 digits: GitHub issue numbers have never come close.
// Trailing prose punctuation (`#46.`) is left outside the match by `\b` — no
// peeling needed, unlike `@session:` whose greedy value branch swallows it.
export const GH_ISSUE_REF_RE = /(?<![\w/&?=#.-])(?<!]\()#(\d{1,7})\b/g

const GH_REF_HREF_PREFIX = '#gh-ref/'

// A complete labeled markdown link (`[PR #46](https://…)`) is atomic: a `#N`
// inside its label must not be re-linked into a nested, mangled construct.
const MD_LINK_SPAN_RE = /(\[[^\]\n]*\]\([^)\n]*\))/g

/**
 * Rewrites bare `#123` tokens into `#gh-ref/` links so `MarkdownLink` can render
 * them as repo-resolving chips. Callers must exclude code spans/fences —
 * `preprocessMarkdown` already splits those out.
 */
export function linkifyGhIssueRefs(text: string): string {
  if (!text.includes('#')) {
    return text
  }

  return text
    .split(MD_LINK_SPAN_RE)
    .map((part, index) =>
      index % 2 === 1 ? part : part.replace(GH_ISSUE_REF_RE, (match, digits: string) => `[${match}](#gh-ref/${digits})`)
    )
    .join('')
}

export function ghRefFromMarkdownHref(href?: string): number | null {
  if (!href?.startsWith(GH_REF_HREF_PREFIX)) {
    return null
  }

  const digits = href.slice(GH_REF_HREF_PREFIX.length)

  return /^\d{1,7}$/.test(digits) ? Number(digits) : null
}

export function ghIssueUrl(owner: string, repo: string, number: number): string {
  return `https://github.com/${owner}/${repo}/issues/${number}`
}