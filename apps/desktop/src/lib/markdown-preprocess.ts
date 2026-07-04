import { isLikelyProseFence, sanitizeLanguageTag } from '@/lib/markdown-code'
import { stripPreviewTargets } from '@/lib/preview-targets'

const REASONING_BLOCK_RE = /<(think|thinking|reasoning|scratchpad|analysis)>[\s\S]*?<\/\1>\s*/gi
const PREVIEW_MARKER_RE = /\[Preview:[^\]]+\]\(#preview[:/][^)]+\)/gi

const FENCE_LINE_RE = /^([ \t]*)(`{3,}|~{3,})([^\n]*)$/
const EMPTY_FENCE_BLOCK_RE = /(^|\n)[ \t]*(?:`{3,}|~{3,})[^\n]*\n[ \t]*(?:`{3,}|~{3,})[ \t]*(?=\n|$)/g
const CODE_FENCE_SPLIT_RE = /((?:```|~~~)[\s\S]*?(?:```|~~~))/g
const INLINE_CODE_SPLIT_RE = /(`[^`\n]+`)/g
// Bare-URL autolink matcher. The character classes EXCLUDE `*` so a URL that
// abuts markdown emphasis with no separating space (e.g. `**label: https://x**`,
// a very common LLM pattern) doesn't swallow the trailing `**` into the href.
// `*` is never meaningful in a real URL path, and GFM's own autolink extension
// likewise strips trailing emphasis/punctuation — so dropping it here is safe
// and keeps the emphasis run intact. Other trailing punctuation is still peeled
// off by the final `[^\s<>"'`*.,;:!?]` class.
const RAW_URL_RE = /https?:\/\/[^\s<>"'`*]+[^\s<>"'`*.,;:!?]/g

// File-path autolink matcher. Wraps filesystem paths the model references in
// prose as `[path](file://path)` markdown so MarkdownLink can render them as
// hyperlinks with a right-click context menu (copy / reveal / open / open-with).
//
// Scope is deliberately narrow to keep false positives low — this runs on prose
// only (code fences and inline code spans are carved out by CODE_FENCE_SPLIT_RE
// / INLINE_CODE_SPLIT_RE before this runs). Four unambiguous path shapes are
// matched:
//
//   1. Windows drive paths  — C:\Users\me\src\main.ts
//   2. UNC paths            — \\server\share\file.txt
//   3. Unix absolute paths  — /home/me/src/main.ts
//   4. Dot-relative paths   — ./config/settings.yaml, ../src/App.tsx
//
// Bare relative paths (src/components/App.tsx) are intentionally NOT matched —
// their false-positive rate against natural language and code-like prose is too
// high. Models wrap such paths in backticks often enough that the inline-code
// carve-out is the safer home for a future phase-2 pass.
//
// Each branch captures an optional `:line` / `:line:col` suffix (e.g.
// `src/App.tsx:42:10`); the suffix travels in the visible text but, per the
// file:// URL spec, is NOT encoded into the href (there's no standard way to
// carry line info in a file:// URL — Claude Code and the community `termlink`
// tool handle it the same way). The component keeps the suffix in the label.
//
// Lookbehinds keep the path from matching mid-token: a Windows drive letter or
// Unix root must not be preceded by a path character (so we don't double-link a
// path that's already part of a URL or another path). The character classes
// exclude whitespace, quotes, and shell glob/redirect chars (`*?<>|"`) that
// can't appear in a real path. Trailing punctuation is peeled off so a path at
// the end of a sentence (`.../main.ts.`) doesn't swallow the period.
const FILE_PATH_RE =
  /(?<![\w/\\])(?:([A-Za-z]:[\\/](?:[^\s*?<>|"]*[\\/])+[^\s*?<>|"\\/]+(?:\.[A-Za-z0-9]+))|(\\\\[^\s*?<>|"\\]+(?:\\[^\s*?<>|"\\]+)+)|(\/(?:[^\s*?<>|"]+\/)+[^\s*?<>|"/]+(?:\.[A-Za-z0-9]+))|(\.{1,2}\/(?:[^\s*?<>|"]+\/)*[^\s*?<>|"/]+(?:\.[A-Za-z0-9]+)))(?::(\d+)(?::(\d+))?)?/g

// Trailing punctuation/quote characters that can legitimately end a sentence
// containing a path but are never part of the path itself. Peeled off after
// matching so the href stays clean.
const FILE_PATH_TRAILING_PUNCT = /[.,;:!?)\]"'`*]+$/

// Encode a filesystem path into a `file://` URL. Mirrors Node's
// `pathToFileURL` for the cases we handle: backslashes → forward slashes,
// percent-encode spaces and other reserved chars, and keep the leading
// drive-letter form intact. We don't import url/pathToFileURL here because this
// module runs in both the renderer bundle and vitest (jsdom) — a hand-rolled
// encodeURI keeps the behaviour identical across both.
function pathToFileUrl(filePath: string): string {
  const forwardSlashed = filePath.replace(/\\/g, '/')
  // drive-letter paths (C:/...) become file:///C:/... ; UNC (//server/...)
  // becomes file://server/... ; POSIX (/home/...) becomes file:///home/...
  const prefix = /^[A-Za-z]:\//.test(forwardSlashed) ? 'file:///' : 'file://'
  return prefix + encodeURI(forwardSlashed)
}

function linkifyFilePaths(text: string): string {
  return text.replace(FILE_PATH_RE, match => {
    let path = match
    let trailing = ''

    // Separate the path from its optional :line:col tail so the href targets
    // the file only, while the label keeps the suffix the model wrote.
    const lineMatch = path.match(/^(.*?)(?::(\d+)(?::(\d+))?)?$/)
    if (lineMatch) {
      path = lineMatch[1]
    }

    // Peel trailing punctuation that isn't part of the path.
    const punctMatch = path.match(FILE_PATH_TRAILING_PUNCT)
    if (punctMatch) {
      trailing = punctMatch[0]
      path = path.slice(0, -punctMatch[0].length)
    }

    if (!path) return match

    const href = pathToFileUrl(path)
    // Use the original matched text (path + optional :line:col) as the label
    // so the user sees exactly what the model wrote. Escape any `]` in the
    // label so it doesn't break the `[label](href)` markdown structure.
    const label = (path + (match.slice(path.length) || '')).replace(/([[\]])/g, '\\$1')

    return `[${label}](${href})${trailing}`
  })
}
const LOCAL_PREVIEW_URL_RE = /(^|\s)https?:\/\/(?:localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\])(?::\d+)?\/?[^\s<>"'`]*/gi
const LOCAL_PREVIEW_ONLY_RE = /^https?:\/\/(?:localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\])(?::\d+)?\/?$/i
const URL_ONLY_LINE_RE = /^\s*https?:\/\/\S+\s*$/i
const CITATION_MARKER_RE = /(?<=[\p{L}\p{N})\].,!?:;"'”’])\[(?:\d+(?:\s*,\s*\d+)*)\](?!\()/gu

/**
 * Returns true when `body` contains a line that's exactly `marker` (modulo
 * leading/trailing horizontal whitespace) — i.e. an unambiguous close fence
 * for an opening fence with the same marker.
 *
 * Implemented with string comparisons (not RegExp) so that input-derived
 * `marker` values can never bleed into a regex pattern. This matters for
 * CodeQL's `js/incomplete-hostname-regexp` dataflow, which would otherwise
 * trace test-fixture URLs from the input through `marker` into the regex
 * source, even though `marker` is captured by `(`{3,}|~{3,})` and can only
 * ever be backticks or tildes.
 */
function hasCloseFenceLine(body: string, marker: string): boolean {
  const lines = body.split('\n')

  // Original regex required `\n` immediately before the close fence, so the
  // first line of `body` (which has no preceding newline within `body`)
  // cannot itself be the close fence.
  for (let i = 1; i < lines.length; i += 1) {
    const line = lines[i]
    let lo = 0
    let hi = line.length

    while (lo < hi && (line[lo] === ' ' || line[lo] === '\t')) {
      lo += 1
    }

    while (hi > lo && (line[hi - 1] === ' ' || line[hi - 1] === '\t')) {
      hi -= 1
    }

    if (line.slice(lo, hi) === marker) {
      return true
    }
  }

  return false
}

function scrubBacktickNoise(text: string): string {
  const balancedFenceRe = /(^|\n)([ \t]*)(`{3,}|~{3,})([^\n]*)\n([\s\S]*?)\n[ \t]*\3[ \t]*(?=\n|$)/g
  const protectedRanges: { end: number; start: number }[] = []
  let match: RegExpExecArray | null

  while ((match = balancedFenceRe.exec(text)) !== null) {
    const start = match.index + match[1].length

    protectedRanges.push({ end: balancedFenceRe.lastIndex, start })
  }

  const danglingCodeFenceRe = /(^|\n)[ \t]*(`{3,}|~{3,})([a-z0-9][a-z0-9+#-]{0,15})[ \t]*\n([\s\S]*)$/gi

  while ((match = danglingCodeFenceRe.exec(text)) !== null) {
    const start = match.index + match[1].length
    const marker = match[2] || '```'
    const info = match[3] || ''
    const body = match[4] || ''

    if (!hasCloseFenceLine(body, marker) && sanitizeLanguageTag(info) && !isLikelyProseFence(info, body)) {
      protectedRanges.push({ end: text.length, start })

      break
    }
  }

  protectedRanges.sort((a, b) => a.start - b.start)

  const fenceNoiseRe = /`{3,}/g
  let out = ''
  let cursor = 0

  for (const range of protectedRanges) {
    out += text.slice(cursor, range.start).replace(fenceNoiseRe, '')
    out += text.slice(range.start, range.end)
    cursor = range.end
  }

  out += text.slice(cursor).replace(fenceNoiseRe, '')

  for (let pass = 0; pass < 2; pass += 1) {
    // Match EXACTLY 2 backticks (not part of a longer run) on each side.
    // Without the lookbehind/lookahead, two adjacent triple-backtick
    // fences with only whitespace between them get spliced together —
    // e.g. ```bash\n...\n```\n\n```latex matches the regex's
    // last-2-of-bash-close + \n\n + first-2-of-latex-open and the
    // surrounding fence markers collapse into a single longer block,
    // which the markdown parser then treats as ONE giant code block.
    out = out.replace(/(?<!`)``(?!`)\s*(?<!`)``(?!`)/g, '')
    out = out.replace(/(^|[^`])``(?=\s|[.,;:!?)\]'"\u2014\u2013-]|$)/g, '$1')
  }

  return out
}

function stripEmptyFenceBlocks(text: string): string {
  return text.replace(EMPTY_FENCE_BLOCK_RE, '$1')
}

function isUrlOnlyBlock(lines: string[]): boolean {
  const nonEmpty = lines.filter(line => line.trim())

  return nonEmpty.length > 0 && nonEmpty.every(line => URL_ONLY_LINE_RE.test(line))
}

function autoLinkRawUrls(text: string): string {
  return text.replace(RAW_URL_RE, (url: string, index: number) => {
    const previous = text[index - 1] || ''
    const beforePrevious = text[index - 2] || ''

    if (previous === '<' || (beforePrevious === ']' && previous === '(')) {
      return url
    }

    return `<${url}>`
  })
}

function normalizeVisibleProse(text: string): string {
  return text
    .split(INLINE_CODE_SPLIT_RE)
    .map(part =>
      part.startsWith('`')
        ? part
        : linkifyFilePaths(
            autoLinkRawUrls(
              part.replace(/`{3,}/g, '').replace(LOCAL_PREVIEW_URL_RE, '$1').replace(CITATION_MARKER_RE, '')
            )
          )
    )
    .join('')
}

function extend(out: string[], lines: string[]) {
  for (const line of lines) {
    out.push(line)
  }
}

function pushProseFence(out: string[], indent: string, info: string, lines: string[]) {
  if (info) {
    out.push(`${indent}${info}`.trimEnd())
  }

  extend(out, lines)
}

function findClosingFence(lines: string[], start: number, marker: string): number {
  for (let cursor = start + 1; cursor < lines.length; cursor += 1) {
    const closeMatch = (lines[cursor] || '').match(FENCE_LINE_RE)

    if (!closeMatch) {
      continue
    }

    const closeMarker = closeMatch[2] || ''
    const closeInfo = (closeMatch[3] || '').trim()

    if (!closeInfo && closeMarker[0] === marker[0] && closeMarker.length >= marker.length) {
      return cursor
    }
  }

  return -1
}

// Languages that should be routed to the math (KaTeX) renderer instead of
// being shown as a syntax-highlighted code block.
//
// We deliberately recognize ONLY `math` here, not `latex` or `tex`.
// Reasoning: GitHub-style markdown uses ` ```math ` to mean "render as
// math" and ` ```latex `/` ```tex ` to mean "show LaTeX/TeX source code"
// (syntax highlighted). Conflating the two breaks code blocks where a
// user is *discussing* LaTeX rather than embedding it (e.g.,
// ```latex\n\begin{equation}\n  E = mc^2\n\end{equation}``` shown as a
// teaching example). Anyone who wants math rendered should use ```math.
const MATH_FENCE_LANGUAGES = new Set(['math'])

function isMathFence(language: string): boolean {
  return MATH_FENCE_LANGUAGES.has(language.toLowerCase())
}

function normalizeFenceBlocks(text: string): string {
  const sourceLines = text.split('\n')
  const out: string[] = []
  let index = 0

  while (index < sourceLines.length) {
    const line = sourceLines[index] || ''
    const match = line.match(FENCE_LINE_RE)

    if (!match) {
      out.push(line)
      index += 1

      continue
    }

    const indent = match[1] || ''
    const marker = match[2] || '```'
    const infoRaw = (match[3] || '').trim()
    const languageToken = infoRaw.split(/\s+/, 1)[0] || ''
    const language = sanitizeLanguageTag(languageToken)
    const openerValid = !infoRaw || Boolean(language)

    if (!openerValid) {
      out.push(`${indent}${infoRaw}`.trimEnd())
      index += 1

      continue
    }

    const closeIndex = findClosingFence(sourceLines, index, marker)
    const bodyLines = sourceLines.slice(index + 1, closeIndex === -1 ? sourceLines.length : closeIndex)
    const body = bodyLines.join('\n')

    if (closeIndex !== -1 && !body.trim()) {
      index = closeIndex + 1

      continue
    }

    if (closeIndex !== -1 && LOCAL_PREVIEW_ONLY_RE.test(body.trim())) {
      index = closeIndex + 1

      continue
    }

    if (closeIndex !== -1 && isUrlOnlyBlock(bodyLines)) {
      extend(out, bodyLines)
      index = closeIndex + 1

      continue
    }

    if (closeIndex === -1) {
      if (!body.trim()) {
        index += 1

        continue
      }

      if (isLikelyProseFence(infoRaw, body)) {
        pushProseFence(out, indent, infoRaw, bodyLines)
      } else if (isMathFence(language)) {
        // Streaming math fence — rewrite the language tag to "math".
        // remark-math + rehype-katex pick up ```math fenced blocks via
        // the language-math class on the resulting <code> element. We
        // keep the fence intact (instead of converting to $$..$$) so
        // any literal `$$` characters in the body don't collide with
        // an outer math wrapper. No close emitted yet — streaming.
        out.push(`${indent}${marker}math`)
        extend(out, bodyLines)
      } else {
        out.push(`${indent}${marker}${language}`)
        extend(out, bodyLines)
      }

      break
    }

    if (isLikelyProseFence(infoRaw, body)) {
      pushProseFence(out, indent, infoRaw, bodyLines)
      index = closeIndex + 1

      continue
    }

    if (isMathFence(language)) {
      // Closed math fence — rewrite the language tag to "math" so
      // rehype-katex's language-math class detection picks it up.
      // Body stays untouched (no $$..$$ rewrite) so authors can write
      // arbitrary LaTeX including `$$display$$` markers without them
      // colliding with our wrapper. Without this rewrite the block
      // would render as a syntax-highlighted "latex" code listing.
      out.push(`${indent}${marker}math`)
      extend(out, bodyLines)
      out.push(`${indent}${marker}`)
      index = closeIndex + 1

      continue
    }

    out.push(`${indent}${marker}${language}`)
    extend(out, bodyLines)
    out.push(`${indent}${marker}`)
    index = closeIndex + 1
  }

  return out.join('\n')
}

// Convert LaTeX bracket delimiters to remark-math's dollar-sign syntax.
// Models often emit `\(...\)` for inline math and `\[...\]` for display
// math (the standard LaTeX convention) instead of `$...$` / `$$...$$`.
// remark-math only natively recognizes the dollar form, so we rewrite at
// preprocess time. Done with simple non-greedy matches keyed on the
// escaped-bracket sequences — these are rare enough in non-math content
// (you'd have to write a literal `\(` followed eventually by a literal
// `\)` with NO interleaving newline-paragraph-break) that false positives
// are extremely unlikely.
const LATEX_INLINE_RE = /\\\(([^\n]+?)\\\)/g
const LATEX_DISPLAY_RE = /\\\[([\s\S]+?)\\\]/g

function rewriteLatexBracketDelimiters(text: string): string {
  return text
    .replace(LATEX_INLINE_RE, (_, body: string) => `$${body}$`)
    .replace(LATEX_DISPLAY_RE, (_, body: string) => `$$${body}$$`)
}

// Escape `$<digit>` patterns so they don't get eaten as math delimiters.
// Models commonly write currency amounts ($5, $19.99, $1,299) in prose.
// With `singleDollarTextMath: true`, remark-math is greedy and matches
// EVERY pair of `$`s — including the open of `$5` to the next `$10`,
// rendering "5 in my pocket and you have " as italicized math text.
// The de-facto convention across math-supporting LLM UIs is to treat
// `$` followed by a digit as currency rather than math, since math
// expressions almost always start with a letter or `\command`. Trade-
// off: a math expression like `$5x = 10$` would have its leading 5
// escaped — annoying but rare. The escape `\$` survives to render as
// a literal `$` in the final output.
const CURRENCY_DOLLAR_RE = /(^|[^\\])\$(?=\d)/g

function escapeCurrencyDollars(text: string): string {
  return text.replace(CURRENCY_DOLLAR_RE, '$1\\$')
}

export function preprocessMarkdown(text: string): string {
  const cleaned = text.replace(REASONING_BLOCK_RE, '').replace(PREVIEW_MARKER_RE, '')
  const scrubbed = scrubBacktickNoise(cleaned)
  const normalizedFences = normalizeFenceBlocks(scrubbed)
  const strippedEmptyFences = stripEmptyFenceBlocks(normalizedFences)

  return strippedEmptyFences
    .split(CODE_FENCE_SPLIT_RE)
    .map(part => {
      // Fence blocks pass through untouched.
      if (/^(?:```|~~~)/.test(part)) {
        return part
      }

      // Whitespace-only segments (e.g. the `\n\n` between two adjacent
      // fences) must NOT go through stripPreviewTargets — its internal
      // .trim() would collapse them to '' and glue the surrounding
      // fences together, producing things like ``````math which the
      // markdown parser then reads as a single 6-backtick block.
      if (!part.trim()) {
        return part
      }

      // Preserve leading/trailing whitespace around the prose body so
      // that fence-prose-fence sequences keep their blank-line gaps.
      // stripPreviewTargets internally calls .trim() on its result for
      // the benefit of its other (single-segment) callers; here we're
      // operating on a SEGMENT of a larger document where outer
      // whitespace is structural and must survive.
      const leading = part.match(/^\s*/)?.[0] ?? ''
      const trailing = part.match(/\s*$/)?.[0] ?? ''

      // rewriteLatexBracketDelimiters runs only on prose segments so
      // we don't accidentally touch `\(` inside a code block.
      // escapeCurrencyDollars likewise only runs on prose, so legit
      // `$5` literals inside fenced code stay intact.
      const transformed = normalizeVisibleProse(
        stripPreviewTargets(rewriteLatexBracketDelimiters(escapeCurrencyDollars(part)))
      )

      return leading + transformed + trailing
    })
    .join('')
    .replace(/[ \t]+\n/g, '\n')
}
