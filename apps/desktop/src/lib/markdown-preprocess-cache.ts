import { preprocessMarkdown } from './markdown-preprocess'

type FullPreprocess = (text: string) => string

export interface IncrementalMarkdownPreprocessor {
  (text: string): string
  clear: () => void
}

type MarkdownPreprocessor = (text: string) => string

export function selectMarkdownPreprocessor(
  isStreaming: boolean,
  incremental: MarkdownPreprocessor,
  completed: MarkdownPreprocessor
): MarkdownPreprocessor {
  return isStreaming ? incremental : completed
}

interface AppendEntry {
  output: string
  refreshAt: number
  text: string
}

const APPEND_CACHE_MAX = 8
const APPEND_LINEAGE_MAX = 8
const APPEND_CACHE_MIN_LENGTH = 2048
const APPEND_CACHE_REFRESH_CHARS = 1024
const MAX_RETAINED_MARKDOWN_CHARS = 200_000
const PLAIN_PROSE_UNSAFE_RE = /[<`$\\[]|~{3,}|https?:\/\/|\n{3,}|[ \t]+\n/iu

/**
 * Finds a prefix whose preprocessing cannot be changed by later appended text.
 *
 * The proof is intentionally narrow. The prefix contains only plain prose and
 * stops before the whitespace of a blank-line boundary. That retained boundary
 * keeps all suffix work separated from the cached bytes, including whitespace
 * collapsing. Tokens that can open state across chunks (code, math, raw
 * HTML/reasoning tags, links/citations/preview markers, or tilde fences) end the
 * candidate prefix, but may safely remain in the reprocessed tail.
 */
function settledPlainProsePrefixLength(text: string): number {
  const unsafeIndex = text.search(PLAIN_PROSE_UNSAFE_RE)
  const plainCandidate = unsafeIndex === -1 ? text : text.slice(0, unsafeIndex)
  const lastBoundaryStart = plainCandidate.lastIndexOf('\n\n')

  if (lastBoundaryStart === -1) {
    return 0
  }

  // Keep the final completed paragraph in the reprocessed tail. Without that
  // non-whitespace context, leading-newline preservation in preprocessMarkdown
  // would prevent a later 3-newline run from collapsing exactly as it does in
  // the full document.
  let lastWhitespaceRunStart = lastBoundaryStart

  while (lastWhitespaceRunStart > 0 && /[\t\n\r ]/u.test(text[lastWhitespaceRunStart - 1])) {
    lastWhitespaceRunStart -= 1
  }

  const previousBoundaryStart = text.lastIndexOf('\n\n', lastWhitespaceRunStart - 2)

  if (previousBoundaryStart === -1) {
    return 0
  }

  let prefixLength = previousBoundaryStart

  // Leave the entire preceding whitespace run in the reprocessed tail too, so
  // the cached bytes always end at settled, non-whitespace prose.
  while (prefixLength > 0 && /[\t\n\r ]/u.test(text[prefixLength - 1])) {
    prefixLength -= 1
  }

  return prefixLength >= APPEND_CACHE_MIN_LENGTH ? prefixLength : 0
}

export function createIncrementalMarkdownPreprocessor(fullPreprocess: FullPreprocess = preprocessMarkdown) {
  const appendCache: AppendEntry[] = []
  const previousTexts: string[] = []

  function findSettledPrefix(text: string): AppendEntry | undefined {
    let longest: AppendEntry | undefined

    for (const entry of appendCache) {
      if (text.startsWith(entry.text) && (!longest || entry.text.length > longest.text.length)) {
        longest = entry
      }
    }

    return longest
  }

  function findAppendLineage(text: string): number {
    let longestIndex = -1

    for (let index = 0; index < previousTexts.length; index += 1) {
      const previous = previousTexts[index]!

      if (
        text.startsWith(previous) &&
        (longestIndex === -1 || previous.length > previousTexts[longestIndex]!.length)
      ) {
        longestIndex = index
      }
    }

    return longestIndex
  }

  function rememberLineage(text: string, existingIndex: number): void {
    if (existingIndex !== -1) {
      previousTexts.splice(existingIndex, 1)
    }

    previousTexts.push(text)

    if (previousTexts.length > APPEND_LINEAGE_MAX) {
      previousTexts.shift()
    }
  }

  function rememberSettled(text: string, source: AppendEntry | undefined): AppendEntry | undefined {
    if (source && text.length < source.refreshAt) {
      return source
    }

    const prefixLength = settledPlainProsePrefixLength(text)

    if (!prefixLength || (source && prefixLength <= source.text.length)) {
      if (source) {
        source.refreshAt = text.length + APPEND_CACHE_REFRESH_CHARS
      }

      return source
    }

    const settledText = text.slice(0, prefixLength)
    const existingIndex = appendCache.findIndex(entry => entry.text === settledText)

    if (existingIndex !== -1) {
      const [existing] = appendCache.splice(existingIndex, 1)
      appendCache.push(existing)

      return existing
    }

    const settledOutput =
      source && settledText.startsWith(source.text)
        ? source.output + fullPreprocess(settledText.slice(source.text.length))
        : fullPreprocess(settledText)

    for (let index = appendCache.length - 1; index >= 0; index -= 1) {
      if (settledText.startsWith(appendCache[index].text)) {
        appendCache.splice(index, 1)
      }
    }

    const entry = {
      output: settledOutput,
      refreshAt: text.length + APPEND_CACHE_REFRESH_CHARS,
      text: settledText
    }

    appendCache.push(entry)

    if (appendCache.length > APPEND_CACHE_MAX) {
      appendCache.shift()
    }

    return entry
  }

  const preprocess: IncrementalMarkdownPreprocessor = (text: string): string => {
    if (text.length > MAX_RETAINED_MARKDOWN_CHARS) {
      appendCache.length = 0
      previousTexts.length = 0

      return fullPreprocess(text)
    }

    const appendLineage = findAppendLineage(text)
    const isAppend = previousTexts.length === 0 || appendLineage !== -1

    const settledPrefix = isAppend ? findSettledPrefix(text) : undefined
    const reusablePrefix = isAppend ? rememberSettled(text, settledPrefix) : undefined

    const output = reusablePrefix
      ? reusablePrefix.output + fullPreprocess(text.slice(reusablePrefix.text.length))
      : fullPreprocess(text)

    rememberLineage(text, appendLineage)

    return output
  }

  preprocess.clear = () => {
    appendCache.length = 0
    previousTexts.length = 0
  }

  return preprocess
}
