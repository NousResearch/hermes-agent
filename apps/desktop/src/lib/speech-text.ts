const EMOJI_RE = /(?:[\u{1F000}-\u{1FAFF}\u{2600}-\u{27BF}]|[\u{FE0F}\u{200D}]|[\u{E0020}-\u{E007F}])+/gu

const FENCED_CODE_RE = /```[\s\S]*?(?:```|$)/g
const INLINE_CODE_RE = /`([^`]+)`/g
const MARKDOWN_LINK_RE = /\[([^\]]+)\]\(([^)]+)\)/g
const PARAGRAPH_BREAK_RE = /[ \t]*\n{2,}[ \t]*/g
const PUNCTUATED_PARAGRAPH_BREAK_RE = /([.!?])([*_~`>"'’”)}\]]*)[ \t]*\n{2,}[ \t]*/g
const SOFT_BREAK_RE = /[ \t]*\n[ \t]*/g

const MEDIA_PATH_RE = /MEDIA:\S+/g
const LINE_FINAL_COLON_RE = /:\s*$/gm

// Bare filesystem paths in prose ("~/.config/himalaya/config.toml", "/etc/hosts",
// "src/lib/app.ts") have no MEDIA: marker. A path is an address for the screen,
// not speech: the voice loops on "slash dot config slash himalaya slash config
// dot toml". Tilde paths are unambiguous; absolute and dot-slash paths need at
// least one letter (so "/06/02" is left alone); bare relative paths must end in
// a file extension, which keeps "and/or", "N/A", "2026/06/02", "1.5/2.5" and
// "5/month" intact. The replacement is "the path": the prose carries the meaning.
const PATH_TOKEN_RE =
  /~\/(?:[\w.-]+\/)*[\w.-]+(?:\.[A-Za-z]{1,8})?|(?:\.\/|\/)(?=[\w.-]*[A-Za-z])(?:[\w.-]+\/)+[\w.-]+(?:\.[A-Za-z]{1,8})?|(?:\.\/|\/)[\w.-]+\.[A-Za-z]{1,8}|(?<![\w/])[\w.-]+(?:\/[\w.-]+)+\.[A-Za-z]{1,8}/g

const THINKING_PREFIX_RE =
  /^\s*(?:\([^)\n]{1,48}\)\s*)?(?:processing|thinking|reasoning|analyzing|pondering|contemplating|musing|cogitating|ruminating|deliberating|mulling|reflecting|computing|synthesizing|formulating|brainstorming)\.\.\.\s*/i

const URL_RE = /\bhttps?:\/\/\S+/gi

const MARKDOWN_TABLE_DELIMITER_CELL_RE = /^:?-{3,}:?$/

interface MarkdownTableRow {
  blockquoteDepth: number
  cells: string[]
}

function isUnescapedPipe(row: string, index: number): boolean {
  let backslashes = 0

  for (let cursor = index - 1; cursor >= 0 && row[cursor] === '\\'; cursor -= 1) {
    backslashes += 1
  }

  return backslashes % 2 === 0
}

function splitMarkdownTableCells(row: string): string[] {
  const cells: string[] = []
  let cellStart = 0

  for (let index = 0; index < row.length; index += 1) {
    if (row[index] === '|' && isUnescapedPipe(row, index)) {
      cells.push(row.slice(cellStart, index).trim())
      cellStart = index + 1
    }
  }

  cells.push(row.slice(cellStart).trim())

  return cells
}

function parseMarkdownTableRow(line: string): MarkdownTableRow | null {
  let row = line
  let blockquoteDepth = 0

  while (true) {
    const indentation = row.match(/^[ \t]*/)?.[0] ?? ''

    if (indentation.includes('\t') || indentation.length > 3) {
      return null
    }

    row = row.slice(indentation.length)

    if (!row.startsWith('>')) {
      break
    }

    blockquoteDepth += 1
    row = row.slice(1)

    if (row.startsWith(' ')) {
      row = row.slice(1)
    }
  }

  row = row.trimEnd()

  const pipeIndexes = [...row.matchAll(/\|/g)].map(match => match.index).filter(index => isUnescapedPipe(row, index))

  if (pipeIndexes.length === 0) {
    return null
  }

  const hasLeadingPipe = pipeIndexes[0] === 0
  const hasTrailingPipe = pipeIndexes.at(-1) === row.length - 1

  if (hasLeadingPipe) {
    row = row.slice(1)
  }

  if (hasTrailingPipe) {
    row = row.slice(0, -1)
  }

  const cells = splitMarkdownTableCells(row)

  if (cells.length < 2 && !(hasLeadingPipe && hasTrailingPipe && cells.length === 1)) {
    return null
  }

  return { blockquoteDepth, cells }
}

function stripMarkdownTables(text: string): string {
  const lines = text.replace(/\r\n?/g, '\n').split('\n')
  const tableLines = new Set<number>()

  let index = 1

  while (index < lines.length) {
    const delimiterRow = parseMarkdownTableRow(lines[index])
    const headerRow = parseMarkdownTableRow(lines[index - 1])

    if (
      !delimiterRow ||
      !headerRow ||
      !delimiterRow.cells.every(cell => MARKDOWN_TABLE_DELIMITER_CELL_RE.test(cell)) ||
      headerRow.cells.length !== delimiterRow.cells.length ||
      headerRow.blockquoteDepth !== delimiterRow.blockquoteDepth
    ) {
      index += 1

      continue
    }

    tableLines.add(index - 1)
    tableLines.add(index)

    let rowIndex = index + 1

    for (; rowIndex < lines.length; rowIndex += 1) {
      const bodyRow = parseMarkdownTableRow(lines[rowIndex])

      if (!bodyRow || bodyRow.blockquoteDepth !== delimiterRow.blockquoteDepth) {
        break
      }

      tableLines.add(rowIndex)
    }

    index = rowIndex
  }

  return lines.filter((_, index) => !tableLines.has(index)).join('\n')
}

function normalizeLineBreaks(text: string): string {
  return text
    .replace(/\r\n?/g, '\n')
    .replace(/(\p{L})-\n(\p{L})/gu, '$1$2')
    .replace(PUNCTUATED_PARAGRAPH_BREAK_RE, '$1$2 ')
    .replace(PARAGRAPH_BREAK_RE, '. ')
    .replace(SOFT_BREAK_RE, ' ')
}

function expandSymbols(text: string): string {
  return text
    .replace(/~(?=\s*\d)/g, ' about ') // "~100" -> "about 100"
    .replace(/~/g, '') // stray tildes (e.g. ~~strike~~) are silence
    .replace(/(?<=\d)\s*×\s*/g, ' times ') // "2×5090" -> "2 times 5090"
    .replace(/→/g, ' to ')
    .replace(/⇒/g, ' to ')
    .replace(/(\d)\s*[—–]\s*(\d)/g, '$1 to $2') // ranges: "pages 5–10" -> "5 to 10"
    .replace(/\s*[—–]\s*(?=[A-Z])/g, '. ') // "peek — Done" -> "peek. Done"
    .replace(/\s*[—–]\s*/g, ', ') // everything left is a comma pause
    .replace(/≈/g, ' about ')
    .replace(/&/g, ' and ')
    .replace(/(?<=\d)\s*%/g, ' percent ')
    .replace(/€\s*([\d,]*\d)/g, '$1 euros') // "€5" -> "5 euros"
    .replace(/([\d,]*\d)\s*€/g, '$1 euros') // PT/ES suffix: "1.499,90 €" -> "1.499,90 euros"
    .replace(/€/g, ' euros ') // bare remainder ("prices in €")
    .replace(/…/g, '...')
}

export function sanitizeTextForSpeech(text: string): string {
  // Tables first: their right-align marker is a trailing colon (":-"), and
  // closing colons before the table detector runs would mangle it.
  const withoutTables = stripMarkdownTables(String(text))

  // Close line-final colons BEFORE newlines are flattened: "the regex list:"
  // followed by a code block keeps its colon if this runs after the flatten,
  // and the voice hangs on it. Closing early turns it into "the regex list.".
  const pre = withoutTables.replace(LINE_FINAL_COLON_RE, '.')

  const cleaned = normalizeLineBreaks(pre)
    .replace(FENCED_CODE_RE, '')
    .replace(THINKING_PREFIX_RE, ' ')
    .replace(MARKDOWN_LINK_RE, '$1')
    .replace(INLINE_CODE_RE, '$1')
    .replace(URL_RE, '')
    .replace(MEDIA_PATH_RE, '')
    .replace(PATH_TOKEN_RE, ' the path ')
    .replace(EMOJI_RE, ' ')
    .replace(/^#{1,6}\s+/gm, '')
    .replace(/[*_>#]/g, '')
    .replace(/^\s*[-+*]\s+/gm, '')

  return expandSymbols(cleaned)
    .replace(/:\s*$/, '.') // colon orphaned when its link/code was stripped
    .replace(/\s+/g, ' ')
    .trim()
}
