/**
 * One section of a Markdown document as paragraphs and (GFM) tables, for a page that shows a part of a
 * checked-in document (DESIGN.md's "Still to unify") instead of restating it.
 */

export interface MarkdownTable {
  head: string[]
  rows: string[][]
}

export type MarkdownBlock = { kind: 'paragraph'; text: string } | { kind: 'table'; table: MarkdownTable }

const HEADING = /^(#{1,6})\s+(.*?)\s*$/

/** A table row's cells; `\|` is a literal pipe inside a cell. */
export function tableCells(line: string): string[] {
  const inner = line.trim().replace(/^\|/, '').replace(/\|$/, '')

  return inner.split(/(?<!\\)\|/).map(cell => cell.trim().replace(/\\\|/g, '|'))
}

const isDelimiterRow = (cells: string[]) => cells.every(cell => /^:?-{3,}:?$/.test(cell))

function toBlocks(lines: string[]): MarkdownBlock[] {
  const blocks: MarkdownBlock[] = []
  let paragraph: string[] = []
  let table: string[][] = []

  const end = () => {
    if (paragraph.length) {
      blocks.push({ kind: 'paragraph', text: paragraph.join(' ') })
    }

    if (table.length) {
      const [head, ...rest] = table
      blocks.push({ kind: 'table', table: { head, rows: rest.filter(row => !isDelimiterRow(row)) } })
    }

    paragraph = []
    table = []
  }

  for (const raw of lines) {
    const line = raw.trim()

    if (!line) {
      end()
    } else if (line.startsWith('|')) {
      if (paragraph.length) {
        end()
      }

      table.push(tableCells(line))
    } else {
      if (table.length) {
        end()
      }

      paragraph.push(line)
    }
  }

  end()

  return blocks
}

/** The blocks under the heading whose text is `title`, up to the next heading of the same or a higher
 *  level; `null` when the document has no such heading. */
export function markdownSection(markdown: string, title: string): MarkdownBlock[] | null {
  const lines = markdown.split('\n')
  const start = lines.findIndex(line => HEADING.exec(line)?.[2] === title)

  if (start < 0) {
    return null
  }

  const level = HEADING.exec(lines[start])![1].length

  const end = lines.findIndex((line, index) => index > start && (HEADING.exec(line)?.[1].length ?? Infinity) <= level)

  return toBlocks(lines.slice(start + 1, end < 0 ? undefined : end))
}

export type InlineSpan = { code: boolean; text: string }

/** Inline code spans (`` `…` ``) split out of a line; everything else is plain text. */
export function inlineCode(text: string): InlineSpan[] {
  return text
    .split(/(`[^`]+`)/)
    .filter(Boolean)
    .map(part =>
      part.startsWith('`') && part.endsWith('`') ? { code: true, text: part.slice(1, -1) } : { code: false, text: part }
    )
}
