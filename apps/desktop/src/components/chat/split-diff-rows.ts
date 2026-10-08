import type { DiffLine } from './diff-lines'

export interface SplitDiffLine extends DiffLine {
  index: number
}

export interface SplitDiffRow {
  before?: SplitDiffLine
  after?: SplitDiffLine
}

/** Pair each replacement run without pairing across context or hunk boundaries. */
export function pairDiffLines(lines: DiffLine[]): SplitDiffRow[] {
  const rows: SplitDiffRow[] = []
  let cursor = 0

  while (cursor < lines.length) {
    const line = { ...lines[cursor], index: cursor }

    if (line.kind === 'context') {
      rows.push({ before: line, after: line })
      cursor += 1

      continue
    }

    const before: SplitDiffLine[] = []
    const after: SplitDiffLine[] = []

    while (cursor < lines.length && lines[cursor].kind === 'remove') {
      before.push({ ...lines[cursor], index: cursor++ })
    }

    while (cursor < lines.length && lines[cursor].kind === 'add') {
      after.push({ ...lines[cursor], index: cursor++ })
    }

    for (let offset = 0; offset < Math.max(before.length, after.length); offset += 1) {
      rows.push({ before: before[offset], after: after[offset] })
    }
  }

  return rows
}
