/** A unified diff split into its file header and its `@@` hunks, each hunk kept
 *  verbatim (header line, `+`/`-`/` ` rows, `\ No newline` markers). */
export interface SplitDiff {
  header: string
  hunks: string[]
}

export function splitDiffHunks(diff: string): SplitDiff {
  const lines = String(diff ?? '').split('\n')

  if (lines.at(-1) === '') {
    lines.pop()
  }

  const first = lines.findIndex(line => line.startsWith('@@'))

  if (first === -1) {
    return { header: '', hunks: [] }
  }

  const hunks: string[] = []
  let current: string[] = []

  for (const line of lines.slice(first)) {
    if (line.startsWith('@@') && current.length > 0) {
      hunks.push(current.join('\n'))
      current = []
    }

    current.push(line)
  }

  hunks.push(current.join('\n'))

  return { header: lines.slice(0, first).join('\n'), hunks }
}

/** The patch `git apply` takes for ONE hunk of `diff`, or null when `hunk` is
 *  not one of its hunks verbatim (the file changed since it was shown). */
export function singleHunkPatch(diff: string, hunk: string): null | string {
  const { header, hunks } = splitDiffHunks(diff)

  if (!header.includes('\n+++ ') || !hunks.includes(hunk)) {
    return null
  }

  return `${header}\n${hunk}\n`
}
