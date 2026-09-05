// Pure derivation for the assistant message's "N files changed" card: fold a
// turn's file-edit tool parts into one row per file. No React/DOM.

import {
  countDiffLineStats,
  fileEditBasename,
  fileEditPath,
  inlineDiffFromResult,
  isFileEditTool,
  parseMaybeObject
} from '@/components/assistant-ui/tool/fallback-model'

export interface ChangedFile {
  added: number
  /** Tool-reported unified diffs, retained in edit order for non-Git review. */
  diff: string
  /** Basename, for the row label. */
  name: string
  /** Path exactly as the tool reported it (absolute or repo-relative). */
  path: string
  removed: number
}

interface ChangedFilePart {
  args?: unknown
  result?: unknown
  toolName?: unknown
  type?: unknown
}

function withoutLeadingDiffPreamble(diff: string): string {
  const lines = diff.split('\n')
  const firstHunk = lines.findIndex(line => line.startsWith('@@'))

  return firstHunk > 0 ? lines.slice(firstHunk).join('\n') : diff
}

/**
 * One row per file the turn edited, in first-touched order, with the +/- of
 * every edit to that file summed. Only landed edits with a diff count: a call
 * still running has no result, and a failed one changed nothing.
 */
export function deriveChangedFiles(parts: readonly unknown[]): ChangedFile[] {
  const byPath = new Map<string, ChangedFile>()

  for (const raw of parts) {
    const part = (raw ?? {}) as ChangedFilePart

    if (part.type !== 'tool-call' || typeof part.toolName !== 'string' || !isFileEditTool(part.toolName)) {
      continue
    }

    const result = parseMaybeObject(part.result)
    const diff = inlineDiffFromResult(result)

    if (!diff) {
      continue
    }

    const path = fileEditPath(parseMaybeObject(part.args), result)

    if (!path) {
      continue
    }

    const stats = countDiffLineStats(diff)
    const existing = byPath.get(path)

    if (existing) {
      existing.added += stats.added
      existing.diff = `${existing.diff}\n${withoutLeadingDiffPreamble(diff)}`
      existing.removed += stats.removed
    } else {
      byPath.set(path, { added: stats.added, diff, name: fileEditBasename(path), path, removed: stats.removed })
    }
  }

  return [...byPath.values()]
}
