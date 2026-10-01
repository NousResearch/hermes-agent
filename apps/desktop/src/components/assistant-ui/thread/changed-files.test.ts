import { describe, expect, it } from 'vitest'

import { deriveChangedFiles } from '@/components/assistant-ui/thread/changed-files'

// Shaped like the parts the transcript actually carries: a `patch` tool-call
// with its diff in the gateway's display hint.
function patchPart(path: string, diff: string) {
  return { args: { path }, toolName: 'patch', toolResultMetadata: { inline_diff: diff }, type: 'tool-call' }
}

const EDIT_1 = '--- a/a.ts\n+++ b/a.ts\n@@ -1 +1,2 @@\n-old\n+new\n+more'
const EDIT_2 = '--- a/a.ts\n+++ b/a.ts\n@@ -9 +9 @@\n-x\n+y'
const OTHER = '--- a/b.ts\n+++ b/b.ts\n@@ -1 +1 @@\n-1\n+2'

describe('deriveChangedFiles', () => {
  it('carries every edit diff of a file, in order, so the review pane can replay them', () => {
    const files = deriveChangedFiles([
      patchPart('/repo/a.ts', EDIT_1),
      patchPart('/repo/b.ts', OTHER),
      patchPart('/repo/a.ts', EDIT_2)
    ])

    // First-touched order, one row per file.
    expect(files.map(f => f.path)).toEqual(['/repo/a.ts', '/repo/b.ts'])
    expect(files[0]?.diffs).toEqual([EDIT_1, EDIT_2])
    expect(files[1]?.diffs).toEqual([OTHER])
  })

  it('keeps the row +/- equal to the diffs it carries', () => {
    const [file] = deriveChangedFiles([patchPart('/repo/a.ts', EDIT_1), patchPart('/repo/a.ts', EDIT_2)])

    expect(file?.added).toBe(3)
    expect(file?.removed).toBe(2)
  })
})
