import { describe, expect, it } from 'vitest'

import { deriveChangedFiles } from './changed-files'

const edit = (path: string, diff: string) => ({
  args: { path },
  result: { diff },
  toolName: 'patch',
  type: 'tool-call'
})

describe('deriveChangedFiles', () => {
  it('retains repeated tool diffs in edit order', () => {
    const first = '--- a/note.md\n+++ b/note.md\n@@ -1 +1 @@\n-old\n+middle'
    const second = '--- a/note.md\n+++ b/note.md\n@@ -1 +1 @@\n-middle\n+new'

    expect(deriveChangedFiles([edit('/workspace/note.md', first), edit('/workspace/note.md', second)])).toEqual([
      {
        added: 2,
        diff: `${first}\n@@ -1 +1 @@\n-middle\n+new`,
        name: 'note.md',
        path: '/workspace/note.md',
        removed: 2
      }
    ])
  })

  it('counts header-shaped content lines inside a hunk', () => {
    const diff = '--- a/options.txt\n+++ b/options.txt\n@@ -1 +1 @@\n--- option\n+++ option'

    expect(deriveChangedFiles([edit('/workspace/options.txt', diff)])).toEqual([
      {
        added: 1,
        diff,
        name: 'options.txt',
        path: '/workspace/options.txt',
        removed: 1
      }
    ])
  })
})
