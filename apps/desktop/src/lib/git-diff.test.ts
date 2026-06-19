import { describe, expect, it } from 'vitest'

import { buildHunkPatch, diffStats, parseUnifiedDiff } from './git-diff'

const SAMPLE = `diff --git a/file.txt b/file.txt
index 0000001..0000002 100644
--- a/file.txt
+++ b/file.txt
@@ -1,4 +1,4 @@
 line one
-line two
+line TWO
 line three
 line four
@@ -10,2 +10,3 @@
 ten
+inserted
 eleven`

describe('parseUnifiedDiff', () => {
  it('extracts file paths and hunks', () => {
    const d = parseUnifiedDiff(SAMPLE)
    expect(d.oldPath).toBe('file.txt')
    expect(d.newPath).toBe('file.txt')
    expect(d.binary).toBe(false)
    expect(d.hunks).toHaveLength(2)
  })

  it('parses hunk headers into line ranges', () => {
    const d = parseUnifiedDiff(SAMPLE)
    expect(d.hunks[0].oldStart).toBe(1)
    expect(d.hunks[0].oldLines).toBe(4)
    expect(d.hunks[0].newStart).toBe(1)
    expect(d.hunks[0].newLines).toBe(4)
  })

  it('classifies add / del / context lines with line numbers', () => {
    const d = parseUnifiedDiff(SAMPLE)
    const h = d.hunks[0]
    const del = h.lines.find(l => l.kind === 'del')
    const add = h.lines.find(l => l.kind === 'add')
    expect(del?.text).toBe('line two')
    expect(del?.oldLine).toBe(2)
    expect(del?.newLine).toBeNull()
    expect(add?.text).toBe('line TWO')
    expect(add?.newLine).toBe(2)
    expect(add?.oldLine).toBeNull()
  })

  it('handles a single-line hunk count default (@@ -a +c @@)', () => {
    const d = parseUnifiedDiff('--- a/x\n+++ b/x\n@@ -5 +5 @@\n-old\n+new')
    expect(d.hunks[0].oldLines).toBe(1)
    expect(d.hunks[0].newLines).toBe(1)
  })

  it('detects binary files', () => {
    const d = parseUnifiedDiff('diff --git a/img.png b/img.png\nBinary files a/img.png and b/img.png differ')
    expect(d.binary).toBe(true)
  })

  it('returns empty for empty input', () => {
    const d = parseUnifiedDiff('')
    expect(d.hunks).toHaveLength(0)
  })
})

describe('diffStats', () => {
  it('counts additions and deletions', () => {
    const d = parseUnifiedDiff(SAMPLE)
    const s = diffStats(d)
    expect(s.additions).toBe(2) // line TWO + inserted
    expect(s.deletions).toBe(1) // line two
  })
})

describe('buildHunkPatch', () => {
  it('reconstructs an apply-able single-hunk patch with the file header', () => {
    const d = parseUnifiedDiff(SAMPLE)
    const patch = buildHunkPatch(d, d.hunks[0])
    expect(patch).toContain('diff --git a/file.txt b/file.txt')
    expect(patch).toContain('--- a/file.txt')
    expect(patch).toContain('+++ b/file.txt')
    expect(patch).toContain('@@ -1,4 +1,4 @@')
    expect(patch).toContain('-line two')
    expect(patch).toContain('+line TWO')
    // Only the first hunk — the second hunk's content must be absent.
    expect(patch).not.toContain('inserted')
    // Trailing newline so git apply is happy.
    expect(patch.endsWith('\n')).toBe(true)
  })

  it('preserves context lines with a leading space', () => {
    const d = parseUnifiedDiff(SAMPLE)
    const patch = buildHunkPatch(d, d.hunks[0])
    expect(patch).toContain(' line one')
    expect(patch).toContain(' line three')
  })
})
