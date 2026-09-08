import { act, cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

// Regression coverage for #93479: a failed dynamic import of the lazily-loaded
// Shiki diff module (packaged asar/asar.unpacked path mismatch, or any other
// fetch failure) rejects the `React.lazy()` promise. React.Suspense only
// covers the *pending* state, so the rejection throws past it to the nearest
// error boundary — which in production is the whole workspace `ContribBoundary`
// — and blanks the transcript instead of degrading to the plain colored diff.
//
// The mock resolves to a component that throws the fetch error during render —
// the same way React surfaces a rejected lazy payload (a rejected import
// re-throws at render time). Do NOT throw inside the factory itself: a
// throwing factory leaves rejected promises in the vitest mocker registry,
// and under CI load one escapes as an "unhandled error during the test run"
// attributed to whichever sibling test file the worker is running (#94415).
vi.mock('./syntax-diff', () => ({
  default: () => {
    throw new Error(
      'Failed to fetch dynamically imported module: file:///Hermes.app/Contents/Resources/app.asar/dist/assets/syntax-diff-Bo0962zh.js'
    )
  }
}))

import { deriveChangedFiles } from '@/components/assistant-ui/thread/changed-files'
import { ErrorBoundary } from '@/components/error-boundary'

import { FileDiffPanel } from './diff-lines'

afterEach(cleanup)

const DIFF = [
  'diff --git a/file.ts b/file.ts',
  '--- a/file.ts',
  '+++ b/file.ts',
  '@@ -1,2 +1,2 @@',
  ' const a = 1',
  '-const b = 2',
  '+const b = 3'
].join('\n')

const WORKSPACE_FALLBACK_TEXT = 'workspace failed to render'

// The failure surfaces only from console.error noise, not from the assertion.
function renderQuietly(node: Parameters<typeof render>[0]) {
  const spy = vi.spyOn(console, 'error').mockImplementation(() => undefined)

  try {
    return render(node)
  } finally {
    spy.mockRestore()
  }
}

describe('FileDiffPanel survives a failed lazy syntax-diff chunk', () => {
  it('degrades to the plain colored diff instead of taking down the surrounding boundary', async () => {
    const { container } = renderQuietly(
      <ErrorBoundary fallback={() => <div>{WORKSPACE_FALLBACK_TEXT}</div>} label="workspace">
        <FileDiffPanel diff={DIFF} path="file.ts" />
      </ErrorBoundary>
    )

    // The rejection settles a tick after the initial Suspense-pending render
    // (which coincidentally shows the same plain text already) — give it
    // real time to propagate before asserting nothing regressed.
    await act(() => new Promise(resolve => setTimeout(resolve, 300)))

    expect(container.textContent).toContain('const a = 1')
    expect(container.textContent).toContain('const b = 2')
    expect(container.textContent).toContain('const b = 3')
    expect(container.textContent).not.toContain(WORKSPACE_FALLBACK_TEXT)
  })

  it('renders repeated tool diffs as hunks without exposing later file headers', async () => {
    const first = '--- a/note.md\n+++ b/note.md\n@@ -1 +1 @@\n-old\n+middle'
    const second = '--- a/note.md\n+++ b/note.md\n@@ -1 +1 @@\n-middle\n+new'

    const edit = (diff: string) => ({
      args: { path: '/workspace/note.md' },
      result: { diff },
      toolName: 'patch',
      type: 'tool-call'
    })

    const [changed] = deriveChangedFiles([edit(first), edit(second)])

    const { container } = renderQuietly(<FileDiffPanel diff={changed!.diff} path="note.md" />)

    await act(() => new Promise(resolve => setTimeout(resolve, 300)))

    expect(container.textContent).toContain('old')
    expect(container.textContent).toContain('middle')
    expect(container.textContent).toContain('new')
    expect(container.textContent).not.toContain('--- a/note.md')
    expect(container.textContent).not.toContain('+++ b/note.md')
  })

  it('keeps legitimate changed lines that resemble file headers', async () => {
    const optionDiff = '--- a/options.txt\n+++ b/options.txt\n@@ -1 +1 @@\n--- option\n+++ option'
    const { container } = renderQuietly(<FileDiffPanel diff={optionDiff} path="options.txt" />)

    await act(() => new Promise(resolve => setTimeout(resolve, 300)))

    const lines = Array.from(container.querySelectorAll('span'))
    const removed = lines.find(line => line.textContent === '-- option')
    const added = lines.find(line => line.textContent === '++ option')

    expect(removed?.className).toContain('ui-diff-remove')
    expect(added?.className).toContain('ui-diff-add')
    expect(lines.some(line => line.textContent === '--- option')).toBe(false)
    expect(lines.some(line => line.textContent === '+++ option')).toBe(false)
  })

  it('does not render concatenated multi-file headers as changed content', async () => {
    const multiFileDiff = [
      '--- a/one.txt',
      '+++ b/one.txt',
      '@@ -1 +1 @@',
      '-old one',
      '+new one',
      '--- a/two.txt',
      '+++ b/two.txt',
      '@@ -1 +1 @@',
      '-old two',
      '+new two'
    ].join('\n')
    const { container } = renderQuietly(<FileDiffPanel diff={multiFileDiff} path="combined.patch" />)

    await act(() => new Promise(resolve => setTimeout(resolve, 300)))

    expect(container.textContent).toContain('old one')
    expect(container.textContent).toContain('new two')
    expect(container.textContent).not.toContain('a/two.txt')
    expect(container.textContent).not.toContain('b/two.txt')
  })
})
