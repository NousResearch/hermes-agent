import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

/**
 * Perf regression guard for #95595: switching to a warm session remounts the
 * incoming transcript, and every mounted code block used to re-tokenize from
 * scratch on the main thread. The content-keyed cache must make a remount of
 * an unchanged block a cache hit — ZERO highlighter calls.
 *
 * shiki itself is mocked (jsdom cannot run the oniguruma wasm engine); the
 * mock counts `codeToHtml` invocations, which is the cost we are guarding.
 */
const { codeToHtml } = vi.hoisted(() => ({
  codeToHtml: vi.fn((code: string) => {
    const escaped = String(code).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')

    return `<pre class="shiki"><code>${escaped}</code></pre>`
  })
}))

vi.mock('shiki', () => ({
  bundledLanguages: { typescript: 'typescript-loader', text: 'text-loader' },
  getSingletonHighlighter: vi.fn(async () => ({
    codeToHtml: (code: string) => codeToHtml(code),
    getLoadedLanguages: () => ['text', 'typescript'],
    loadLanguage: vi.fn(async () => undefined)
  }))
}))

vi.mock('shiki/engine/oniguruma', () => ({
  createOnigurumaEngine: vi.fn(() => ({}) as never)
}))

import CachedShikiBlock from '@/components/chat/shiki-block'
import { highlightCache } from '@/components/chat/shiki-highlight-cache'

const TS_BLOCK = { language: 'typescript', code: 'const answer: number = 42\n' }

async function waitForHighlighted(): Promise<void> {
  await screen.findByTestId('shiki-container', undefined, { timeout: 2_000 })
}

beforeEach(() => {
  codeToHtml.mockClear()
  highlightCache.clear()
})

afterEach(() => {
  cleanup()
})

describe('CachedShikiBlock (warm-switch perf guard)', () => {
  it('highlights on first mount and reuses the cached HTML on remount', async () => {
    const { unmount } = render(<CachedShikiBlock {...TS_BLOCK} />)
    await waitForHighlighted()

    expect(codeToHtml).toHaveBeenCalledTimes(1)
    expect(screen.getByTestId('shiki-container').innerHTML).toContain('const answer')

    // Warm session switch: the row unmounts and the SAME block mounts again.
    unmount()
    render(<CachedShikiBlock {...TS_BLOCK} />)
    await waitForHighlighted()

    // The guard: remounting an unchanged block must NOT re-tokenize.
    expect(codeToHtml).toHaveBeenCalledTimes(1)
  })

  it('re-highlights a block whose code changed (cache miss)', async () => {
    const { unmount } = render(<CachedShikiBlock {...TS_BLOCK} />)
    await waitForHighlighted()

    unmount()
    render(<CachedShikiBlock code="const other = true\n" language="typescript" />)
    await waitForHighlighted()

    expect(codeToHtml).toHaveBeenCalledTimes(2)
  })

  it('keeps blocks independent: N blocks highlight N times across two mounts', async () => {
    const { unmount } = render(
      <>
        <CachedShikiBlock {...TS_BLOCK} />
        <CachedShikiBlock code="function two(): void {}\n" language="typescript" />
      </>
    )

    await screen.findAllByTestId('shiki-container', undefined, { timeout: 2_000 })

    expect(codeToHtml).toHaveBeenCalledTimes(2)

    unmount()
    render(
      <>
        <CachedShikiBlock {...TS_BLOCK} />
        <CachedShikiBlock code="function two(): void {}\n" language="typescript" />
      </>
    )
    await screen.findAllByTestId('shiki-container', undefined, { timeout: 2_000 })

    // Two mounts of the same two blocks: exactly two tokenizations total.
    expect(codeToHtml).toHaveBeenCalledTimes(2)
  })

  it('does not cache a failed highlight, so a retry can succeed', async () => {
    codeToHtml.mockRejectedValueOnce(new Error('boom'))

    const { unmount } = render(<CachedShikiBlock {...TS_BLOCK} />)
    await waitForHighlighted()

    // The failure degrades to escaped plain text (and is NOT cached).
    expect(screen.getByTestId('shiki-container').innerHTML).toContain('const answer')
    expect(codeToHtml).toHaveBeenCalledTimes(1)

    unmount()
    render(<CachedShikiBlock {...TS_BLOCK} />)
    await waitForHighlighted()

    // Second mount tries the highlighter again instead of serving stale HTML.
    expect(codeToHtml).toHaveBeenCalledTimes(2)
  })
})
