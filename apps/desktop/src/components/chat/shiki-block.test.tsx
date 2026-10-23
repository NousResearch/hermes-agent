import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
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
  bundledLanguages: { python: 'python-loader', text: 'text-loader', typescript: 'typescript-loader' },
  getSingletonHighlighter: vi.fn(async () => ({
    codeToHtml: (code: string) => codeToHtml(code),
    getLoadedLanguages: () => ['python', 'text', 'typescript'],
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

/**
 * jsdom has no layout, so the shim in vitest.setup.ts never delivers an
 * intersection and a block that IS on screen would defer forever. Model a
 * visible block: report intersection as soon as observing starts. Tests that
 * exercise the deferred path install their own silent observer over this.
 */
function stubVisibleIntersection(): void {
  vi.stubGlobal(
    'IntersectionObserver',
    class {
      private readonly callback: IntersectionObserverCallback

      constructor(callback: IntersectionObserverCallback) {
        this.callback = callback
      }

      disconnect() {}

      observe(target: Element) {
        this.callback(
          [{ intersectionRatio: 1, isIntersecting: true, target } as IntersectionObserverEntry],
          this as unknown as IntersectionObserver
        )
      }

      takeRecords(): IntersectionObserverEntry[] {
        return []
      }

      unobserve() {}
    }
  )
}

beforeEach(() => {
  codeToHtml.mockClear()
  highlightCache.clear()
  stubVisibleIntersection()
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe('CachedShikiBlock (warm-switch perf guard)', () => {
  it('keeps escaped code visible while a cache miss waits for highlighting', () => {
    const code = '<script>not executable</script>\nconst two = 2\n'
    const { container } = render(<CachedShikiBlock code={code} language="typescript" />)
    expect(container.querySelector('pre code')?.textContent).toBe(code)
    expect(container.querySelector('script')).toBeNull()
    expect(codeToHtml).not.toHaveBeenCalled()
  })

  it('defers tokenization for an offscreen cache miss', async () => {
    vi.stubGlobal(
      'IntersectionObserver',
      class {
        disconnect() {}
        observe() {}
        unobserve() {}
      }
    )

    render(<CachedShikiBlock {...TS_BLOCK} />)

    await new Promise(resolve => window.setTimeout(resolve, 250))

    expect(screen.getByTestId('shiki-placeholder').textContent).toContain('const answer')
    expect(codeToHtml).not.toHaveBeenCalled()
  })

  it('highlights once the block enters the viewport margin', async () => {
    let deliver!: (entries: IntersectionObserverEntry[]) => void
    vi.stubGlobal(
      'IntersectionObserver',
      class {
        constructor(callback: IntersectionObserverCallback) {
          deliver = entries => callback(entries, this as unknown as IntersectionObserver)
        }
        disconnect() {}
        observe() {}
        unobserve() {}
      }
    )

    render(<CachedShikiBlock {...TS_BLOCK} />)
    expect(codeToHtml).not.toHaveBeenCalled()

    act(() => deliver([{ intersectionRatio: 1, isIntersecting: true } as IntersectionObserverEntry]))
    await waitForHighlighted()

    expect(codeToHtml).toHaveBeenCalledTimes(1)
  })

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

  /**
   * Warm switch → live edit. A block that MOUNTS on a cache hit paints from the
   * cache and never arms its viewport observer (there is nothing to wait for),
   * so when the same mounted row then receives props that are NOT cached the
   * stale HTML must not be allowed to win: the observer has to arm and the new
   * content has to tokenize. This is the regression the review caught.
   */
  it('re-highlights in place when a cache-hit block gets new uncached code', async () => {
    const warm = render(<CachedShikiBlock {...TS_BLOCK} />)
    await waitForHighlighted()
    expect(codeToHtml).toHaveBeenCalledTimes(1)
    warm.unmount()

    // Cache-hit initial mount: no highlighter work, no viewport wait.
    const { rerender } = render(<CachedShikiBlock {...TS_BLOCK} />)
    expect(screen.getByTestId('shiki-container').innerHTML).toContain('const answer')

    rerender(<CachedShikiBlock code="const edited = 7\n" language="typescript" />)

    await waitFor(() => expect(codeToHtml).toHaveBeenCalledTimes(2))
    expect(screen.getByTestId('shiki-container').innerHTML).toContain('const edited')
    expect(screen.getByTestId('shiki-container').innerHTML).not.toContain('const answer')
  })

  it('re-highlights in place when a cache-hit block changes code and language', async () => {
    const warm = render(<CachedShikiBlock {...TS_BLOCK} />)
    await waitForHighlighted()
    expect(codeToHtml).toHaveBeenCalledTimes(1)
    warm.unmount()

    const { rerender } = render(<CachedShikiBlock {...TS_BLOCK} />)
    expect(screen.getByTestId('shiki-container').innerHTML).toContain('const answer')

    rerender(<CachedShikiBlock code="print('hi')\n" language="python" />)

    await waitFor(() => expect(codeToHtml).toHaveBeenCalledTimes(2))
    expect(screen.getByTestId('shiki-container').innerHTML).toContain("print('hi')")
    expect(screen.getByTestId('shiki-container').innerHTML).not.toContain('const answer')
  })

  it('paints an already-cached entry when the props change to cached content', async () => {
    const other = { language: 'python', code: "print('cached')\n" }

    const both = render(
      <>
        <CachedShikiBlock {...TS_BLOCK} />
        <CachedShikiBlock {...other} />
      </>
    )

    await screen.findAllByTestId('shiki-container', undefined, { timeout: 2_000 })
    expect(codeToHtml).toHaveBeenCalledTimes(2)
    both.unmount()

    // Cache-hit mount of the first block, then re-props to the other cached one.
    const { rerender } = render(<CachedShikiBlock {...TS_BLOCK} />)
    expect(screen.getByTestId('shiki-container').innerHTML).toContain('const answer')

    rerender(<CachedShikiBlock {...other} />)

    // Already highlighted earlier this session: swap the HTML, tokenize nothing.
    expect(screen.getByTestId('shiki-container').innerHTML).toContain("print('cached')")
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
