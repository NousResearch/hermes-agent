import { act, cleanup, fireEvent, render, waitFor } from '@testing-library/react'
import type * as Shiki from 'shiki'
import { afterEach, describe, expect, it, vi } from 'vitest'

const grammarStates = new Map<string, Shiki.GrammarState>()

const highlightNormally = async (
  code: string,
  _options?: { grammarState?: Shiki.GrammarState }
): Promise<{
  grammarState?: Shiki.GrammarState
  tokens: Array<Array<{ content: string; offset: number }>>
}> => {
  const grammarState = ({ code } as unknown) as Shiki.GrammarState

  grammarStates.set(code, grammarState)

  return {
    grammarState,
    tokens: code.split('\n').map(line => [{ content: line, offset: 0 }])
  }
}

const codeToTokens = vi.fn(highlightNormally)

vi.mock('shiki', async importOriginal => {
  const actual = await importOriginal<typeof Shiki>()

  return { ...actual, codeToTokens }
})

import { FileDiffPanel } from './diff-lines'

afterEach(() => {
  cleanup()
  codeToTokens.mockReset()
  codeToTokens.mockImplementation(highlightNormally)
  grammarStates.clear()
})

describe('FileDiffPanel windowed syntax highlighting', () => {
  it('tokenizes only the visible and overscan chunks, not the whole permitted diff', async () => {
    const diff = Array.from({ length: 1_000 }, (_, index) => `+const line${index} = ${index}`).join('\n')

    render(<FileDiffPanel diff={diff} path="file.ts" virtualized />)

    await waitFor(() => expect(codeToTokens).toHaveBeenCalledTimes(3))

    const highlightedChunks = codeToTokens.mock.calls.map(([code]) => code)
    const highlighted = highlightedChunks.join('\n')

    expect(highlightedChunks).toHaveLength(3)
    expect(highlightedChunks.every(code => code.split('\n').length <= 200)).toBe(true)
    expect(highlighted).toContain('const line0 = 0')
    expect(highlighted).toContain('const line599 = 599')
    expect(highlighted).not.toContain('const line600 = 600')
    expect(highlighted).not.toContain('const line999 = 999')
  })

  it('continues tokenizer grammar state across visible chunk boundaries', async () => {
    const diff = Array.from({ length: 600 }, (_, index) => `+const continuity${index} = ${index}`).join('\n')

    render(<FileDiffPanel diff={diff} path="continuity.ts" virtualized />)

    await waitFor(() => expect(codeToTokens).toHaveBeenCalledTimes(3))

    const [first, second, third] = codeToTokens.mock.calls

    expect(second?.[1]).toMatchObject({ grammarState: grammarStates.get(first?.[0] ?? '') })
    expect(third?.[1]).toMatchObject({ grammarState: grammarStates.get(second?.[0] ?? '') })
  })

  it('continues from the preceding cached boundary when the visible window moves', async () => {
    vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) =>
      window.setTimeout(() => callback(performance.now()), 0)
    )
    vi.stubGlobal('cancelAnimationFrame', (id: number) => window.clearTimeout(id))

    try {
      const diff = [
        '@@ -0,0 +1,1600 @@',
        ...Array.from({ length: 1_600 }, (_, index) => `+const scrolled${index} = ${index}`)
      ].join('\n')

      const { container } = render(<FileDiffPanel diff={diff} path="scrolled.ts" showLineNumbers />)

      await waitFor(() => expect(codeToTokens).toHaveBeenCalledTimes(3))

      const initialThirdCode = codeToTokens.mock.calls[2]?.[0] ?? ''
      const initialThirdState = grammarStates.get(initialThirdCode)
      const scroller = container.querySelector('[data-slot="file-diff-panel"] > div')

      expect(scroller).toBeInstanceOf(HTMLElement)
      Object.defineProperties(scroller, {
        clientHeight: { configurable: true, value: 800 },
        scrollTop: { configurable: true, value: 16_000, writable: true }
      })
      fireEvent.scroll(scroller as HTMLElement)

      await waitFor(() => expect(codeToTokens.mock.calls.length).toBeGreaterThan(3))

      const firstNewCall = codeToTokens.mock.calls[3]

      expect(firstNewCall?.[0]).toContain('const scrolled600 = 600')
      expect(firstNewCall?.[1]).toMatchObject({ grammarState: initialThirdState })
      expect(container.querySelector('[data-glass-opaque]')?.textContent).toMatch(/^401402/)

      await waitFor(() => {
        const row = [...container.querySelectorAll('span.block')].find(
          candidate => candidate.textContent === 'const scrolled600 = 600'
        )

        expect(row?.querySelector('span')).not.toBeNull()
      })
    } finally {
      vi.unstubAllGlobals()
    }
  })

  it('leaves a hard-jumped window plain when no preceding grammar boundary is cached', async () => {
    vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) =>
      window.setTimeout(() => callback(performance.now()), 0)
    )
    vi.stubGlobal('cancelAnimationFrame', (id: number) => window.clearTimeout(id))

    try {
      const diff = [
        '@@ -0,0 +1,2400 @@',
        ...Array.from({ length: 2_400 }, (_, index) => `+const jumped${index} = ${index}`)
      ].join('\n')

      const { container } = render(<FileDiffPanel diff={diff} path="jumped.ts" showLineNumbers />)

      await waitFor(() => expect(codeToTokens).toHaveBeenCalledTimes(3))
      const scroller = container.querySelector('[data-slot="file-diff-panel"] > div')

      expect(scroller).toBeInstanceOf(HTMLElement)
      Object.defineProperties(scroller, {
        clientHeight: { configurable: true, value: 800 },
        scrollTop: { configurable: true, value: 40_000, writable: true }
      })
      fireEvent.scroll(scroller as HTMLElement)

      await waitFor(() => expect(container.querySelector('[data-glass-opaque]')?.textContent).toMatch(/^16011602/))
      expect(codeToTokens).toHaveBeenCalledTimes(3)

      const visibleRow = [...container.querySelectorAll('span.block')].find(
        candidate => candidate.textContent === 'const jumped1600 = 1600'
      )

      expect(visibleRow?.querySelector('span')).toBeNull()
    } finally {
      vi.unstubAllGlobals()
    }
  })

  it('does not let a stale highlight overwrite a newer diff', async () => {
    let resolveOld!: (result: Awaited<ReturnType<typeof highlightNormally>>) => void
    let resolveNew!: (result: Awaited<ReturnType<typeof highlightNormally>>) => void
    const oldDiff = Array.from({ length: 100 }, (_, index) => `+const old${index} = ${index}`).join('\n')
    const newDiff = Array.from({ length: 100 }, (_, index) => `+const new${index} = ${index}`).join('\n')

    codeToTokens.mockImplementation(
      code =>
        new Promise(resolve => {
          if (code.includes('old0')) {
            resolveOld = resolve
          } else {
            resolveNew = resolve
          }
        })
    )

    const { container, rerender } = render(<FileDiffPanel diff={oldDiff} path="stale.ts" virtualized />)

    await waitFor(() => expect(resolveOld).toBeTypeOf('function'))
    rerender(<FileDiffPanel diff={newDiff} path="stale.ts" virtualized />)
    await waitFor(() => expect(resolveNew).toBeTypeOf('function'))

    resolveNew({ tokens: [[{ content: 'new highlight won', offset: 0 }]] })
    await waitFor(() => expect(container.textContent).toContain('new highlight won'))

    await act(async () => {
      resolveOld({ tokens: [[{ content: 'stale highlight won', offset: 0 }]] })
      await Promise.resolve()
    })

    expect(container.textContent).toContain('new highlight won')
    expect(container.textContent).not.toContain('stale highlight won')
  })
})
