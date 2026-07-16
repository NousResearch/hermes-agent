// Shiki HTML cache (the "no plain->color pop" layer). Contract:
//  1. First render of a fence: html=null (plain fallback) -> async fill.
//  2. Remount of the SAME fence: html available SYNCHRONOUSLY (initial state).
//  3. Unknown language falls back to 'text' (never throws).
//  4. Bounded LRU.
import { act, renderHook, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('shiki', () => ({
  bundledLanguages: { python: {}, typescript: {} },
  codeToHtml: vi.fn((code: string, opts: { lang: string }) =>
    Promise.resolve(`<pre class="shiki"><code>${opts.lang}:${code}</code></pre>`)
  )
}))

import { codeToHtml } from 'shiki'

import {
  clearShikiHtmlCache,
  MAX_HTML_ENTRIES,
  seedShikiHtmlCacheForTest,
  shikiHtmlCacheSize,
  useCachedShikiHtml
} from './shiki-html-cache'

describe('useCachedShikiHtml', () => {
  beforeEach(() => {
    clearShikiHtmlCache()
    vi.mocked(codeToHtml).mockClear()
  })

  it('first render is a miss (plain fallback), then fills async', async () => {
    const { result } = renderHook(() => useCachedShikiHtml('print(1)', 'python'))

    expect(result.current.html).toBeNull()
    await waitFor(() => expect(result.current.html).toContain('python:print(1)'))
    expect(shikiHtmlCacheSize()).toBe(1)
  })

  it('a remount of the same fence gets highlighted HTML synchronously', async () => {
    const first = renderHook(() => useCachedShikiHtml('x = 1', 'python'))

    await waitFor(() => expect(first.result.current.html).not.toBeNull())
    first.unmount()

    // Remount: initial state must be the cached HTML — no plain pass, no swap.
    const second = renderHook(() => useCachedShikiHtml('x = 1', 'python'))

    expect(second.result.current.html).toContain('python:x = 1')
    // And no second tokenize was issued for the cached content.
    expect(vi.mocked(codeToHtml)).toHaveBeenCalledTimes(1)
  })

  it('unknown languages fall back to text', async () => {
    const { result } = renderHook(() => useCachedShikiHtml('hello', 'made-up-lang'))

    await waitFor(() => expect(result.current.html).toContain('text:hello'))
  })

  it('different code strings are cached independently', async () => {
    const a = renderHook(() => useCachedShikiHtml('a', 'python'))
    const b = renderHook(() => useCachedShikiHtml('b', 'python'))

    await waitFor(() => {
      expect(a.result.current.html).toContain('python:a')
      expect(b.result.current.html).toContain('python:b')
    })
    expect(shikiHtmlCacheSize()).toBe(2)
  })

  it('tokenize failure stays null (plain code is the fail-open state)', async () => {
    vi.mocked(codeToHtml).mockRejectedValueOnce(new Error('grammar exploded'))

    const { result } = renderHook(() => useCachedShikiHtml('boom', 'python'))

    expect(result.current.html).toBeNull()
    // Give the rejection a tick to settle; html must remain null, no throw.
    await act(() => new Promise(resolve => setTimeout(resolve, 10)))
    expect(result.current.html).toBeNull()
  })

  it('bounds the cache with an LRU (oldest evicted past MAX_HTML_ENTRIES)', async () => {
    // Seed the cache to exactly the cap without driving the async hook N times.
    for (let i = 0; i < MAX_HTML_ENTRIES; i++) {
      seedShikiHtmlCacheForTest('python', `code-${i}`, `<pre>${i}</pre>`)
    }

    expect(shikiHtmlCacheSize()).toBe(MAX_HTML_ENTRIES)

    // One more unique fence evicts the oldest (code-0), size stays capped.
    seedShikiHtmlCacheForTest('python', 'code-overflow', '<pre>ovf</pre>')
    expect(shikiHtmlCacheSize()).toBe(MAX_HTML_ENTRIES)

    // The evicted fence is a MISS again (re-tokenizes); the newest is a HIT.
    vi.mocked(codeToHtml).mockClear()

    const evicted = renderHook(() => useCachedShikiHtml('code-0', 'python'))

    expect(evicted.result.current.html).toBeNull() // evicted -> miss -> plain pass
    await waitFor(() => expect(evicted.result.current.html).not.toBeNull())
    expect(vi.mocked(codeToHtml)).toHaveBeenCalledTimes(1) // had to re-tokenize

    // A still-resident fence is a synchronous HIT (no tokenize).
    vi.mocked(codeToHtml).mockClear()

    const resident = renderHook(() => useCachedShikiHtml('code-overflow', 'python'))

    expect(resident.result.current.html).toBe('<pre>ovf</pre>')
    expect(vi.mocked(codeToHtml)).not.toHaveBeenCalled()
  })
})
