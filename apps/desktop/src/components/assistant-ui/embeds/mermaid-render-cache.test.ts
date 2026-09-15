import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// The cache module must never pull the real mermaid runtime into the test
// bundle: tests drive the deferred-import seam with a stubbed renderer.
vi.mock('mermaid', () => ({
  default: {
    initialize: vi.fn(),
    render: vi.fn()
  }
}))

import { default as mermaidStub } from 'mermaid'

import {
  cachedMermaidSvgCount,
  nextPaint,
  renderMermaidSvg,
  resetMermaidRenderCacheForTests
} from './mermaid-render-cache'

const render = vi.mocked(mermaidStub.render)
const initialize = vi.mocked(mermaidStub.initialize)

beforeEach(() => {
  resetMermaidRenderCacheForTests()
  vi.clearAllMocks()
})

afterEach(() => {
  vi.restoreAllMocks()
})

const okSvg = (tag: string) => `<svg data-tag="${tag}"></svg>`

describe('renderMermaidSvg', () => {
  it('renders uncached source and caches the completed SVG', async () => {
    render.mockResolvedValueOnce({ svg: okSvg('one') } as never)

    const first = await renderMermaidSvg('graph TD;A-->B', 'default')

    expect(first.svg).toBe(okSvg('one'))
    expect(render).toHaveBeenCalledTimes(1)
    expect(cachedMermaidSvgCount()).toBe(1)
  })

  it('reuses the cached SVG on remount instead of re-rendering', async () => {
    render.mockResolvedValueOnce({ svg: okSvg('one') } as never)
    await renderMermaidSvg('graph TD;A-->B', 'default')
    render.mockClear()

    const second = await renderMermaidSvg('graph TD;A-->B', 'default')

    expect(second.svg).toBe(okSvg('one'))
    expect(render).not.toHaveBeenCalled()
  })

  it('shares one in-flight render between identical concurrent requests', async () => {
    let release: (svg: string) => void = () => {}

    const gate = new Promise<string>((resolve) => {
      release = resolve
    })

    render.mockImplementationOnce(async () => ({ svg: await gate } as never))

    const first = renderMermaidSvg('graph TD;A-->B', 'dark')
    const second = renderMermaidSvg('graph TD;A-->B', 'dark')
    release(okSvg('shared'))

    await expect(first).resolves.toEqual({ svg: okSvg('shared') })
    await expect(second).resolves.toEqual({ svg: okSvg('shared') })
    expect(render).toHaveBeenCalledTimes(1)
  })

  it('caches light and dark themes separately', async () => {
    render.mockResolvedValueOnce({ svg: okSvg('light') } as never)
    render.mockResolvedValueOnce({ svg: okSvg('dark') } as never)

    await renderMermaidSvg('graph TD;A-->B', 'default')
    await renderMermaidSvg('graph TD;A-->B', 'dark')

    expect(render).toHaveBeenCalledTimes(2)

    await expect(renderMermaidSvg('graph TD;A-->B', 'default')).resolves.toEqual({
      svg: okSvg('light')
    })
    await expect(renderMermaidSvg('graph TD;A-->B', 'dark')).resolves.toEqual({
      svg: okSvg('dark')
    })
  })

  it('does not poison the cache when a render fails', async () => {
    render.mockRejectedValueOnce(new Error('parse error') as never)

    await expect(renderMermaidSvg('not a diagram', 'default')).rejects.toThrow(
      'parse error'
    )
    expect(cachedMermaidSvgCount()).toBe(0)

    render.mockResolvedValueOnce({ svg: okSvg('retry') } as never)
    await expect(renderMermaidSvg('not a diagram', 'default')).resolves.toEqual({
      svg: okSvg('retry')
    })
  })

  it('serializes uncached renders instead of starting them together', async () => {
    const order: string[] = []

    const slow = (tag: string, ms: number) => {
      let release!: (svg: string) => void

      const gate = new Promise<string>((resolve) => {
        release = resolve
      })

      render.mockImplementationOnce(async () => {
        order.push(`start:${tag}`)
        const svg = await gate
        order.push(`end:${tag}`)

        return { svg } as never
      })
      // Resolve on the next microtask unless explicitly released.
      queueMicrotask(() => release(okSvg(tag)))
      void ms

      return gate
    }

    slow('a', 0)
    slow('b', 0)

    const first = renderMermaidSvg('graph TD;A-->A', 'default')
    const second = renderMermaidSvg('graph TD;B-->B', 'default')
    await Promise.all([first, second])

    expect(order).toEqual([
      'start:a',
      'end:a',
      'start:b',
      'end:b'
    ])
  })

  it('evicts the oldest entry once the cache passes its bound', async () => {
    for (let i = 0; i < 66; i++) {
      render.mockResolvedValueOnce({ svg: okSvg(`svg-${i}`) } as never)
      await renderMermaidSvg(`graph TD;N${i}-->N${i}`, 'default')
    }

    expect(cachedMermaidSvgCount()).toBeLessThanOrEqual(64)

    render.mockClear()
    render.mockResolvedValue({ svg: okSvg('re-rendered') } as never)
    const oldest = await renderMermaidSvg('graph TD;N0-->N0', 'default')
    const newest = await renderMermaidSvg('graph TD;N65-->N65', 'default')
    expect(oldest.svg).not.toBe(okSvg('svg-0'))
    expect(newest.svg).toBe(okSvg('svg-65'))
    // Only the evicted oldest entry re-rendered; the newest survived.
    expect(render).toHaveBeenCalledTimes(1)
  })
})

describe('nextPaint', () => {
  it('resolves without requestAnimationFrame (node fallback)', async () => {
    const raf = globalThis.requestAnimationFrame
    // @ts-expect-error exercise the timer fallback branch
    delete globalThis.requestAnimationFrame

    try {
      await expect(nextPaint()).resolves.toBeUndefined()
    } finally {
      if (raf) {
        globalThis.requestAnimationFrame = raf
      }
    }
  })
})
