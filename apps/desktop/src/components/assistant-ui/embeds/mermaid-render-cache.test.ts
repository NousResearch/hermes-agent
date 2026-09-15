import { describe, expect, it, vi } from 'vitest'

import { createMermaidRenderCache, createRetryableLoader } from './mermaid-render-cache'

const flushTasks = async () => {
  await Promise.resolve()
  await Promise.resolve()
}

describe('createMermaidRenderCache', () => {
  it('shares in-flight work and reuses the completed SVG', async () => {
    let release!: (svg: string) => void
    const render = vi.fn(() => new Promise<string>(resolve => (release = resolve)))
    const defer = vi.fn(async () => undefined)
    const cache = createMermaidRenderCache({ defer, maxEntries: 8, render })

    const first = cache.render('graph TD; A-->B', 'dark')
    const concurrent = cache.render('graph TD; A-->B', 'dark')
    await flushTasks()

    expect(first).toBe(concurrent)
    expect(defer).toHaveBeenCalledTimes(1)
    expect(render).toHaveBeenCalledTimes(1)

    release('<svg>dark</svg>')
    await expect(first).resolves.toBe('<svg>dark</svg>')

    const cached = cache.render('graph TD; A-->B', 'dark')
    expect(cached).toBe(first)
    await expect(cached).resolves.toBe('<svg>dark</svg>')
    expect(render).toHaveBeenCalledTimes(1)
  })

  it('caches themes separately and serializes cache misses', async () => {
    const releases: Array<(svg: string) => void> = []
    let active = 0
    let maxActive = 0

    const render = vi.fn(
      (_code: string, theme: 'dark' | 'default') =>
        new Promise<string>(resolve => {
          active += 1
          maxActive = Math.max(maxActive, active)
          releases.push(svg => {
            active -= 1
            resolve(`${svg}-${theme}`)
          })
        })
    )

    const cache = createMermaidRenderCache({
      defer: async () => undefined,
      maxEntries: 8,
      render,
      yieldToMainThread: async () => undefined
    })

    const dark = cache.render('sequenceDiagram\nA->>B: hello', 'dark')
    const light = cache.render('sequenceDiagram\nA->>B: hello', 'default')
    await flushTasks()

    expect(render).toHaveBeenCalledTimes(1)
    expect(maxActive).toBe(1)

    releases[0]('<svg>first</svg>')
    await dark
    await flushTasks()

    expect(render).toHaveBeenCalledTimes(2)
    expect(maxActive).toBe(1)

    releases[1]('<svg>second</svg>')
    await expect(light).resolves.toBe('<svg>second</svg>-default')
  })

  it('removes failed work so a later mount can retry', async () => {
    const render = vi
      .fn<(code: string, theme: 'dark' | 'default') => Promise<string>>()
      .mockRejectedValueOnce(new Error('temporary failure'))
      .mockResolvedValueOnce('<svg>retry</svg>')

    const cache = createMermaidRenderCache({ defer: async () => undefined, maxEntries: 8, render })

    await expect(cache.render('flowchart LR; A-->B', 'dark')).rejects.toThrow('temporary failure')
    await expect(cache.render('flowchart LR; A-->B', 'dark')).resolves.toBe('<svg>retry</svg>')
    expect(render).toHaveBeenCalledTimes(2)
  })

  it('evicts the least recently used result at the configured bound', async () => {
    const render = vi.fn(async (code: string, theme: 'dark' | 'default') => `<svg>${theme}:${code}</svg>`)
    const cache = createMermaidRenderCache({ defer: async () => undefined, maxEntries: 2, render })

    await cache.render('A', 'dark')
    await cache.render('B', 'dark')
    await cache.render('A', 'dark')
    await cache.render('C', 'dark')
    await cache.render('B', 'dark')

    expect(render.mock.calls.map(([code]) => code)).toEqual(['A', 'B', 'C', 'B'])
  })

  it('keeps in-flight work deduplicated when the completed-cache bound is exceeded', async () => {
    const render = vi.fn(async (code: string) => `<svg>${code}</svg>`)
    const cache = createMermaidRenderCache({ defer: async () => undefined, maxEntries: 2, render })

    const firstA = cache.render('A', 'dark')

    const all = Promise.all([firstA, cache.render('B', 'dark'), cache.render('C', 'dark'), cache.render('A', 'dark')])

    await all
    expect(render.mock.calls.map(([code]) => code)).toEqual(['A', 'B', 'C'])
  })

  it('admits misses in parallel but keeps Mermaid rendering serialized', async () => {
    const admissions: Array<() => void> = []

    const defer = vi.fn(
      () =>
        new Promise<void>(resolve => {
          admissions.push(resolve)
        })
    )

    let active = 0
    let maxActive = 0

    const render = vi.fn(async (code: string) => {
      active += 1
      maxActive = Math.max(maxActive, active)
      await Promise.resolve()
      active -= 1

      return `<svg>${code}</svg>`
    })

    const cache = createMermaidRenderCache({ defer, maxEntries: 8, render })

    const pending = [cache.render('A', 'dark'), cache.render('B', 'dark'), cache.render('C', 'dark')]
    await flushTasks()

    expect(defer).toHaveBeenCalledTimes(3)
    expect(render).not.toHaveBeenCalled()

    admissions.forEach(admit => admit())
    await Promise.all(pending)

    expect(render.mock.calls.map(([code]) => code)).toEqual(['A', 'B', 'C'])
    expect(maxActive).toBe(1)
  })

  it('yields a task between CPU-heavy Mermaid renders', async () => {
    const releaseYields: Array<() => void> = []

    const yieldToMainThread = vi.fn(
      () =>
        new Promise<void>(resolve => {
          releaseYields.push(resolve)
        })
    )

    const render = vi.fn(async (code: string) => `<svg>${code}</svg>`)

    const cache = createMermaidRenderCache({
      defer: async () => undefined,
      maxEntries: 8,
      render,
      yieldToMainThread
    })

    const first = cache.render('A', 'dark')
    const second = cache.render('B', 'dark')

    await first
    await flushTasks()
    expect(render.mock.calls.map(([code]) => code)).toEqual(['A'])
    expect(yieldToMainThread).toHaveBeenCalledTimes(1)

    releaseYields.shift()?.()
    await second
    expect(render.mock.calls.map(([code]) => code)).toEqual(['A', 'B'])
  })

  it('cancels unobserved work before it enters the Mermaid render mutex', async () => {
    const defer = vi.fn(
      (signal?: AbortSignal) =>
        new Promise<void>((resolve, reject) => {
          signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')), { once: true })
        })
    )

    const render = vi.fn(async () => '<svg>stale</svg>')
    const cache = createMermaidRenderCache({ defer, maxEntries: 8, render })
    const controller = new AbortController()

    const stale = cache.render('stale', 'dark', controller.signal)
    controller.abort()

    await expect(stale).rejects.toMatchObject({ name: 'AbortError' })
    expect(render).not.toHaveBeenCalled()
  })

  it('starts fresh work when the same key remounts immediately after cancellation', async () => {
    let attempt = 0

    const defer = vi.fn((signal?: AbortSignal) => {
      attempt += 1

      if (attempt > 1) {
        return Promise.resolve()
      }

      return new Promise<void>((_resolve, reject) => {
        signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')), { once: true })
      })
    })

    const render = vi.fn(async () => '<svg>fresh</svg>')
    const cache = createMermaidRenderCache({ defer, maxEntries: 8, render })
    const controller = new AbortController()

    const stale = cache.render('strict-mode', 'dark', controller.signal)
    controller.abort()
    const remounted = cache.render('strict-mode', 'dark')

    await expect(stale).rejects.toMatchObject({ name: 'AbortError' })
    await expect(remounted).resolves.toBe('<svg>fresh</svg>')
    expect(render).toHaveBeenCalledTimes(1)
  })
})

describe('createRetryableLoader', () => {
  it('shares a load attempt but retries after a transient rejection', async () => {
    const load = vi
      .fn<() => Promise<string>>()
      .mockRejectedValueOnce(new Error('chunk unavailable'))
      .mockResolvedValueOnce('mermaid')

    const get = createRetryableLoader(load)

    const first = get()
    expect(get()).toBe(first)
    await expect(first).rejects.toThrow('chunk unavailable')
    await expect(get()).resolves.toBe('mermaid')
    expect(load).toHaveBeenCalledTimes(2)
  })
})
