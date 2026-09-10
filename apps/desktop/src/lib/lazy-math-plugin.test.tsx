import { act, renderHook, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import {
  createLazyMathPluginLoader,
  hasRenderableMath,
  type MathPlugin,
  useLazyMathPlugin,
  useLazyMathPluginState
} from './lazy-math-plugin'

const plugin = { name: 'katex' as const, type: 'math' as const } as MathPlugin

describe('hasRenderableMath', () => {
  it.each([
    'Inline $x^2$ math',
    'Display $$E = mc^2$$',
    String.raw`Inline \(x + y\) math`,
    String.raw`Display \[x + y\] math`,
    '```math\nx^2\n```',
    '$ x$',
    '$a\nb$',
    '[/math]\nx^2\n[/math]',
    '````math title="equation"\r\nx^2\r\n`````'
  ])('detects supported math syntax in %s', text => {
    expect(hasRenderableMath(text)).toBe(true)
  })

  it('does not treat ordinary prose or code without math markers as renderable math', () => {
    expect(hasRenderableMath('ordinary prose')).toBe(false)
    expect(hasRenderableMath('`echo HOME`')).toBe(false)
  })
})

describe('createLazyMathPluginLoader', () => {
  it('does not import KaTeX for prose-only markdown', async () => {
    const importer = vi.fn()
    const loader = createLazyMathPluginLoader(importer)

    await expect(loader.load('ordinary prose')).resolves.toBeUndefined()
    expect(importer).not.toHaveBeenCalled()
  })

  it('deduplicates concurrent imports and reuses the completed plugin', async () => {
    const importer = vi.fn(async () => ({ createMemoizedMathPlugin: () => plugin }))
    const loader = createLazyMathPluginLoader(importer)

    const [first, second] = await Promise.all([loader.load('$x$'), loader.load('$$y$$')])

    expect(first).toBe(plugin)
    expect(second).toBe(plugin)
    await expect(loader.load('$z$')).resolves.toBe(plugin)
    expect(importer).toHaveBeenCalledOnce()
  })

  it('does not re-request a module URL that Chromium has already poisoned', async () => {
    const failure = new Error('chunk failed')
    const importer = vi.fn().mockRejectedValue(failure)
    const loader = createLazyMathPluginLoader(importer)

    await expect(loader.load('$x$')).rejects.toBe(failure)
    await expect(loader.load('$x$')).rejects.toBe(failure)
    expect(importer).toHaveBeenCalledOnce()
  })
})

describe('useLazyMathPlugin', () => {
  it('loads the plugin using delimiters after math preprocessing', async () => {
    const load = vi.fn(async () => plugin)

    const loader = {
      load,
      peek: () => undefined
    }

    const { result } = renderHook(() => useLazyMathPlugin('[/inline]x + y[/inline]', loader))

    await waitFor(() => expect(result.current).toBe(plugin))
    expect(load).toHaveBeenCalledWith('$x + y$')
  })

  it('publishes a failed state so poisoned imports can use the renderer fallback', async () => {
    const failure = new Error('chunk failed')

    const loader = {
      load: vi.fn().mockRejectedValue(failure),
      peek: () => undefined
    }

    const { result } = renderHook(() => useLazyMathPluginState('$x$', loader))

    await waitFor(() => expect(result.current.failed).toBe(true))
    expect(result.current.loading).toBe(false)
    expect(result.current.plugin).toBeUndefined()
  })

  it('does not render raw delimiters during a prose-to-math transition', () => {
    const renders: Array<{
      failed: boolean
      loading: boolean
      plugin: MathPlugin | undefined
      renderedText: string
    }> = []

    const loader = {
      load: vi.fn(() => new Promise<MathPlugin>(() => undefined)),
      peek: () => undefined
    }

    const { rerender } = renderHook(
      ({ text }) => {
        const state = useLazyMathPluginState(text, loader)
        renders.push({ ...state, renderedText: state.loading ? 'loading' : text })

        return state
      },
      { initialProps: { text: 'ordinary prose' } }
    )

    const rendersBeforeMath = renders.length
    rerender({ text: 'Now $x$ appears' })

    expect(renders[rendersBeforeMath]).toMatchObject({ failed: false, loading: true, plugin: undefined })
    expect(renders[rendersBeforeMath].renderedText).not.toContain('$x$')
  })

  it('publishes the plugin after eligible markdown loads it', async () => {
    const importer = vi.fn(async () => ({ createMemoizedMathPlugin: () => plugin }))
    const loader = createLazyMathPluginLoader(importer)

    const { result, rerender } = renderHook(({ text }) => useLazyMathPlugin(text, loader), {
      initialProps: { text: 'ordinary prose' }
    })

    expect(result.current).toBeUndefined()
    expect(importer).not.toHaveBeenCalled()

    rerender({ text: 'Now $x$ appears' })

    await waitFor(() => expect(result.current).toBe(plugin))
    expect(importer).toHaveBeenCalledOnce()

    act(() => rerender({ text: 'prose again' }))
    expect(result.current).toBeUndefined()
  })
})
