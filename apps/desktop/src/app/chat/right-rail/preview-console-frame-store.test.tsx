import { act, cleanup, fireEvent, render, renderHook } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { isNearConsoleBottom, PreviewConsolePanel, useFrameStore } from './preview-console'
import { type ConsoleEntry, createPreviewConsoleState } from './preview-console-state'

const entry = (id: number): ConsoleEntry => ({ id, level: 0, message: `line-${id}` })

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe('preview console frame coalescing', () => {
  it('renders once per frame for a burst of log appends', () => {
    const frames: FrameRequestCallback[] = []
    vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
      frames.push(callback)

      return frames.length
    })
    vi.stubGlobal('cancelAnimationFrame', () => {})

    const store = atom<ConsoleEntry[]>([])
    let renders = 0

    const { result } = renderHook(() => {
      renders += 1

      return useFrameStore(store)
    })

    const before = renders

    act(() => {
      for (let id = 1; id <= 5; id += 1) {
        store.set([...store.get(), entry(id)])
      }
    })

    // Nothing is painted until the frame callback runs.
    expect(renders - before).toBe(0)
    expect(frames).toHaveLength(1)

    act(() => {
      frames.splice(0).forEach(callback => callback(0))
    })

    expect(renders - before).toBe(1)
    expect(result.current.map(item => item.id)).toEqual([1, 2, 3, 4, 5])
  })
})

describe('preview console scroll following', () => {
  it.each([true, false])('preserves pinned=%s across consecutive frame appends and resizing', pinned => {
    const frames = new Map<number, FrameRequestCallback>()
    let frameId = 0
    vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
      frames.set(++frameId, callback)

      return frameId
    })
    vi.stubGlobal('cancelAnimationFrame', (id: number) => frames.delete(id))

    const advanceFrame = () =>
      act(() => {
        const pending = [...frames.values()]
        frames.clear()
        pending.forEach(callback => callback(0))
      })

    const consoleState = createPreviewConsoleState()
    consoleState.append(entry(1))
    consoleState.append(entry(2))
    const consoleBodyRef = { current: null as HTMLDivElement | null }
    const consoleShouldStickRef = { current: true }
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <PreviewConsolePanel
          consoleBodyRef={consoleBodyRef}
          consoleShouldStickRef={consoleShouldStickRef}
          consoleState={consoleState}
          startConsoleResize={() => {}}
        />
      </I18nProvider>
    )
    const body = consoleBodyRef.current!
    // jsdom has no layout: each committed row contributes a tall log's height.
    Object.defineProperties(body, {
      scrollHeight: { get: () => body.querySelectorAll('[data-selectable-text]').length * 200 },
      clientHeight: { get: () => consoleState.$height.get() - 32 }
    })
    body.scrollTo = ((options: ScrollToOptions) => {
      body.scrollTop = Math.max(0, Math.min(options.top ?? 0, body.scrollHeight - body.clientHeight))
    }) as typeof body.scrollTo
    body.scrollTo({ top: body.scrollHeight })
    advanceFrame()

    if (!pinned) {
      body.scrollTop = 0
      fireEvent.scroll(body)
    }

    for (const id of [3, 4]) {
      act(() => {
        // PreviewPane owns the intent ref and samples it before each append.
        consoleShouldStickRef.current = isNearConsoleBottom(body)
        consoleState.append(entry(id))
      })
      advanceFrame()
    }

    expect(body.textContent).toContain('line-4')
    expect(consoleShouldStickRef.current).toBe(pinned)
    expect(body.scrollTop).toBe(pinned ? body.scrollHeight - body.clientHeight : 0)

    act(() => consoleState.setHeight(160))
    expect(body.scrollTop).toBe(pinned ? body.scrollHeight - body.clientHeight : 0)
  })
})
