import { renderHook } from '@testing-library/react'
import { createElement, type ReactNode } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/themes/context', () => ({
  useTheme: () => ({ resolvedMode: 'dark' as const, setMode: () => undefined })
}))

const toggleSelectedUnread = vi.hoisted(() => vi.fn())

import { useKeybinds } from '@/app/hooks/use-keybinds'
import { isMacPlatform } from '@/lib/platform'

// ⌘⇧U / Ctrl+Shift+U flips the selected session's persisted unread flag. The
// handler is wired in wiring.tsx beside its pin/archive twins; this exercises
// the dispatch: the chord reaches the handler through the global listener.

const wrapper = ({ children }: { children: ReactNode }) => createElement(MemoryRouter, null, children)

let unmountKeybinds: (() => void) | undefined

function pressUnread() {
  window.dispatchEvent(
    new KeyboardEvent('keydown', {
      bubbles: true,
      cancelable: true,
      code: 'KeyU',
      ctrlKey: !isMacPlatform(),
      key: 'u',
      metaKey: isMacPlatform(),
      shiftKey: true
    })
  )
}

beforeEach(() => {
  toggleSelectedUnread.mockClear()
  unmountKeybinds = renderHook(
    () =>
      useKeybinds({
        archiveSelectedSession: () => undefined,
        openNewSessionTab: () => undefined,
        requestGateway: <T>() => Promise.resolve(undefined as T),
        startFreshSession: () => undefined,
        toggleCommandCenter: () => undefined,
        toggleSelectedPin: () => undefined,
        toggleSelectedUnread
      }),
    { wrapper }
  ).unmount
})

afterEach(() => {
  unmountKeybinds?.()
})

describe('session.toggleUnread', () => {
  it('dispatches ⌘⇧U / Ctrl+Shift+U to the unread handler', () => {
    pressUnread()

    expect(toggleSelectedUnread).toHaveBeenCalledTimes(1)
  })

  it('leaves the handler alone for the unshifted chord', () => {
    window.dispatchEvent(
      new KeyboardEvent('keydown', {
        bubbles: true,
        cancelable: true,
        code: 'KeyU',
        ctrlKey: !isMacPlatform(),
        key: 'u',
        metaKey: isMacPlatform()
      })
    )

    expect(toggleSelectedUnread).not.toHaveBeenCalled()
  })
})
