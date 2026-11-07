import { renderHook } from '@testing-library/react'
import type { ReactNode } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { isMacPlatform } from '@/lib/platform'
import { setBinding } from '@/store/keybinds'
import { $newChatConnectionId, $newChatProfile, $newChatRoute, captureNewChatSource } from '@/store/profile'

import { useKeybinds } from './use-keybinds'

vi.mock('@/app/right-sidebar/terminal/terminals', () => ({
  closeActiveTerminal: vi.fn(),
  createTerminal: vi.fn(),
  cycleTerminal: vi.fn()
}))

vi.mock('@/themes/context', () => ({
  useTheme: () => ({ resolvedMode: 'light', setMode: vi.fn() })
}))

const deps = {
  archiveSelectedSession: vi.fn(),
  openNewSessionTab: vi.fn(),
  requestGateway: vi.fn(),
  startFreshSession: vi.fn(),
  toggleCommandCenter: vi.fn(),
  toggleSelectedPin: vi.fn()
}

function pressNewSession(target: HTMLElement): KeyboardEvent {
  const event = new KeyboardEvent('keydown', {
    bubbles: true,
    cancelable: true,
    code: 'KeyN',
    ctrlKey: !isMacPlatform(),
    key: 'n',
    metaKey: isMacPlatform()
  })

  target.dispatchEvent(event)

  return event
}

describe('session.new chord', () => {
  let unmount: (() => void) | undefined

  beforeEach(() => {
    setBinding('session.new', ['mod+n'])
    unmount = renderHook(() => useKeybinds(deps), {
      wrapper: ({ children }: { children: ReactNode }) => <MemoryRouter>{children}</MemoryRouter>
    }).unmount
  })

  afterEach(() => {
    unmount?.()
    unmount = undefined
    $newChatProfile.set(null)
    $newChatRoute.set(null)
    $newChatConnectionId.set(null)
    deps.startFreshSession.mockReset()
  })

  it('clears a stale owner pin and starts a fresh draft', () => {
    // A previous draft pinned the local backend; the window has since moved to
    // another source. Ctrl-N must not keep minting on the stale route.
    $newChatProfile.set('default')
    $newChatConnectionId.set('local')
    $newChatRoute.set({ connectionId: 'local', profile: 'default' })
    captureNewChatSource('local')

    const input = window.document.createElement('textarea')
    window.document.body.append(input)

    pressNewSession(input)

    expect(deps.startFreshSession).toHaveBeenCalledOnce()
    expect($newChatProfile.get()).toBeNull()
    expect($newChatRoute.get()).toBeNull()
    expect($newChatConnectionId.get()).toBeNull()

    window.document.body.replaceChildren()
  })
})
