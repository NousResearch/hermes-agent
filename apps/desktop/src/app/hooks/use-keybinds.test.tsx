import { renderHook } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { IS_MAC } from '@/lib/keybinds/combo'
import { resetAllBindings } from '@/store/keybinds'

import { useKeybinds } from './use-keybinds'

const composerRequests = vi.hoisted(() => ({
  focus: vi.fn(),
  toggleDictation: vi.fn(),
  toggleVoice: vi.fn()
}))

vi.mock('../chat/composer/focus', () => ({
  requestComposerFocus: composerRequests.focus,
  requestDictationToggle: composerRequests.toggleDictation,
  requestVoiceToggle: composerRequests.toggleVoice
}))

vi.mock('react-router-dom', () => ({ useNavigate: () => vi.fn() }))

vi.mock('@/themes/context', () => ({
  useTheme: () => ({ resolvedMode: 'light', setMode: vi.fn() })
}))

beforeEach(() => {
  vi.clearAllMocks()
  resetAllBindings()
})

function dictationKeydown(repeat = false): KeyboardEvent {
  return new KeyboardEvent('keydown', {
    bubbles: true,
    cancelable: true,
    code: 'KeyD',
    metaKey: IS_MAC,
    ctrlKey: !IS_MAC,
    repeat,
    shiftKey: true
  })
}

describe('useKeybinds dictation dispatch', () => {
  it('dispatches Ctrl+Shift+D once and swallows OS key-repeat', () => {
    const { unmount } = renderHook(() =>
      useKeybinds({
        openNewSessionTab: vi.fn(),
        startFreshSession: vi.fn(),
        toggleCommandCenter: vi.fn(),
        toggleSelectedPin: vi.fn()
      })
    )

    const first = dictationKeydown()
    window.dispatchEvent(first)
    expect(first.defaultPrevented).toBe(true)
    expect(composerRequests.toggleDictation).toHaveBeenCalledOnce()

    const repeated = dictationKeydown(true)
    window.dispatchEvent(repeated)
    expect(repeated.defaultPrevented).toBe(true)
    expect(composerRequests.toggleDictation).toHaveBeenCalledOnce()

    unmount()
  })
})
