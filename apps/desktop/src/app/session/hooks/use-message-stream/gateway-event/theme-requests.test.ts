import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { registerLayoutPresets } from '@/app/contrib/layout-presets'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { ScopedServerRequest } from '@/store/gateway'
import { resetServerRequestsForTests } from '@/store/server-requests'
import { $shellChoice } from '@/store/shell-variant'
import { $themePicks, cancelThemePick, pickTheme } from '@/store/theme-pick'
import { $pendingSkinApply } from '@/themes/backend-sync'

import { handleServerRequest, type ServerRequestContext } from './server-requests'

// SAFETY: the set_theme handlers read only activeSessionIdRef, sessionInterrupted and updateSessionState.
const deps = {
  activeSessionIdRef: { current: 'session-a' },
  sessionInterrupted: () => false,
  sessionStateByRuntimeIdRef: { current: new Map() },
  updateSessionState: (_sessionId, update) => update(createClientSessionState('stored-session')),
  upsertToolCall: () => undefined
} as ServerRequestContext['deps']

function deliver(method: string, params: ScopedServerRequest['params']) {
  const respond = vi.fn()
  const decline = vi.fn()

  handleServerRequest(
    { decline, fail: vi.fn(), id: 'srq-1', method, params, profile: 'default', respond },
    deps,
    'session-a'
  )

  return Object.assign(respond, { decline })
}

describe('set_theme requests', () => {
  beforeAll(() => {
    registerLayoutPresets()
  })

  afterEach(() => {
    $themePicks.set({})
    $pendingSkinApply.set(null)
    $shellChoice.set(null)
    resetServerRequestsForTests()
  })

  // An unpicked skin resolves to the shell's own default at paint time, so the accent lands on that skin.
  it.each([
    { shell: 'classic', choice: false, saved: 'nous-accent-e0218a' },
    { shell: 'redesign', choice: null, saved: 'portal-accent-e0218a' }
  ])('parks a picker card and answers it with the applied pick ($shell)', ({ choice, saved }) => {
    $shellChoice.set(choice)
    const respond = deliver('theme.pick', { picker: 'accent', session_id: 'session-a' })

    expect($themePicks.get()['session-a']).toMatchObject({ picker: 'accent', requestId: 'srq-1' })
    expect(respond).not.toHaveBeenCalled()

    expect(pickTheme('session-a', { accent: 'pink' })).toBe(true)
    expect(respond).toHaveBeenCalledWith({ picked: { accent: 'pink' } })
    expect($pendingSkinApply.get()).toBe(saved)
    expect($themePicks.get()['session-a']).toBeUndefined()
  })

  it('answers a dismissed card as cancelled', () => {
    const respond = deliver('theme.pick', { picker: 'layout', session_id: 'session-a' })

    expect(cancelThemePick('session-a')).toBe(true)
    expect(respond).toHaveBeenCalledWith({ cancelled: true })
  })

  it('applies values only in the window showing the chat, and nothing when one value is unknown', () => {
    const elsewhere = deliver('theme.apply', { accent: 'violet', session_id: 'session-b' })
    const onScreen = deliver('theme.apply', { accent: 'violet', layout: 'cockpit', session_id: 'session-a' })

    expect(elsewhere.decline).toHaveBeenCalled()
    expect(elsewhere).not.toHaveBeenCalled()
    expect(JSON.parse(onScreen.mock.calls[0][0].value)).toMatchObject({
      success: false,
      error: 'Unknown layout "cockpit".'
    })
    expect($pendingSkinApply.get()).toBeNull()
  })
})
