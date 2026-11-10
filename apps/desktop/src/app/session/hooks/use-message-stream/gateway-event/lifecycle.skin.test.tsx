import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { __resetBackendSkinSync } from '@/themes/backend-sync'
import { ThemeProvider } from '@/themes/context'

import { handleLifecycleEvent } from './lifecycle'
import type { GatewayEventContext } from './types'

const ready = (name: string, active: boolean): GatewayEventContext => {
  const payload = { skin: { name, colors: { background: '#121212', ui_text: '#f5f5f5' } } }

  return {
    deps: {} as GatewayEventContext['deps'],
    event: { type: 'gateway.ready', payload } as GatewayEventContext['event'],
    explicitSid: '',
    fromActiveSource: () => active,
    isActiveEvent: false,
    occurredAt: 0,
    payload: payload as GatewayEventContext['payload'],
    scheduleConfigRefresh: vi.fn(),
    sessionId: null
  }
}

beforeEach(() => {
  window.localStorage.clear()
  __resetBackendSkinSync()
})
afterEach(cleanup)

describe('Arabic skin activation from the active gateway', () => {
  it.each(['hermes-arabi', 'hermes-arabi-light'])(
    'activates %s on connect instead of leaving the default skin painted',
    name => {
      window.localStorage.setItem('hermes-desktop-theme-v2', 'nous')
      render(
        <ThemeProvider>
          <div />
        </ThemeProvider>
      )
      act(() => {
        handleLifecycleEvent(ready(name, true))
      })
      expect(window.document.documentElement.dataset.hermesTheme).toBe(name)
    }
  )

  it.each([
    ['hermes-arabi', false],
    ['hermes-arabi-light', false],
    ['everforest', true]
  ] as const)('does not replace the desktop skin for %s (active=%s)', (name, active) => {
    render(
      <ThemeProvider>
        <div />
      </ThemeProvider>
    )
    act(() => {
      handleLifecycleEvent(ready(name, active))
    })
    expect(window.document.documentElement.dataset.hermesTheme).toBe('nous')
  })
})
