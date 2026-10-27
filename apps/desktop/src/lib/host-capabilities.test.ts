// @vitest-environment jsdom
import { afterEach, describe, expect, it } from 'vitest'

import { HOST_CAPABILITIES, type HostCapability, hostHas } from './host-capabilities'
import { isBrowserHostedDesktop } from './platform'

interface HostGlobals {
  __HERMES_AUTH_REQUIRED__?: boolean
  __HERMES_UI_SURFACE__?: string
  hermesDesktop?: unknown
}

const win = window as unknown as HostGlobals

function reset() {
  document.documentElement.removeAttribute('data-hermes-desktop-host')
  delete win.__HERMES_AUTH_REQUIRED__
  delete win.__HERMES_UI_SURFACE__
  delete win.hermesDesktop
}

// Every way the renderer meets its host: before and after the bridge writes the marker.
const HOST_STATES: { browser: boolean; name: string; setup: () => void }[] = [
  { browser: true, name: 'browser marker', setup: () => (document.documentElement.dataset.hermesDesktopHost = 'browser') },
  { browser: true, name: 'gated host before the marker', setup: () => (win.__HERMES_AUTH_REQUIRED__ = true) },
  { browser: true, name: 'Webapp surface before the marker', setup: () => (win.__HERMES_UI_SURFACE__ = 'webapp') },
  { browser: false, name: 'electron marker', setup: () => (document.documentElement.dataset.hermesDesktopHost = 'electron') },
  {
    browser: false,
    name: 'Electron preload with a stale Webapp global',
    setup: () => {
      win.hermesDesktop = {}
      win.__HERMES_UI_SURFACE__ = 'webapp'
    }
  },
  { browser: false, name: 'no host signal', setup: () => undefined }
]

const CAPABILITIES = Object.keys(HOST_CAPABILITIES) as HostCapability[]

afterEach(reset)

describe('hostHas', () => {
  // Parity with the predicate every call site used before: a browser capability is exactly
  // `isBrowserHostedDesktop()`, a native one exactly its negation, in every host state.
  for (const state of HOST_STATES) {
    it(`matches isBrowserHostedDesktop() for every capability: ${state.name}`, () => {
      state.setup()
      expect(isBrowserHostedDesktop()).toBe(state.browser)

      for (const capability of CAPABILITIES) {
        const old = isBrowserHostedDesktop()
        const expected = HOST_CAPABILITIES[capability] === 'browser' ? old : !old

        expect([capability, hostHas(capability)]).toEqual([capability, expected])
      }
    })
  }

  it('follows the host marker written after import', () => {
    expect(hostHas('clipboardIpc')).toBe(true)
    document.documentElement.dataset.hermesDesktopHost = 'browser'
    expect(hostHas('clipboardIpc')).toBe(false)
    expect(hostHas('webappCopy')).toBe(true)
  })
})
