import { afterEach, describe, expect, it, vi } from 'vitest'

import { createBrowserWindowOpener } from './browser-window'

const POPUP_BLOCKED = 'Webapp window did not open. Allow popups for this site and try again.'

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

function opener(privateSession: boolean, api = vi.fn()) {
  return createBrowserWindowOpener({ api, basePath: '/one', privateSession })
}

function target(): URL {
  return new URL('/?win=secondary#/session-1', window.location.origin)
}

// The browser's side of window.open: `child` is null when the popup blocker stops the window, and a
// `noopener` or `noreferrer` open returns null even when the window opens (HTML window open steps).
function browserOpens(child: Window | null) {
  return vi.spyOn(window, 'open').mockImplementation((_url, _target, features) =>
    /noopener|noreferrer/.test(features ?? '') ? null : child
  )
}

describe('browser pop-out', () => {
  it('reports a blocked popup instead of success', async () => {
    browserOpens(null)

    await expect(opener(false)(target())).rejects.toThrow(POPUP_BLOCKED)
  })

  it('resolves when the browser opens the window and severs its opener', async () => {
    const child = { opener: window } as Window
    const open = browserOpens(child)
    const href = target().href

    await expect(opener(false)(new URL(href))).resolves.toEqual({ ok: true })
    expect(open.mock.calls[0]?.[0]).toBe(href)
    expect(child.opener).toBeNull()
  })

  it('reports a blocked private-session popup without waiting for the handoff', async () => {
    browserOpens(null)
    vi.stubGlobal(
      'BroadcastChannel',
      class {
        onmessage: ((event: { data: unknown }) => void) | null = null
        close() {}
      }
    )
    const api = vi.fn()

    await expect(opener(true, api)(target())).rejects.toThrow(POPUP_BLOCKED)
    expect(api).not.toHaveBeenCalled()
  })
})
