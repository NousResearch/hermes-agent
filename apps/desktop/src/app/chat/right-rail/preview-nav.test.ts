import { afterEach, describe, expect, it, vi } from 'vitest'

import { commandFocusedPreview, PREVIEW_BROWSER_ATTR, registerPreviewNav } from './preview-nav'

const nav = () => ({ back: vi.fn(), forward: vi.fn(), reload: vi.fn() })

/** A focusable stand-in for a browser pane, marked the way PreviewPane marks it. */
function mountPane(tabId: string) {
  const host = document.createElement('div')

  host.setAttribute(PREVIEW_BROWSER_ATTR, tabId)
  host.tabIndex = -1
  document.body.append(host)

  return host
}

afterEach(() => {
  document.body.replaceChildren()
})

// This registry answers only for focus in Hermes' OWN chrome. A gesture made
// inside the guest page never reaches this renderer at all — main handles that
// against the focused webContents (see `commandFocusedGuest` in main.ts).
describe('commandFocusedPreview', () => {
  it('reports no handler when focus is elsewhere, so ⌘R falls back to the window', () => {
    const handle = nav()

    registerPreviewNav('url:browser', handle)

    expect(commandFocusedPreview('reload')).toBe(false)
    expect(handle.reload).not.toHaveBeenCalled()
  })

  it.each(['back', 'forward', 'reload'] as const)('routes %s to the focused pane', command => {
    const handle = nav()
    const host = mountPane('url:browser')

    registerPreviewNav('url:browser', handle)
    host.focus()

    expect(commandFocusedPreview(command)).toBe(true)
    expect(handle[command]).toHaveBeenCalledOnce()
  })

  // The common chrome case: focus is in the address field, not the page.
  it('answers for focus nested inside the pane', () => {
    const handle = nav()
    const host = mountPane('url:browser')
    const input = document.createElement('input')

    host.append(input)
    registerPreviewNav('url:browser', handle)
    input.focus()

    expect(commandFocusedPreview('reload')).toBe(true)
    expect(handle.reload).toHaveBeenCalledOnce()
  })

  it('commands only the focused pane when several are open', () => {
    const first = nav()
    const second = nav()

    mountPane('url:one')
    const secondHost = mountPane('url:two')

    registerPreviewNav('url:one', first)
    registerPreviewNav('url:two', second)
    secondHost.focus()

    expect(commandFocusedPreview('back')).toBe(true)
    expect(second.back).toHaveBeenCalledOnce()
    expect(first.back).not.toHaveBeenCalled()
  })

  // A parked/unmounted pane must not answer for a tab that is no longer live —
  // the gesture has to fall through instead of hitting a dead handle.
  it('stops answering once the pane unregisters', () => {
    const handle = nav()
    const host = mountPane('url:browser')
    const unregister = registerPreviewNav('url:browser', handle)

    host.focus()
    unregister()

    expect(commandFocusedPreview('reload')).toBe(false)
    expect(handle.reload).not.toHaveBeenCalled()
  })

  it('leaves a later registration in place when an older one unregisters', () => {
    const stale = nav()
    const live = nav()
    const host = mountPane('url:browser')

    const unregisterStale = registerPreviewNav('url:browser', stale)

    registerPreviewNav('url:browser', live)
    unregisterStale()
    host.focus()

    expect(commandFocusedPreview('reload')).toBe(true)
    expect(live.reload).toHaveBeenCalledOnce()
    expect(stale.reload).not.toHaveBeenCalled()
  })
})
