import { beforeEach, describe, expect, it, vi } from 'vitest'

import { isMacPlatform } from '@/lib/platform'

import { $rightRailActiveTabId } from '@/store/layout'
import { closeRightRail, openPreview, type PreviewTarget } from '@/store/preview'

import { actOnActivePreview } from './preview-act'
import { registerPreviewInput } from './preview-input'
import { registerPreviewNav } from './preview-nav'
import { registerPreviewScriptRunner } from './preview-script-runner'

function urlTarget(url: string): PreviewTarget {
  return { kind: 'url', label: 'Browser', source: url, url }
}

describe('actOnActivePreview (drive_preview tool)', () => {
  // URL targets share the singleton Browser tab id, so anything a test
  // registers would answer the next one.
  let cleanups: Array<() => void> = []

  const openBrowserTab = () => {
    openPreview(urlTarget('https://example.com'), 'tool-result')

    return $rightRailActiveTabId.get()!
  }

  const withRunner = (runner: (code: string) => Promise<unknown>) =>
    cleanups.push(registerPreviewScriptRunner(openBrowserTab(), runner))

  beforeEach(() => {
    vi.useRealTimers()

    for (const cleanup of cleanups) {
      cleanup()
    }

    cleanups = []
    closeRightRail()
    window.localStorage.clear()
  })

  it('tells the agent to open a page when no live pane is behind the tab', async () => {
    const result = await actOnActivePreview({ kind: 'elements' })

    expect(result.success).toBe(false)
    expect(result.error).toContain('open_preview')
  })

  it('injects the engine and returns the page’s answer', async () => {
    let injected = ''

    withRunner(async code => {
      injected = code

      return JSON.stringify({ acted: 'clicked button "Save"', success: true })
    })

    const result = await actOnActivePreview({ kind: 'click', ref: '@e1' })

    expect(result).toMatchObject({ acted: 'clicked button "Save"', success: true })
    // Self-contained payload: the engine source and the action travel together,
    // and the holder keeps refs alive across calls on the same page.
    expect(injected).toContain('__hermesActHolder')
    expect(injected).toContain('"ref":"@e1"')
  })

  it('re-inventories after a mutating action so the next ref is current', async () => {
    const actions: string[] = []

    withRunner(code => {
      // Stand in for the guest page: run the script's own settle/rescan shape
      // by answering each act() call in order.
      actions.push(...(code.match(/"kind":"(\w+)"/g) ?? []))

      return Promise.resolve(
        JSON.stringify({
          acted: 'clicked',
          elements: [{ label: 'Log out', ref: '@e1', role: 'button', selector: '#out' }],
          success: true,
          url: 'https://example.com/app'
        })
      )
    })

    const result = await actOnActivePreview({ kind: 'click', ref: '@e1' })

    expect(result.elements?.[0].label).toBe('Log out')
    expect(result.url).toBe('https://example.com/app')
  })

  it('does not pay the settle delay for a plain inventory', async () => {
    let injected = ''
    withRunner(async code => {
      injected = code

      return JSON.stringify({ elements: [], success: true })
    })

    await actOnActivePreview({ kind: 'elements' })

    expect(injected).toContain('0 <= 0')
  })

  it('reports a page that answers with nothing', async () => {
    withRunner(async () => '')

    expect((await actOnActivePreview({ kind: 'click', ref: '@e1' })).error).toContain('did not answer')
  })

  /** A pane that answers the locate trip with a fixed on-screen point, and the
   *  read-back trip with an empty inventory. Returns the input spy. */
  const withDrivenPane = () => {
    const tabId = openBrowserTab()
    const send = vi.fn()

    cleanups.push(
      registerPreviewScriptRunner(tabId, async code =>
        code.includes('"kind":"locate"')
          ? JSON.stringify({ acted: 'looking at button "Save"', point: { x: 120, y: 80 }, success: true })
          : // `hit` is the page's witness that the real pointerdown arrived.
            JSON.stringify({ elements: [], hit: { tag: 'BUTTON', trusted: true }, success: true })
      )
    )
    cleanups.push(registerPreviewInput(tabId, { focus: vi.fn(), send }))

    return send
  }

  const sentTypes = (send: ReturnType<typeof vi.fn>) => send.mock.calls.map(([event]) => event.type)

  it('clicks with real input, walking the pointer to where the page said', async () => {
    const send = withDrivenPane()

    const result = await actOnActivePreview({ kind: 'click', ref: '@e1' })
    const types = sentTypes(send)

    // Stepped rather than teleported: a page learns it is hovered from a stream
    // of moves, and one jump to the target skips everything in between.
    expect(types.filter(type => type === 'mouseMove').length).toBeGreaterThan(1)
    expect(types).toContain('mouseDown')
    expect(types).toContain('mouseUp')
    expect(send.mock.calls.map(([event]) => event).find(event => event.type === 'mouseDown')).toMatchObject({
      x: 120,
      y: 80
    })
    expect(result.acted).toBe('clicked button "Save"')
  })

  it('types by pressing keys, after selecting whatever the field held', async () => {
    const send = withDrivenPane()

    await actOnActivePreview({ kind: 'type', ref: '@e1', submit: true, text: 'hi' })

    const events = send.mock.calls.map(([event]) => event)
    const chars = events.filter(event => event.type === 'char').map(event => event.keyCode)

    // One click to focus, then select-all by keyboard — NOT the triple-click
    // this used to do. A triple-click is a pointer gesture, so it grabs the
    // paragraph under the cursor whenever the target turns out not to be a
    // field, and the agent was leaving pages with their body text highlighted.
    expect(events.filter(event => event.type === 'mouseDown').map(event => event.clickCount)).toEqual([1])
    // Select-all must use the OS's real chord — Cmd+A on macOS, Ctrl+A elsewhere.
    // Sending BOTH modifiers is not a chord any platform acts on, so the select
    // silently no-ops and typing APPENDS to the field (9.99 + "19.99" = corruption).
    expect(events.filter(event => event.type === 'keyDown' && event.keyCode === 'a')[0]).toMatchObject({
      modifiers: isMacPlatform() ? ['meta'] : ['control']
    })
    // The chord must not send a `char` phase, or select-all types a literal 'a'.
    expect(chars).toEqual(['h', 'i', 'Enter'])
  })

  /** A pane whose field STATE is scriptable, so the driver's verify-and-
   *  fallback clear logic can be exercised. `selectWorks` decides whether the
   *  select-all chord empties the field (a re-render that throws the selection
   *  away makes it false); `clearable` decides whether End/Backspace actually
   *  empties it (false models a field that keeps refilling). The field starts
   *  at `initial` chars. Every input event is recorded on `send`. */
  const withTypingPane = (opts: { initial: number; selectWorks: boolean; clearable: boolean }) => {
    const tabId = openBrowserTab()
    const send = vi.fn()
    let fieldLen = opts.initial
    const isSelectAll = (e: { type: string; keyCode: string; modifiers?: string[] }) =>
      e.type === 'keyDown' && e.keyCode === 'a' && !!e.modifiers?.length

    send.mockImplementation((e: { type: string; keyCode: string; modifiers?: string[] }) => {
      if (isSelectAll(e)) {
        if (opts.selectWorks && opts.clearable) fieldLen = 0
        return
      }
      if (e.type === 'keyDown' && e.keyCode === 'Backspace' && fieldLen > 0 && opts.clearable) fieldLen--
    })

    cleanups.push(
      registerPreviewScriptRunner(tabId, async code =>
        code.includes('document.activeElement')
          ? JSON.stringify({ success: true, field: true, len: fieldLen })
          : code.includes('"kind":"locate"')
            ? JSON.stringify({
                acted: 'looking at textbox "Price"',
                point: { x: 120, y: 80 },
                success: true,
                typable: true
              })
            : JSON.stringify({ elements: [], hit: { tag: 'INPUT', trusted: true }, success: true })
      )
    )
    cleanups.push(registerPreviewInput(tabId, { focus: vi.fn(), send }))

    return send
  }

  const charKeys = (send: ReturnType<typeof vi.fn>) =>
    send.mock.calls.map(([e]) => e).filter((e: { type: string }) => e.type === 'char').map((e: { keyCode: string }) => e.keyCode)

  const keyDowns = (send: ReturnType<typeof vi.fn>) =>
    send.mock.calls.map(([e]) => e).filter((e: { type: string }) => e.type === 'keyDown').map((e: { keyCode: string }) => e.keyCode)

  it('verifies the select emptied the field before typing (select works)', async () => {
    const send = withTypingPane({ initial: 4, selectWorks: true, clearable: true })

    await actOnActivePreview({ kind: 'type', ref: '@e1', text: 'hi' })
    const keys = keyDowns(send)

    // Select-all cleared it on the first pass (len read back as 0) — no End or
    // Backspace fallback should fire, and the typed chars should just land.
    expect(keys).not.toContain('End')
    expect(keys).not.toContain('Backspace')
    expect(charKeys(send)).toEqual(['h', 'i'])
  })

  it('falls back to End+Backspace when a re-render throws the select away', async () => {
    const send = withTypingPane({ initial: 4, selectWorks: false, clearable: true })

    await actOnActivePreview({ kind: 'type', ref: '@e1', text: 'hi' })
    const keys = keyDowns(send)

    // select-all did nothing (the page replaced the node), so the driver must
    // End + Backspace the whole old value (4 chars) before typing — and never
    // append onto stale content.
    expect(keys.filter(k => k === 'Backspace').length).toBeGreaterThanOrEqual(4)
    expect(charKeys(send)).toEqual(['h', 'i'])
  })

  it('fails the action instead of typing when the field keeps refilling AND direct-set also fails', async () => {
    const tabId = openBrowserTab()
    const send = vi.fn()

    // Every read reports the field as non-empty (it never clears), and the
    // direct-set fallback reports failure too — so the action must hard-stop.
    cleanups.push(
      registerPreviewScriptRunner(tabId, async code =>
        code.includes('"kind":"locate"')
          ? JSON.stringify({
              acted: 'looking at textbox "Price"',
              point: { x: 120, y: 80 },
              success: true,
              typable: true
            })
          : code.includes('getOwnPropertyDescriptor') // direct-set also fails
            ? JSON.stringify({ success: false, error: 'mask refuses' })
            : code.includes('document.activeElement')
              ? JSON.stringify({ success: true, field: true, len: 6 })
              : JSON.stringify({ elements: [], hit: { tag: 'INPUT', trusted: true }, success: true })
      )
    )
    cleanups.push(registerPreviewInput(tabId, { focus: vi.fn(), send }))

    const result = await actOnActivePreview({ kind: 'type', ref: '@e1', text: 'hi' })

    // Nothing can clear it and the fallback failed, so nothing may be typed.
    expect(result.success).toBe(false)
    expect(charKeys(send)).not.toContain('h')
  })

  it('falls back to a direct DOM value-set when real keys cannot clear a masked field', async () => {
    const tabId = openBrowserTab()
    const send = vi.fn()

    cleanups.push(
      registerPreviewScriptRunner(tabId, async code =>
        code.includes('"kind":"locate"')
          ? JSON.stringify({
              acted: 'looking at textbox "Price"',
              point: { x: 120, y: 80 },
              success: true,
              typable: true
            })
          : !code.includes('__hermesAct') && code.includes('getOwnPropertyDescriptor')
            ? JSON.stringify({ success: true }) // the direct-set fallback succeeded (no preamble)
            : !code.includes('__hermesAct') && code.includes('document.activeElement')
              ? JSON.stringify({ success: true, field: true, len: 4 }) // field-state read
              : JSON.stringify({ elements: [], hit: { tag: 'INPUT', trusted: true }, success: true }) // finish trip
      )
    )
    cleanups.push(registerPreviewInput(tabId, { focus: vi.fn(), send }))

    const result = await actOnActivePreview({ kind: 'type', ref: '@e1', text: '19.99' })

    // Real keystrokes couldn't clear the mask, but the direct-set fallback
    // committed the value, so the action reports success without typing chars.
    expect(result.success).toBe(true)
    expect(charKeys(send)).not.toContain('1')
  })

  it('hovers by walking the pointer over and leaving it there', async () => {
    const send = withDrivenPane()

    const result = await actOnActivePreview({ kind: 'hover', ref: '@e1' })
    const types = sentTypes(send)

    expect(types).toContain('mouseMove')
    // The whole request is "be on it" — a click here would open the dropdown the
    // agent was trying to reveal, or worse, activate it.
    expect(types).not.toContain('mouseDown')
    expect(result.acted).toBe('hovered over button "Save"')
  })

  // The witness only speaks for verbs that put the button down. Demanding one
  // from a key press reported every press as a failure.
  it('does not expect a click witness from a verb that never clicks', async () => {
    const tabId = openBrowserTab()

    cleanups.push(
      registerPreviewScriptRunner(tabId, async code =>
        code.includes('"kind":"locate"')
          ? JSON.stringify({ acted: 'looking at textbox "Search"', point: { x: 40, y: 20 }, success: true })
          : JSON.stringify({ elements: [], hit: null, success: true })
      )
    )
    cleanups.push(registerPreviewInput(tabId, { focus: vi.fn(), send: vi.fn() }))

    for (const action of [
      { key: 'Escape', kind: 'press', ref: '@e1' },
      { kind: 'hover', ref: '@e1' }
    ]) {
      expect(await actOnActivePreview(action)).toMatchObject({ success: true })
    }
  })

  it('scrolls by wheeling for real, not by scripting the page', async () => {
    const tabId = openBrowserTab()
    const send = vi.fn()
    let scripted = false

    cleanups.push(
      registerPreviewScriptRunner(tabId, async code => {
        scripted ||= code.includes('"kind":"scroll"')

        return code.includes('scrollHeight')
          ? JSON.stringify({ page: 700, point: { x: 500, y: 400 }, span: 4_000, success: true })
          : JSON.stringify({ elements: [], success: true })
      })
    )
    cleanups.push(registerPreviewInput(tabId, { focus: vi.fn(), send }))

    await actOnActivePreview({ kind: 'scroll' })

    const wheels = send.mock.calls.map(([event]) => event).filter(event => event.type === 'mouseWheel')

    // A stream of notches, not one delta: scroll-linked headers and lazy loaders
    // only react to the events, so a scripted scrollBy leaves them asleep.
    expect(wheels.length).toBeGreaterThan(1)
    // Electron's wheel delta is wheelDelta-signed, so scrolling DOWN is negative.
    expect(wheels.every(event => event.deltaY < 0)).toBe(true)
    expect(scripted).toBe(false)
  })

  it('says so plainly when the page has nothing to scroll', async () => {
    const tabId = openBrowserTab()

    cleanups.push(
      registerPreviewScriptRunner(tabId, async () =>
        JSON.stringify({ page: 700, point: { x: 500, y: 400 }, span: 0, success: true })
      )
    )
    cleanups.push(registerPreviewInput(tabId, { focus: vi.fn(), send: vi.fn() }))

    expect(await actOnActivePreview({ kind: 'scroll' })).toMatchObject({
      note: expect.stringContaining('nothing to scroll')
    })
  })

  it('fails loudly when the pointer input never reaches the page', async () => {
    const tabId = openBrowserTab()

    cleanups.push(
      registerPreviewScriptRunner(tabId, async code =>
        code.includes('"kind":"locate"')
          ? JSON.stringify({ acted: 'looking at button "Save"', point: { x: 12, y: 8 }, success: true })
          : JSON.stringify({ elements: [], hit: null, success: true })
      )
    )
    cleanups.push(registerPreviewInput(tabId, { focus: vi.fn(), send: vi.fn() }))

    const result = await actOnActivePreview({ kind: 'click', ref: '@e1' })

    // Everything else about this action travels on the script channel and would
    // report success whether or not a single event landed.
    expect(result.success).toBe(false)
    expect(result.error).toContain('never reached the page')
  })

  it('says so when the overlay itself swallowed the click', async () => {
    const tabId = openBrowserTab()

    cleanups.push(
      registerPreviewScriptRunner(tabId, async code =>
        code.includes('"kind":"locate"')
          ? JSON.stringify({ acted: 'looking at button "Save"', point: { x: 12, y: 8 }, success: true })
          : JSON.stringify({ elements: [], hit: { tag: 'HERMES-WATCH', trusted: true }, success: true })
      )
    )
    cleanups.push(registerPreviewInput(tabId, { focus: vi.fn(), send: vi.fn() }))

    expect((await actOnActivePreview({ kind: 'click', ref: '@e1' })).note).toContain('overlay intercepted')
  })

  it('falls back to scripted events when the pane exposes no input channel', async () => {
    let injected = ''

    withRunner(async code => {
      injected = code

      return JSON.stringify({ acted: 'clicked', success: true })
    })

    await actOnActivePreview({ kind: 'click', ref: '@e1' })

    // The one-trip shape: the engine both acts and re-reads, no locate handshake.
    expect(injected).toContain('"kind":"click"')
    expect(injected).not.toContain('"kind":"locate"')
  })

  it('routes history verbs to the pane instead of the guest page', async () => {
    const back = vi.fn()
    const runner = vi.fn()

    const tabId = openBrowserTab()
    cleanups.push(registerPreviewNav(tabId, { back, forward: vi.fn(), reload: vi.fn() }))
    cleanups.push(registerPreviewScriptRunner(tabId, runner))

    const result = await actOnActivePreview({ kind: 'back' })

    expect(back).toHaveBeenCalledOnce()
    expect(runner).not.toHaveBeenCalled()
    expect(result.success).toBe(true)
    expect(result.note).toContain('elements')
  })

  it('reports history verbs with no pane to drive', async () => {
    expect((await actOnActivePreview({ kind: 'reload' })).error).toContain('open_preview')
  })
})
