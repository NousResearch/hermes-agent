import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  appendBrowserConsoleEntry,
  appendBrowserNetworkEvent,
  clearBrowserTabs,
  createBrowserTab,
  getBrowserActionEvents,
  getBrowserConsoleEntries,
  getBrowserNetworkEvents,
  getBrowserScreenshotEntries,
  setBrowserEnabled,
  updateBrowserTab
} from './browser'
import { registerBrowserWebview, runBrowserBridgeCommand } from './browser-bridge'

describe('browser bridge', () => {
  beforeEach(() => {
    window.localStorage.clear()
    clearBrowserTabs()
    setBrowserEnabled(true)
  })

  afterEach(() => {
    clearBrowserTabs()
    setBrowserEnabled(false)
    document.body.innerHTML = ''
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it('rejects commands for unregistered tabs', async () => {
    const tab = createBrowserTab({ sessionId: 'session-1', url: 'https://example.com' })

    await expect(runBrowserBridgeCommand(tab.id, 'getState')).rejects.toThrow('not registered')
  })

  it('allows observe commands only after observe or control consent', async () => {
    const tab = createBrowserTab({ sessionId: 'session-1', title: 'Example', url: 'https://example.com' })

    const unregister = registerBrowserWebview(tab.id, {
      getTitle: () => 'Example',
      getURL: () => 'https://example.com/app'
    })

    await expect(runBrowserBridgeCommand(tab.id, 'getState')).rejects.toThrow('not bound')

    updateBrowserTab(tab.id, { controlMode: 'observe' })

    await expect(runBrowserBridgeCommand(tab.id, 'getState')).resolves.toMatchObject({
      title: 'Example',
      url: 'https://example.com/app'
    })

    unregister()
  })

  it('requires control consent for input commands and forwards sanitized events', async () => {
    const tab = createBrowserTab({ sessionId: 'session-1', url: 'https://example.com' })
    const sendInputEvent = vi.fn()

    registerBrowserWebview(tab.id, { sendInputEvent })
    updateBrowserTab(tab.id, { controlMode: 'observe' })

    await expect(runBrowserBridgeCommand(tab.id, 'click', { x: 12, y: 34 })).rejects.toThrow('requires control')

    updateBrowserTab(tab.id, { controlMode: 'control' })
    await runBrowserBridgeCommand(tab.id, 'click', { x: 12, y: 34 })

    expect(sendInputEvent).toHaveBeenCalledWith({ button: 'left', clickCount: 1, type: 'mouseDown', x: 12, y: 34 })
    expect(sendInputEvent).toHaveBeenCalledWith({ button: 'left', clickCount: 1, type: 'mouseUp', x: 12, y: 34 })
  })

  it('normalizes unsafe navigate commands before touching the visible webview', async () => {
    const tab = createBrowserTab({ sessionId: 'session-1', url: 'https://example.com' })
    const setAttribute = vi.fn()

    registerBrowserWebview(tab.id, { setAttribute })
    updateBrowserTab(tab.id, { controlMode: 'control' })

    await expect(runBrowserBridgeCommand(tab.id, 'navigate', { url: 'file:///etc/passwd' })).resolves.toMatchObject({
      url: expect.stringMatching(/^https:\/\/www\.google\.com\/search\?q=file%3A%2F%2F%2Fetc%2Fpasswd$/)
    })
    expect(setAttribute).toHaveBeenCalledWith('src', expect.stringMatching(/^https:\/\/www\.google\.com\/search\?/))

    await expect(runBrowserBridgeCommand(tab.id, 'navigate', { url: 'localhost:3000' })).resolves.toEqual({
      url: 'http://localhost:3000'
    })
  })

  it('captures screenshots as data URLs for vision feedback', async () => {
    const tab = createBrowserTab({ sessionId: 'session-1', url: 'https://example.com' })

    registerBrowserWebview(tab.id, {
      capturePage: async () => ({ toDataURL: () => 'data:image/png;base64,abc123' })
    })
    updateBrowserTab(tab.id, { controlMode: 'observe' })

    await expect(runBrowserBridgeCommand(tab.id, 'screenshot')).resolves.toEqual({ dataUrl: 'data:image/png;base64,abc123' })
  })

  it('keeps snapshot refs aligned with clickRef/fillRef actions', async () => {
    const tab = createBrowserTab({ sessionId: 'session-1', url: 'https://example.com' })

    document.body.innerHTML = `
      <a href="/alpha">Alpha link</a>
      <button>Beta button</button>
      <input aria-label="Gamma input" value="" />
    `
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue({
      bottom: 20,
      height: 20,
      left: 0,
      right: 100,
      toJSON: () => ({}),
      top: 0,
      width: 100,
      x: 0,
      y: 0
    } as DOMRect)

    registerBrowserWebview(tab.id, {
      executeJavaScript: async script => (0, eval)(script)
    })
    updateBrowserTab(tab.id, { controlMode: 'observe' })

    const snapshot = await runBrowserBridgeCommand(tab.id, 'snapshot') as { elements: Array<{ ref: string; text: string }> }

    expect(snapshot.elements.map(element => element.ref)).toEqual(['@e0', '@e1', '@e2'])
    expect(snapshot.elements.map(element => element.text)).toContain('Beta button')

    updateBrowserTab(tab.id, { controlMode: 'control' })
    await runBrowserBridgeCommand(tab.id, 'fillRef', { ref: '@e2', text: 'Cloud' })

    expect((document.querySelector('input') as HTMLInputElement).value).toBe('Cloud')
  })

  it('supports hover, double-click, right-click, richer scroll, and evaluation commands', async () => {
    const tab = createBrowserTab({ sessionId: 'session-1', url: 'https://example.com' })
    const sendInputEvent = vi.fn()

    registerBrowserWebview(tab.id, {
      executeJavaScript: async script => (0, eval)(script),
      sendInputEvent
    })
    updateBrowserTab(tab.id, { controlMode: 'control' })

    document.body.innerHTML = '<button id="target">Target</button>'
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue({
      bottom: 20,
      height: 20,
      left: 0,
      right: 100,
      toJSON: () => ({}),
      top: 0,
      width: 100,
      x: 0,
      y: 0
    } as DOMRect)

    await runBrowserBridgeCommand(tab.id, 'hover', { x: 5, y: 6 })
    await runBrowserBridgeCommand(tab.id, 'doubleClick', { x: 7, y: 8 })
    await runBrowserBridgeCommand(tab.id, 'rightClick', { x: 9, y: 10 })
    await runBrowserBridgeCommand(tab.id, 'scroll', { direction: 'right', amount: 321 })
    await expect(runBrowserBridgeCommand(tab.id, 'evaluate', { expression: 'document.querySelector("#target")?.id' }))
      .resolves.toBe('target')

    expect(sendInputEvent).toHaveBeenCalledWith({ type: 'mouseMove', x: 5, y: 6 })
    expect(sendInputEvent).toHaveBeenCalledWith({ button: 'left', clickCount: 2, type: 'mouseDown', x: 7, y: 8 })
    expect(sendInputEvent).toHaveBeenCalledWith({ button: 'right', clickCount: 1, type: 'mouseDown', x: 9, y: 10 })
  })

  it('emits stable refs and accepts stableRef targeting for agent workflows', async () => {
    const tab = createBrowserTab({ sessionId: 'session-1', url: 'https://example.com' })
    const clicks: string[] = []

    document.body.innerHTML = `
      <button aria-label="Save settings">Save</button>
      <button aria-label="Cancel dialog">Cancel</button>
    `
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue({
      bottom: 20,
      height: 20,
      left: 0,
      right: 100,
      toJSON: () => ({}),
      top: 0,
      width: 100,
      x: 0,
      y: 0
    } as DOMRect)
    document.querySelectorAll('button').forEach(button => {
      button.addEventListener('click', event => clicks.push((event.target as HTMLButtonElement).ariaLabel ?? ''))
    })

    registerBrowserWebview(tab.id, {
      executeJavaScript: async script => (0, eval)(script)
    })
    updateBrowserTab(tab.id, { controlMode: 'observe' })

    const snapshot = await runBrowserBridgeCommand(tab.id, 'snapshot') as {
      elements: Array<{ ref: string; stableRef: string; text: string }>
    }

    const save = snapshot.elements.find(element => element.text.includes('Save'))

    expect(save?.ref).toBe('@e0')
    expect(save?.stableRef).toMatch(/^@s/)

    if (!save) {
      throw new Error('Save element missing from browser snapshot')
    }

    updateBrowserTab(tab.id, { controlMode: 'control' })
    await runBrowserBridgeCommand(tab.id, 'clickRef', { ref: save.stableRef })

    expect(clicks).toEqual(['Save settings'])
  })

  it('inspects and selects elements for a safe Hermes design-mode handoff', async () => {
    const tab = createBrowserTab({ sessionId: 'session-1', url: 'https://example.com' })

    document.body.innerHTML = `
      <main>
        <button id="cta" class="primary" aria-label="Start checkout" style="color: rgb(255, 255, 255); background-color: rgb(0, 0, 0);">Buy now</button>
        <button id="secondary" class="ghost" aria-label="Compare plans">Compare</button>
      </main>
    `
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue({
      bottom: 44,
      height: 32,
      left: 12,
      right: 132,
      toJSON: () => ({}),
      top: 12,
      width: 120,
      x: 12,
      y: 12
    } as DOMRect)

    registerBrowserWebview(tab.id, {
      executeJavaScript: async script => (0, eval)(script)
    })
    updateBrowserTab(tab.id, { controlMode: 'observe' })

    const snapshot = await runBrowserBridgeCommand(tab.id, 'snapshot') as {
      elements: Array<{ ref: string; stableRef: string; text: string }>
    }

    const target = snapshot.elements.find(element => element.text.includes('Buy now'))
    const secondary = snapshot.elements.find(element => element.text.includes('Compare'))

    expect(target).toBeTruthy()
    expect(secondary).toBeTruthy()

    const inspected = await runBrowserBridgeCommand(tab.id, 'inspectElement', { ref: target?.stableRef }) as {
      element: {
        accessibility: { ariaLabel: string; name: string }
        htmlPreview: string
        tag: string
        text: string
        styles: Record<string, string>
        layout: { width: number }
      }
    }

    expect(inspected.element.tag).toBe('button')
    expect(inspected.element.text).toBe('Buy now')
    expect(inspected.element.layout.width).toBe(120)
    expect(inspected.element.styles.color).toBeTruthy()
    expect(inspected.element.htmlPreview.length).toBeLessThanOrEqual(1200)
    expect(inspected.element.htmlPreview).toContain('id="cta"')
    expect(inspected.element.accessibility).toMatchObject({ ariaLabel: 'Start checkout', name: 'Buy now' })

    await expect(runBrowserBridgeCommand(tab.id, 'selectElement', { ref: target?.stableRef })).resolves.toMatchObject({
      selected: [expect.objectContaining({ tag: 'button', text: 'Buy now' })]
    })

    await expect(runBrowserBridgeCommand(tab.id, 'selectElement', { refs: [target?.stableRef, secondary?.stableRef] })).resolves.toMatchObject({
      selected: [
        expect.objectContaining({ tag: 'button', text: 'Buy now' }),
        expect.objectContaining({ tag: 'button', text: 'Compare' })
      ]
    })

    await expect(runBrowserBridgeCommand(tab.id, 'designHandoff', {
      goal: 'Make the CTA more prominent',
      refs: [target?.stableRef]
    })).resolves.toMatchObject({
      mode: 'agent-mediated',
      unsafeDirectDomMutation: false,
      prompt: expect.stringContaining('Do not mutate the live DOM as the source of truth'),
      selected: [expect.objectContaining({ tag: 'button', text: 'Buy now' })]
    })

    await expect(runBrowserBridgeCommand(tab.id, 'mutateDom' as never, { ref: target?.stableRef })).rejects.toThrow('Unsupported browser command')
    expect(document.querySelector('#cta')?.textContent).toBe('Buy now')
  })

  it('returns accessibility audit findings for common browser workflow issues', async () => {
    const tab = createBrowserTab({ sessionId: 'session-1', url: 'https://example.com' })

    document.body.innerHTML = `
      <img src="/hero.png">
      <button></button>
      <input id="email">
      <a href="/checkout"></a>
      <div aria-hidden="true"><button>Hidden focus target</button></div>
    `
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue({
      bottom: 20,
      height: 20,
      left: 0,
      right: 100,
      toJSON: () => ({}),
      top: 0,
      width: 100,
      x: 0,
      y: 0
    } as DOMRect)

    registerBrowserWebview(tab.id, {
      executeJavaScript: async script => (0, eval)(script)
    })
    updateBrowserTab(tab.id, { controlMode: 'observe' })

    const audit = await runBrowserBridgeCommand(tab.id, 'accessibilityAudit') as {
      findings: Array<{ rule: string; severity: string }>
      summary: { error: number; warning: number }
    }

    expect(audit.findings.map(finding => finding.rule)).toEqual(expect.arrayContaining([
      'image-alt',
      'control-name',
      'input-label',
      'link-name',
      'aria-hidden-focusable'
    ]))
    expect(audit.summary.error).toBeGreaterThanOrEqual(4)
    expect(audit.summary.warning).toBeGreaterThanOrEqual(1)
  })

  it('records bridge action timeline entries and screenshot history through command execution', async () => {
    const tab = createBrowserTab({ sessionId: 'session-1', title: 'Visible', url: 'https://example.com' })
    const png = 'data:image/png;base64,abc123'

    registerBrowserWebview(tab.id, {
      capturePage: async () => png,
      getTitle: () => 'Visible',
      getURL: () => 'https://example.com/live',
      setAttribute: () => {
        throw new Error('blocked navigation')
      }
    })
    updateBrowserTab(tab.id, { controlMode: 'control' })

    await expect(runBrowserBridgeCommand(tab.id, 'screenshot')).resolves.toMatchObject({ dataUrl: png })
    await expect(runBrowserBridgeCommand(tab.id, 'navigate', { url: 'https://example.com/next' })).rejects.toThrow('blocked navigation')

    expect(getBrowserScreenshotEntries(tab.id)).toEqual([
      expect.objectContaining({ dataUrl: png, title: 'Visible', url: 'https://example.com/live' })
    ])
    expect(getBrowserActionEvents(tab.id)).toEqual([
      expect.objectContaining({ command: 'screenshot', status: 'success' }),
      expect.objectContaining({ command: 'navigate', error: 'blocked navigation', status: 'error', target: 'https://example.com/next' })
    ])
  })

  it('returns and clears visible browser console and network histories', async () => {
    const tab = createBrowserTab({ sessionId: 'session-1', url: 'https://example.com' })

    registerBrowserWebview(tab.id, {})
    updateBrowserTab(tab.id, { controlMode: 'observe' })
    appendBrowserConsoleEntry(tab.id, { level: 'warn', message: 'careful', source: 'console', url: 'https://example.com' })
    appendBrowserNetworkEvent(tab.id, { method: 'GET', status: 204, type: 'response', url: 'https://example.com/ping' })

    await expect(runBrowserBridgeCommand(tab.id, 'getConsole')).resolves.toMatchObject({
      messages: [expect.objectContaining({ level: 'warn', message: 'careful' })]
    })
    await expect(runBrowserBridgeCommand(tab.id, 'getNetwork')).resolves.toMatchObject({
      events: [expect.objectContaining({ status: 204, url: 'https://example.com/ping' })]
    })

    await runBrowserBridgeCommand(tab.id, 'clearConsole')
    await runBrowserBridgeCommand(tab.id, 'clearNetwork')

    expect(getBrowserConsoleEntries(tab.id)).toEqual([])
    expect(getBrowserNetworkEvents(tab.id)).toEqual([])
  })
})
