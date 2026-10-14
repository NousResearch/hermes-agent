import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $rightRailActiveTabId } from '@/store/layout'
import { closeRightRail, openPreview } from '@/store/preview'

import { actOnActivePreview } from './preview-act'
import { dragFrom } from './preview-drive'
import { type PreviewInputEvent, registerPreviewInput } from './preview-input'
import { capturePreviewGuest, type PreviewGuest } from './preview-input-guest'
import { registerPreviewScriptRunner } from './preview-script-runner'

const action = { kind: 'drag', selector: '#handle', dx: 40, dy: 20 }
const cleanups: Array<() => void> = []

function pane() {
  openPreview({ kind: 'url', label: 'Fixture', source: 'https://example.test', url: 'https://example.test' })
  const id = $rightRailActiveTabId.get()!
  const events: PreviewInputEvent[] = []
  let generation = 7

  const guest: PreviewGuest = {
    isConnected: true,
    getWebContentsId: () => generation,
    getZoomFactor: () => 1,
    focus: vi.fn(),
    sendInputEvent: vi.fn(event => { events.push(event) }),
    executeJavaScript: vi.fn(async () => JSON.stringify({ success: true, point: { x: 100, y: 80 }, hit: { trusted: true } }))
  }

  let current: PreviewGuest | null = guest
  cleanups.push(registerPreviewInput(id, () => capturePreviewGuest(() => current)))
  const mutableRunner = vi.fn(async () => 'unused')
  cleanups.push(registerPreviewScriptRunner(id, mutableRunner))

  return { events, guest, mutableRunner, replace: (other: PreviewGuest | null) => { current = other }, regenerate: () => { generation++ } }
}

beforeEach(() => {
  vi.useFakeTimers()
  closeRightRail()
})

afterEach(() => {
  for (const cleanup of cleanups.splice(0)) {cleanup()}
  vi.restoreAllMocks()
  vi.useRealTimers()
})

async function finish<T>(pending: Promise<T>): Promise<T> {
  await vi.advanceTimersByTimeAsync(2000)

  return pending
}

describe('native preview drag boundary and ownership', () => {
  it.each([
    { dx: '40' }, { dx: true }, { dy: null }, { dx: NaN }, { dy: Infinity }, { dx: 2001 },
    { dy: -2001 }, { dx: 0, dy: 0 }, { selector: '' }, { selector: '  ' }, { selector: null },
    { ref: 'btn-handle' }, { text: null }, { amount: 0 }, { full: true }, { unknown: 1 },
    { kind: 'click' }, { dx: undefined }
  ])('rejects raw invalid payload before locate/focus/input: %j', async change => {
    const { guest } = pane()
    const result = await actOnActivePreview({ ...action, ...change } as never)
    expect(result.success).toBe(false)
    expect(guest.executeJavaScript).not.toHaveBeenCalled()
    expect(guest.focus).not.toHaveBeenCalled()
    expect(guest.sendInputEvent).not.toHaveBeenCalled()
  })

  it('does not substitute a synthetic drag when native input is absent', async () => {
    const p = pane()
    p.replace(null)
    expect((await actOnActivePreview(action)).success).toBe(false)
    expect(p.mutableRunner).not.toHaveBeenCalled()
  })

  it('pins original readback and scales guest CSS coordinates once', async () => {
    const p = pane()
    p.guest.getZoomFactor = () => 1.25
    const pending = actOnActivePreview(action)
    await vi.advanceTimersByTimeAsync(50)
    $rightRailActiveTabId.set(null)
    const result = await finish(pending)
    expect(result.success).toBe(true)
    expect(result.acted).toBe('drag input delivered')
    expect(p.mutableRunner).not.toHaveBeenCalled()
    expect(p.guest.executeJavaScript).toHaveBeenCalledTimes(2)
    const down = p.events.filter(e => e.type === 'mouseDown')
    const up = p.events.filter(e => e.type === 'mouseUp')
    expect(down).toEqual([{ button: 'left', clickCount: 1, type: 'mouseDown', x: 125, y: 100 }])
    expect(up).toEqual([{ button: 'left', clickCount: 1, type: 'mouseUp', x: 175, y: 125 }])
    const held = p.events.filter(e => e.type === 'mouseMove' && e.modifiers?.includes('leftbuttondown'))
    expect(held.length).toBeGreaterThan(1)
  })

  it.each(['drag', 'click', 'hover', 'type', 'scroll', 'press'])('excludes overlapping %s before locate', async kind => {
    const p = pane()
    let unblock!: (value: string) => void
    vi.mocked(p.guest.executeJavaScript).mockImplementationOnce(() => new Promise(resolve => { unblock = resolve }))
    const first = actOnActivePreview(action)
    const calls = vi.mocked(p.guest.executeJavaScript).mock.calls.length
    const second = await actOnActivePreview(kind === 'drag' ? action : { kind, selector: '#handle', ...(kind === 'type' ? { text: 'x' } : {}) })
    expect(second.error).toMatch(/Another interaction/)
    expect(p.guest.executeJavaScript).toHaveBeenCalledTimes(calls)
    unblock(JSON.stringify({ success: true, point: { x: 100, y: 80 } }))
    expect((await finish(first)).success).toBe(true)
    expect((await finish(actOnActivePreview(action))).success).toBe(true)
  })

  it.each(['before', 'approach', 'down', 'held', 'readback'])('cancels at %s and releases only when down was attempted', async stage => {
    const p = pane()
    const abort = new AbortController()

    if (stage === 'before') {abort.abort()}
    vi.mocked(p.guest.sendInputEvent).mockImplementation(event => {
      p.events.push(event)

      if ((stage === 'approach' && event.type === 'mouseMove') ||
          (stage === 'down' && event.type === 'mouseDown') ||
          (stage === 'held' && event.type === 'mouseMove' && event.modifiers?.length)) {abort.abort()}
    })

    if (stage === 'readback') {vi.mocked(p.guest.executeJavaScript).mockImplementationOnce(async () =>
      JSON.stringify({ success: true, point: { x: 100, y: 80 } })).mockImplementationOnce(async () => {
        abort.abort()

        return JSON.stringify({ success: true, hit: { trusted: true } })
      })}

    const result = await finish(actOnActivePreview(action, abort.signal))
    expect(result.success).toBe(false)
    const down = p.events.some(e => e.type === 'mouseDown')
    const up = p.events.filter(e => e.type === 'mouseUp')
    expect(up).toHaveLength(down ? 1 : 0)

    if (down) {
      const last = p.events.filter(e => e.type === 'mouseDown' || (e.type === 'mouseMove' && e.modifiers?.length)).at(-1)!
      expect(up[0]).toMatchObject({ x: 'x' in last ? last.x : 0, y: 'y' in last ? last.y : 0 })
    }

    expect((await finish(actOnActivePreview(action))).success).toBe(true)
  })

  it.each(['mouseDown', 'held', 'readback', 'mouseUp'])('unwinds a %s exception and permits a later gesture', async stage => {
    const p = pane()
    vi.mocked(p.guest.sendInputEvent).mockImplementation(event => {
      p.events.push(event)

      if (event.type === stage || (stage === 'held' && event.type === 'mouseMove' && event.modifiers?.length)) {throw new Error('fixture failure')}
    })

    if (stage === 'readback') {vi.mocked(p.guest.executeJavaScript).mockResolvedValueOnce(JSON.stringify({ success: true, point: { x: 100, y: 80 } })).mockRejectedValueOnce(new Error('readback failure'))}
    const result = await finish(actOnActivePreview(action))
    expect(result.success).toBe(false)
    expect(p.events.filter(e => e.type === 'mouseUp')).toHaveLength(1)

    if (stage === 'mouseUp') {expect(result.error).toContain('release')}
    vi.mocked(p.guest.sendInputEvent).mockImplementation(event => { p.events.push(event) })
    expect((await finish(actOnActivePreview(action))).success).toBe(true)
  })

  it.each(['mouseDown', 'held', 'mouseUp'])('awaits an asynchronous %s rejection from the Electron webview', async stage => {
    const p = pane()
    vi.mocked(p.guest.sendInputEvent).mockImplementation(async event => {
      p.events.push(event)

      if (event.type === stage || (stage === 'held' && event.type === 'mouseMove' && event.modifiers?.length)) {
        throw new Error('asynchronous guest failure')
      }
    })
    const result = await finish(actOnActivePreview(action))
    expect(result.success).toBe(false)
    expect(result.error).toContain('asynchronous guest failure')
    expect(p.events.filter(event => event.type === 'mouseUp')).toHaveLength(1)
  })

  it('retains the guest lease until the asynchronous release has settled', async () => {
    const p = pane()
    let released!: () => void
    vi.mocked(p.guest.sendInputEvent).mockImplementation(event => {
      p.events.push(event)

      if (event.type === 'mouseUp') {return new Promise<void>(resolve => { released = resolve })}
    })
    const first = actOnActivePreview(action)
    await vi.advanceTimersByTimeAsync(1000)
    expect(released).toBeTypeOf('function')
    expect((await actOnActivePreview(action)).error).toContain('Another interaction')
    released()
    expect((await finish(first)).success).toBe(true)
  })

  it.each(['object', 'generation', 'destroyed'])('never sends replacement input when the %s changes', async mode => {
    const p = pane()
    const other = { ...p.guest, sendInputEvent: vi.fn() }
    vi.mocked(p.guest.sendInputEvent).mockImplementation(event => {
      p.events.push(event)

      if (event.type === 'mouseDown') {
        if (mode === 'object') {p.replace(other)}

        if (mode === 'generation') {p.regenerate()}

        if (mode === 'destroyed') {p.guest.isConnected = false}
      }
    })
    const result = await finish(actOnActivePreview(action))
    expect(result.success).toBe(false)
    expect(other.sendInputEvent).not.toHaveBeenCalled()
    expect(p.events.filter(e => e.type === 'mouseUp')).toHaveLength(mode === 'object' ? 1 : 0)
  })

  it('different guests can receive independent gestures', async () => {
    const a = pane()
    const first = actOnActivePreview(action)
    const b = pane()
    const second = actOnActivePreview(action)
    const results = await finish(Promise.all([first, second]))
    expect(results.every(result => result.success)).toBe(true)
    expect(a.events.filter(e => e.type === 'mouseUp')).toHaveLength(1)
    expect(b.events.filter(e => e.type === 'mouseUp')).toHaveLength(1)
  })

  it.each([null, { trusted: false }])('does not credit an absent or synthetic witness: %j', async hit => {
    const p = pane()
    vi.mocked(p.guest.executeJavaScript).mockResolvedValueOnce(JSON.stringify({ success: true, point: { x: 100, y: 80 } })).mockResolvedValueOnce(JSON.stringify({ success: true, hit }))
    expect((await finish(actOnActivePreview(action))).success).toBe(false)
  })
})

it('the driver preserves both the movement error and release failure', async () => {
  const input = { focus: vi.fn(), send: vi.fn((event: PreviewInputEvent) => {
    if (event.type === 'mouseDown') {throw new Error('down failed')}

    if (event.type === 'mouseUp') {throw new Error('up failed')}
  }) }

  const result = dragFrom(input, { x: 20, y: 20 }, { x: 10, y: 0 }).catch(error => error)
  expect((await finish(result)).message).toMatch(/down failed.*release.*up failed/)
})

it('resolves inventory refs and non-inventoried SVG through the serialized production locator', async () => {
  vi.useRealTimers()
  const p = pane()
  document.body.innerHTML = '<button id="handle">Resize</button><svg><circle id="svg-handle" /></svg>'
  vi.spyOn(Element.prototype, 'getBoundingClientRect').mockReturnValue({ x: 90, y: 70, left: 90, top: 70, width: 20, height: 20, right: 110, bottom: 90, toJSON: () => ({}) })
  vi.mocked(p.guest.executeJavaScript).mockImplementation(async code => new Function('return ' + code)())
  const inventory = await actOnActivePreview({ kind: 'elements' })
  const ref = inventory.elements?.find(el => el.label === 'Resize')?.ref
  expect(ref).toBeTruthy()

  for (const target of [{ ref }, { selector: '#svg-handle' }]) {
    p.events.length = 0
    // jsdom cannot produce trusted input. The actual locator executes, but
    // readback must honestly refuse success without a trusted witness.
    const result = await actOnActivePreview({ kind: 'drag', ...target, dx: 40, dy: 20 })
    expect(p.events.find(e => e.type === 'mouseDown')).toMatchObject({ x: 100, y: 80 })
    expect(result.success).toBe(false)
    expect(result.error).toContain('trusted')
  }
})
