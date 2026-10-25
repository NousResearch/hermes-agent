/**
 * The picker open-bridge: the frame-identity gate and the injected shim.
 *
 * The shim is asserted by RUNNING the shipped source string against a minimal
 * fake window/document — a string match would happily pass on a shim that never
 * posts anything. `new Function` is used only with `HUB_PICKER_OPEN_BRIDGE_SOURCE`,
 * a constant defined in this repo (electron/hub-picker-bridge.ts); no test input
 * and nothing from a guest frame ever reaches the function body.
 */
import { describe, expect, it } from 'vitest'

import { isBridgedPickerNavigation, isHermesHubPickerUrl } from './hub-iframe-policy'
import { HUB_PICKER_OPEN_BRIDGE_SOURCE, HUB_PICKER_OPEN_MESSAGE } from './hub-picker-bridge'

const HUB = 'https://hermes-agent.nousresearch.com'
const FALLBACK = 'https://nousresearch.github.io'
const PICKER = `${HUB}/docs/plugins?embed=picker`

describe('isHermesHubPickerUrl', () => {
  it('accepts both deployments of both pickers', () => {
    expect(isHermesHubPickerUrl(`${HUB}/docs/skills?embed=picker`)).toBe(true)
    expect(isHermesHubPickerUrl(`${HUB}/hermes-agent/docs/skills`)).toBe(true)
    expect(isHermesHubPickerUrl(PICKER)).toBe(true)
    expect(isHermesHubPickerUrl(`${FALLBACK}/hermes-agent/docs/plugins`)).toBe(true)
  })

  it('rejects the full docs site, other origins and junk', () => {
    expect(isHermesHubPickerUrl(`${HUB}/docs/plugins/accent-picker`)).toBe(false)
    expect(isHermesHubPickerUrl(`${HUB}/docs/api/plugins.json`)).toBe(false)
    expect(isHermesHubPickerUrl('https://evil.example/docs/plugins')).toBe(false)
    expect(isHermesHubPickerUrl('not a url')).toBe(false)
  })
})

describe('isBridgedPickerNavigation', () => {
  it('bridges a committed hub picker navigation', () => {
    expect(isBridgedPickerNavigation(PICKER, HUB)).toBe(true)
  })

  it('never bridges without a live hub-origin frame', () => {
    // A sandboxed artifact frame reports an opaque origin, whatever URL it
    // navigated to — the whole point of checking the live frame, not the URL.
    expect(isBridgedPickerNavigation(PICKER, null)).toBe(false)
    expect(isBridgedPickerNavigation(PICKER, 'null')).toBe(false)
    expect(isBridgedPickerNavigation(PICKER, 'https://evil.example')).toBe(false)
  })

  it('never bridges a non-picker navigation from a hub frame', () => {
    expect(isBridgedPickerNavigation(`${HUB}/docs/plugins/accent-picker`, HUB)).toBe(false)
  })
})

type PostedMessage = { type?: unknown; url?: unknown }

/** Install the shipped shim over a fake window/document and hand back the seams. */
function installBridge() {
  const posted: PostedMessage[] = []
  const nativeOpens: string[] = []
  const clickHandlers: Array<(event: unknown) => void> = []

  const fakeWindow: Record<string, unknown> = {
    location: { href: PICKER },
    open: (url: unknown) => {
      nativeOpens.push(String(url))

      return null
    },
    parent: { postMessage: (message: PostedMessage) => posted.push(message) }
  }

  const fakeDocument = {
    addEventListener: (type: string, handler: (event: unknown) => void) => {
      if (type === 'click') {
        clickHandlers.push(handler)
      }
    }
  }

  // `return` because the shim ships as an IIFE: Electron's executeJavaScript
  // yields its completion value, while a plain function body would discard it.
  const install = new Function('window', 'document', `return ${HUB_PICKER_OPEN_BRIDGE_SOURCE}`) as (
    w: unknown,
    d: unknown
  ) => string

  const result = install(fakeWindow, fakeDocument)

  const clicks = { prevented: 0, stopped: 0 }

  const click = (href: string, target: string) => {
    for (const handler of clickHandlers) {
      handler({
        preventDefault: () => {
          clicks.prevented += 1
        },
        stopPropagation: () => {
          clicks.stopped += 1
        },
        target: { closest: () => ({ href, target }) }
      })
    }
  }

  return {
    click,
    clickHandlerCount: clickHandlers.length,
    clicks,
    fakeWindow,
    install,
    installAgain: () => install(fakeWindow, fakeDocument),
    nativeOpens,
    open: fakeWindow.open as (url: string) => unknown,
    posted,
    result
  }
}

describe('HUB_PICKER_OPEN_BRIDGE_SOURCE', () => {
  it('installs a click listener once and reports it', () => {
    const bridge = installBridge()

    expect(bridge.result).toBe('installed')
    expect(bridge.clickHandlerCount).toBe(1)
  })

  it('does not double-wrap on re-injection', () => {
    const bridge = installBridge()

    expect(bridge.installAgain()).toBe('present')

    bridge.open('https://github.com/NousResearch/hermes-agent')

    expect(bridge.posted).toEqual([
      { type: HUB_PICKER_OPEN_MESSAGE, url: 'https://github.com/NousResearch/hermes-agent' }
    ])
  })

  it('bridges http, mailto and relative window.open targets', () => {
    const bridge = installBridge()

    bridge.open('https://github.com/NousResearch/hermes-agent')
    bridge.open('mailto:plugins@nousresearch.com')
    bridge.open('/docs/plugins/accent-picker')

    expect(bridge.posted).toEqual([
      { type: HUB_PICKER_OPEN_MESSAGE, url: 'https://github.com/NousResearch/hermes-agent' },
      { type: HUB_PICKER_OPEN_MESSAGE, url: 'mailto:plugins@nousresearch.com' },
      { type: HUB_PICKER_OPEN_MESSAGE, url: `${HUB}/docs/plugins/accent-picker` }
    ])
    // Nothing was attempted as a real popup.
    expect(bridge.nativeOpens).toEqual([])
  })

  it('leaves non-openable schemes to the native path the CVE handler denies', () => {
    const bridge = installBridge()

    bridge.open('javascript:alert(1)')
    bridge.open('file:///etc/passwd')
    bridge.open('')

    expect(bridge.posted).toEqual([])
    expect(bridge.nativeOpens).toEqual(['javascript:alert(1)', 'file:///etc/passwd'])
  })

  it('intercepts target=_blank anchors in the capture phase, and only those', () => {
    const bridge = installBridge()

    bridge.click('https://github.com/NousResearch/hermes-agent', '_blank')
    bridge.click('https://github.com/NousResearch/hermes-agent', '')
    bridge.click('javascript:alert(1)', '_blank')

    expect(bridge.posted).toEqual([
      { type: HUB_PICKER_OPEN_MESSAGE, url: 'https://github.com/NousResearch/hermes-agent' }
    ])
    expect(bridge.clicks).toEqual({ prevented: 1, stopped: 1 })
  })
})
