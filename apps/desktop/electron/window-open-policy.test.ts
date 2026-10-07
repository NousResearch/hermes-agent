// @vitest-environment node
import { describe, expect, it, vi } from 'vitest'

import {
  RABBIT_HUB_ORIGIN,
  isRabbitHubClipboardWrite,
  isRabbitHubExternalUrl,
  isRabbitHubOrigin
} from './hub-iframe-policy'
import { createWindowOpenHandler, describeDeniedUrl } from './window-open-policy'

describe('hub-iframe-policy predicates', () => {
  it('admits exactly the hub origin and nothing else', () => {
    expect(isRabbitHubOrigin(RABBIT_HUB_ORIGIN)).toBe(true)

    // Opaque sandboxed frames, data URLs, look-alikes, and absent origins
    // never qualify.
    expect(isRabbitHubOrigin('null')).toBe(false)
    expect(isRabbitHubOrigin('')).toBe(false)
    expect(isRabbitHubOrigin(null)).toBe(false)
    expect(isRabbitHubOrigin(undefined)).toBe(false)
    expect(isRabbitHubOrigin('https://github.com/seven0070/Rabbit-.evil.example')).toBe(false)
    expect(isRabbitHubOrigin('https://evil.example')).toBe(false)
    expect(isRabbitHubOrigin('file://')).toBe(false)
  })

  it('delegates only http/https/mailto external URLs', () => {
    expect(isRabbitHubExternalUrl('https://github.com/seven0070/Rabbit-')).toBe(true)
    expect(isRabbitHubExternalUrl('http://example.com/docs')).toBe(true)
    expect(isRabbitHubExternalUrl('mailto:support@example.com')).toBe(true)

    expect(isRabbitHubExternalUrl('file:///etc/passwd')).toBe(false)
    expect(isRabbitHubExternalUrl('javascript:alert(1)')).toBe(false)
    expect(isRabbitHubExternalUrl('rabbit://internal')).toBe(false)
    expect(isRabbitHubExternalUrl('not a url')).toBe(false)
  })

  it('grants clipboard-write to hub origins only', () => {
    expect(isRabbitHubClipboardWrite(RABBIT_HUB_ORIGIN)).toBe(true)
    expect(isRabbitHubClipboardWrite('null')).toBe(false)
    expect(isRabbitHubClipboardWrite('https://artifact-preview.invalid')).toBe(false)
    expect(isRabbitHubClipboardWrite(null)).toBe(false)
  })
})

describe('createWindowOpenHandler trusted-hub delegation', () => {
  const baseDetails = { url: 'https://github.com/seven0070/Rabbit-' }

  it('still denies artifact frames (opaque origin) with NO external open', () => {
    const openExternalUrl = vi.fn()

    const handler = createWindowOpenHandler(undefined, {
      getOpenerOrigin: () => 'null',
      openExternalUrl
    })

    expect(handler(baseDetails)).toEqual({ action: 'deny' })
    expect(openExternalUrl).not.toHaveBeenCalled()
  })

  it('delegates a hub-origin http(s) open but still denies the window', () => {
    const openExternalUrl = vi.fn()

    const handler = createWindowOpenHandler(undefined, {
      getOpenerOrigin: () => RABBIT_HUB_ORIGIN,
      openExternalUrl
    })

    expect(handler(baseDetails)).toEqual({ action: 'deny' })
    expect(openExternalUrl).toHaveBeenCalledExactlyOnceWith('https://github.com/seven0070/Rabbit-')
  })

  it('delegates from the fallback (GitHub Pages) hub origin too', () => {
    const openExternalUrl = vi.fn()

    const handler = createWindowOpenHandler(undefined, {
      getOpenerOrigin: () => RABBIT_HUB_ORIGIN,
      openExternalUrl
    })

    expect(handler({ url: 'https://docs.example.com/x' })).toEqual({ action: 'deny' })
    expect(openExternalUrl).toHaveBeenCalledExactlyOnceWith('https://docs.example.com/x')
  })

  it('never delegates file:// or unknown schemes from a hub-origin opener', () => {
    const openExternalUrl = vi.fn()

    const handler = createWindowOpenHandler(undefined, {
      getOpenerOrigin: () => RABBIT_HUB_ORIGIN,
      openExternalUrl
    })

    expect(handler({ url: 'file:///C:/x.html' })).toEqual({ action: 'deny' })
    expect(handler({ url: 'javascript:alert(1)' })).toEqual({ action: 'deny' })
    expect(handler({ url: 'not a url' })).toEqual({ action: 'deny' })
    expect(openExternalUrl).not.toHaveBeenCalled()
  })

  it('never delegates when the opener origin is not exactly the hub', () => {
    const openExternalUrl = vi.fn()

    const handler = createWindowOpenHandler(undefined, {
      getOpenerOrigin: () => 'https://github.com/seven0070/Rabbit-.evil.example',
      openExternalUrl
    })

    expect(handler(baseDetails)).toEqual({ action: 'deny' })
    expect(openExternalUrl).not.toHaveBeenCalled()
  })

  it('a throwing opener probe or external open stays deny-only', () => {
    const openExternalUrl = vi.fn(() => {
      throw new Error('boom')
    })

    const throwingProbe = createWindowOpenHandler(undefined, {
      getOpenerOrigin: () => {
        throw new Error('probe failed')
      },
      openExternalUrl
    })

    expect(throwingProbe(baseDetails)).toEqual({ action: 'deny' })
    expect(openExternalUrl).not.toHaveBeenCalled()

    const throwingOpen = createWindowOpenHandler(undefined, {
      getOpenerOrigin: () => RABBIT_HUB_ORIGIN,
      openExternalUrl
    })

    expect(throwingOpen(baseDetails)).toEqual({ action: 'deny' })
  })

  it('a throwing logging observer cannot change the decision or the delegation', () => {
    const openExternalUrl = vi.fn()

    const handler = createWindowOpenHandler(
      () => {
        throw new Error('log failed')
      },
      { getOpenerOrigin: () => RABBIT_HUB_ORIGIN, openExternalUrl }
    )

    expect(handler(baseDetails)).toEqual({ action: 'deny' })
    expect(openExternalUrl).toHaveBeenCalledOnce()
  })

  it('without trusted options the handler is side-effect-free deny (CVE-2026-70608 posture)', () => {
    const handler = createWindowOpenHandler()

    expect(handler({ url: 'https://anything.example' })).toEqual({ action: 'deny' })
  })
})

describe('describeDeniedUrl', () => {
  it('logs origin only, never the full URL', () => {
    expect(describeDeniedUrl('https://example.com/path?token=secret')).toBe('https://example.com')
    expect(describeDeniedUrl('data:text/html,foo')).toBe('data:')
    expect(describeDeniedUrl('not a url')).toBe('<unparseable>')
  })
})
