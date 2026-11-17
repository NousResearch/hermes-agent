import { describe, expect, it, vi } from 'vitest'

import {
  denyHostedArtifactPermissionChecks,
  denyHostedArtifactPermissionRequests,
  type FrameNavigationDetails,
  installSubframeNavigationFence,
  type NavigatingFrame
} from './hosted-artifact-fence'

const ID = '01j9z8y7x6w5v4t3s2r1q0p9n8'
const DOMAIN = 'frame.example'
const POLICY = { dev: false, frameDomain: DOMAIN }
const ARTIFACT = `https://${ID}.${DOMAIN}`
const VIEWER = 'file://'

type Listener = (details: FrameNavigationDetails) => void

/** A stand-in webContents: records listeners by event, emits Electron-shaped details events. */
function fakeContents() {
  const listeners = new Map<string, Listener[]>()

  return {
    on(event: string, listener: Listener) {
      listeners.set(event, [...(listeners.get(event) ?? []), listener])

      return this
    },
    emit(
      event: 'will-frame-navigate' | 'will-redirect',
      url: string,
      frame: NavigatingFrame | null,
      isMainFrame = false
    ) {
      const details = { url, isMainFrame, frame, preventDefault: vi.fn() }

      for (const listener of listeners.get(event) ?? []) {
        listener(details)
      }

      return details
    }
  }
}

function fenced(otherNavigation = vi.fn()) {
  const contents = fakeContents()
  const log = vi.fn()

  installSubframeNavigationFence(contents, { policy: POLICY, log, otherNavigation })

  return { contents, log, otherNavigation }
}

describe('subframe navigation fence', () => {
  it('stops an artifact frame from navigating itself to another origin', () => {
    const { contents, log, otherNavigation } = fenced()
    const details = contents.emit('will-frame-navigate', 'https://evil.test/steal?x=1', { origin: ARTIFACT })

    expect(details.preventDefault).toHaveBeenCalledOnce()
    expect(otherNavigation).not.toHaveBeenCalled()
    expect(log.mock.calls[0][0]).toContain('https://evil.test')
    expect(log.mock.calls[0][0]).not.toContain('steal')
  })

  it('lets an artifact frame move within its own origin', () => {
    const { contents, otherNavigation } = fenced()
    const details = contents.emit('will-frame-navigate', `${ARTIFACT}/page.html`, { origin: ARTIFACT })

    expect(details.preventDefault).not.toHaveBeenCalled()
    expect(otherNavigation).not.toHaveBeenCalled()
  })

  it('lets the viewer enter an artifact origin this build frames, and nothing under the domain else', () => {
    const { contents } = fenced()

    expect(
      contents.emit('will-frame-navigate', `${ARTIFACT}/__av?t=x`, { origin: VIEWER }).preventDefault
    ).not.toHaveBeenCalled()
    expect(
      contents.emit('will-frame-navigate', `https://notanid.${DOMAIN}/`, { origin: VIEWER }).preventDefault
    ).toHaveBeenCalledOnce()
  })

  it('fences server redirects of an artifact frame too', () => {
    const { contents } = fenced()

    expect(
      contents.emit('will-redirect', 'https://evil.test/', { origin: ARTIFACT }).preventDefault
    ).toHaveBeenCalledOnce()
    expect(contents.emit('will-redirect', `${ARTIFACT}/`, { origin: VIEWER }).preventDefault).not.toHaveBeenCalled()
    expect(contents.emit('will-redirect', `${ARTIFACT}/`, { origin: ARTIFACT }).preventDefault).not.toHaveBeenCalled()
  })

  it('hands every other subframe navigation on with the frame origin, and leaves the main frame alone', () => {
    const { contents, otherNavigation } = fenced()
    const hub = { origin: 'https://hermes-agent.nousresearch.com' }
    const details = contents.emit('will-frame-navigate', 'https://example.com/', hub)

    expect(otherNavigation).toHaveBeenCalledWith(details, hub.origin)
    expect(details.preventDefault).not.toHaveBeenCalled()

    otherNavigation.mockClear()
    const main = contents.emit('will-frame-navigate', 'https://evil.test/', { origin: ARTIFACT }, true)

    expect(main.preventDefault).not.toHaveBeenCalled()
    expect(otherNavigation).not.toHaveBeenCalled()
  })

  it('treats a destroyed frame as having no origin', () => {
    const { contents, otherNavigation } = fenced()

    contents.emit('will-frame-navigate', 'https://example.com/', null)
    contents.emit('will-frame-navigate', 'https://example.com/', { origin: 'null' })

    expect(otherNavigation.mock.calls.map(call => call[1])).toEqual([null, null])
  })
})

describe('hosted artifact permission fence', () => {
  it('denies an artifact frame every permission request and passes the rest through', () => {
    const inner = vi.fn((_wc: unknown, _p: string, callback: (granted: boolean) => void) => callback(true))
    const handler = denyHostedArtifactPermissionRequests(POLICY, inner)
    const callback = vi.fn()

    handler({}, 'media', callback, { requestingUrl: `${ARTIFACT}/` })
    expect(callback).toHaveBeenLastCalledWith(false)
    expect(inner).not.toHaveBeenCalled()

    handler({}, 'media', callback, { requestingUrl: 'file:///index.html' })
    expect(callback).toHaveBeenLastCalledWith(true)
    expect(inner).toHaveBeenCalledOnce()
  })

  it('answers every permission check from an artifact frame with false', () => {
    const inner = vi.fn(() => true)
    const check = denyHostedArtifactPermissionChecks(POLICY, inner)

    expect(check({}, 'media', ARTIFACT)).toBe(false)
    expect(inner).not.toHaveBeenCalled()
    expect(check({}, 'media', 'file://')).toBe(true)
  })
})
