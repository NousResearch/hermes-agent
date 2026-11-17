/**
 * Main-process wiring of the hosted artifact frame policy (hosted-artifact-frame.ts):
 * subframe navigations and redirects, and permission requests. Takes Electron's
 * objects structurally, so it is exercised with plain fakes; no Electron import.
 */

import {
  hostedArtifactFrameNavigation,
  type HostedArtifactFramePolicy,
  isHostedArtifactRequestingUrl
} from './hosted-artifact-frame'
import { safeFrameOrigin, safeUrlOrigin } from './subframe'

/** The frame being navigated (Electron's `WebFrameMain`), as far as the fence reads it. */
export interface NavigatingFrame {
  origin?: string
  reload?: () => unknown
}

/** Electron's `will-frame-navigate` / `will-redirect` details event. */
export interface FrameNavigationDetails {
  url: string
  isMainFrame: boolean
  frame?: NavigatingFrame | null
  preventDefault: () => void
}

/** The webContents events the fence listens to. */
export interface FrameNavigationSource {
  on(event: 'will-frame-navigate' | 'will-redirect', listener: (details: FrameNavigationDetails) => void): unknown
}

export interface SubframeNavigationFenceOptions {
  policy: HostedArtifactFramePolicy
  log: (line: string) => void
  /** Subframe navigations the artifact policy leaves alone, with the navigating frame's origin. */
  otherNavigation?: (details: FrameNavigationDetails, frameOrigin: string | null) => void
}

/**
 * Keep hosted artifact frames on their own origin. The navigating frame is the
 * one Electron names in the event (`details.frame`): an artifact that moves
 * itself, a parent that retargets it, and a server redirect on the way are all
 * judged against that frame's current origin.
 */
export function installSubframeNavigationFence(
  contents: FrameNavigationSource,
  { policy, log, otherNavigation }: SubframeNavigationFenceOptions
): void {
  const fenced = (details: FrameNavigationDetails, frameOrigin: string | null): boolean => {
    const verdict = hostedArtifactFrameNavigation(frameOrigin, details.url, policy)

    if (verdict === 'deny') {
      details.preventDefault()
      log(`[hosted-artifact] denied frame navigation to ${safeUrlOrigin(details.url)}`)
    }

    return verdict !== null
  }

  contents.on('will-frame-navigate', details => {
    if (details.isMainFrame) {
      return
    }

    const frameOrigin = safeFrameOrigin(details.frame)

    if (!fenced(details, frameOrigin)) {
      otherNavigation?.(details, frameOrigin)
    }
  })

  contents.on('will-redirect', details => {
    if (!details.isMainFrame) {
      fenced(details, safeFrameOrigin(details.frame))
    }
  })
}

/** Wrap a permission request handler so a hosted artifact frame is granted nothing. */
export function denyHostedArtifactPermissionRequests<W, P, D extends { requestingUrl?: string } | undefined>(
  policy: HostedArtifactFramePolicy,
  handler: (webContents: W, permission: P, callback: (granted: boolean) => void, details: D) => void
): (webContents: W, permission: P, callback: (granted: boolean) => void, details: D) => void {
  return (webContents, permission, callback, details) => {
    if (isHostedArtifactRequestingUrl(details?.requestingUrl, policy)) {
      callback(false)

      return
    }

    handler(webContents, permission, callback, details)
  }
}

/** Wrap a permission check handler so a hosted artifact frame is granted nothing. */
export function denyHostedArtifactPermissionChecks<W, P, R extends unknown[]>(
  policy: HostedArtifactFramePolicy,
  handler: (webContents: W, permission: P, requestingOrigin: string, ...rest: R) => boolean
): (webContents: W, permission: P, requestingOrigin: string, ...rest: R) => boolean {
  return (webContents, permission, requestingOrigin, ...rest) =>
    !isHostedArtifactRequestingUrl(requestingOrigin, policy) &&
    handler(webContents, permission, requestingOrigin, ...rest)
}
