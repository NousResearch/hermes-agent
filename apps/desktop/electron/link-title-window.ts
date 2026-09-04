// Hidden BrowserWindow used by tier-2 link-title resolution: when curl can't
// read a page <title> (bot walls, JS-rendered pages), we briefly load the URL
// in an offscreen window and read its title. That window loads arbitrary
// user-linked pages, so it must never emit sound or trigger real downloads,
// and its requests are gated through the same SSRF verdict as the tier-1
// fetch (see guardLinkTitleSession — review B6).

import { isPrivateAddress, isPrivateHostname } from './link-preview'

// Resource types we cancel before the network even fires — keeps the hidden
// renderer fast and cuts third-party tracking noise.
export const RENDER_TITLE_BLOCKED_RESOURCES = new Set([
  'cspReport',
  'font',
  'imageset',
  'media',
  'object',
  'ping',
  'stylesheet'
])

export function linkTitleWindowOptions(partitionSession) {
  return {
    show: false,
    width: 1280,
    height: 800,
    webPreferences: {
      // Deliberately throttled: this hidden window loads arbitrary user-linked
      // pages, and an unthrottled heavy page burns full CPU for the window's
      // whole lifetime. Title resolution rides load events
      // (page-title-updated / did-finish-load) plus main-process timers, none
      // of which the renderer clamp touches — hidden-page throttling only
      // slows the page's own timer-driven JS, and the grace window already
      // absorbs that.
      contextIsolation: true,
      javascript: true,
      nodeIntegration: false,
      sandbox: true,
      session: partitionSession,
      webSecurity: true
    }
  }
}

// Create the offscreen title-fetch window and immediately mute it. Without the
// mute, autoplaying media on the loaded page (e.g. a YouTube link) leaks ~2s of
// audio every time a session containing such links is re-rendered. See #49505.
export function createLinkTitleWindow(BrowserWindow, partitionSession) {
  const window = new BrowserWindow(linkTitleWindowOptions(partitionSession))

  try {
    window.webContents.setAudioMuted(true)
  } catch {
    // webContents may be unavailable in degraded/headless environments; muting
    // is best-effort and the window is destroyed within a few seconds anyway.
  }

  return window
}

// Cancel any download the title-fetch window triggers. Without this, a link
// artifact URL served with Content-Disposition: attachment auto-downloads every
// time the Artifacts page renders and fetchLinkTitle loads it.
//
// The SSRF half (review B6): tier 1's per-hop curl guard (fetchWithGuardedRedirects)
// vets only URLs *we* fetch. When tier 1 comes back empty, runRenderTitleJob loads
// the original URL here and CHROMIUM walks the redirect chain — every hop is a
// fresh request, and Chromium resolves each target's DNS itself. So every request
// this window makes (initial load, each 30x hop, allowed subresources) is gated
// through the SAME verdict tier 1 applies, via webRequest.onBeforeRequest — whose
// callback may be invoked asynchronously, which is what makes our own DNS
// resolution possible at the chokepoint. No dependency on details.ip timing:
// when Chromium does supply it, it only ever tightens the verdict.
//
// Residual (same class as tier 1, out of threat model): our resolution and
// Chromium's are separate lookups, so a TOCTOU swap between them is theoretical;
// pinning Chromium's connection to our answers is not worth the resolver-rules
// plumbing for a short-lived title window.
export function decideLinkTitleRequest(
  resourceType: string,
  url: string,
  { connectedIp, resolvedAddresses }: { connectedIp?: string; resolvedAddresses?: string[] } = {}
) {
  if (RENDER_TITLE_BLOCKED_RESOURCES.has(resourceType)) {
    return true
  }

  let hostname = ''

  try {
    hostname = new URL(url).hostname
  } catch {
    // A request without a parsable URL is not one we can vouch for.
    return true
  }

  // Name-level half: localhost shapes, single labels, .local/.internal, and
  // literal private IPs.
  if (isPrivateHostname(hostname)) {
    return true
  }

  // Chromium's own view of the connection, when available.
  if (connectedIp && isPrivateAddress(connectedIp)) {
    return true
  }

  // DNS half — only judged once resolution has actually run (resolvedAddresses
  // undefined = not yet resolved, e.g. the synchronous pre-check). An empty
  // answer denies: unresolvable names would fail to connect anyway, so denying
  // is the fail-safe direction.
  if (resolvedAddresses !== undefined && (!resolvedAddresses.length || resolvedAddresses.some(isPrivateAddress))) {
    return true
  }

  return false
}

export function guardLinkTitleSession(
  partitionSession: any,
  { resolveHost }: { resolveHost?: (hostname: string) => Promise<string[]> } = {}
) {
  try {
    partitionSession.on('will-download', (_event, item) => item.cancel())
  } catch {
    // best-effort; worst case is a spurious download
  }

  try {
    const webRequest = partitionSession.webRequest

    if (!webRequest || typeof webRequest.onBeforeRequest !== 'function') {
      return
    }

    const resolveHostOrDeny = resolveHost || (() => Promise.resolve([]))

    webRequest.onBeforeRequest((details, callback) => {
      // Synchronous verdicts first: resource blocks and hostname-shaped
      // refusals answer without a DNS round-trip.
      if (decideLinkTitleRequest(details.resourceType, details.url, { connectedIp: details.ip })) {
        callback({ cancel: true })

        return
      }

      const { hostname } = new URL(details.url)

      resolveHostOrDeny(hostname)
        .catch(() => [])
        .then(addresses => callback({ cancel: decideLinkTitleRequest(details.resourceType, details.url, { resolvedAddresses: addresses }) }))
    })
  } catch {
    // best-effort; worst case the window loads unguarded behind tier 1's
    // per-hop curl guard — never let a guard failure take down the window.
  }
}

// Read the page title from a title-fetch window. Callers schedule this from
// timers that can fire after finish() destroys the window, so every access must
// guard isDestroyed and swallow Electron's "Object has been destroyed" throws.
export function readLinkTitleWindowTitle(window) {
  try {
    if (!window || window.isDestroyed()) {
      return ''
    }

    const contents = window.webContents

    if (!contents || contents.isDestroyed()) {
      return ''
    }

    return contents.getTitle() || ''
  } catch {
    return ''
  }
}
