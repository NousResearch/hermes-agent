import type { WebFrameMain } from 'electron'
import { getDomain } from 'tldts'

export interface LinkTitleRequestDetails {
  frame?: WebFrameMain | { top?: { url?: string } | null; url?: string } | null
  resourceType?: string
  url?: string
  webContentsId?: number
}

/**
 * Compute the registrable domain (eTLD+1) for public suffixes, or the exact
 * hostname for IP literals, localhost, and single-label hosts.
 *
 * Uses the Public Suffix List via tldts to distinguish eTLD+1 for multi-label
 * public suffixes (*.com.cn, *.co.uk) without treating subdomains as distinct sites.
 */
export function computeSite(rawUrl: string): string {
  if (!rawUrl) {
    return ''
  }

  let hostname = ''

  try {
    const url = new URL(rawUrl)

    if (url.protocol === 'blob:' && url.origin && url.origin !== 'null') {
      try {
        hostname = new URL(url.origin).hostname.toLowerCase()
      } catch {
        hostname = ''
      }
    } else {
      hostname = url.hostname.toLowerCase()
    }
  } catch {
    hostname = String(rawUrl).trim().toLowerCase()
  }

  if (!hostname) {
    return ''
  }

  const cleanHost = hostname.replace(/^\[|\]$/g, '')
  const domain = getDomain(cleanHost)

  return domain || cleanHost
}

/** True when both targets resolve to the same registrable domain or exact host. */
export function isSameSite(urlA: string, urlB: string): boolean {
  const siteA = computeSite(urlA)
  const siteB = computeSite(urlB)

  return Boolean(siteA && siteA === siteB)
}

/**
 * Pure decision for link-title requests.
 *
 * - mainFrame requests (initial navigations and redirect hops) are never blocked.
 * - Non-mainFrame requests with unresolvable top-level site fail closed.
 * - Subresource requests whose site matches the top-level document are allowed;
 *   third-party trackers, beacons, ad scripts, and cross-site frames are cancelled.
 */
export function shouldBlockThirdPartyTitleRequest(
  details: { resourceType?: string; url?: string },
  topLevelUrl?: string
): boolean {
  if (details.resourceType === 'mainFrame') {
    return false
  }

  if (!topLevelUrl) {
    return true
  }

  return !isSameSite(details.url ?? '', topLevelUrl)
}

export class LinkTitleRequestTracker {
  private static readonly MAX_TRACKED_WINDOWS = 50
  private readonly topLevelUrls = new Map<number, string>()

  recordMainFrame(webContentsId: number, url: string): void {
    if (this.topLevelUrls.size >= LinkTitleRequestTracker.MAX_TRACKED_WINDOWS) {
      const oldestKey = this.topLevelUrls.keys().next().value

      if (oldestKey !== undefined) {
        this.topLevelUrls.delete(oldestKey)
      }
    }

    this.topLevelUrls.set(webContentsId, url)
  }

  getTopLevelUrl(webContentsId?: number): string | undefined {
    return webContentsId !== undefined ? this.topLevelUrls.get(webContentsId) : undefined
  }

  deleteWebContents(webContentsId: number): void {
    this.topLevelUrls.delete(webContentsId)
  }

  reset(): void {
    this.topLevelUrls.clear()
  }

  shouldBlock(details: LinkTitleRequestDetails): boolean {
    if (details.resourceType === 'mainFrame') {
      if (details.webContentsId && details.url) {
        this.recordMainFrame(details.webContentsId, details.url)
      }

      return false
    }

    const topLevelUrl =
      (details.webContentsId ? this.getTopLevelUrl(details.webContentsId) : undefined) ||
      details.frame?.top?.url ||
      details.frame?.url

    return shouldBlockThirdPartyTitleRequest(details, topLevelUrl)
  }

  bindSession(
    partitionSession?: any,
    electronApp?: any
  ): void {
    try {
      electronApp?.on?.('web-contents-created', (_event: unknown, wc: { id?: number; once?: (event: string, fn: () => void) => void; session?: unknown }) => {
        try {
          if (partitionSession && wc?.session === partitionSession && typeof wc?.id === 'number') {
            const id = wc.id
            wc.once?.('destroyed', () => {
              this.deleteWebContents(id)
            })
          }
        } catch {
          // best-effort cleanup
        }
      })
    } catch {
      // best-effort cleanup
    }
  }
}

export function createLinkTitleRequestTracker(): LinkTitleRequestTracker {
  return new LinkTitleRequestTracker()
}
