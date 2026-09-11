/**
 * Runtime owner for link titles and link previews (D7).
 *
 * link-preview.ts holds the POLICY — pure, injected I/O, fully unit-tested.
 * This file holds the RUNTIME: the production I/O those pure functions are
 * wired to (the pinned one-hop curl fetcher, the DNS resolver, the hidden
 * title window, the durable cache) and the IPC surface that exposes them.
 * main.ts stays the narrow composition seam: it constructs the runtime and
 * registers it, nothing more.
 *
 * Everything that touches the network for a preview goes through the same
 * per-hop admission with DNS pinning (fetchWithGuardedRedirects /
 * resolveThumbnail): a connection is only ever opened to an address the guard
 * just vetted, never to a hostname the transport resolves on its own.
 */

import { spawn } from 'node:child_process'
import dns from 'node:dns'
import fs from 'node:fs'
import path from 'node:path'

import {
  fetchWithGuardedRedirects,
  HostRateLimiter,
  type HttpHopResponse,
  type LinkPreviewIo,
  type LinkPreviewResult,
  LinkPreviewStore,
  PREVIEW_MAX_REDIRECTS,
  resolveLinkPreview,
  resolveThumbnail
} from './link-preview'
import { createLinkTitleWindow, guardLinkTitleSession, readLinkTitleWindowTitle } from './link-title-window'
import { hiddenWindowsChildOptions } from './windows-child-options'

// Browser-shaped UA — many bot-walled sites (GetYourGuide, Cloudflare-protected
// pages) refuse anything that doesn't look like a real Chrome.
export const TITLE_USER_AGENT =
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36'

const TITLE_BYTE_BUDGET = 96 * 1024
const TITLE_TIMEOUT_MS = 5000
const TITLE_CACHE_LIMIT = 500

const TITLE_ERROR_RE =
  /\b(access denied|attention required|captcha|error|forbidden|just a moment|request blocked|too many requests)\b/i

const HTML_ENTITIES = { amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", nbsp: ' ', '#39': "'" }

// Tier-2 renderer fallback config. Only invoked when curl came back empty or
// matched TITLE_ERROR_RE — keeps cold/CDN-cached pages on the cheap path.
const RENDER_TITLE_MAX_CONCURRENT = 2
const RENDER_TITLE_TIMEOUT_MS = 8000
const RENDER_TITLE_GRACE_MS = 700

// Resolved thumbnails live in memory only: the durable preview cache keeps the
// raw og:image URL (never the bytes — 400 base64 images would weight the cache
// file in the hundreds of megabytes), so a preview expanded after a restart
// shows its card without the thumbnail rather than re-downloading it.
const IMAGE_MEMORY_LIMIT = 64

/** Minimal surface of the Electron modules the runtime needs. */
export interface LinkPreviewRuntimeDeps {
  app: { isReady(): boolean; getPath(name: string): string }
  // Electron's BrowserWindow constructor and session module, passed rather
  // than imported so this module stays unit-testable without Electron.
  BrowserWindow: any
  session: { fromPartition(partition: string, options?: unknown): any }
}

export interface LinkPreviewRuntime {
  fetchLinkTitle: (rawUrl: string) => Promise<string>
  fetchLinkPreview: (rawUrl: string) => Promise<LinkPreviewResult>
}

export function registerLinkPreviewRuntime(ipcMain: any, runtime: LinkPreviewRuntime): void {
  ipcMain.handle('hermes:fetchLinkTitle', (_event: unknown, url: string) => runtime.fetchLinkTitle(url))
  ipcMain.handle('hermes:fetchLinkPreview', (_event: unknown, url: string) => runtime.fetchLinkPreview(url))
}

export function createLinkPreviewRuntime(deps: LinkPreviewRuntimeDeps): LinkPreviewRuntime {
  // ── Title resolution state ────────────────────────────────────────────────
  const titleCache = new Map()
  const titleInflight = new Map()
  let linkTitleSession: any = null
  let renderTitleInFlight = 0
  const renderTitleQueue: { resolve: (title: string) => void; url: string }[] = []

  function canonicalTitleCacheKey(rawUrl: string) {
    const value = String(rawUrl || '').trim()

    if (!value) {
      return ''
    }

    try {
      const url = new URL(value)
      const host = url.hostname.replace(/^www\./i, '').toLowerCase()
      const pathname = url.pathname === '/' ? '/' : url.pathname.replace(/\/+$/, '') || '/'

      return `${host}${pathname}${url.search || ''}`
    } catch {
      return value
    }
  }

  function cacheTitle(key: string, title: string) {
    if (titleCache.size >= TITLE_CACHE_LIMIT) {
      titleCache.delete(titleCache.keys().next().value)
    }

    titleCache.set(key, title)
  }

  function decodeHtmlEntities(value: string) {
    return value
      .replace(/&(amp|lt|gt|quot|apos|nbsp|#39);/gi, (_, k: string) => HTML_ENTITIES[k.toLowerCase()] ?? '')
      .replace(/&#x([0-9a-f]+);/gi, (_, hex: string) => String.fromCodePoint(parseInt(hex, 16) || 32))
      .replace(/&#(\d+);/g, (_, dec: string) => String.fromCodePoint(parseInt(dec, 10) || 32))
  }

  function parseHtmlTitle(html: string) {
    const raw = html.match(/<title[^>]*>([\s\S]*?)<\/title>/i)?.[1]

    return raw ? decodeHtmlEntities(raw).replace(/\s+/g, ' ').trim() : ''
  }

  // Strips known error/captcha titles (e.g. "GetYourGuide – Error", "Just a
  // moment...") so they don't get cached as the resolved title.
  function usableTitle(value: string): string {
    return value && !TITLE_ERROR_RE.test(value) ? value : ''
  }

  // ── One-hop curl transport (binary-safe, DNS-pinned) ──────────────────────
  // curl is driven ONE HOP at a time and every hop is re-validated and PINNED
  // by the guard (fetchWithGuardedRedirects / resolveThumbnail). `--include`
  // puts the status line and headers on stdout so the hop walk can read them;
  // the byte budget covers the header block too, which keeps the cap honest.
  // TITLE_TIMEOUT_MS is the budget for the WHOLE walk.

  function curlHop(
    url: string,
    timeoutMs: number,
    pinnedAddresses: string[]
  ): Promise<{ status: number; location: string; bytes: Buffer }> {
    return new Promise(resolve => {
      // DNS pinning (shape from upstream #63171's fetchPinnedLinkTitle): the
      // connection is bound to the addresses the guard just vetted via
      // --resolve, so curl never resolves the hostname itself and an attacker
      // controlling DNS cannot swap the answer between the verdict and the
      // request. Every vetted address is pinned; curl fails over among them
      // without a fresh lookup. A literal-IP hop has no name to pin.
      let resolveArgs: string[] = []

      try {
        const parsed = new URL(url)
        const port = parsed.port || (parsed.protocol === 'https:' ? '443' : '80')
        const hostname = parsed.hostname.replace(/^\[|\]$/g, '')

        resolveArgs = pinnedAddresses.flatMap(address => ['--resolve', `${hostname}:${port}:${address}`])
      } catch {
        // Unparsable URL: the guard refuses it before any request anyway.
      }

      const args = [
        '--silent',
        '--show-error',
        '--include',
        '--max-time',
        String(Math.max(1, Math.ceil(timeoutMs / 1000))),
        '--connect-timeout',
        '4',
        ...resolveArgs,
        '--user-agent',
        TITLE_USER_AGENT,
        '--header',
        'Accept: text/html,application/xhtml+xml;q=0.9,*/*;q=0.5',
        '--header',
        'Accept-Language: en-US,en;q=0.7',
        '--header',
        'Accept-Encoding: identity',
        '--raw',
        url
      ]

      const child = spawn('curl', args, hiddenWindowsChildOptions({ stdio: ['ignore', 'pipe', 'ignore'] }))
      const chunks: Buffer[] = []
      let bytes = 0

      child.stdout.on('data', (chunk: Buffer | string) => {
        if (bytes >= TITLE_BYTE_BUDGET) {
          return
        }

        const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)
        const remaining = TITLE_BYTE_BUDGET - bytes
        const next = buffer.length > remaining ? buffer.subarray(0, remaining) : buffer

        chunks.push(next)
        bytes += next.length
      })

      child.on('error', () => resolve({ status: 0, location: '', bytes: Buffer.alloc(0) }))
      child.on('close', () => resolve(splitHopResponse(Buffer.concat(chunks))))
    })
  }

  function splitHopResponse(raw: Buffer): { status: number; location: string; bytes: Buffer } {
    // Binary-safe header/body split: the body may be image bytes, so the
    // terminator is located in the raw buffer, never in a decoded string.
    const separator = raw.indexOf('\r\n\r\n')

    if (separator < 0) {
      // No header block: malformed response, treat as a transport failure.
      return { status: 0, location: '', bytes: Buffer.alloc(0) }
    }

    const head = raw.subarray(0, separator).toString('latin1')
    const statusLine = head.split('\n')[0] ?? ''
    const status = Number(statusLine.match(/\b(\d{3})\b/)?.[1] ?? 0)
    const location = head.match(/^location:[ \t]*(.*)$/im)?.[1]?.trim() ?? ''

    return { status, location, bytes: raw.subarray(separator + 4) }
  }

  function resolveHostAddresses(hostname: string): Promise<string[]> {
    return new Promise(resolve => {
      try {
        dns.lookup(hostname, { all: true }, (error, addresses) => {
          resolve(error || !addresses?.length ? [] : addresses.map(address => address.address))
        })
      } catch {
        resolve([])
      }
    })
  }

  async function fetchPageHtmlWithCurl(rawUrl: string): Promise<string> {
    const url = String(rawUrl || '').trim()

    if (!url) {
      return ''
    }

    const deadline = Date.now() + TITLE_TIMEOUT_MS

    const result = await fetchWithGuardedRedirects(
      url,
      {
        fetchOnce: async (hopUrl, addresses) => {
          const hop = await curlHop(hopUrl, deadline - Date.now(), addresses)

          return { status: hop.status, location: hop.location, body: hop.bytes.toString('utf8') }
        },
        resolveHost: resolveHostAddresses
      },
      { maxRedirects: PREVIEW_MAX_REDIRECTS }
    )

    // Refusals (private redirect hop, exhausted redirect budget) and transport
    // failures both surface as '' — the same miss signal the old curl error
    // leg produced, so tier 2 and the envelope see nothing new.
    return result.ok ? result.body : ''
  }

  // ── Tier-2 rendered title (hidden BrowserWindow) ──────────────────────────
  function getLinkTitleSession() {
    if (linkTitleSession || !deps.app.isReady()) {
      return linkTitleSession
    }

    linkTitleSession = deps.session.fromPartition('hermes:link-titles', { cache: false })
    // One onBeforeRequest registration owns everything this window's requests
    // must pass: resource blocks AND the B6 SSRF guard (per-request hostname +
    // DNS verdict, every redirect hop included). A second registration would
    // replace this one, not stack on it.
    guardLinkTitleSession(linkTitleSession, { resolveHost: resolveHostAddresses })

    return linkTitleSession
  }

  function dequeueRenderTitle() {
    while (renderTitleInFlight < RENDER_TITLE_MAX_CONCURRENT && renderTitleQueue.length) {
      const item = renderTitleQueue.shift()

      if (!item) {
        return
      }

      renderTitleInFlight += 1
      runRenderTitleJob(item.url).then((title: string) => {
        renderTitleInFlight -= 1
        item.resolve(title)
        dequeueRenderTitle()
      })
    }
  }

  function runRenderTitleJob(rawUrl: string): Promise<string> {
    return new Promise(resolve => {
      if (!deps.app.isReady()) {
        return resolve('')
      }

      const partitionSession = getLinkTitleSession()

      if (!partitionSession) {
        return resolve('')
      }

      let settled = false
      let window: any = null
      let hardTimer: any = null
      let graceTimer: any = null

      const finish = (title: string) => {
        if (settled) {
          return
        }

        settled = true

        if (hardTimer) {
          clearTimeout(hardTimer)
        }

        if (graceTimer) {
          clearTimeout(graceTimer)
        }

        const value = (title || '').replace(/\s+/g, ' ').trim()

        try {
          if (window && !window.isDestroyed()) {
            window.destroy()
          }
        } catch {
          // BrowserWindow may already be torn down; ignore.
        }

        resolve(value)
      }

      try {
        window = createLinkTitleWindow(deps.BrowserWindow, partitionSession)
      } catch {
        return finish('')
      }

      const finishWithTitle = () => finish(readLinkTitleWindowTitle(window))

      const scheduleGrace = () => {
        if (graceTimer) {
          clearTimeout(graceTimer)
        }

        graceTimer = setTimeout(finishWithTitle, RENDER_TITLE_GRACE_MS)
      }

      hardTimer = setTimeout(finishWithTitle, RENDER_TITLE_TIMEOUT_MS)

      window.webContents.setUserAgent(TITLE_USER_AGENT)
      window.webContents.on('page-title-updated', scheduleGrace)
      window.webContents.on('did-finish-load', scheduleGrace)
      window.webContents.on('did-fail-load', (_event: unknown, _code: unknown, _desc: unknown, _validatedURL: unknown, isMainFrame: boolean) => {
        if (isMainFrame) {
          finish('')
        }
      })

      window
        .loadURL(rawUrl, {
          httpReferrer: 'https://www.google.com/',
          userAgent: TITLE_USER_AGENT
        })
        .catch(() => finish(''))
    })
  }

  function fetchHtmlTitleWithRenderer(rawUrl: string): Promise<string> {
    return new Promise(resolve => {
      renderTitleQueue.push({ resolve, url: rawUrl })
      dequeueRenderTitle()
    })
  }

  function fetchLinkTitle(rawUrl: string): Promise<string> {
    const url = String(rawUrl || '').trim()
    const key = canonicalTitleCacheKey(url)

    if (!key) {
      return Promise.resolve('')
    }

    if (titleCache.has(key)) {
      return Promise.resolve(titleCache.get(key))
    }

    if (titleInflight.has(key)) {
      return titleInflight.get(key)
    }

    const pending = fetchPageHtmlWithCurl(url)
      .catch(() => '')
      .then(value => usableTitle(parseHtmlTitle(value).slice(0, 240)))
      .then(
        async value => value || usableTitle(((await fetchHtmlTitleWithRenderer(url).catch(() => '')) || '').slice(0, 240))
      )
      .then(clean => {
        cacheTitle(key, clean)
        titleInflight.delete(key)

        return clean
      })

    titleInflight.set(key, pending)

    return pending
  }

  // ── Link previews (D7 click-to-expand unfurl) ─────────────────────────────
  // The policy (SSRF guard, per-hop redirect validation, per-host pacing, cache,
  // field caps) lives in link-preview.ts; this is its I/O. Tier 1 shares
  // fetchPageHtmlWithCurl with the title fetcher; the thumbnail rides the same
  // pinned one-hop transport through resolveThumbnail — the renderer never
  // talks to the network for a preview, it renders a validated data URL.

  const LINK_PREVIEW_CACHE_PATH = path.join(deps.app.getPath('userData'), 'link-preview-cache.json')

  const previewPersistence = {
    read: () => {
      try {
        const raw = JSON.parse(fs.readFileSync(LINK_PREVIEW_CACHE_PATH, 'utf8'))

        return raw?.previews && typeof raw.previews === 'object' ? raw.previews : {}
      } catch {
        return {}
      }
    },
    write: (entries: Record<string, unknown>) => {
      try {
        fs.writeFileSync(LINK_PREVIEW_CACHE_PATH, JSON.stringify({ previews: entries }), 'utf8')
      } catch {
        // Cache is an optimization; failing to persist it costs one refetch.
      }
    }
  }

  const previewStore = new LinkPreviewStore(previewPersistence, { ttlMs: 24 * 60 * 60 * 1000 })
  const previewLimiter = new HostRateLimiter(10_000, 4)
  const imageMemory = new Map<string, string>()

  const previewIo: LinkPreviewIo = {
    fetchHtml: fetchPageHtmlWithCurl,
    fetchRenderedTitle: fetchHtmlTitleWithRenderer,
    resolveHost: resolveHostAddresses,
    fetchThumbnail: async url =>
      resolveThumbnail(url, {
        fetchOnce: async (hopUrl, addresses): Promise<HttpHopResponse<Uint8Array>> => {
          const hop = await curlHop(hopUrl, TITLE_TIMEOUT_MS, addresses)

          return { status: hop.status, location: hop.location, body: new Uint8Array(hop.bytes) }
        },
        resolveHost: resolveHostAddresses
      }).catch(() => '')
  }

  async function fetchLinkPreview(rawUrl: string): Promise<LinkPreviewResult> {
    const result = await resolveLinkPreview(rawUrl, previewIo, { store: previewStore, limiter: previewLimiter })

    if (result.ok) {
      const key = canonicalTitleCacheKey(rawUrl)

      if (result.meta.image) {
        // LRU the validated data URL so re-expands in this session (and cache
        // hits, which legitimately carry no image) still render a thumbnail.
        if (imageMemory.size >= IMAGE_MEMORY_LIMIT) {
          imageMemory.delete(imageMemory.keys().next().value)
        }

        imageMemory.set(key, result.meta.image)
      } else {
        const remembered = imageMemory.get(key)

        if (remembered) {
          result.meta.image = remembered
        }
      }
    }

    return result
  }

  return { fetchLinkTitle, fetchLinkPreview }
}

// Re-exported for main.ts's favicon I/O, which shares the browser-shaped UA
// and the same byte budget the title fetcher uses.
export { TITLE_BYTE_BUDGET, TITLE_TIMEOUT_MS }
