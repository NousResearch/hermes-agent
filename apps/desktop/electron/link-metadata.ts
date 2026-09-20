import { spawn } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

import { app, BrowserWindow, net as electronNet, session } from 'electron'

import { type FaviconIo, resolveFavicon } from './favicon'
import { createLinkTitleWindow, guardLinkTitleSession, readLinkTitleWindowTitle } from './link-title-window'
import { hiddenWindowsChildOptions } from './windows-child-options'

// Link title resolution — curl (tier 1) → hidden BrowserWindow (tier 2).
const titleCache = new Map()
const titleInflight = new Map()
const TITLE_CACHE_LIMIT = 500
const TITLE_BYTE_BUDGET = 96 * 1024
const TITLE_TIMEOUT_MS = 5000
const TITLE_MAX_REDIRECTS = 3

// Browser-shaped UA — many bot-walled sites (GetYourGuide, Cloudflare-protected
// pages) refuse anything that doesn't look like a real Chrome.
const TITLE_USER_AGENT =
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36'

const TITLE_ERROR_RE =
  /\b(access denied|attention required|captcha|error|forbidden|just a moment|request blocked|too many requests)\b/i

const HTML_ENTITIES = { amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", nbsp: ' ', '#39': "'" }

// Tier-2 renderer fallback config. Only invoked when curl came back empty or
// matched TITLE_ERROR_RE — keeps cold/CDN-cached pages on the cheap path.
const RENDER_TITLE_MAX_CONCURRENT = 2
const RENDER_TITLE_TIMEOUT_MS = 8000
const RENDER_TITLE_GRACE_MS = 700

// Resource types we cancel before the network even fires — keeps the hidden
// renderer fast and cuts third-party tracking noise.
const RENDER_TITLE_BLOCKED_RESOURCES = new Set([
  'cspReport',
  'font',
  'imageset',
  'media',
  'object',
  'ping',
  'stylesheet'
])

let linkTitleSession = null
let renderTitleInFlight = 0
const renderTitleQueue = []

function canonicalTitleCacheKey(rawUrl) {
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

function cacheTitle(key, title) {
  if (titleCache.size >= TITLE_CACHE_LIMIT) {
    titleCache.delete(titleCache.keys().next().value)
  }

  titleCache.set(key, title)
}

function decodeHtmlEntities(value) {
  return value
    .replace(/&(amp|lt|gt|quot|apos|nbsp|#39);/gi, (_, k) => HTML_ENTITIES[k.toLowerCase()] ?? '')
    .replace(/&#x([0-9a-f]+);/gi, (_, hex) => String.fromCodePoint(parseInt(hex, 16) || 32))
    .replace(/&#(\d+);/g, (_, dec) => String.fromCodePoint(parseInt(dec, 10) || 32))
}

function parseHtmlTitle(html) {
  const raw = html.match(/<title[^>]*>([\s\S]*?)<\/title>/i)?.[1]

  return raw ? decodeHtmlEntities(raw).replace(/\s+/g, ' ').trim() : ''
}

function fetchHtmlTitleWithCurl(rawUrl: string): Promise<string> {
  return new Promise(resolve => {
    const url = String(rawUrl || '').trim()

    if (!url) {
      return resolve('')
    }

    const args = [
      '--silent',
      '--show-error',
      '--location',
      '--max-redirs',
      String(TITLE_MAX_REDIRECTS),
      '--max-time',
      String(Math.max(2, Math.ceil(TITLE_TIMEOUT_MS / 1000))),
      '--connect-timeout',
      '4',
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
    const chunks = []
    let bytes = 0

    child.stdout.on('data', chunk => {
      if (bytes >= TITLE_BYTE_BUDGET) {
        return
      }

      const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)
      const remaining = TITLE_BYTE_BUDGET - bytes
      const next = buffer.length > remaining ? buffer.subarray(0, remaining) : buffer
      chunks.push(next)
      bytes += next.length
    })

    child.on('error', () => resolve(''))
    child.on('close', () => {
      if (!chunks.length) {
        return resolve('')
      }

      resolve(parseHtmlTitle(Buffer.concat(chunks).toString('utf8')))
    })
  })
}

function getLinkTitleSession() {
  if (linkTitleSession || !app.isReady()) {
    return linkTitleSession
  }

  linkTitleSession = session.fromPartition('hermes:link-titles', { cache: false })
  linkTitleSession.webRequest.onBeforeRequest((details, callback) => {
    callback({ cancel: RENDER_TITLE_BLOCKED_RESOURCES.has(details.resourceType) })
  })
  guardLinkTitleSession(linkTitleSession)

  return linkTitleSession
}

function dequeueRenderTitle() {
  while (renderTitleInFlight < RENDER_TITLE_MAX_CONCURRENT && renderTitleQueue.length) {
    const item = renderTitleQueue.shift()
    renderTitleInFlight += 1
    runRenderTitleJob(item.url).then(title => {
      renderTitleInFlight -= 1
      item.resolve(title)
      dequeueRenderTitle()
    })
  }
}

function runRenderTitleJob(rawUrl) {
  return new Promise(resolve => {
    if (!app.isReady()) {
      return resolve('')
    }

    const partitionSession = getLinkTitleSession()

    if (!partitionSession) {
      return resolve('')
    }

    let settled = false
    let window = null
    let hardTimer = null
    let graceTimer = null

    const finish = title => {
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
      window = createLinkTitleWindow(BrowserWindow, partitionSession)
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
    window.webContents.on('did-fail-load', (_event, _code, _desc, _validatedURL, isMainFrame) => {
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

// Strips known error/captcha titles (e.g. "GetYourGuide – Error", "Just a
// moment...") so they don't get cached as the resolved title.
function usableTitle(value: string): string {
  return value && !TITLE_ERROR_RE.test(value) ? value : ''
}

function fetchLinkTitle(rawUrl) {
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

  const pending = fetchHtmlTitleWithCurl(url)
    .catch(() => '')
    .then(value => usableTitle((value || '').slice(0, 240)))
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

// ─── Favicon resolution ──────────────────────────────────────────────────────
// The ladder itself is electron/favicon.ts; this is its I/O, its cache, and
// the one rule that belongs to the app rather than the algorithm: one icon
// per host. A connector's mark doesn't vary by path, and hosting the cache on
// the host key means Linear's docs page and Linear's MCP endpoint cost one
// lookup between them.

const FAVICON_CACHE_PATH = path.join(app.getPath('userData'), 'favicon-cache.json')
const FAVICON_CACHE_LIMIT = 400
const FAVICON_TTL_MS = 30 * 24 * 60 * 60 * 1000
// A miss is cheap to re-check and expensive to be wrong about (a site that
// was behind a captcha yesterday has a logo today), so it expires fast.
const FAVICON_MISS_TTL_MS = 12 * 60 * 60 * 1000
const FAVICON_TIMEOUT_MS = 6000
const FAVICON_MAX_BYTES = 256 * 1024
const FAVICON_WRITE_DEBOUNCE_MS = 3000

let faviconCache: Map<string, { at: number; icon: string }> | null = null
let faviconWriteTimer: null | ReturnType<typeof setTimeout> = null
const faviconInflight = new Map<string, Promise<string>>()

function faviconCacheKey(rawUrl: string): string {
  try {
    return new URL(rawUrl).hostname.replace(/^www\./i, '').toLowerCase()
  } catch {
    return ''
  }
}

function loadFaviconCache(): Map<string, { at: number; icon: string }> {
  if (faviconCache) {
    return faviconCache
  }

  faviconCache = new Map()

  try {
    const raw = JSON.parse(fs.readFileSync(FAVICON_CACHE_PATH, 'utf8'))

    for (const [host, entry] of Object.entries(raw?.icons ?? {})) {
      const at = Number((entry as { at?: number })?.at)
      const icon = String((entry as { icon?: string })?.icon ?? '')

      if (Number.isFinite(at) && Date.now() - at < (icon ? FAVICON_TTL_MS : FAVICON_MISS_TTL_MS)) {
        faviconCache.set(host, { at, icon })
      }
    }
  } catch {
    // No cache yet, or it's unreadable — resolving again is the whole cost.
  }

  return faviconCache
}

function saveFaviconCacheSoon() {
  if (faviconWriteTimer) {
    return
  }

  faviconWriteTimer = setTimeout(() => {
    faviconWriteTimer = null

    try {
      const icons = Object.fromEntries(loadFaviconCache())

      fs.writeFileSync(FAVICON_CACHE_PATH, JSON.stringify({ icons }), 'utf8')
    } catch {
      // Cache is an optimization; failing to persist it costs one refetch.
    }
  }, FAVICON_WRITE_DEBOUNCE_MS)

  faviconWriteTimer.unref?.()
}

async function faviconFetch(url: string, accept: string) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), FAVICON_TIMEOUT_MS)

  try {
    return await electronNet.fetch(url, {
      // Same browser-shaped identity the title fetcher uses: a plain Electron
      // UA gets a challenge page from anything behind a bot wall.
      headers: { Accept: accept, 'Accept-Language': 'en-US,en;q=0.7', 'User-Agent': TITLE_USER_AGENT },
      redirect: 'follow',
      signal: controller.signal
    })
  } finally {
    clearTimeout(timer)
  }
}

const faviconIo: FaviconIo = {
  fetchImage: async url => {
    const response = await faviconFetch(url, 'image/avif,image/webp,image/svg+xml,image/*;q=0.8,*/*;q=0.5')

    if (!response.ok) {
      return null
    }

    const buffer = await response.arrayBuffer()

    if (buffer.byteLength === 0 || buffer.byteLength > FAVICON_MAX_BYTES) {
      return null
    }

    return { bytes: new Uint8Array(buffer), mime: response.headers.get('content-type') ?? '' }
  },
  fetchText: async url => {
    const response = await faviconFetch(url, 'text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.5')

    return response.ok ? (await response.text()).slice(0, TITLE_BYTE_BUDGET * 2) : ''
  }
}

function resolveFaviconCached(rawUrl: string): Promise<string> {
  const key = faviconCacheKey(String(rawUrl || '').trim())

  if (!key) {
    return Promise.resolve('')
  }

  const cache = loadFaviconCache()
  const hit = cache.get(key)

  if (hit && Date.now() - hit.at < (hit.icon ? FAVICON_TTL_MS : FAVICON_MISS_TTL_MS)) {
    return Promise.resolve(hit.icon)
  }

  const inflight = faviconInflight.get(key)

  if (inflight) {
    return inflight
  }

  const pending = resolveFavicon(rawUrl, faviconIo)
    .catch(() => '')
    .then(icon => {
      if (cache.size >= FAVICON_CACHE_LIMIT) {
        cache.delete(cache.keys().next().value)
      }

      cache.set(key, { at: Date.now(), icon })
      saveFaviconCacheSoon()
      faviconInflight.delete(key)

      return icon
    })

  faviconInflight.set(key, pending)

  return pending
}

export { fetchLinkTitle, resolveFaviconCached }
