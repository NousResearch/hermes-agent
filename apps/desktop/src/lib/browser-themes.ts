/**
 * VS Code Marketplace color themes, fetched in-tab (webapp renderer).
 *
 * The Marketplace answers any origin (`access-control-allow-origin: *` on both
 * the gallery and the VSIX CDN) and the webapp CSP's `connect-src` already
 * covers third-party origins (update notes fetch `api.github.com`), so the
 * renderer can query the gallery, download the `.vsix` and unzip it itself —
 * no gateway proxy, and a bad package costs this tab, never the gateway
 * process that serves every profile and messaging platform.
 *
 * Behavior mirrors `electron/vscode-marketplace.ts` (the native main-process
 * fetcher) so both surfaces install themes identically; `fflate` (already an
 * `apps/desktop` dependency) replaces `node:zlib` for the zip reads.
 */
import { unzipSync } from 'fflate'

import type { DesktopMarketplaceSearchItem, DesktopMarketplaceThemeResult } from '@/global'

export interface BrowserThemesBridge {
  themes: {
    fetchMarketplace: (id: string) => Promise<DesktopMarketplaceThemeResult>
    searchMarketplace: (query: string) => Promise<DesktopMarketplaceSearchItem[]>
  }
}

const GALLERY_QUERY_URL = 'https://marketplace.visualstudio.com/_apis/public/gallery/extensionquery'
const VSIX_ASSET_TYPE = 'Microsoft.VisualStudio.Services.VSIXPackage'
const MAX_VSIX_BYTES = 40 * 1024 * 1024 // 40 MB — themes are tiny; this is paranoia.
/** Inflated bytes read out of one package, every entry together (see #49594). */
const MAX_EXTRACTED_BYTES = 16 * 1024 * 1024
const MAX_GALLERY_BYTES = 4 * 1024 * 1024
const REQUEST_TIMEOUT_MS = 20_000

const ID_RE = /^[\w-]+\.[\w-]+$/
// The "Themes" category also ships file-icon and product-icon themes; we can't
// see contributions without downloading, so filter the obvious packs out.
const ICON_THEME_RE = /\b(icon theme|file icons?|product icons?|icon pack|fileicons)\b/i

/** Fetch a URL with a timeout and a hard cap on the bytes we buffer. */
async function fetchCapped(url: string, init: RequestInit = {}, maxBytes: number): Promise<Uint8Array> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(new Error('Request timed out.')), REQUEST_TIMEOUT_MS)

  try {
    // `fetch` follows redirects (gallery → CDN) and enforces the CSP for us.
    const response = await fetch(url, { ...init, signal: controller.signal })

    if (!response.ok) {
      throw new Error(`Request failed (${response.status}) for ${url}`)
    }

    const reader = response.body?.getReader()

    if (!reader) {
      const buffer = new Uint8Array(await response.arrayBuffer())

      if (buffer.byteLength > maxBytes) {
        throw new Error('Response exceeded the size limit.')
      }

      return buffer
    }

    const chunks: Uint8Array[] = []
    let total = 0

    for (;;) {
      const { done, value } = await reader.read()

      if (done) {
        break
      }

      total += value.byteLength

      if (total > maxBytes) {
        controller.abort(new Error('Response exceeded the size limit.'))
        throw new Error('Response exceeded the size limit.')
      }

      chunks.push(value)
    }

    const out = new Uint8Array(total)
    let offset = 0

    for (const chunk of chunks) {
      out.set(chunk, offset)
      offset += chunk.byteLength
    }

    return out
  } finally {
    clearTimeout(timer)
  }
}

/** POST an ExtensionQuery payload and return the parsed gallery response. */
async function queryGallery(payload: unknown): Promise<any> {
  const raw = await fetchCapped(
    GALLERY_QUERY_URL,
    {
      method: 'POST',
      headers: {
        Accept: 'application/json;api-version=3.0-preview.1',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    },
    MAX_GALLERY_BYTES
  )

  return JSON.parse(new TextDecoder().decode(raw))
}

/** True when an extension is obviously a file/product icon pack, not colors. */
function looksLikeIconTheme(extension: any): boolean {
  const tags = (extension.tags ?? []).map((tag: unknown) => String(tag).toLowerCase())

  if (tags.includes('icon-theme') || tags.includes('product-icon-theme')) {
    return true
  }

  const text = `${extension.displayName ?? ''} ${extension.shortDescription ?? ''}`.toLowerCase()

  return ICON_THEME_RE.test(text)
}

interface MarketplaceCard {
  extensionId: string
  displayName: string
  publisher: string
  description: string
  installs: number
}

/**
 * Search the Marketplace for color-theme extensions. With an empty query this
 * returns the most-installed themes; with a query it's a full-text search
 * scoped to the Themes category. Returns lightweight cards (no download).
 */
async function searchMarketplaceThemes(query: string, limit = 20): Promise<MarketplaceCard[]> {
  const text = String(query ?? '').trim()
  const pageSize = Math.min(Math.max(Number(limit) || 20, 1), 50)

  // FilterType: 8=Target, 5=Category, 10=SearchText, 12=ExcludeWithFlags.
  const criteria: Array<{ filterType: number; value: string }> = [
    { filterType: 8, value: 'Microsoft.VisualStudio.Code' },
    { filterType: 5, value: 'Themes' },
    { filterType: 12, value: '4096' } // Exclude unpublished (Unpublished = 0x1000).
  ]

  if (text) {
    criteria.push({ filterType: 10, value: text })
  }

  const json = await queryGallery({
    // Over-fetch so the icon-theme filter below still leaves a full page.
    filters: [{ criteria, pageNumber: 1, pageSize: Math.min(pageSize * 2, 50), sortBy: 4, sortOrder: 0 }],
    // IncludeStatistics (0x100) | IncludeLatestVersionOnly (0x200) | IncludeCategoryAndTags (0x4).
    flags: 772
  })

  const extensions: any[] = json?.results?.[0]?.extensions ?? []

  return extensions
    .filter(extension => !looksLikeIconTheme(extension))
    .slice(0, pageSize)
    .map(extension => {
      const publisherName = extension.publisher?.publisherName ?? ''
      const installStat = (extension.statistics ?? []).find((stat: any) => stat.statisticName === 'install')

      return {
        extensionId: `${publisherName}.${extension.extensionName}`,
        displayName: extension.displayName || extension.extensionName,
        publisher: extension.publisher?.displayName || publisherName,
        description: extension.shortDescription || '',
        installs: Math.round(installStat?.value ?? 0)
      }
    })
}

/** Resolve `{ displayName, vsixUrl }` for the latest version of `id`. */
async function resolveExtension(id: string): Promise<{ displayName: string; vsixUrl: string }> {
  const json = await queryGallery({
    // FilterType 7 = ExtensionName (the full publisher.extension id).
    filters: [{ criteria: [{ filterType: 7, value: id }], pageNumber: 1, pageSize: 1 }],
    // IncludeFiles | IncludeVersionProperties | IncludeAssetUri |
    // IncludeCategoryAndTags | IncludeLatestVersionOnly = 914.
    flags: 914
  })

  const extension = json?.results?.[0]?.extensions?.[0]

  if (!extension) {
    throw new Error(`Extension "${id}" was not found on the Marketplace.`)
  }

  const version = extension.versions?.[0]

  if (!version) {
    throw new Error(`Extension "${id}" has no published versions.`)
  }

  const asset = (version.files ?? []).find((file: any) => file.assetType === VSIX_ASSET_TYPE)
  const vsixUrl = asset?.source

  if (!vsixUrl) {
    throw new Error(`Could not find a downloadable package for "${id}".`)
  }

  return { displayName: extension.displayName || id, vsixUrl }
}

/** Normalize a package.json theme path to its zip entry name. */
function themeEntryName(themePath: unknown): string {
  let clean = String(themePath ?? '')

  if (clean.startsWith('./')) {
    clean = clean.slice(2)
  }

  if (clean.startsWith('/')) {
    clean = clean.slice(1)
  }

  return `extension/${clean}`
}

/** Text-decode one inflated zip entry (theme JSON is JSONC text). */
function decodeEntry(data: Uint8Array): string {
  return new TextDecoder().decode(data)
}

/**
 * Extract every contributed color theme from a `.vsix` buffer.
 *
 * One budget covers the whole package: each entry's declared size is checked
 * against the remaining budget before it is inflated, so a small archive can't
 * cost unbounded tab memory (#49594).
 */
function extractThemes(vsix: Uint8Array): DesktopMarketplaceThemeResult['themes'] {
  // Pass 1: the manifest, so we know which entries the package contributes.
  const manifest = unzipSync(vsix, { filter: file => file.name === 'extension/package.json' })

  if (!manifest['extension/package.json']) {
    throw new Error('Package manifest missing from the extension.')
  }

  const pkg = JSON.parse(decodeEntry(manifest['extension/package.json']))
  const contributed = pkg?.contributes?.themes

  if (!Array.isArray(contributed) || contributed.length === 0) {
    return []
  }

  let budget = MAX_EXTRACTED_BYTES
  const wanted = new Set<string>()
  const entries: Array<{ label: string; uiTheme?: string; name: string }> = []

  for (const entry of contributed) {
    if (!entry?.path) {
      continue
    }

    const name = themeEntryName(entry.path)

    wanted.add(name)
    entries.push({
      label: entry.label || entry.id || pkg.displayName || pkg.name || 'VS Code Theme',
      uiTheme: entry.uiTheme,
      name
    })
  }

  // Pass 2: the theme entries only, each checked against the running budget
  // before it is inflated (a declared size the budget can't fit is skipped).
  const inflated = unzipSync(vsix, {
    filter: file => {
      if (!wanted.has(file.name)) {
        return false
      }

      if (file.originalSize > budget) {
        return false
      }

      budget -= file.originalSize

      return true
    }
  })

  const themes: DesktopMarketplaceThemeResult['themes'] = []

  for (const entry of entries) {
    const data = inflated[entry.name]

    if (!data) {
      continue // Over budget or missing — skip rather than fail the install.
    }

    themes.push({
      label: entry.label,
      uiTheme: entry.uiTheme,
      contents: decodeEntry(data)
    })
  }

  return themes
}

/**
 * Public entry: resolve, download, and extract color themes for `id`
 * (`publisher.extension`). Returns `{ extensionId, displayName, themes }`.
 */
async function fetchMarketplaceThemes(id: string): Promise<DesktopMarketplaceThemeResult> {
  const trimmed = String(id ?? '').trim()

  if (!ID_RE.test(trimmed)) {
    throw new Error('Expected a Marketplace id like "publisher.extension".')
  }

  const { displayName, vsixUrl } = await resolveExtension(trimmed)
  const vsix = await fetchCapped(vsixUrl, {}, MAX_VSIX_BYTES)

  return { extensionId: trimmed, displayName, themes: extractThemes(vsix) }
}

/**
 * Connect the webapp renderer to the Marketplace, fetched in-tab.
 */
export function createBrowserThemesBridge(): BrowserThemesBridge {
  return {
    themes: {
      searchMarketplace: (query: string) => searchMarketplaceThemes(query),
      fetchMarketplace: (id: string) => fetchMarketplaceThemes(id)
    }
  }
}

const __testing = { extractThemes, looksLikeIconTheme, searchMarketplaceThemes, themeEntryName }

export { __testing }