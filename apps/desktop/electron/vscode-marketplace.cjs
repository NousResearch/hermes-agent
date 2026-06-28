'use strict'

/**
 * VS Code Marketplace color-theme fetcher (main process).
 *
 * Resolves an extension's latest version via the (undocumented but stable)
 * gallery ExtensionQuery API, downloads the `.vsix` (a zip), and extracts the
 * color-theme JSON files it contributes. No theme code is ever executed — we
 * only read `package.json` + the referenced `*.json` theme files out of the
 * archive and hand their text back to the renderer to convert.
 *
 * Dependency-free on purpose: a `.vsix` is a plain zip, so we parse the central
 * directory and inflate just the entries we need with `zlib`. Avoids pulling a
 * zip library into the desktop bundle for a feature this small.
 */

const https = require('node:https')
const zlib = require('node:zlib')

const GALLERY_QUERY_URL = 'https://marketplace.visualstudio.com/_apis/public/gallery/extensionquery'
const VSIX_ASSET_TYPE = 'Microsoft.VisualStudio.Services.VSIXPackage'
const MAX_VSIX_BYTES = 40 * 1024 * 1024 // 40 MB — themes are tiny; this is paranoia.
const MAX_REDIRECTS = 5
const REQUEST_TIMEOUT_MS = 20_000
const MAX_THEME_ENTRY_BYTES = 2 * 1024 * 1024
const MAX_THEME_TOTAL_BYTES = 8 * 1024 * 1024
const MAX_THEME_COUNT = 25

const ID_RE = /^[\w-]+\.[\w-]+$/

/** Minimal HTTPS helper with redirect-following, timeout, and a size cap. */
function request(
  url,
  { method = 'GET', headers = {}, body = null, maxBytes = MAX_VSIX_BYTES } = {},
  redirectsLeft = MAX_REDIRECTS
) {
  return new Promise((resolve, reject) => {
    const req = https.request(url, { method, headers }, res => {
      const status = res.statusCode ?? 0

      if (status >= 300 && status < 400 && res.headers.location) {
        if (redirectsLeft <= 0) {
          res.resume()
          reject(new Error('Too many redirects.'))

          return
        }

        const next = new URL(res.headers.location, url).toString()
        res.resume()
        // Redirects to the CDN are plain GETs (drop the POST body).
        resolve(
          request(
            next,
            { method: 'GET', headers: { 'User-Agent': headers['User-Agent'] }, maxBytes },
            redirectsLeft - 1
          )
        )

        return
      }

      if (status < 200 || status >= 300) {
        res.resume()
        reject(new Error(`Request failed (${status}) for ${url}`))

        return
      }

      const chunks = []
      let total = 0

      res.on('data', chunk => {
        total += chunk.length

        if (total > maxBytes) {
          req.destroy()
          reject(new Error('Response exceeded the size limit.'))

          return
        }

        chunks.push(chunk)
      })
      res.on('end', () => resolve(Buffer.concat(chunks)))
    })

    req.on('error', reject)
    req.setTimeout(REQUEST_TIMEOUT_MS, () => req.destroy(new Error('Request timed out.')))

    if (body) {
      req.write(body)
    }

    req.end()
  })
}

/** Resolve `{ displayName, vsixUrl }` for the latest version of `id`. */
async function resolveExtension(id) {
  const json = await queryGallery({
    // FilterType 7 = ExtensionName (the full publisher.extension id).
    filters: [{ criteria: [{ filterType: 7, value: id }], pageNumber: 1, pageSize: 1 }],
    // Flags: IncludeFiles | IncludeVersionProperties | IncludeAssetUri |
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

  const asset = (version.files ?? []).find(file => file.assetType === VSIX_ASSET_TYPE)
  const vsixUrl = asset?.source

  if (!vsixUrl) {
    throw new Error(`Could not find a downloadable package for "${id}".`)
  }

  return { displayName: extension.displayName || id, vsixUrl }
}

/** POST an ExtensionQuery payload and return the parsed gallery response. */
async function queryGallery(payload, { maxBytes = 4 * 1024 * 1024 } = {}) {
  const body = JSON.stringify(payload)
  const raw = await request(GALLERY_QUERY_URL, {
    method: 'POST',
    headers: {
      Accept: 'application/json;api-version=3.0-preview.1',
      'Content-Type': 'application/json',
      'Content-Length': Buffer.byteLength(body),
      'User-Agent': 'Hermes-Desktop'
    },
    body,
    maxBytes
  })

  return JSON.parse(raw.toString('utf8'))
}

/**
 * Search the Marketplace for color-theme extensions. With an empty query this
 * returns the most-installed themes; with a query it's a full-text search
 * scoped to the Themes category. Returns lightweight cards (no download).
 */
/**
 * The "Themes" category also contains file-icon and product-icon themes (the
 * gallery has no color-only category). We can't see an extension's actual
 * contributions without downloading it, so filter the obvious icon packs out by
 * tag + name/description. Color themes that also ship icons are rare; worst case
 * a user installs them by exact id from settings.
 */
function looksLikeIconTheme(extension) {
  const tags = (extension.tags ?? []).map(tag => String(tag).toLowerCase())

  if (tags.includes('icon-theme') || tags.includes('product-icon-theme')) {
    return true
  }

  const text = `${extension.displayName ?? ''} ${extension.shortDescription ?? ''}`.toLowerCase()

  return /\b(icon theme|file icons?|product icons?|icon pack|fileicons)\b/.test(text)
}

async function searchMarketplaceThemes(query, limit = 20) {
  const text = String(query || '').trim()
  const pageSize = Math.min(Math.max(Number(limit) || 20, 1), 50)

  // FilterType: 8=Target, 5=Category, 10=SearchText, 12=ExcludeWithFlags.
  const criteria = [
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

  const extensions = json?.results?.[0]?.extensions ?? []

  return extensions
    .filter(extension => !looksLikeIconTheme(extension))
    .slice(0, pageSize)
    .map(extension => {
      const publisherName = extension.publisher?.publisherName ?? ''
      const installStat = (extension.statistics ?? []).find(stat => stat.statisticName === 'install')

      return {
        extensionId: `${publisherName}.${extension.extensionName}`,
        displayName: extension.displayName || extension.extensionName,
        publisher: extension.publisher?.displayName || publisherName,
        description: extension.shortDescription || '',
        installs: Math.round(installStat?.value ?? 0)
      }
    })
}

// ─── Minimal zip reader ─────────────────────────────────────────────────────

function findEndOfCentralDirectory(buf) {
  // EOCD signature 0x06054b50, scanning back from the end (comment is rare).
  for (let i = buf.length - 22; i >= 0; i--) {
    if (buf.readUInt32LE(i) === 0x06054b50) {
      return i
    }
  }

  throw new Error('Not a valid zip archive (no end-of-central-directory).')
}

/** Parse the central directory into a name → record map. */
function readCentralDirectory(buf) {
  const eocd = findEndOfCentralDirectory(buf)
  const count = buf.readUInt16LE(eocd + 10)
  let offset = buf.readUInt32LE(eocd + 16)
  const records = new Map()

  for (let i = 0; i < count; i++) {
    if (buf.readUInt32LE(offset) !== 0x02014b50) {
      break
    }

    const method = buf.readUInt16LE(offset + 10)
    const compressedSize = buf.readUInt32LE(offset + 20)
    const uncompressedSize = buf.readUInt32LE(offset + 24)
    const nameLen = buf.readUInt16LE(offset + 28)
    const extraLen = buf.readUInt16LE(offset + 30)
    const commentLen = buf.readUInt16LE(offset + 32)
    const localOffset = buf.readUInt32LE(offset + 42)
    const name = buf.toString('utf8', offset + 46, offset + 46 + nameLen)

    records.set(name, { method, compressedSize, uncompressedSize, localOffset })
    offset += 46 + nameLen + extraLen + commentLen
  }

  return records
}

/** Inflate a single entry to a string. */
function extractEntry(buf, record) {
  if (record.method !== 0 && record.method !== 8) {
    throw new Error('Unsupported zip compression method.')
  }
  if (record.uncompressedSize > MAX_THEME_ENTRY_BYTES) {
    throw new Error('Zip entry exceeds the uncompressed size limit.')
  }
  // The local header's name/extra lengths can differ from the central record,
  // so re-read them here to locate the compressed payload.
  if (buf.readUInt32LE(record.localOffset) !== 0x04034b50) {
    throw new Error('Corrupt zip: bad local file header.')
  }

  const nameLen = buf.readUInt16LE(record.localOffset + 26)
  const extraLen = buf.readUInt16LE(record.localOffset + 28)
  const dataStart = record.localOffset + 30 + nameLen + extraLen
  if (dataStart + record.compressedSize > buf.length) {
    throw new Error('Corrupt zip: entry exceeds archive bounds.')
  }
  const data = buf.subarray(dataStart, dataStart + record.compressedSize)

  // 0 = stored, 8 = deflate. Theme files are one or the other.
  return record.method === 0
    ? data.toString('utf8')
    : zlib.inflateRawSync(data, { maxOutputLength: MAX_THEME_ENTRY_BYTES }).toString('utf8')
}

/** Normalize a package.json theme path to its zip entry name. */
function themeEntryName(themePath) {
  const clean = String(themePath).replace(/^\.\//, '').replace(/^\//, '')

  return `extension/${clean}`
}

/** Extract every contributed color theme from a `.vsix` buffer. */
function extractThemes(vsixBuffer) {
  const records = readCentralDirectory(vsixBuffer)
  const pkgRecord = records.get('extension/package.json')

  if (!pkgRecord) {
    throw new Error('Package manifest missing from the extension.')
  }

  const pkg = JSON.parse(extractEntry(vsixBuffer, pkgRecord))
  const contributed = pkg?.contributes?.themes

  if (!Array.isArray(contributed) || contributed.length === 0) {
    return []
  }

  const themes = []

  let totalBytes = 0

  for (const entry of contributed.slice(0, MAX_THEME_COUNT)) {
    if (!entry?.path) {
      continue
    }

    const record = records.get(themeEntryName(entry.path))

    if (!record) {
      continue
    }

    try {
      const contents = extractEntry(vsixBuffer, record)
      totalBytes += Buffer.byteLength(contents, 'utf8')
      if (totalBytes > MAX_THEME_TOTAL_BYTES) {
        break
      }
      themes.push({
        label: entry.label || entry.id || pkg.displayName || pkg.name || 'VS Code Theme',
        uiTheme: entry.uiTheme,
        contents
      })
    } catch {
      // Skip an entry we can't inflate rather than failing the whole install.
    }
  }

  return themes
}

/**
 * Public entry: resolve, download, and extract color themes for `id`
 * (`publisher.extension`). Returns `{ extensionId, displayName, themes }`.
 */
async function fetchMarketplaceThemes(id) {
  const trimmed = String(id || '').trim()

  if (!ID_RE.test(trimmed)) {
    throw new Error('Expected a Marketplace id like "publisher.extension".')
  }

  const { displayName, vsixUrl } = await resolveExtension(trimmed)
  const vsix = await request(vsixUrl, { headers: { 'User-Agent': 'Hermes-Desktop' } })
  const themes = extractThemes(vsix)

  return { extensionId: trimmed, displayName, themes }
}

module.exports = {
  fetchMarketplaceThemes,
  searchMarketplaceThemes,
  extractThemes,
  readCentralDirectory,
  __testing: { themeEntryName, looksLikeIconTheme }
}
