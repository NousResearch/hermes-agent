import crypto from 'node:crypto'
import fs from 'node:fs'
import https from 'node:https'
import path from 'node:path'

import { unzipSync } from 'fflate'

export type UblockInstallIntent = 'cached' | 'pinned'
export type PreviewUblockFailureCode = 'compatibility' | 'integrity' | 'network' | 'storage' | 'timeout'

export interface PreviewUblockProgress {
  receivedBytes: number
  totalBytes: number | null
}

export type PreviewUblockInstallPhase = 'checking-cache' | 'downloading' | 'extracting' | 'preparing' | 'verifying'

export interface PreviewUblockInstallCallbacks {
  onPhase?: (phase: PreviewUblockInstallPhase) => void
  onProgress?: (progress: PreviewUblockProgress) => void
}

export interface InstalledUblock {
  path: string
  popupPath?: string
  version: string
}

export interface PreviewUblockInstallCandidate {
  finalPath: string
  installFinal(): void
  commitActive(): void
  discard(): void
  popupPath?: string
  stagedPath: string
  version: string
}

export type PreviewUblockInstallResult = InstalledUblock | PreviewUblockInstallCandidate

export interface PreviewUblockInstaller {
  resolve(
    intent: UblockInstallIntent,
    callbacks?: PreviewUblockInstallCallbacks
  ): Promise<PreviewUblockInstallResult | null>
}

export interface PreviewUblockHttpRequestOptions {
  headers?: Record<string, string>
  maxBytes: number
  maxRedirects: number
  onProgress?: (progress: PreviewUblockProgress) => void
  timeoutMs: number
}

export type PreviewUblockHttpRequest = (url: string, options: PreviewUblockHttpRequestOptions) => Promise<Uint8Array>

export interface PreviewUblockInstallerOptions {
  cacheDirectory?: string
  request?: PreviewUblockHttpRequest
  userDataPath?: string
}

export interface InstalledUblockMetadata {
  archiveSha256: string
  archiveUrl: string
  compatibilityRevision: string
  schemaVersion: 2
  version: string
}

export const PREVIEW_UBLOCK_VERSION = '2026.825.1619'
export const PREVIEW_UBLOCK_ARCHIVE_URL = `https://github.com/uBlockOrigin/uBOL-home/releases/download/${PREVIEW_UBLOCK_VERSION}/uBOLite_${PREVIEW_UBLOCK_VERSION}.chromium.zip`
export const PREVIEW_UBLOCK_ARCHIVE_SHA256 = '9f0acbe3eabd4ba1c1c0629438cfacafbdaf04cd150769932d5d265b2fac117e'
export const PREVIEW_UBLOCK_COMPATIBILITY_REVISION = 'electron-permission-shims-v2'
export const PREVIEW_UBLOCK_MAX_ARCHIVE_BYTES = 80 * 1024 * 1024
export const PREVIEW_UBLOCK_MAX_EXPANDED_BYTES = 128 * 1024 * 1024
export const PREVIEW_UBLOCK_MAX_FILE_BYTES = 32 * 1024 * 1024
export const PREVIEW_UBLOCK_MAX_FILES = 2_000

const PREVIEW_UBLOCK_CACHE_DIRNAME = 'preview-ublock'
const VERSIONS_DIRNAME = 'versions'
const ACTIVE_POINTER_FILENAME = 'active.json'
const INSTALLED_METADATA_FILENAME = 'installed.json'
const ORIGINAL_SERVICE_WORKER_FILENAME = 'js/background.js'
const WRAPPER_FILENAME = 'js/hermes-service-worker.js'
const EXT_COMPAT_FILENAME = 'js/ext-compat.js'
const PRESERVED_EXT_COMPAT_FILENAME = 'js/hermes-ext-compat-source.js'
const MAX_REDIRECTS = 5
const DOWNLOAD_TIMEOUT_MS = 30_000
const RELEASE_TAG_RE = /^[A-Za-z0-9._-]+$/
const CACHE_KEY_RE = /^[A-Za-z0-9._-]+$/

const ALLOWED_NETWORK_HOSTS = new Set([
  'github.com',
  'release-assets.githubusercontent.com',
  'objects.githubusercontent.com'
])

const REQUIRED_FILES = [
  'LICENSE.txt',
  'manifest.json',
  'dashboard.html',
  EXT_COMPAT_FILENAME,
  'rulesets/main/easylist.json',
  'rulesets/main/easyprivacy.json',
  'rulesets/main/ublock-filters.json'
]

export function safeError(
  message: string,
  code: PreviewUblockFailureCode = 'compatibility'
): Error & { code: PreviewUblockFailureCode } {
  const error = new Error(`uBlock Origin Lite could not be installed: ${message}`) as Error & {
    code: PreviewUblockFailureCode
  }

  error.code = code

  return error
}

function classifyRequestError(error: unknown): Error & { code: PreviewUblockFailureCode } {
  if (
    error &&
    typeof error === 'object' &&
    'code' in error &&
    ['compatibility', 'integrity', 'network', 'storage', 'timeout'].includes(String(error.code))
  ) {
    return error as Error & { code: PreviewUblockFailureCode }
  }

  const message = error instanceof Error ? error.message : String(error)

  return safeError(message || 'the release request failed', /timed out|timeout/i.test(message) ? 'timeout' : 'network')
}

function isPreviewUblockFailure(error: unknown): error is Error & { code: PreviewUblockFailureCode } {
  return (
    error !== null &&
    typeof error === 'object' &&
    'code' in error &&
    ['compatibility', 'integrity', 'network', 'storage', 'timeout'].includes(String(error.code))
  )
}

function isAllowedUrl(rawUrl: string): URL {
  let url: URL

  try {
    url = new URL(rawUrl)
  } catch {
    throw safeError('the release server returned an invalid URL', 'network')
  }

  if (url.protocol !== 'https:' || url.username || url.password || !ALLOWED_NETWORK_HOSTS.has(url.hostname)) {
    throw safeError('the release download returned a disallowed URL', 'network')
  }

  return url
}

function requestHttps(urlString: string, options: PreviewUblockHttpRequestOptions): Promise<Uint8Array> {
  const visit = (currentUrl: string, redirects: number): Promise<Uint8Array> =>
    new Promise((resolve, reject) => {
      let url: URL

      try {
        url = isAllowedUrl(currentUrl)
      } catch (error) {
        reject(error)

        return
      }

      const request = https.request(url, { headers: options.headers, method: 'GET' }, response => {
        const status = response.statusCode ?? 0

        if (status >= 300 && status < 400) {
          const location = response.headers.location
          response.resume()

          if (!location || redirects >= options.maxRedirects) {
            reject(safeError('the release download redirected too many times', 'network'))

            return
          }

          try {
            const nextUrl = new URL(location, url)
            isAllowedUrl(nextUrl.toString())
            void visit(nextUrl.toString(), redirects + 1).then(resolve, reject)
          } catch (error) {
            reject(error)
          }

          return
        }

        if (status < 200 || status >= 300) {
          response.resume()
          reject(safeError(`the release server returned HTTP ${status}`, 'network'))

          return
        }

        const header = response.headers['content-length']
        const parsedLength = typeof header === 'string' && /^\d+$/.test(header) ? Number(header) : NaN
        const totalBytes = Number.isSafeInteger(parsedLength) && parsedLength >= 0 ? parsedLength : null
        options.onProgress?.({ receivedBytes: 0, totalBytes })
        const chunks: Buffer[] = []
        let receivedBytes = 0
        let tooLarge = false
        response.on('data', chunk => {
          if (tooLarge) {
            return
          }
          const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)
          receivedBytes += buffer.length

          if (receivedBytes > options.maxBytes) {
            tooLarge = true
            response.destroy(safeError('the release response is too large', 'integrity'))

            return
          }

          chunks.push(buffer)
          options.onProgress?.({ receivedBytes, totalBytes })
        })
        response.on('end', () => {
          if (tooLarge) {
            return
          }

          if (totalBytes !== null && receivedBytes !== totalBytes) {
            reject(safeError('the release response size did not match Content-Length', 'integrity'))

            return
          }

          resolve(new Uint8Array(Buffer.concat(chunks)))
        })
        response.on('error', error => reject(classifyRequestError(error)))
      })

      request.setTimeout(options.timeoutMs, () =>
        request.destroy(safeError('the release request timed out', 'timeout'))
      )
      request.on('error', error => reject(classifyRequestError(error)))
      request.end()
    })

  return visit(urlString, 0)
}

function checkedZipPath(name: string): string {
  if (!name || name.includes('\\') || name.includes('\0') || name.startsWith('/') || /^[A-Za-z]:/.test(name)) {
    throw safeError('the archive contains an unsafe path', 'integrity')
  }

  const pathWithoutMarker = name.endsWith('/') ? name.slice(0, -1) : name
  const segments = pathWithoutMarker.split('/')

  if (!pathWithoutMarker || segments.some(segment => segment === '' || segment === '.' || segment === '..')) {
    throw safeError('the archive contains an unsafe path', 'integrity')
  }

  return segments.join('/')
}

function decodeZipName(bytes: Uint8Array): string {
  try {
    return new TextDecoder('utf-8', { fatal: true }).decode(bytes)
  } catch {
    throw safeError('the archive contains an invalid UTF-8 path', 'integrity')
  }
}

function readU16(data: Uint8Array, offset: number): number {
  if (offset < 0 || offset + 2 > data.length) {
    throw safeError('the archive is truncated', 'integrity')
  }

  return data[offset] | (data[offset + 1] << 8)
}

function readU32(data: Uint8Array, offset: number): number {
  if (offset < 0 || offset + 4 > data.length) {
    throw safeError('the archive is truncated', 'integrity')
  }

  return (data[offset] | (data[offset + 1] << 8) | (data[offset + 2] << 16) | (data[offset + 3] << 24)) >>> 0
}

interface ZipEntry {
  isDirectory: boolean
  name: string
  uncompressedSize: number
}

export function inspectZip(archive: Uint8Array): ZipEntry[] {
  if (archive.length > PREVIEW_UBLOCK_MAX_ARCHIVE_BYTES) {
    throw safeError('the archive is too large', 'integrity')
  }
  const minimumEnd = Math.max(0, archive.length - 22 - 65_535)
  let endOffset = -1

  for (let offset = archive.length - 22; offset >= minimumEnd; offset -= 1) {
    if (offset >= 0 && readU32(archive, offset) === 0x06054b50) {
      endOffset = offset

      break
    }
  }

  if (endOffset < 0) {
    throw safeError('the archive has no valid ZIP directory', 'integrity')
  }
  const diskNumber = readU16(archive, endOffset + 4)
  const directoryDisk = readU16(archive, endOffset + 6)
  const entriesOnDisk = readU16(archive, endOffset + 8)
  const entryCount = readU16(archive, endOffset + 10)
  const directorySize = readU32(archive, endOffset + 12)
  const directoryOffset = readU32(archive, endOffset + 16)

  if (diskNumber !== 0 || directoryDisk !== 0 || entriesOnDisk !== entryCount || entryCount === 0xffff) {
    throw safeError('the archive uses unsupported ZIP features', 'integrity')
  }

  if (directoryOffset + directorySize > archive.length) {
    throw safeError('the archive directory is truncated', 'integrity')
  }

  const entries: ZipEntry[] = []
  const names = new Set<string>()
  let offset = directoryOffset
  let expandedBytes = 0
  let fileCount = 0

  for (let index = 0; index < entryCount; index += 1) {
    if (readU32(archive, offset) !== 0x02014b50) {
      throw safeError('the archive has an invalid directory entry', 'integrity')
    }
    const flags = readU16(archive, offset + 8)
    const compression = readU16(archive, offset + 10)
    const compressedSize = readU32(archive, offset + 20)
    const uncompressedSize = readU32(archive, offset + 24)
    const nameLength = readU16(archive, offset + 28)
    const extraLength = readU16(archive, offset + 30)
    const commentLength = readU16(archive, offset + 32)
    const externalAttributes = readU32(archive, offset + 38)
    const localOffset = readU32(archive, offset + 42)
    const nameStart = offset + 46
    const nextOffset = nameStart + nameLength + extraLength + commentLength

    if (nextOffset > directoryOffset + directorySize || nextOffset > archive.length) {
      throw safeError('the archive is truncated', 'integrity')
    }

    if (compressedSize === 0xffffffff || uncompressedSize === 0xffffffff || localOffset === 0xffffffff) {
      throw safeError('the archive uses unsupported ZIP64 features', 'integrity')
    }

    const rawName = decodeZipName(archive.slice(nameStart, nameStart + nameLength))
    const name = checkedZipPath(rawName)

    if (names.has(name)) {
      throw safeError('the archive contains duplicate paths', 'integrity')
    }
    names.add(name)
    const unixMode = (externalAttributes >>> 16) & 0xffff

    if ((unixMode & 0xf000) === 0xa000) {
      throw safeError('the archive contains a symlink', 'integrity')
    }

    if (flags & 1) {
      throw safeError('the archive contains an encrypted entry', 'integrity')
    }

    if (compression !== 0 && compression !== 8) {
      throw safeError('the archive uses unsupported compression', 'integrity')
    }

    if (readU32(archive, localOffset) !== 0x04034b50) {
      throw safeError('the archive has an invalid local entry', 'integrity')
    }
    const localNameLength = readU16(archive, localOffset + 26)
    const localExtraLength = readU16(archive, localOffset + 28)
    const dataOffset = localOffset + 30 + localNameLength + localExtraLength

    if (dataOffset > archive.length || compressedSize > archive.length - dataOffset) {
      throw safeError('the archive entry is truncated', 'integrity')
    }
    const isDirectory = rawName.endsWith('/') || (externalAttributes & 0x10) !== 0 || (unixMode & 0xf000) === 0x4000

    if (!isDirectory) {
      fileCount += 1

      if (fileCount > PREVIEW_UBLOCK_MAX_FILES) {
        throw safeError('the archive contains too many files', 'integrity')
      }

      if (uncompressedSize > PREVIEW_UBLOCK_MAX_FILE_BYTES) {
        throw safeError('an archive file is too large', 'integrity')
      }
      expandedBytes += uncompressedSize

      if (expandedBytes > PREVIEW_UBLOCK_MAX_EXPANDED_BYTES) {
        throw safeError('the expanded archive is too large', 'integrity')
      }
    }

    entries.push({ isDirectory, name, uncompressedSize })
    offset = nextOffset
  }

  return entries
}

function ensureInside(root: string, candidate: string): void {
  const resolvedRoot = path.resolve(root)
  const resolvedCandidate = path.resolve(candidate)

  if (resolvedCandidate !== resolvedRoot && !resolvedCandidate.startsWith(`${resolvedRoot}${path.sep}`)) {
    throw safeError('the archive escaped its staging directory', 'integrity')
  }
}

function validateExtensionTree(root: string): void {
  const rootStat = fs.lstatSync(root)

  if (!rootStat.isDirectory() || rootStat.isSymbolicLink()) {
    throw safeError('the extension root is not a directory', 'integrity')
  }

  const visit = (directory: string): void => {
    for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
      const entryPath = path.join(directory, entry.name)
      ensureInside(root, entryPath)
      const stat = fs.lstatSync(entryPath)

      if (stat.isSymbolicLink()) {
        throw safeError('the extension contains a symlink', 'integrity')
      }

      if (stat.isDirectory()) {
        visit(entryPath)
      } else if (!stat.isFile()) {
        throw safeError('the extension contains a non-regular file', 'integrity')
      }
    }
  }

  visit(root)
}

function readManifest(root: string): Record<string, any> {
  try {
    const value: unknown = JSON.parse(fs.readFileSync(path.join(root, 'manifest.json'), 'utf8'))

    if (!value || typeof value !== 'object' || Array.isArray(value)) {
      throw new Error('not an object')
    }

    return value as Record<string, any>
  } catch (error) {
    const code = error && typeof error === 'object' && 'code' in error ? String(error.code) : ''

    if (code && code !== 'ENOENT') {
      throw safeError(
        `could not read the extension manifest: ${error instanceof Error ? error.message : String(error)}`,
        'storage'
      )
    }

    throw safeError('manifest.json is invalid', 'compatibility')
  }
}

function safeManifestPath(value: unknown, fallback: string): string {
  if (value === undefined) {
    return fallback
  }

  if (typeof value !== 'string' || value.includes('\\')) {
    throw safeError('the manifest contains an unsafe path', 'compatibility')
  }

  return checkedZipPath(value.replace(/^\/+/, ''))
}

export function validateExtensionDirectory(root: string, tag: string, requireWrapper = true): string | null {
  if (!RELEASE_TAG_RE.test(tag)) {
    throw safeError('the release tag is invalid', 'compatibility')
  }
  validateExtensionTree(root)
  const manifest = readManifest(root)

  if (manifest.manifest_version !== 3 || manifest.version !== tag) {
    throw safeError('the extension manifest is not the requested MV3 release', 'compatibility')
  }

  if (typeof manifest.name !== 'string' || manifest.name.trim() === '') {
    throw safeError('the extension name is missing', 'compatibility')
  }
  const dashboard = safeManifestPath(manifest.dashboard, 'dashboard.html')
  const serviceWorker = safeManifestPath(manifest.background?.service_worker, '')
  const declaredPopup =
    manifest.action && typeof manifest.action === 'object' && !Array.isArray(manifest.action)
      ? manifest.action.default_popup
      : undefined
  const popup = declaredPopup === undefined ? null : safeManifestPath(declaredPopup, '')

  if (!serviceWorker) {
    throw safeError('the background service worker is missing', 'compatibility')
  }
  const required = new Set([...REQUIRED_FILES, dashboard, serviceWorker])

  if (popup) {
    required.add(popup)
  }

  if (requireWrapper) {
    required.add(WRAPPER_FILENAME)
    required.add(ORIGINAL_SERVICE_WORKER_FILENAME)
    required.add(PRESERVED_EXT_COMPAT_FILENAME)
  }

  for (const relativePath of required) {
    const absolutePath = path.join(root, relativePath)
    ensureInside(root, absolutePath)
    let stat: fs.Stats

    try {
      stat = fs.lstatSync(absolutePath)
    } catch (error) {
      const code = error && typeof error === 'object' && 'code' in error ? String(error.code) : ''

      if (code && code !== 'ENOENT') {
        throw safeError(
          `could not inspect the extension cache: ${error instanceof Error ? error.message : String(error)}`,
          'storage'
        )
      }

      throw safeError(`required extension file is missing: ${relativePath}`, 'compatibility')
    }

    if (!stat.isFile() || stat.isSymbolicLink()) {
      throw safeError(`required extension file is not regular: ${relativePath}`, 'integrity')
    }
  }

  if (requireWrapper && serviceWorker !== WRAPPER_FILENAME) {
    throw safeError('the manifest does not use the Hermes compatibility wrapper', 'compatibility')
  }

  if (requireWrapper) {
    let wrapper: Buffer
    let expectedWrapper: Buffer

    try {
      wrapper = fs.readFileSync(path.join(root, WRAPPER_FILENAME))
      expectedWrapper = createCompatibilityWorker(fs.readFileSync(path.join(root, ORIGINAL_SERVICE_WORKER_FILENAME)))
    } catch (error) {
      throw safeError(
        `could not read the compatibility worker: ${error instanceof Error ? error.message : String(error)}`,
        'storage'
      )
    }

    if (!wrapper.equals(expectedWrapper)) {
      throw safeError('the Hermes compatibility wrapper is invalid', 'compatibility')
    }

    let extensionCompatSource: Buffer
    let generatedExtensionCompat: Buffer

    try {
      extensionCompatSource = fs.readFileSync(path.join(root, PRESERVED_EXT_COMPAT_FILENAME))
      generatedExtensionCompat = fs.readFileSync(path.join(root, EXT_COMPAT_FILENAME))
    } catch (error) {
      throw safeError(
        `could not read the extension-page compatibility module: ${error instanceof Error ? error.message : String(error)}`,
        'storage'
      )
    }

    if (!generatedExtensionCompat.equals(createCompatibilityExtensionModule(extensionCompatSource))) {
      throw safeError('the extension-page compatibility module is invalid', 'compatibility')
    }
  }

  return popup
}

function createCompatibilityWrapperForRevision(revision: string): string {
  return (
    `// Hermes compatibility revision: ${revision}\n` +
    `const createEvent = () => {\n` +
    `  const listeners = new Set();\n` +
    `  return { addListener(listener) { listeners.add(listener); }, removeListener(listener) { listeners.delete(listener); }, hasListener(listener) { return listeners.has(listener); } };\n` +
    `};\n` +
    `const api = globalThis.browser ?? globalThis.chrome;\n` +
    `if (!api) throw new Error('Chrome extension API is unavailable');\n` +
    `if (!globalThis.browser) globalThis.browser = api;\n` +
    `const permissions = api.permissions ?? (api.permissions = {});\n` +
    `if (typeof permissions.getAll !== 'function') permissions.getAll = async () => ({ origins: ['<all_urls>'], permissions: [] });\n` +
    `if (!permissions.onAdded) permissions.onAdded = createEvent();\n` +
    `if (!permissions.onRemoved) permissions.onRemoved = createEvent();\n` +
    `const commands = api.commands ?? (api.commands = {});\n` +
    `if (!commands.onCommand) commands.onCommand = createEvent();\n`
  )
}

export function createCompatibilityWrapper(): string {
  return createCompatibilityWrapperForRevision(PREVIEW_UBLOCK_COMPATIBILITY_REVISION)
}

export function createCompatibilityWorker(originalWorker: Uint8Array | string): Buffer {
  const source = typeof originalWorker === 'string' ? Buffer.from(originalWorker, 'utf8') : Buffer.from(originalWorker)

  return Buffer.concat([Buffer.from(createCompatibilityWrapper(), 'utf8'), source])
}

export function createCompatibilityExtensionModule(source: Uint8Array | string): Buffer {
  const original = typeof source === 'string' ? Buffer.from(source, 'utf8') : Buffer.from(source)
  const prefix =
    `// Hermes extension-page compatibility revision: ${PREVIEW_UBLOCK_COMPATIBILITY_REVISION}\n` +
    `const createEvent = () => {\n` +
    `  const listeners = new Set();\n` +
    `  return { addListener(listener) { listeners.add(listener); }, removeListener(listener) { listeners.delete(listener); }, hasListener(listener) { return listeners.has(listener); } };\n` +
    `};\n` +
    `const api = globalThis.browser ?? globalThis.chrome;\n` +
    `if (!api) throw new Error('Chrome extension API is unavailable');\n` +
    `if (!globalThis.browser) globalThis.browser = api;\n` +
    `const permissions = api.permissions ?? (api.permissions = {});\n` +
    `if (typeof permissions.getAll !== 'function') permissions.getAll = async () => ({ origins: ['<all_urls>'], permissions: [] });\n` +
    `if (!permissions.onAdded) permissions.onAdded = createEvent();\n` +
    `if (!permissions.onRemoved) permissions.onRemoved = createEvent();\n` +
    `if (typeof permissions.request !== 'function') permissions.request = async details => {\n` +
    `  if (!details || typeof details !== 'object' || Array.isArray(details)) throw new TypeError('Invalid permission request');\n` +
    `  const origins = details.origins;\n` +
    `  const requestedPermissions = details.permissions;\n` +
    `  if (!Array.isArray(origins) || origins.some(origin => typeof origin !== 'string' || !/^<all_urls>$|^(?:\\*|https?|ftp):\\/\\/(?:\\*|\\*\\.[A-Za-z0-9.-]+|[A-Za-z0-9.-]+)(?::\\d+)?\\/\\*$|^file:\\/\\/\\/.*$/.test(origin))) throw new TypeError('Unsupported permission origin');\n` +
    `  if (requestedPermissions !== undefined && (!Array.isArray(requestedPermissions) || requestedPermissions.length !== 0)) throw new TypeError('Unsupported permission request');\n` +
    `  return true;\n` +
    `};\n` +
    `const commands = api.commands ?? (api.commands = {});\n` +
    `if (!commands.onCommand) commands.onCommand = createEvent();\n`

  return Buffer.concat([Buffer.from(prefix, 'utf8'), original])
}

export function rewriteManifestServiceWorker(root: string): void {
  const manifest = readManifest(root)

  if (!manifest.background || typeof manifest.background !== 'object' || Array.isArray(manifest.background)) {
    throw safeError('the extension manifest has no background service worker', 'compatibility')
  }

  manifest.background.service_worker = WRAPPER_FILENAME
  manifest.background.type = 'module'
  fs.writeFileSync(path.join(root, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`, {
    encoding: 'utf8',
    mode: 0o600
  })
}

export function extractArchive(archive: Uint8Array, stagingPath: string, tag: string): void {
  const entries = inspectZip(archive)
  let files: Record<string, Uint8Array>

  try {
    files = unzipSync(archive)
  } catch {
    throw safeError('the archive could not be extracted', 'integrity')
  }

  let expandedBytes = 0
  let fileCount = 0
  fs.mkdirSync(stagingPath, { recursive: true, mode: 0o700 })
  fs.chmodSync(stagingPath, 0o700)

  for (const entry of entries) {
    const destination = path.join(stagingPath, entry.name)
    ensureInside(stagingPath, destination)

    if (entry.isDirectory) {
      fs.mkdirSync(destination, { recursive: true, mode: 0o700 })

      continue
    }

    const data = files[entry.name]

    if (!data || data.length !== entry.uncompressedSize) {
      throw safeError('the archive contents do not match its directory', 'integrity')
    }
    fileCount += 1
    expandedBytes += data.length

    if (
      fileCount > PREVIEW_UBLOCK_MAX_FILES ||
      expandedBytes > PREVIEW_UBLOCK_MAX_EXPANDED_BYTES ||
      data.length > PREVIEW_UBLOCK_MAX_FILE_BYTES
    ) {
      throw safeError('the expanded archive is too large', 'integrity')
    }

    fs.mkdirSync(path.dirname(destination), { recursive: true, mode: 0o700 })
    fs.writeFileSync(destination, Buffer.from(data), { encoding: null, mode: 0o600 })
  }

  validateExtensionDirectory(stagingPath, tag, false)
}

export function prepareExtension(root: string, tag: string): void {
  const extensionCompatPath = path.join(root, EXT_COMPAT_FILENAME)
  const preservedExtensionCompatPath = path.join(root, PRESERVED_EXT_COMPAT_FILENAME)

  if (!fs.existsSync(preservedExtensionCompatPath)) {
    fs.renameSync(extensionCompatPath, preservedExtensionCompatPath)
  }
  const extensionCompatSource = fs.readFileSync(preservedExtensionCompatPath)
  const originalWorkerPath = path.join(root, ORIGINAL_SERVICE_WORKER_FILENAME)
  const originalWorker = fs.readFileSync(originalWorkerPath)

  fs.writeFileSync(extensionCompatPath, createCompatibilityExtensionModule(extensionCompatSource), { mode: 0o600 })
  fs.writeFileSync(path.join(root, WRAPPER_FILENAME), createCompatibilityWorker(originalWorker), { mode: 0o600 })
  rewriteManifestServiceWorker(root)
  validateExtensionDirectory(root, tag, true)
}

interface ActivePointer {
  cacheKey: string
  schemaVersion: 2
}

function cacheKey(): string {
  return `${PREVIEW_UBLOCK_VERSION}-${PREVIEW_UBLOCK_COMPATIBILITY_REVISION}`
}

function metadataPath(directory: string): string {
  return path.join(directory, INSTALLED_METADATA_FILENAME)
}

function validateMetadata(value: unknown): value is InstalledUblockMetadata {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return false
  }
  const candidate = value as Record<string, unknown>

  return (
    Object.keys(candidate).sort().join(',') ===
      'archiveSha256,archiveUrl,compatibilityRevision,schemaVersion,version' &&
    candidate.schemaVersion === 2 &&
    candidate.version === PREVIEW_UBLOCK_VERSION &&
    candidate.archiveUrl === PREVIEW_UBLOCK_ARCHIVE_URL &&
    candidate.archiveSha256 === PREVIEW_UBLOCK_ARCHIVE_SHA256 &&
    candidate.compatibilityRevision === PREVIEW_UBLOCK_COMPATIBILITY_REVISION
  )
}

function readJson(filePath: string): unknown {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'))
  } catch (error) {
    const code = error && typeof error === 'object' && 'code' in error ? String(error.code) : ''

    if (code && code !== 'ENOENT') {
      throw safeError(`could not read the cache: ${error instanceof Error ? error.message : String(error)}`, 'storage')
    }

    return null
  }
}

function readActive(
  cacheDirectory: string
): { directory: string; metadata: InstalledUblockMetadata; popupPath?: string } | null {
  const pointer = readJson(path.join(cacheDirectory, ACTIVE_POINTER_FILENAME)) as Partial<ActivePointer> | null

  if (
    !pointer ||
    pointer.schemaVersion !== 2 ||
    typeof pointer.cacheKey !== 'string' ||
    !CACHE_KEY_RE.test(pointer.cacheKey)
  ) {
    return null
  }
  const directory = path.join(cacheDirectory, VERSIONS_DIRNAME, pointer.cacheKey)
  const metadata = readJson(metadataPath(directory))

  if (!validateMetadata(metadata)) {
    return null
  }

  try {
    const popupPath = validateExtensionDirectory(directory, metadata.version, true)

    return { directory, metadata, ...(popupPath ? { popupPath } : {}) }
  } catch (error) {
    if (error && typeof error === 'object' && 'code' in error && error.code === 'storage') {
      throw error
    }

    const code = error && typeof error === 'object' && 'code' in error ? String(error.code) : ''

    if (['EACCES', 'EIO', 'ENOSPC', 'ENOTDIR', 'EPERM', 'EROFS'].includes(code)) {
      throw safeError(
        `could not inspect the active uBlock cache: ${error instanceof Error ? error.message : String(error)}`,
        'storage'
      )
    }

    return null
  }
}

function removePath(target: string): void {
  fs.rmSync(target, { force: true, recursive: true })
}

/**
 * Remove immutable cache versions that are no longer reachable from the
 * validated active pointer. An unreadable or invalid pointer intentionally
 * leaves all validly named versions alone: there is no safe basis for
 * choosing which one to preserve.
 */
export function prunePreviewUblockCache(cacheDirectory: string): void {
  const active = readActive(cacheDirectory)

  if (!active) {
    return
  }

  const versionsDirectory = path.join(cacheDirectory, VERSIONS_DIRNAME)
  const activeKey = path.basename(active.directory)
  let cleanupError: unknown = null

  for (const entry of fs.readdirSync(versionsDirectory, { withFileTypes: true })) {
    if (entry.name === activeKey || !CACHE_KEY_RE.test(entry.name) || !entry.isDirectory()) {
      continue
    }

    const target = path.join(versionsDirectory, entry.name)
    ensureInside(versionsDirectory, target)
    try {
      removePath(target)
    } catch (error) {
      cleanupError ??= error
    }
  }

  if (cleanupError) {
    throw cleanupError
  }
}

function discardPath(target: string): void {
  try {
    removePath(target)
  } catch (error) {
    throw safeError(
      `could not clean up the uBlock cache: ${error instanceof Error ? error.message : String(error)}`,
      'storage'
    )
  }
}

function recoverCache(cacheDirectory: string): void {
  fs.mkdirSync(cacheDirectory, { recursive: true, mode: 0o700 })
  fs.mkdirSync(path.join(cacheDirectory, VERSIONS_DIRNAME), { recursive: true, mode: 0o700 })

  for (const entry of fs.readdirSync(cacheDirectory)) {
    if (entry.startsWith('.staging-') || entry.startsWith('.active-')) {
      removePath(path.join(cacheDirectory, entry))
    }
  }

  // Schema-v1 stored a mutable `current` tree and was produced by the old text patcher.
  removePath(path.join(cacheDirectory, 'current'))
  removePath(path.join(cacheDirectory, INSTALLED_METADATA_FILENAME))

  for (const entry of fs.readdirSync(path.join(cacheDirectory, VERSIONS_DIRNAME))) {
    if (!CACHE_KEY_RE.test(entry)) {
      removePath(path.join(cacheDirectory, VERSIONS_DIRNAME, entry))
    }
  }

  prunePreviewUblockCache(cacheDirectory)
}

function randomSuffix(): string {
  return `${process.pid}-${crypto.randomBytes(8).toString('hex')}`
}

function prepareCandidate(
  cacheDirectory: string,
  stagingPath: string,
  metadata: InstalledUblockMetadata
): PreviewUblockInstallCandidate {
  const versionsDirectory = path.join(cacheDirectory, VERSIONS_DIRNAME)
  const candidateKey = `${cacheKey()}-${randomSuffix()}`
  const finalDirectory = path.join(versionsDirectory, candidateKey)

  fs.writeFileSync(metadataPath(stagingPath), `${JSON.stringify(metadata, null, 2)}\n`, {
    encoding: 'utf8',
    mode: 0o600
  })
  const popupPath = validateExtensionDirectory(stagingPath, metadata.version, true)

  const pointerPath = path.join(cacheDirectory, ACTIVE_POINTER_FILENAME)
  let installed = false
  let committed = false

  return {
    finalPath: finalDirectory,
    installFinal() {
      if (installed) {
        return
      }

      try {
        fs.renameSync(stagingPath, finalDirectory)
        installed = true
      } catch (error) {
        throw safeError(
          `could not prepare the immutable cache: ${error instanceof Error ? error.message : String(error)}`,
          'storage'
        )
      }
    },
    commitActive() {
      if (!installed) {
        throw safeError('the immutable cache was not prepared', 'storage')
      }
      if (committed) {
        return
      }

      const temporaryPointerPath = path.join(cacheDirectory, `.active-${randomSuffix()}.json`)

      try {
        fs.writeFileSync(
          temporaryPointerPath,
          `${JSON.stringify({ cacheKey: candidateKey, schemaVersion: 2 } satisfies ActivePointer)}\n`,
          {
            encoding: 'utf8',
            mode: 0o600
          }
        )
        fs.renameSync(temporaryPointerPath, pointerPath)
        committed = true
      } catch (error) {
        discardPath(temporaryPointerPath)
        throw safeError(
          `could not activate the verified cache: ${error instanceof Error ? error.message : String(error)}`,
          'storage'
        )
      }

      try {
        prunePreviewUblockCache(cacheDirectory)
      } catch {
        // The pointer and verified extension are already committed. Cleanup
        // is retried by recoverCache during the next installer startup.
        console.warn('[preview] could not prune obsolete uBlock cache versions; will retry on next launch')
      }
    },
    discard() {
      if (committed) {
        return
      }

      discardPath(stagingPath)

      if (installed) {
        discardPath(finalDirectory)
      }
    },
    ...(popupPath ? { popupPath } : {}),
    stagedPath: stagingPath,
    version: metadata.version
  }
}

function verifyArchive(archive: Uint8Array): void {
  if (archive.length < 1 || archive.length > PREVIEW_UBLOCK_MAX_ARCHIVE_BYTES) {
    throw safeError('the archive size is invalid', 'integrity')
  }
  const actual = crypto.createHash('sha256').update(archive).digest('hex')

  if (actual !== PREVIEW_UBLOCK_ARCHIVE_SHA256) {
    throw safeError('the downloaded archive checksum did not match the pinned release', 'integrity')
  }
}

export function createPreviewUblockInstaller({
  cacheDirectory,
  request = requestHttps,
  userDataPath
}: PreviewUblockInstallerOptions): PreviewUblockInstaller {
  const resolvedCacheDirectory =
    cacheDirectory ?? (userDataPath ? path.join(userDataPath, PREVIEW_UBLOCK_CACHE_DIRNAME) : null)

  if (!resolvedCacheDirectory) {
    const unavailable = safeError('the uBlock cache directory is not configured', 'storage')

    return {
      resolve: async () => {
        throw unavailable
      }
    }
  }

  let cacheRecoveryError: (Error & { code: PreviewUblockFailureCode }) | null = null

  try {
    recoverCache(resolvedCacheDirectory)
  } catch (error) {
    cacheRecoveryError = safeError(
      `could not prepare the uBlock cache: ${error instanceof Error ? error.message : String(error)}`,
      'storage'
    )
  }

  let pinnedPromise: Promise<PreviewUblockInstallResult> | null = null

  const resolveCached = (): PreviewUblockInstallResult | null => {
    if (cacheRecoveryError) {
      throw cacheRecoveryError
    }

    const active = readActive(resolvedCacheDirectory)

    if (active) {
      return {
        path: active.directory,
        ...(active.popupPath ? { popupPath: active.popupPath } : {}),
        version: active.metadata.version
      }
    }

    return null
  }

  const installPinned = async (callbacks: PreviewUblockInstallCallbacks): Promise<PreviewUblockInstallResult> => {
    if (cacheRecoveryError) {
      throw cacheRecoveryError
    }

    callbacks.onPhase?.('checking-cache')
    const cached = resolveCached()

    if (cached) {
      return cached
    }
    callbacks.onPhase?.('downloading')
    let archive: Uint8Array

    try {
      archive = await request(PREVIEW_UBLOCK_ARCHIVE_URL, {
        maxBytes: PREVIEW_UBLOCK_MAX_ARCHIVE_BYTES,
        maxRedirects: MAX_REDIRECTS,
        onProgress: callbacks.onProgress,
        timeoutMs: DOWNLOAD_TIMEOUT_MS
      })
    } catch (error) {
      throw classifyRequestError(error)
    }

    callbacks.onPhase?.('verifying')
    verifyArchive(archive)
    const stagingPath = path.join(resolvedCacheDirectory, `.staging-${randomSuffix()}`)

    try {
      callbacks.onPhase?.('extracting')
      extractArchive(archive, stagingPath, PREVIEW_UBLOCK_VERSION)
      callbacks.onPhase?.('preparing')
      prepareExtension(stagingPath, PREVIEW_UBLOCK_VERSION)
      const metadata: InstalledUblockMetadata = {
        archiveSha256: PREVIEW_UBLOCK_ARCHIVE_SHA256,
        archiveUrl: PREVIEW_UBLOCK_ARCHIVE_URL,
        compatibilityRevision: PREVIEW_UBLOCK_COMPATIBILITY_REVISION,
        schemaVersion: 2,
        version: PREVIEW_UBLOCK_VERSION
      }

      return prepareCandidate(resolvedCacheDirectory, stagingPath, metadata)
    } catch (error) {
      discardPath(stagingPath)

      if (isPreviewUblockFailure(error)) {
        throw error
      }
      throw safeError(error instanceof Error ? error.message : String(error), 'storage')
    }
  }

  return {
    async resolve(intent, callbacks = {}) {
      if (intent === 'cached') {
        callbacks.onPhase?.('checking-cache')

        return resolveCached()
      }

      if (!pinnedPromise) {
        pinnedPromise = installPinned(callbacks).finally(() => {
          pinnedPromise = null
        })
      }

      return pinnedPromise
    }
  }
}

export { PREVIEW_UBLOCK_CACHE_DIRNAME as PREVIEW_UBLOCK_CACHE_NAME, WRAPPER_FILENAME }
