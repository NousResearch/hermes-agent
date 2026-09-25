#!/usr/bin/env node
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { createHash } from 'node:crypto'
import { createRequire } from 'node:module'
import { pipeline } from 'node:stream/promises'
import { Readable } from 'node:stream'
import { pathToFileURL } from 'node:url'
import { parseArgs } from 'node:util'
import { isMain } from './utils.mjs'
import { publishPackagingInputs } from './prepared-packaging.mjs'
import { ensureWindowsBundleTools } from './windows-bundle-tools.mjs'
import { prepareDmgbuild } from './prepare-dmgbuild.mjs'

/** @param {string} source @param {string} name @returns {string} */
export function pinnedPackageRoot(source, name) {
  const require = createRequire(path.join(source, 'apps/desktop/package.json'))
  const entry = require.resolve(name)
  let directory = path.dirname(entry)
  while (!fs.existsSync(path.join(directory, 'package.json'))) {
    const parent = path.dirname(directory)
    if (parent === directory) throw new Error(`Cannot locate installed ${name}`)
    directory = parent
  }
  const installed = JSON.parse(fs.readFileSync(path.join(directory, 'package.json'), 'utf8'))
  const lock = JSON.parse(fs.readFileSync(path.join(source, 'package-lock.json'), 'utf8'))
  const key = path.relative(source, directory).split(path.sep).join('/')
  if (!lock.packages?.[key] || lock.packages[key].version !== installed.version || installed.name !== name) {
    throw new Error(`Installed ${name} is not the source's lock-pinned package; prepare Node dependencies first`)
  }
  return directory
}

/** @param {string} target @returns {'x64' | 'arm64'} */
export function packagingTargetArch(target) {
  if (target === `${process.platform}-x64`) return 'x64'
  if (target === `${process.platform}-arm64`) return 'arm64'
  throw new Error(`Packaging preparation requires a same-OS x64/arm64 target, got ${target}`)
}

/** @param {string} from @param {string} to @returns {string} */
function copyTool(from, to) {
  fs.rmSync(to, { recursive: true, force: true })
  fs.cpSync(from, to, { recursive: true, verbatimSymlinks: true })
  return to
}

const ELECTRON_DOWNLOAD_ATTEMPTS = 4
const ELECTRON_DOWNLOAD_BASE_DELAY_MS = 1000
const ELECTRON_DEFAULT_MIRROR = 'https://github.com/electron/electron/releases/download'

/** @param {number} ms @returns {Promise<void>} */
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms))
}

/**
 * Run an async download step with exponential backoff and jitter. Partial
 * files are left in place between attempts so resumable downloads continue
 * where they stopped instead of restarting.
 * @template T
 * @param {() => Promise<T>} step
 * @param {{ label?: string, attempts?: number, baseDelayMs?: number }} [options]
 * @returns {Promise<T>}
 */
export async function withDownloadRetry(step, { label = 'download', attempts = ELECTRON_DOWNLOAD_ATTEMPTS, baseDelayMs = ELECTRON_DOWNLOAD_BASE_DELAY_MS } = {}) {
  let lastError
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      return await step()
    } catch (error) {
      lastError = error
      if (attempt === attempts) break
      const delay = baseDelayMs * 2 ** (attempt - 1) + Math.floor(Math.random() * baseDelayMs)
      console.warn(`${label} failed (attempt ${attempt}/${attempts}); retrying in ${delay}ms: ${error instanceof Error ? error.message : String(error)}`)
      await sleep(delay)
    }
  }
  throw lastError
}

/**
 * @param {{ version: string, platformName: string, arch: string, artifactName?: string }} artifact
 * @returns {string}
 */
export function electronArtifactFileName({ version, platformName, arch, artifactName = 'electron' }) {
  return `${artifactName}-v${version}-${platformName}-${arch}.zip`
}

/** @returns {string} */
export function electronMirrorRoot() {
  return (process.env.ELECTRON_MIRROR || ELECTRON_DEFAULT_MIRROR).replace(/\/+$/, '')
}

/**
 * @param {string} version
 * @returns {string}
 */
export function electronCustomDir(version) {
  const template = process.env.ELECTRON_CUSTOM_DIR || `v${version}`
  return template.replace('{{ version }}', version).replace(/^\/+|\/+$/g, '')
}

/**
 * Replicates @electron/get Cache.getCacheDirectory: sha256 of the download
 * URL with hash/search stripped and the pathname reduced to its directory.
 * @param {string} downloadFileUrl
 * @returns {string}
 */
export function electronCacheDirName(downloadFileUrl) {
  const parsed = new URL(downloadFileUrl)
  parsed.hash = ''
  parsed.search = ''
  parsed.pathname = path.posix.dirname(parsed.pathname)
  return createHash('sha256').update(parsed.toString()).digest('hex')
}

/**
 * Exact private-cache slot @electron/get will use for this artifact, so a
 * seeded or resumed file resolves as a cache hit without re-downloading.
 * @param {{ version: string, platformName: string, arch: string, cacheDir: string, artifactName?: string }} artifact
 * @returns {{ fileName: string, dirUrl: string, fileUrl: string, cachePath: string }}
 */
export function electronArtifactCacheSlot({ version, platformName, arch, cacheDir, artifactName = 'electron' }) {
  const fileName = electronArtifactFileName({ version, platformName, arch, artifactName })
  const dirUrl = `${electronMirrorRoot()}/${electronCustomDir(version)}`
  const fileUrl = `${dirUrl}/${fileName}`
  const cachePath = path.join(cacheDir, electronCacheDirName(fileUrl), fileName)
  return { fileName, dirUrl, fileUrl, cachePath }
}

/**
 * Candidate roots for the default @electron/get cache: the ELECTRON_CACHE
 * override first, then the per-OS default location.
 * @returns {string[]}
 */
export function defaultElectronCacheRoots() {
  const roots = []
  if (process.env.ELECTRON_CACHE) roots.push(process.env.ELECTRON_CACHE)
  const home = os.homedir()
  if (process.platform === 'darwin') roots.push(path.join(home, 'Library', 'Caches', 'electron'))
  else if (process.platform === 'win32') {
    if (process.env.LOCALAPPDATA) roots.push(path.join(process.env.LOCALAPPDATA, 'electron', 'Cache'))
  } else {
    if (process.env.XDG_CACHE_HOME) roots.push(path.join(process.env.XDG_CACHE_HOME, 'electron'))
    roots.push(path.join(home, '.cache', 'electron'))
  }
  return [...new Set(roots)]
}

/**
 * @param {string} dir
 * @param {string} fileName
 * @param {string[]} [hits]
 * @returns {string[]}
 */
function findFilesByName(dir, fileName, hits = []) {
  let entries
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true })
  } catch {
    return hits
  }
  for (const entry of entries) {
    const full = path.join(dir, entry.name)
    if (entry.isDirectory()) findFilesByName(full, fileName, hits)
    else if (entry.isFile() && entry.name === fileName) hits.push(full)
  }
  return hits
}

/** @param {string} filePath @returns {Promise<string>} */
function sha256File(filePath) {
  return new Promise((resolve, reject) => {
    const hash = createHash('sha256')
    const stream = fs.createReadStream(filePath)
    stream.on('error', reject)
    stream.on('data', chunk => hash.update(chunk))
    stream.on('end', () => {
      try {
        resolve(hash.digest('hex'))
      } catch (error) {
        reject(error)
      }
    })
  })
}

/**
 * @param {string} text
 * @returns {Map<string, string>}
 */
export function parseShasums(text) {
  const entries = new Map()
  for (const line of text.split('\n')) {
    const match = line.match(/^([0-9a-fA-F]{64})\s+\*?(\S+)\s*$/)
    if (match) entries.set(match[2], match[1].toLowerCase())
  }
  return entries
}

/**
 * @param {string} filePath
 * @param {string} fileName
 * @param {string} shasumsText
 * @returns {Promise<void>}
 */
async function verifyFileAgainstShasums(filePath, fileName, shasumsText) {
  const expected = parseShasums(shasumsText).get(fileName)
  if (!expected) throw new Error(`No checksum entry for ${fileName} in SHASUMS256.txt`)
  const actual = (await sha256File(filePath)).toLowerCase()
  if (actual !== expected) throw new Error(`Checksum mismatch for ${fileName}: expected ${expected}, got ${actual}`)
}

/**
 * Fetch the SHASUMS256.txt index for an electron release. Prefers copies
 * already present in the default @electron/get cache so a seeded restore
 * needs no network; falls back to the mirror with retry.
 * @param {{ dirUrl: string, fileName: string, cacheRoots: string[] }} options
 * @returns {Promise<string | null>}
 */
async function loadShasumsText({ dirUrl, fileName, cacheRoots }) {
  for (const root of cacheRoots) {
    for (const candidate of findFilesByName(root, 'SHASUMS256.txt')) {
      try {
        const text = fs.readFileSync(candidate, 'utf8')
        if (parseShasums(text).has(fileName)) return text
      } catch {
        // unreadable copy: keep looking
      }
    }
  }
  try {
    const response = await withDownloadRetry(async () => {
      const result = await fetch(`${dirUrl}/SHASUMS256.txt`, { redirect: 'follow' })
      if (!result.ok) throw new Error(`SHASUMS256.txt request failed with HTTP ${result.status}`)
      return result
    }, { label: 'SHASUMS256.txt download' })
    const text = await response.text()
    if (!parseShasums(text).has(fileName)) return null
    return text
  } catch {
    return null
  }
}

/**
 * On a private-cache miss, copy a checksum-verified electron zip from the
 * default @electron/get cache into the exact private-cache slot, so the
 * subsequent download resolves as a cache hit without touching the network.
 * @param {{ version: string, platformName: string, arch: string, cacheDir: string, artifactName?: string }} artifact
 * @returns {Promise<string | null>}
 */
export async function seedElectronCacheFromDefaultCache({ version, platformName, arch, cacheDir, artifactName = 'electron' }) {
  const slot = electronArtifactCacheSlot({ version, platformName, arch, cacheDir, artifactName })
  try {
    if (fs.statSync(slot.cachePath).size > 0) return slot.cachePath
  } catch {
    // private-cache miss: consult the shared default cache below
  }
  const roots = defaultElectronCacheRoots().filter(root => path.resolve(root) !== path.resolve(cacheDir))
  const shasums = await loadShasumsText({ dirUrl: slot.dirUrl, fileName: slot.fileName, cacheRoots: roots })
  if (!shasums) return null
  for (const root of roots) {
    for (const candidate of findFilesByName(root, slot.fileName)) {
      try {
        if (fs.statSync(candidate).size === 0) continue
        await verifyFileAgainstShasums(candidate, slot.fileName, shasums)
        fs.mkdirSync(path.dirname(slot.cachePath), { recursive: true })
        fs.copyFileSync(candidate, slot.cachePath)
        await verifyFileAgainstShasums(slot.cachePath, slot.fileName, shasums)
        return slot.cachePath
      } catch {
        try {
          fs.rmSync(slot.cachePath, { force: true })
        } catch {
          // best effort cleanup of a bad copy
        }
      }
    }
  }
  return null
}

/**
 * Download a URL with HTTP Range resume: an interrupted attempt leaves a
 * `.part` file in place and the next attempt continues from its size, with
 * exponential backoff between attempts.
 * @param {string} url
 * @param {string} destPath
 * @param {{ label?: string, attempts?: number, baseDelayMs?: number }} [options]
 * @returns {Promise<string>}
 */
export async function downloadFileWithResume(url, destPath, { label = url, attempts = ELECTRON_DOWNLOAD_ATTEMPTS, baseDelayMs = ELECTRON_DOWNLOAD_BASE_DELAY_MS } = {}) {
  const partPath = `${destPath}.part`
  let lastError
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      let start = 0
      try {
        start = fs.statSync(partPath).size
      } catch {
        start = 0
      }
      const response = await fetch(url, {
        redirect: 'follow',
        ...(start > 0 ? { headers: { Range: `bytes=${start}-` } } : {}),
      })
      if (response.status === 416) throw new Error('range unsatisfiable; restarting download')
      if (response.status !== 200 && response.status !== 206) throw new Error(`unexpected HTTP ${response.status}`)
      const resumed = response.status === 206 && start > 0
      if (!resumed) start = 0
      if (response.body === null) throw new Error('empty response body')
      const file = fs.createWriteStream(partPath, { flags: resumed ? 'a' : 'w' })
      try {
        await pipeline(Readable.fromWeb(response.body), file)
      } catch (error) {
        try {
          file.destroy()
        } catch {
          // best effort cleanup of a broken stream
        }
        throw error
      }
      const received = fs.statSync(partPath).size - start
      const total = Number(response.headers.get('content-length'))
      if (Number.isFinite(total) && total >= 0 && received < total) {
        throw new Error(`truncated response: received ${received} of ${total} bytes`)
      }
      fs.renameSync(partPath, destPath)
      return destPath
    } catch (error) {
      lastError = error
      if (attempt === attempts) break
      const delay = baseDelayMs * 2 ** (attempt - 1) + Math.floor(Math.random() * baseDelayMs)
      console.warn(`${label} failed (attempt ${attempt}/${attempts}); resuming in ${delay}ms: ${error instanceof Error ? error.message : String(error)}`)
      await sleep(delay)
    }
  }
  throw lastError
}

/**
 * Download the electron artifact zip with backoff retry throughout: seed the
 * private slot from the default @electron/get cache first, retry the builder
 * download, and as a last resort fetch the release asset directly with Range
 * resume plus SHASUMS256 checksum verification.
 * @param {{ downloadElectronArtifactZip: (options: object) => Promise<string> }} electronGet
 * @param {{ version: string, platformName: string, arch: string, cacheDir: string, artifactName?: string }} artifact
 * @returns {Promise<string>}
 */
export async function downloadElectronArtifactZipWithRetry(electronGet, { version, platformName, arch, cacheDir, artifactName = 'electron' }) {
  const slot = electronArtifactCacheSlot({ version, platformName, arch, cacheDir, artifactName })
  let seeded = null
  try {
    seeded = await seedElectronCacheFromDefaultCache({ version, platformName, arch, cacheDir, artifactName })
  } catch (error) {
    console.warn(`default-cache seeding skipped: ${error instanceof Error ? error.message : String(error)}`)
  }
  if (seeded) console.log(`Reusing cached electron artifact ${slot.fileName} from the default @electron/get cache`)
  const options = { version, platformName, arch, artifactName, cacheDir }
  try {
    return await withDownloadRetry(() => electronGet.downloadElectronArtifactZip(options), { label: 'electron artifact download' })
  } catch (error) {
    console.warn(`electron artifact download failed after retries; falling back to resumed direct download: ${error instanceof Error ? error.message : String(error)}`)
  }
  const shasums = await loadShasumsText({ dirUrl: slot.dirUrl, fileName: slot.fileName, cacheRoots: defaultElectronCacheRoots() })
  if (!shasums) throw new Error(`Cannot verify ${slot.fileName}: SHASUMS256.txt unavailable for ${slot.dirUrl}`)
  await downloadFileWithResume(slot.fileUrl, slot.cachePath, { label: `electron artifact resume download (${slot.fileName})` })
  await verifyFileAgainstShasums(slot.cachePath, slot.fileName, shasums)
  return electronGet.downloadElectronArtifactZip(options)
}

/**
 * Acquire bytes without signing credentials. Builder modules are loaded only
 * after the explicit cache root has been selected, before their lazy state runs.
 * @param {{ source: string, out: string, cache: string, target?: string, formats?: string[], dmgbuild?: string }} options
 * @returns {Promise<string>}
 */
async function preparePackagingTools({ source, out, cache, target = `${process.platform}-${process.arch}`, formats, dmgbuild }) {
  source = fs.realpathSync(source)
  out = path.resolve(out)
  cache = path.resolve(cache)
  fs.mkdirSync(out, { recursive: true })
  fs.rmSync(path.join(out, 'prepared.json'), { force: true })
  packagingTargetArch(target)
  const builderRoot = pinnedPackageRoot(source, 'app-builder-lib')
  pinnedPackageRoot(source, 'electron-builder')
  const require = createRequire(path.join(source, 'apps/desktop/package.json'))
  const config = require(path.join(source, 'apps/desktop/electron-builder.config.cjs'))
  formats ??= process.platform === 'win32' ? ['msix'] : process.platform === 'darwin' ? ['dmg', 'zip'] : ['AppImage']
  if (process.env.CUSTOM_DMGBUILD_PATH) throw new Error('Preparation must select the pinned dmgbuild supplier, not CUSTOM_DMGBUILD_PATH')
  const supported = process.platform === 'win32' ? ['dir', 'msix', 'zip'] : process.platform === 'darwin' ? ['dir', 'dmg', 'zip'] : ['dir', 'AppImage', 'deb', 'rpm', 'zip']
  if (formats.some(format => !supported.includes(format))) throw new Error(`Unsupported prepared package formats: ${formats.join(', ')}`)
  const dmg = formats.includes('dmg') ? prepareDmgbuild({ source, out, cache, binary: dmgbuild }) : null
  const previousCache = process.env.ELECTRON_BUILDER_CACHE
  process.env.ELECTRON_BUILDER_CACHE = path.join(cache, 'builder')
  try {
    return await acquirePackagingTools({ source, out, cache, target, formats, builderRoot, config, dmgbuild: dmg })
  } finally {
    if (previousCache === undefined) delete process.env.ELECTRON_BUILDER_CACHE
    else process.env.ELECTRON_BUILDER_CACHE = previousCache
  }
}

/**
 * @param {{ source: string, out: string, cache: string, target: string, formats: string[], builderRoot: string, config: import('app-builder-lib').Configuration, dmgbuild: string | null }} options
 * @returns {Promise<string>}
 */
async function acquirePackagingTools({ source, out, cache, target, formats, builderRoot, config, dmgbuild }) {
  /** @param {string} relative */
  const load = (relative) => import(pathToFileURL(path.join(builderRoot, 'dist', relative)).href)
  const [electronGet, sevenZip, icons] = await Promise.all([
    load('util/electronGet.js'), load('toolsets/7zip.js'), load('toolsets/icons.js'),
  ])
  const resourcesDir = path.join(source, 'apps/desktop', config.directories?.buildResources || 'build')
  const electronCacheDir = path.join(cache, 'electron')
  const electronArch = packagingTargetArch(target)
  const [archive, archiveTool, iconTools] = await Promise.all([
    downloadElectronArtifactZipWithRetry(electronGet, { version: config.electronVersion, platformName: process.platform, arch: electronArch,
      artifactName: 'electron', cacheDir: electronCacheDir }),
    withDownloadRetry(() => sevenZip.getPath7za(), { label: '7za toolset download' }),
    withDownloadRetry(() => icons.getIconsToolsetPath(config.toolsets?.icons, resourcesDir), { label: 'icons toolset download' }),
  ])
  const electron = copyTool(archive, path.join(out, 'electron.zip'))
  /** @type {import('./prepared-packaging.mjs').PackagingToolsets} */
  const toolsets = {
    sevenZip: copyTool(path.dirname(path.dirname(archiveTool)), path.join(out, 'sevenZip')),
    icons: copyTool(iconTools, path.join(out, 'icons')),
  }
  let windows = null
  if (process.platform === 'win32') {
    const builder = await load('toolsets/winCodeSign.js')
    const tools = await ensureWindowsBundleTools({ config, resourcesDir, signing: true, load: async () => builder, prepared: null })
    const kitRoot = copyTool(path.dirname(path.dirname(tools.makeappx)), path.join(out, 'winCodeSign'))
    if (!tools.dlib || !tools.dotnetRoot) throw new Error('Windows preparation requires the ATS dlib and paired .NET runtime')
    fs.cpSync(path.dirname(tools.dlib), path.join(kitRoot, path.basename(path.dirname(tools.signtool))), { recursive: true })
    const rcedit = await builder.getRceditBundle(config.toolsets?.winCodeSign, resourcesDir)
    fs.copyFileSync(rcedit.x64, path.join(kitRoot, 'rcedit-x64.exe'))
    fs.copyFileSync(rcedit.x86, path.join(kitRoot, 'rcedit-x86.exe'))
    const kit = path.join(kitRoot, path.basename(path.dirname(tools.makeappx)))
    windows = { makeappx: path.join(kit, 'makeappx.exe'), signtool: path.join(kit, 'signtool.exe'),
      dlib: path.join(kit, 'Azure.CodeSigning.Dlib.dll'), dotnetRoot: copyTool(tools.dotnetRoot, path.join(out, 'dotnet')) }
    toolsets.winCodeSign = kitRoot
  }
  if (formats.includes('AppImage')) {
    const appimage = await load('toolsets/appimage.js')
    const { Arch } = await import(pathToFileURL(path.join(builderRoot, 'dist/index.js')).href)
    const tools = await appimage.getAppImageTools(config.toolsets?.appimage, Arch[packagingTargetArch(target)], resourcesDir)
    toolsets.appimage = copyTool(path.dirname(tools.mksquashfs), path.join(out, 'appimage'))
  }
  if (formats.some(format => format === 'deb' || format === 'rpm')) {
    const fpm = await load('toolsets/fpm.js')
    toolsets.fpm = copyTool(path.dirname(await fpm.getFpmPath(config.toolsets?.fpm, resourcesDir)), path.join(out, 'fpm'))
  }
  return publishPackagingInputs({ source, out, target, formats, electron, toolsets, windows, dmgbuild })
}

if (isMain(import.meta.url)) {
  const { values } = parseArgs({ options: {
    source: { type: 'string' }, out: { type: 'string' }, cache: { type: 'string' }, target: { type: 'string' },
    format: { type: 'string', multiple: true }, dmgbuild: { type: 'string' },
  } })
  if (!values.source || !values.out || !values.cache) throw new Error('Usage: prepare-packaging-tools.mjs --source REPO --out WORK/packager --cache CACHE/packager [--target same-OS-target] [--format FORMAT] [--dmgbuild PM_BINARY]')
  console.log(await preparePackagingTools({ source: values.source, out: values.out, cache: values.cache, target: values.target, formats: values.format, dmgbuild: values.dmgbuild }))
}
