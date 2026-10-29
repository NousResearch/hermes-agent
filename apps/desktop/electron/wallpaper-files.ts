import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

export const WALLPAPER_PROTOCOL = 'hermes-wallpaper'
export const WALLPAPER_PROTOCOL_PRIVILEGES = {
  secure: true,
  standard: true
} as const
export const WALLPAPER_MAX_SOURCE_BYTES = 32 * 1024 * 1024
export const WALLPAPER_MIN_EDGE = 1920
export const WALLPAPER_MAX_EDGE = 3840
export const WALLPAPER_JPEG_QUALITY = 85

const PROFILE_NAME_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/
const WALLPAPER_ASSET_RE = /^[a-f0-9]{24}$/
const WALLPAPER_EXTENSIONS = new Set(['.jpeg', '.jpg', '.png', '.webp'])

export interface WallpaperFileAsset {
  filePath: string
  url: string
  version: string
}

export interface WallpaperDisplaySize {
  height: number
  scaleFactor: number
  width: number
}

export interface WallpaperSourceStat {
  dev: number
  ino: number
  mtimeMs: number
  size: number
}

interface WallpaperWriteOptions {
  rename?: typeof fs.promises.rename
}

export function normalizeWallpaperProfile(profile: unknown): string {
  const value = String(profile ?? '').trim() || 'default'

  if (!PROFILE_NAME_RE.test(value)) {
    throw new Error('Wallpaper profile is invalid.')
  }

  return value
}

export function wallpaperAssetId(profile: unknown): string {
  return crypto.createHash('sha256').update(normalizeWallpaperProfile(profile)).digest('hex').slice(0, 24)
}

export function wallpaperFilePathFromAsset(userDataDir: string, assetId: string): string {
  if (!WALLPAPER_ASSET_RE.test(assetId)) {
    throw new Error('Wallpaper asset is invalid.')
  }

  return path.join(userDataDir, 'wallpapers', `${assetId}.jpg`)
}

export function wallpaperFilePath(userDataDir: string, profile: unknown): string {
  return wallpaperFilePathFromAsset(userDataDir, wallpaperAssetId(profile))
}

export function wallpaperProtocolUrl(profile: unknown, version: string, accessToken = ''): string {
  const params = new URLSearchParams({ v: version })

  if (accessToken) {
    params.set('token', accessToken)
  }

  return `${WALLPAPER_PROTOCOL}://asset/${wallpaperAssetId(profile)}?${params}`
}

export function isSupportedWallpaperPath(filePath: string): boolean {
  return WALLPAPER_EXTENSIONS.has(path.extname(filePath).toLowerCase())
}

export function fitWallpaperDimensions(
  width: number,
  height: number,
  maxEdge = WALLPAPER_MAX_EDGE
): { height: number; width: number } {
  if (![width, height, maxEdge].every(value => Number.isFinite(value) && value > 0)) {
    throw new Error('Wallpaper dimensions are invalid.')
  }

  const sourceWidth = Math.round(width)
  const sourceHeight = Math.round(height)
  const longestEdge = Math.max(sourceWidth, sourceHeight)

  if (longestEdge <= maxEdge) {
    return { height: sourceHeight, width: sourceWidth }
  }

  const scale = maxEdge / longestEdge

  return {
    height: Math.max(1, Math.round(sourceHeight * scale)),
    width: Math.max(1, Math.round(sourceWidth * scale))
  }
}

export function preferredWallpaperMaxEdge(
  displays: WallpaperDisplaySize[],
  minEdge = WALLPAPER_MIN_EDGE,
  maxEdge = WALLPAPER_MAX_EDGE
): number {
  if (![minEdge, maxEdge].every(value => Number.isFinite(value) && value > 0) || minEdge > maxEdge) {
    throw new Error('Wallpaper edge limits are invalid.')
  }

  const displayEdge = displays.reduce((largest, display) => {
    if (![display.width, display.height, display.scaleFactor].every(value => Number.isFinite(value) && value > 0)) {
      return largest
    }

    return Math.max(largest, Math.ceil(Math.max(display.width, display.height) * display.scaleFactor))
  }, 0)

  return Math.min(maxEdge, Math.max(minEdge, displayEdge))
}

/**
 * A profile name can be deleted and later reused. Since wallpaper filenames
 * are deterministic, an asset older than the new profile directory belongs to
 * the previous profile lifetime and must not be adopted by the replacement.
 */
export function wallpaperAssetPredatesProfile(
  assetModifiedAtMs: number,
  profileCreatedAtMs: number,
  timestampToleranceMs = 1_000
): boolean {
  if (![assetModifiedAtMs, profileCreatedAtMs, timestampToleranceMs].every(Number.isFinite)) {
    return false
  }

  if (assetModifiedAtMs <= 0 || profileCreatedAtMs <= 0 || timestampToleranceMs < 0) {
    return false
  }

  return assetModifiedAtMs + timestampToleranceMs < profileCreatedAtMs
}

export async function readWallpaperFileAsset(
  userDataDir: string,
  profile: unknown,
  accessToken = ''
): Promise<WallpaperFileAsset | null> {
  const filePath = wallpaperFilePath(userDataDir, profile)

  try {
    // Imported assets must stay regular app-owned files. Using lstat here
    // prevents a local symlink/reparse-point replacement from turning the
    // narrow wallpaper protocol into an arbitrary file reader.
    const stat = await fs.promises.lstat(filePath)

    if (!stat.isFile()) {
      return null
    }

    const version = `${stat.mtimeMs}-${stat.size}`

    return {
      filePath,
      url: wallpaperProtocolUrl(profile, version, accessToken),
      version
    }
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      return null
    }

    throw error
  }
}

function isSameFileSnapshot(expected: WallpaperSourceStat, actual: WallpaperSourceStat): boolean {
  const inodeMatches =
    expected.ino === 0 || actual.ino === 0 || (expected.dev === actual.dev && expected.ino === actual.ino)

  return inodeMatches && expected.mtimeMs === actual.mtimeMs && expected.size === actual.size
}

/**
 * Read a previously validated canonical path through a stable file handle.
 * The second stat closes the validation-to-read gap: replacing the canonical
 * target after validation is rejected, and reads stay capped even if a file is
 * modified while the handle is open.
 */
export async function readWallpaperSourceFile(
  realPath: string,
  expectedStat: WallpaperSourceStat,
  maxBytes = WALLPAPER_MAX_SOURCE_BYTES
): Promise<Buffer> {
  const noFollow = typeof fs.constants.O_NOFOLLOW === 'number' ? fs.constants.O_NOFOLLOW : 0
  const handle = await fs.promises.open(realPath, fs.constants.O_RDONLY | noFollow)

  try {
    const openedStat = await handle.stat()

    if (!openedStat.isFile() || !isSameFileSnapshot(expectedStat, openedStat)) {
      throw new Error('Wallpaper source changed during validation.')
    }

    if (openedStat.size > maxBytes) {
      throw new Error(`Wallpaper source is too large (${openedStat.size} bytes; limit ${maxBytes} bytes).`)
    }

    const data = Buffer.alloc(openedStat.size)
    let offset = 0

    while (offset < data.length) {
      const { bytesRead } = await handle.read(data, offset, data.length - offset, offset)

      if (bytesRead === 0) {
        break
      }

      offset += bytesRead
    }

    const finalStat = await handle.stat()

    if (offset !== data.length || !isSameFileSnapshot(openedStat, finalStat)) {
      throw new Error('Wallpaper source changed while it was being read.')
    }

    return data
  } finally {
    await handle.close()
  }
}

export async function writeWallpaperFileAtomically(
  filePath: string,
  data: Uint8Array,
  options: WallpaperWriteOptions = {}
): Promise<void> {
  const tempPath = path.join(
    path.dirname(filePath),
    `.${path.basename(filePath)}.${process.pid}.${crypto.randomUUID()}.tmp`
  )

  let handle: fs.promises.FileHandle | null = null

  try {
    handle = await fs.promises.open(tempPath, 'wx', 0o600)
    await handle.writeFile(data)
    await handle.sync()
    await handle.close()
    handle = null
    await (options.rename ?? fs.promises.rename)(tempPath, filePath)
  } catch (error) {
    await handle?.close().catch(() => undefined)
    await fs.promises.unlink(tempPath).catch(() => undefined)

    throw error
  }
}

export async function writeWallpaperFile(
  userDataDir: string,
  profile: unknown,
  data: Uint8Array,
  options: WallpaperWriteOptions = {}
): Promise<WallpaperFileAsset> {
  const filePath = wallpaperFilePath(userDataDir, profile)

  await fs.promises.mkdir(path.dirname(filePath), { recursive: true })
  await writeWallpaperFileAtomically(filePath, data, options)

  const asset = await readWallpaperFileAsset(userDataDir, profile)

  if (!asset) {
    throw new Error('Wallpaper could not be saved.')
  }

  return asset
}

export async function removeWallpaperFile(userDataDir: string, profile: unknown): Promise<boolean> {
  try {
    await fs.promises.unlink(wallpaperFilePath(userDataDir, profile))

    return true
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      return false
    }

    throw error
  }
}
