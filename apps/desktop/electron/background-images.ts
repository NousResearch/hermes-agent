import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

export const BACKGROUND_PROTOCOL = 'hermes-background'
export const BACKGROUND_MAX_FILE_BYTES = 100 * 1024 * 1024
export const BACKGROUND_MAX_FOLDER_IMAGES = 1_000
const BACKGROUND_TOKEN_LIMIT = 4_096

export const BACKGROUND_IMAGE_EXTENSIONS = new Set(['.avif', '.bmp', '.gif', '.jpeg', '.jpg', '.png', '.webp'])

export type BackgroundSourceKind = 'folder' | 'image'
export type BackgroundResolveError = 'empty' | 'invalid-source' | 'missing' | 'unreadable' | 'unsupported'

export interface BackgroundResolveRequest {
  kind: BackgroundSourceKind
  sourcePath: string
}

export interface BackgroundImageDescriptor {
  fingerprint: string
  id: string
  name: string
  url: string
}

export interface BackgroundResolveResult {
  error?: BackgroundResolveError
  images: BackgroundImageDescriptor[]
  sourcePath: string
  truncated: boolean
}

interface AuthorizedImage {
  filePath: string
}

interface BackgroundFile {
  filePath: string
  fingerprint: string
  id: string
  name: string
}

interface FsLike {
  promises: Pick<typeof fs.promises, 'lstat' | 'readdir' | 'realpath' | 'stat'>
}

const isInside = (root: string, candidate: string): boolean => {
  const relative = path.relative(root, candidate)

  return relative === '' || (!relative.startsWith(`..${path.sep}`) && relative !== '..' && !path.isAbsolute(relative))
}

const imageExtension = (filePath: string): string => path.extname(filePath).toLowerCase()

const stableImageId = (filePath: string): string =>
  crypto.createHash('sha256').update(filePath).digest('hex').slice(0, 24)

async function inspectImage(filePath: string, fsImpl: FsLike, requiredRoot?: string): Promise<BackgroundFile | null> {
  if (!BACKGROUND_IMAGE_EXTENSIONS.has(imageExtension(filePath))) {
    return null
  }

  let canonical: string
  let stat: Awaited<ReturnType<FsLike['promises']['stat']>>

  try {
    canonical = await fsImpl.promises.realpath(filePath)

    if (requiredRoot && !isInside(requiredRoot, canonical)) {
      return null
    }

    stat = await fsImpl.promises.stat(canonical)
  } catch {
    return null
  }

  if (!stat.isFile() || stat.size > BACKGROUND_MAX_FILE_BYTES) {
    return null
  }

  return {
    filePath: canonical,
    fingerprint: `${Math.trunc(stat.mtimeMs)}:${stat.size}`,
    id: stableImageId(canonical),
    name: path.basename(canonical)
  }
}

export class BackgroundImageRegistry {
  private readonly entries = new Map<string, AuthorizedImage>()

  authorize(filePath: string): string {
    const token = crypto.randomBytes(24).toString('base64url')
    this.entries.set(token, { filePath })

    while (this.entries.size > BACKGROUND_TOKEN_LIMIT) {
      const oldest = this.entries.keys().next().value

      if (!oldest) {
        break
      }

      this.entries.delete(oldest)
    }

    return `${BACKGROUND_PROTOCOL}://image/${token}`
  }

  resolve(token: string): string | null {
    const entry = this.entries.get(token)

    if (!entry) {
      return null
    }

    // Refresh insertion order so actively displayed images survive the LRU cap.
    this.entries.delete(token)
    this.entries.set(token, entry)

    return entry.filePath
  }

  get size(): number {
    return this.entries.size
  }
}

function errorResult(error: BackgroundResolveError, sourcePath = ''): BackgroundResolveResult {
  return { error, images: [], sourcePath, truncated: false }
}

export async function resolveBackgroundImages(
  request: BackgroundResolveRequest,
  registry: BackgroundImageRegistry,
  options: { fs?: FsLike } = {}
): Promise<BackgroundResolveResult> {
  const fsImpl = options.fs ?? fs
  const kind = request?.kind
  const rawPath = typeof request?.sourcePath === 'string' ? request.sourcePath.trim() : ''

  if ((kind !== 'folder' && kind !== 'image') || !rawPath) {
    return errorResult('invalid-source')
  }

  let sourcePath: string

  try {
    sourcePath = await fsImpl.promises.realpath(path.resolve(rawPath))
  } catch {
    return errorResult('missing', path.resolve(rawPath))
  }

  if (kind === 'image') {
    const image = await inspectImage(sourcePath, fsImpl)

    if (!image) {
      return errorResult(
        BACKGROUND_IMAGE_EXTENSIONS.has(imageExtension(sourcePath)) ? 'unreadable' : 'unsupported',
        sourcePath
      )
    }

    return {
      images: [{ ...image, url: registry.authorize(image.filePath) }],
      sourcePath,
      truncated: false
    }
  }

  try {
    const sourceStat = await fsImpl.promises.stat(sourcePath)

    if (!sourceStat.isDirectory()) {
      return errorResult('invalid-source', sourcePath)
    }
  } catch {
    return errorResult('unreadable', sourcePath)
  }

  let dirents: fs.Dirent[]

  try {
    dirents = (await fsImpl.promises.readdir(sourcePath, { withFileTypes: true })) as fs.Dirent[]
  } catch {
    return errorResult('unreadable', sourcePath)
  }

  const candidates = dirents
    .filter(entry => !entry.name.startsWith('.') && (entry.isFile() || entry.isSymbolicLink()))
    .map(entry => path.join(sourcePath, entry.name))
    .filter(candidate => BACKGROUND_IMAGE_EXTENSIONS.has(imageExtension(candidate)))
    .sort((left, right) => left.localeCompare(right))

  // Stop after one item beyond the public limit. Sequential stat calls avoid
  // creating an unbounded Promise fan-out for very large selected folders.
  const images: BackgroundFile[] = []

  for (const candidate of candidates) {
    const image = await inspectImage(candidate, fsImpl, sourcePath)

    if (image) {
      images.push(image)
    }

    if (images.length > BACKGROUND_MAX_FOLDER_IMAGES) {
      break
    }
  }

  if (images.length === 0) {
    return errorResult('empty', sourcePath)
  }

  const bounded = images.slice(0, BACKGROUND_MAX_FOLDER_IMAGES)

  return {
    images: bounded.map(image => ({ ...image, url: registry.authorize(image.filePath) })),
    sourcePath,
    truncated: images.length > bounded.length
  }
}

export function backgroundTokenFromUrl(rawUrl: string): string | null {
  try {
    const url = new URL(rawUrl)

    if (url.protocol !== `${BACKGROUND_PROTOCOL}:` || url.hostname !== 'image') {
      return null
    }

    return url.pathname.split('/').filter(Boolean)[0] ?? null
  } catch {
    return null
  }
}
