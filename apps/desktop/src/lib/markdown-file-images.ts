const IMAGE_EXTENSIONS = new Set(['bmp', 'gif', 'jpeg', 'jpg', 'png', 'svg', 'webp'])
const DEFAULT_MAX_IMAGES = 32
const MAX_MAX_IMAGES = 64
const TRANSPARENT_IMAGE_SRC = 'data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs='

export const MARKDOWN_FILE_IMAGE_MAX_BYTES = 2 * 1024 * 1024
export const MARKDOWN_FILE_IMAGE_MAX_CONCURRENT = 2
export const MARKDOWN_FILE_IMAGE_MAX_RETAINED_DATA_URL_BYTES = 8 * 1024 * 1024

type MarkdownFileImageRead = (path: string, sourceFile: string, maxBytes: number) => Promise<string>

interface MarkdownFileImageLoaderOwner {
  disposed: boolean
}

interface MarkdownFileImageScheduledTask {
  owner: MarkdownFileImageLoaderOwner
  reject: (error: Error) => void
  run: () => Promise<void>
}

export interface MarkdownFileImageScheduler {
  cancelQueued: (owner: MarkdownFileImageLoaderOwner) => void
  schedule: (task: MarkdownFileImageScheduledTask) => void
}

export interface MarkdownFileImageLoader {
  dispose: () => void
  load: (path: string, sourceFile: string) => Promise<string>
  retainedDataUrlBytes: () => number
}

export interface MarkdownFileImageLoaderOptions {
  maxConcurrent?: number
  maxImageBytes?: number
  maxRetainedDataUrlBytes?: number
}

function disposedLoaderError() {
  return new Error('Markdown file image loader was disposed')
}

/** Keep the active-read ceiling stable while document-scope loaders rotate. */
export function createMarkdownFileImageScheduler(
  maxConcurrent = MARKDOWN_FILE_IMAGE_MAX_CONCURRENT
): MarkdownFileImageScheduler {
  const concurrency = Math.max(1, Math.floor(maxConcurrent))
  const queue: MarkdownFileImageScheduledTask[] = []
  let active = 0

  const drain = () => {
    while (active < concurrency && queue.length) {
      const task = queue.shift()!

      if (task.owner.disposed) {
        task.reject(disposedLoaderError())

        continue
      }

      active += 1
      void task
        .run()
        .catch(error => task.reject(error instanceof Error ? error : new Error(String(error))))
        .finally(() => {
          active -= 1
          drain()
        })
    }
  }

  return {
    cancelQueued(owner) {
      for (let index = queue.length - 1; index >= 0; index -= 1) {
        const task = queue[index]

        if (task.owner === owner) {
          queue.splice(index, 1)
          task.reject(disposedLoaderError())
        }
      }
    },
    schedule(task) {
      queue.push(task)
      drain()
    }
  }
}

/**
 * Bounds filesystem reads and retained data-URL bytes for one Markdown
 * document. This does not attempt to bound browser image decode memory.
 */
export function createMarkdownFileImageLoader(
  read: MarkdownFileImageRead,
  options: MarkdownFileImageLoaderOptions = {},
  scheduler = createMarkdownFileImageScheduler(options.maxConcurrent)
): MarkdownFileImageLoader {
  const maxImageBytes = Math.max(1, Math.floor(options.maxImageBytes ?? MARKDOWN_FILE_IMAGE_MAX_BYTES))

  const maxRetainedDataUrlBytes = Math.max(
    1,
    Math.floor(options.maxRetainedDataUrlBytes ?? MARKDOWN_FILE_IMAGE_MAX_RETAINED_DATA_URL_BYTES)
  )

  const owner: MarkdownFileImageLoaderOwner = { disposed: false }
  let retainedBytes = 0

  return {
    dispose() {
      if (owner.disposed) {
        return
      }

      owner.disposed = true
      scheduler.cancelQueued(owner)
    },
    load(path, sourceFile) {
      if (owner.disposed) {
        return Promise.reject(disposedLoaderError())
      }

      return new Promise<string>((resolve, reject) => {
        scheduler.schedule({
          owner,
          reject,
          async run() {
            try {
              const dataUrl = await read(path, sourceFile, maxImageBytes)

              if (owner.disposed) {
                throw disposedLoaderError()
              }

              const nextRetainedBytes = retainedBytes + dataUrl.length

              if (nextRetainedBytes > maxRetainedDataUrlBytes) {
                throw new Error('Markdown document image budget exceeded')
              }

              retainedBytes = nextRetainedBytes
              resolve(dataUrl)
            } catch (error) {
              reject(owner.disposed ? disposedLoaderError() : error instanceof Error ? error : new Error(String(error)))
            }
          }
        })
      })
    },
    retainedDataUrlBytes: () => retainedBytes
  }
}

export const FILE_PREVIEW_IMAGE_PATH_ATTR = 'data-hermes-file-image-path'
export const FILE_PREVIEW_IMAGE_SOURCE_ATTR = 'data-hermes-file-image-source'

interface MarkdownHastNode {
  children?: MarkdownHastNode[]
  properties?: Record<string, unknown>
  tagName?: string
  type?: string
}

export interface FilePreviewImagePluginOptions {
  markdownPath: string
  maxImages?: number
}

function decodedPathPart(source: string): string | null {
  const rawPath = source.split(/[?#]/, 1)[0] || ''

  if (!rawPath) {
    return null
  }

  try {
    const decoded = decodeURIComponent(rawPath)

    return decoded.includes('\0') ? null : decoded
  } catch {
    return null
  }
}

function isWindowsPath(path: string): boolean {
  return /^[a-z]:[\\/]/i.test(path) || path.startsWith('\\\\')
}

function isLocalImageSource(source: string): boolean {
  return (
    /^(?:file:|\/|~[\\/]|[a-z]:[\\/]|\\\\)/i.test(source) ||
    (!/^[a-z][a-z0-9+.-]*:/i.test(source) && !source.startsWith('//'))
  )
}

/**
 * Resolve a Markdown image only when it is a descendant of the Markdown
 * document's directory. Absolute, home-relative and parent-traversing sources
 * remain in Streamdown's ordinary hardened path.
 */
export function resolveMarkdownFileImagePath(source: string, markdownPath: string): string | null {
  const value = source.trim()

  if (!value || /^(?:[a-z][a-z0-9+.-]*:|\/|~[\\/]|\\\\)/i.test(value) || /^[a-z]:[\\/]/i.test(value)) {
    return null
  }

  const decoded = decodedPathPart(value)

  if (!decoded) {
    return null
  }

  const parts = decoded
    .replace(/\\/g, '/')
    .split('/')
    .filter(part => part && part !== '.')

  if (!parts.length || parts.some(part => part === '..')) {
    return null
  }

  const extension = parts.at(-1)?.split('.').pop()?.toLowerCase()

  if (!extension || !IMAGE_EXTENSIONS.has(extension)) {
    return null
  }

  const windows = isWindowsPath(markdownPath)
  const normalizedMarkdownPath = windows ? markdownPath.replace(/\//g, '\\') : markdownPath
  const separator = windows ? '\\' : '/'
  const directoryEnd = normalizedMarkdownPath.lastIndexOf(separator)

  if (directoryEnd < 0) {
    return null
  }

  const directory = normalizedMarkdownPath.slice(0, directoryEnd)

  return `${directory}${separator}${parts.join(separator)}`
}

/** Runs after sanitize and immediately before harden in the file-preview pipeline. */
export function filePreviewImageRehypePlugin({
  markdownPath,
  maxImages = DEFAULT_MAX_IMAGES
}: FilePreviewImagePluginOptions) {
  const limit = Math.min(MAX_MAX_IMAGES, Math.max(0, Math.floor(maxImages)))

  return (tree: MarkdownHastNode) => {
    let resolvedImages = 0

    const visit = (node: MarkdownHastNode) => {
      if (node.type === 'element' && node.tagName === 'img' && typeof node.properties?.src === 'string') {
        const source = node.properties.src
        const filePath = resolvedImages < limit ? resolveMarkdownFileImagePath(source, markdownPath) : null

        if (filePath) {
          node.properties[FILE_PREVIEW_IMAGE_PATH_ATTR] = filePath
          node.properties[FILE_PREVIEW_IMAGE_SOURCE_ATTR] = markdownPath
          node.properties.src = TRANSPARENT_IMAGE_SRC
          resolvedImages += 1
        } else if (isLocalImageSource(source)) {
          // Do not let rejected/overflow local paths fall through as browser
          // file-relative requests against the packaged renderer's index.html.
          node.properties.src = ''
        }
      }

      node.children?.forEach(visit)
    }

    visit(tree)
  }
}
