import type { HermesApiRequest, HermesSelectPathsOptions, HermesStagedUpload } from '@/global'
import { blobToDataUrl, bytesToBase64 } from '@/lib/base64'
import { type BrowserBootstrap, browserFetch } from '@/lib/browser-transport'

const STAGED_UPLOAD_CACHE_LIMIT = 256

const IMAGE_MIME_BY_EXTENSION: Record<string, string> = {
  '.bmp': 'image/bmp',
  '.gif': 'image/gif',
  '.jpeg': 'image/jpeg',
  '.jpg': 'image/jpeg',
  '.png': 'image/png',
  '.tif': 'image/tiff',
  '.tiff': 'image/tiff',
  '.webp': 'image/webp'
}

export const IMAGE_EXTENSION_BY_MIME: Record<string, string> = Object.fromEntries(
  Object.entries(IMAGE_MIME_BY_EXTENSION).map(([extension, mime]) => [mime, extension])
)

function normalizedExtension(value: string): string {
  const clean = String(value || '').trim().toLowerCase()

  if (!clean) {return ''}

  return clean.startsWith('.') ? clean : `.${clean}`
}

// Chips retain their own descriptors. Bound the string-path compatibility
// caches without consuming lookups shared by multiple attachment occurrences.
function rememberBounded<K, V>(cache: Map<K, V>, key: K, value: V): void {
  cache.set(key, value)

  if (cache.size > STAGED_UPLOAD_CACHE_LIMIT) {
    const oldest = cache.keys().next().value

    if (oldest !== undefined) {
      cache.delete(oldest)
    }
  }
}

function sandboxedHtmlBlob(bytes: Uint8Array): Blob {
  const source = `data:text/html;charset=utf-8;base64,${bytesToBase64(bytes)}`

  const wrapper = [
    '<!doctype html>',
    '<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
    '<style>html,body,iframe{box-sizing:border-box;width:100%;height:100%;margin:0;border:0}</style>',
    '</head><body>',
    `<iframe sandbox="allow-scripts" referrerpolicy="no-referrer" src="${source}"></iframe>`,
    '</body></html>'
  ].join('')

  return new Blob([wrapper], { type: 'text/html;charset=utf-8' })
}

interface StageResponse {
  detail?: unknown
  path?: string
  staged_upload?: HermesStagedUpload
}

/** A proxy's own error page (nginx's 413, a 502/504) is HTML, not the server's JSON. */
function stageResponse(text: string): StageResponse {
  try {
    const value: unknown = JSON.parse(text)

    return value && typeof value === 'object' ? (value as StageResponse) : {}
  } catch {
    return {}
  }
}

function uploadFailure(status: number, detail: unknown): string {
  if (typeof detail === 'string' && detail) {return detail}

  return status === 413
    ? 'File upload failed (413): the file is larger than the server or a proxy in front of it accepts'
    : `File upload failed (${status})`
}

type OnStaged = (path: string, name: string, staged?: HermesStagedUpload) => void

async function stageBrowserFile(
  bootstrap: BrowserBootstrap,
  file: File,
  profile: null | string,
  onStaged: OnStaged
): Promise<string> {
  const form = new FormData()
  form.append('file', file, file.name || 'attachment')

  const { response, text } = await browserFetch(bootstrap, {
    body: form,
    method: 'POST',
    path: '/api/chat/file-upload',
    profile
  })

  const payload = stageResponse(text)

  if (!response.ok || !payload.path) {
    throw new Error(uploadFailure(response.status, payload.detail))
  }

  onStaged(payload.path, file.name, payload.staged_upload)

  return payload.path
}

function selectBrowserFiles(
  bootstrap: BrowserBootstrap,
  options: HermesSelectPathsOptions | undefined,
  fallbackProfile: null | string,
  onStaged: OnStaged
): Promise<string[]> {
  if (options?.directories) {return Promise.resolve([])}

  return new Promise<string[]>((resolve, reject) => {
    const input = document.createElement('input')
    input.type = 'file'
    input.multiple = Boolean(options?.multiple)
    input.style.display = 'none'

    const extensions = (options?.filters || []).flatMap(filter => filter.extensions || [])

    if (extensions.length) {
      input.accept = extensions.map(extension => `.${extension.replace(/^\./, '')}`).join(',')
    }

    const finish = () => input.remove()
    input.addEventListener(
      'cancel',
      () => {
        finish()
        resolve([])
      },
      { once: true }
    )
    input.addEventListener(
      'change',
      () => {
        const files = Array.from(input.files || [])
        finish()

        const profile = options?.profile?.trim() || fallbackProfile || null

        void Promise.all(files.map(file => stageBrowserFile(bootstrap, file, profile, onStaged))).then(resolve, reject)
      },
      { once: true }
    )
    document.body.append(input)
    input.click()
  })
}

interface BrowserUploadsOptions {
  api: <T>(request: HermesApiRequest) => Promise<T>
  bootstrap: BrowserBootstrap
  currentProfile: () => string | null
  /** Blob URLs handed to the page; the installer revokes them on unload. */
  objectUrls: Set<string>
  /** Only sandbox-wrapped HTML is eligible for the preview opener. */
  previewUrls: Set<string>
}

/** Browser bytes become server-side files the agent can read: images, drops, pastes, picks. */
export function createBrowserUploadsBridge({
  api,
  bootstrap,
  currentProfile,
  objectUrls,
  previewUrls
}: BrowserUploadsOptions): Pick<
  Window['hermesDesktop'],
  'getStagedFileDisplayName' | 'getStagedFileForAttach' | 'saveImageBuffer' | 'savePastedText' | 'selectPaths' | 'stageFileForAttach'
> {
  const displayNames = new Map<string, string>()
  const stagedUploads = new Map<string, HermesStagedUpload>()

  const rememberStaged: OnStaged = (path, name, staged) => {
    // Keep the existing string-path bridge for pickers and drops. Composer chips
    // carry this small source descriptor so draft cloning and retries retain it.
    if (staged?.path === path) {
      rememberBounded(stagedUploads, path, staged)
    }

    rememberBounded(displayNames, path, name)
  }

  const saveBuffer = async (data: ArrayBuffer | Uint8Array, ext: string) => {
    const source = data instanceof Uint8Array ? data : new Uint8Array(data)
    const bytes = new Uint8Array(source.byteLength)
    bytes.set(source)

    const extension = normalizedExtension(ext)
    const imageMime = IMAGE_MIME_BY_EXTENSION[extension]

    if (imageMime) {
      // Route by the profile active at the paste, not after the encode.
      const profile = currentProfile()
      const dataUrl = await blobToDataUrl(new Blob([bytes], { type: imageMime }))

      const uploaded = await api<{ path?: string }>({
        body: {
          data_url: dataUrl,
          filename: `desktop-upload${extension}`
        },
        method: 'POST',
        path: '/api/chat/image-upload',
        profile
      })

      return uploaded.path || ''
    }

    const isHtml = extension === '.htm' || extension === '.html'

    const blob = isHtml
      ? sandboxedHtmlBlob(bytes)
      : new Blob([bytes.buffer], { type: 'application/octet-stream' })

    const url = URL.createObjectURL(blob)
    objectUrls.add(url)

    if (isHtml) { previewUrls.add(url) }

    return url
  }

  return {
    getStagedFileDisplayName: (path: string) => displayNames.get(path),
    getStagedFileForAttach: (path: string) => stagedUploads.get(path),
    saveImageBuffer: saveBuffer,
    savePastedText: (text: string) => stageBrowserFile(
      bootstrap, new File([text], 'pasted.txt', { type: 'text/plain' }), currentProfile(), rememberStaged
    ),
    selectPaths: options => selectBrowserFiles(bootstrap, options, currentProfile(), rememberStaged),
    stageFileForAttach: (file: File) => stageBrowserFile(bootstrap, file, currentProfile(), rememberStaged)
  }
}
