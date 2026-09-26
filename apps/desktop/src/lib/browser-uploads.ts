import type { HermesApiRequest, HermesSelectPathsOptions, HermesStagedUpload } from '@/global'
import { bytesToBase64 } from '@/lib/base64'
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

function bytesToDataUrl(bytes: Uint8Array, mimeType: string): string {
  return `data:${mimeType};base64,${bytesToBase64(bytes)}`
}

function sandboxedHtmlBlob(bytes: Uint8Array): Blob {
  const source = bytesToDataUrl(bytes, 'text/html;charset=utf-8')

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

async function stageBrowserFile(
  bootstrap: BrowserBootstrap,
  file: File,
  profile?: null | string
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

  // Keep the existing string-path bridge for pickers and drops. Composer chips
  // carry this small source descriptor so draft cloning and retries retain it.
  if (payload.staged_upload?.path === payload.path) {
    bootstrap.stagedUploads.set(payload.path, payload.staged_upload)

    // Chips retain their own descriptors. Bound the string-path compatibility
    // cache without consuming lookups shared by multiple attachment occurrences.
    if (bootstrap.stagedUploads.size > STAGED_UPLOAD_CACHE_LIMIT) {
      const oldest = bootstrap.stagedUploads.keys().next().value

      if (oldest !== undefined) { bootstrap.stagedUploads.delete(oldest) }
    }
  }

  return payload.path
}

function selectBrowserFiles(
  bootstrap: BrowserBootstrap,
  options?: HermesSelectPathsOptions,
  fallbackProfile?: null | string
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

        void Promise.all(files.map(file => stageBrowserFile(bootstrap, file, profile))).then(resolve, reject)
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
}

/** Browser bytes become server-side files the agent can read: images, drops, pastes, picks. */
export function createBrowserUploadsBridge({
  api,
  bootstrap,
  currentProfile,
  objectUrls
}: BrowserUploadsOptions): Pick<
  Window['hermesDesktop'],
  'getStagedFileForAttach' | 'saveImageBuffer' | 'savePastedText' | 'selectPaths' | 'stageFileForAttach'
> {
  const saveBuffer = async (data: ArrayBuffer | Uint8Array, ext: string) => {
    const source = data instanceof Uint8Array ? data : new Uint8Array(data)
    const bytes = new Uint8Array(source.byteLength)
    bytes.set(source)

    const extension = normalizedExtension(ext)
    const imageMime = IMAGE_MIME_BY_EXTENSION[extension]

    if (imageMime) {
      const uploaded = await api<{ path?: string }>({
        body: {
          data_url: bytesToDataUrl(bytes, imageMime),
          filename: `desktop-upload${extension}`
        },
        method: 'POST',
        path: '/api/chat/image-upload',
        profile: currentProfile()
      })

      return uploaded.path || ''
    }

    const blob = extension === '.htm' || extension === '.html'
      ? sandboxedHtmlBlob(bytes)
      : new Blob([bytes.buffer], { type: 'application/octet-stream' })

    const url = URL.createObjectURL(blob)
    objectUrls.add(url)

    return url
  }

  return {
    getStagedFileForAttach: (path: string) => bootstrap.stagedUploads.get(path),
    saveImageBuffer: saveBuffer,
    savePastedText: (text: string) => stageBrowserFile(
      bootstrap, new File([text], 'pasted.txt', { type: 'text/plain' }), currentProfile()
    ),
    selectPaths: options => selectBrowserFiles(bootstrap, options, currentProfile()),
    stageFileForAttach: (file: File) => stageBrowserFile(bootstrap, file, currentProfile())
  }
}
