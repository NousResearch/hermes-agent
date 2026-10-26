import type { HermesApiRequest, HermesSelectPathsOptions, HermesStagedUpload } from '@/global'
import { blobToDataUrl } from '@/lib/base64'
import { type BrowserBootstrap, browserFetch } from '@/lib/browser-transport'

const STAGED_UPLOAD_CACHE_LIMIT = 256

/** The largest file the browser stages on the host, and the largest image it can send: sending reads
 *  a staged image back through /api/fs/read-data-url. Mirrors WEBAPP_ATTACHMENT_MAX_BYTES
 *  (hermes_constants.py), the cap on both. */
export const BROWSER_UPLOAD_MAX_BYTES = 16 * 1024 * 1024

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
  const clean = String(value || '')
    .trim()
    .toLowerCase()

  if (!clean) {
    return ''
  }

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

// The page reaches its opaque-origin frame through srcdoc, never a URL: Chromium refuses to
// navigate a frame to a URL over 2 MiB, so a base64 data URL left any page above ~1.5 MB blank.
// Decoded as the data URL's charset=utf-8 did, and escaped for a double-quoted attribute (`&`
// first) so no byte of the page parses as markup of this same-origin wrapper.
// A srcdoc document inherits the wrapper's blob: base URL, so `#anchor` links would resolve to
// blob:…#anchor and do nothing. A page without its own <base> gets about:srcdoc, placed inside
// the page's own <head> start tag (else its <html> tag, else after any doctype): a base before
// <head> opens an implicit head, and the parser then drops the authored tag and its attributes.
// After the doctype, the page keeps its rendering mode.
const SRCDOC_BASE = '<base href="about:srcdoc">'
const LEADING_DOCTYPE = /^\s*<!doctype[^>]*>/i
const BASE_ELEMENT = /<base[\s/>]/i
const START_TAG_REST = String.raw`(?=[\s/>])(?:"[^"]*"|'[^']*'|[^'">])*>`
const HEAD_START = new RegExp(`<head${START_TAG_REST}`, 'i')
const HTML_START = new RegExp(`<html${START_TAG_REST}`, 'i')

function withSrcdocBase(page: string): string {
  if (BASE_ELEMENT.test(page)) {
    return page
  }

  const anchor = HEAD_START.exec(page) ?? HTML_START.exec(page) ?? LEADING_DOCTYPE.exec(page)
  const at = anchor ? anchor.index + anchor[0].length : 0

  return page.slice(0, at) + SRCDOC_BASE + page.slice(at)
}

function sandboxedHtmlBlob(bytes: Uint8Array): Blob {
  const srcdoc = withSrcdocBase(new TextDecoder('utf-8').decode(bytes))
    .replace(/&/g, '&amp;')
    .replace(/"/g, '&quot;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')

  const wrapper = [
    '<!doctype html>',
    '<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
    '<style>html,body,iframe{box-sizing:border-box;width:100%;height:100%;margin:0;border:0}</style>',
    '</head><body>',
    `<iframe sandbox="allow-scripts" referrerpolicy="no-referrer" srcdoc="${srcdoc}"></iframe>`,
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
  if (typeof detail === 'string' && detail) {
    return detail
  }

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
  // The server refuses a larger file anyway, with a 413 that doesn't name the limit: say it here.
  // The reason leaves out the name, which a drop warning adds: a long name would push the reason
  // past readableError's length cap and lose the limit.
  if (file.size > BROWSER_UPLOAD_MAX_BYTES) {
    throw new Error(
      `The file is larger than ${BROWSER_UPLOAD_MAX_BYTES / (1024 * 1024)} MiB, the largest file the browser can attach.`
    )
  }

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
  if (options?.directories) {
    return Promise.resolve([])
  }

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
  | 'getStagedFileDisplayName'
  | 'getStagedFileForAttach'
  | 'saveImageBuffer'
  | 'savePastedText'
  | 'selectPaths'
  | 'stageFileForAttach'
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

    // /api/chat/image-upload stores images up to 25 MiB, but the send reads the staged image back
    // with the 16 MiB file-read cap: refuse a larger one here, before it shows as attached.
    if (imageMime && bytes.byteLength > BROWSER_UPLOAD_MAX_BYTES) {
      throw new Error(
        `The image is larger than ${BROWSER_UPLOAD_MAX_BYTES / (1024 * 1024)} MiB, the largest image the browser can send.`
      )
    }

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

    const blob = isHtml ? sandboxedHtmlBlob(bytes) : new Blob([bytes.buffer], { type: 'application/octet-stream' })

    const url = URL.createObjectURL(blob)
    objectUrls.add(url)

    if (isHtml) {
      previewUrls.add(url)
    }

    return url
  }

  return {
    getStagedFileDisplayName: (path: string) => displayNames.get(path),
    getStagedFileForAttach: (path: string) => stagedUploads.get(path),
    saveImageBuffer: saveBuffer,
    savePastedText: (text: string) =>
      stageBrowserFile(
        bootstrap,
        new File([text], 'pasted.txt', { type: 'text/plain' }),
        currentProfile(),
        rememberStaged
      ),
    selectPaths: options => selectBrowserFiles(bootstrap, options, currentProfile(), rememberStaged),
    stageFileForAttach: (file: File) => stageBrowserFile(bootstrap, file, currentProfile(), rememberStaged)
  }
}
