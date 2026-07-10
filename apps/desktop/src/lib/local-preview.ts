import { isDesktopFsRemoteMode, readDesktopFileText } from '@/lib/desktop-fs'
import type { PreviewTarget } from '@/store/preview'

const HTML_EXTENSIONS = new Set(['.htm', '.html'])
const IMAGE_EXTENSIONS = new Set(['.bmp', '.gif', '.jpeg', '.jpg', '.png', '.svg', '.webp'])

const LANGUAGE_BY_EXT: Record<string, string> = {
  '.c': 'c',
  '.conf': 'ini',
  '.cpp': 'cpp',
  '.css': 'css',
  '.csv': 'csv',
  '.go': 'go',
  '.graphql': 'graphql',
  '.h': 'c',
  '.hpp': 'cpp',
  '.html': 'html',
  '.java': 'java',
  '.js': 'javascript',
  '.json': 'json',
  '.jsx': 'jsx',
  '.log': 'text',
  '.lua': 'lua',
  '.md': 'markdown',
  '.mjs': 'javascript',
  '.py': 'python',
  '.rb': 'ruby',
  '.rs': 'rust',
  '.sh': 'shell',
  '.sql': 'sql',
  '.svg': 'xml',
  '.toml': 'toml',
  '.ts': 'typescript',
  '.tsx': 'tsx',
  '.txt': 'text',
  '.xml': 'xml',
  '.yaml': 'yaml',
  '.yml': 'yaml',
  '.zsh': 'shell'
}

function basename(value: string) {
  return value.split(/[\\/]/).filter(Boolean).pop() || value
}

function extension(value: string) {
  const clean = value.split(/[?#]/, 1)[0] || value
  const idx = clean.lastIndexOf('.')

  return idx >= 0 ? clean.slice(idx).toLowerCase() : ''
}

function joinPath(base: string, rel: string) {
  if (!base) {
    return rel
  }

  return `${base.replace(/\/+$/, '')}/${rel.replace(/^\.?\//, '')}`
}

function encodePathSegments(path: string) {
  return path
    .split(/[\\/]/)
    .map(part => encodeURIComponent(part))
    .join('/')
}

function pathToFileUrl(path: string) {
  const drivePath = /^([a-z]:)[\\/](.*)$/i.exec(path)

  if (drivePath) {
    return `file:///${drivePath[1]}/${encodePathSegments(drivePath[2] || '')}`
  }

  const uncPath = /^\\\\([^\\/]+)[\\/](.*)$/.exec(path)

  if (uncPath) {
    return `file://${uncPath[1]}/${encodePathSegments(uncPath[2] || '')}`
  }

  const encoded = path
    .split('/')
    .map(part => encodeURIComponent(part))
    .join('/')

  return `file://${encoded.startsWith('/') ? encoded : `/${encoded}`}`
}

function fileUrlToPath(raw: string) {
  const url = new URL(raw)
  const pathname = decodeURIComponent(url.pathname)

  if (url.hostname && url.hostname.toLowerCase() !== 'localhost') {
    return `\\\\${url.hostname}${pathname.replaceAll('/', '\\')}`
  }

  return /^\/[a-z]:[\\/]/i.test(pathname) ? pathname.slice(1) : pathname
}

export function localPreviewTarget(rawTarget: string, cwd?: string | null): PreviewTarget | null {
  const raw = rawTarget.trim().replace(/^`|`$/g, '')

  if (!raw) {
    return null
  }

  if (/^https?:\/\//i.test(raw)) {
    return { kind: 'url', label: basename(raw), source: raw, url: raw }
  }

  let path = raw

  if (/^file:\/\//i.test(raw)) {
    try {
      path = fileUrlToPath(raw)
    } catch {
      path = raw.replace(/^file:\/\//i, '')
    }
  } else if (
    !raw.startsWith('/') &&
    !raw.startsWith('\\\\') &&
    !/^[a-z]:[\\/]/i.test(raw) &&
    !raw.startsWith('~/') &&
    !raw.startsWith('~\\') &&
    cwd
  ) {
    path = joinPath(cwd, raw)
  }

  const ext = extension(path)
  const isHtml = HTML_EXTENSIONS.has(ext)
  const isImage = IMAGE_EXTENSIONS.has(ext)

  return {
    kind: 'file',
    label: basename(path),
    language: LANGUAGE_BY_EXT[ext] || 'text',
    path,
    // Renderer fallback can't stat/sniff without reading; assume text unless
    // image/html extension says otherwise. LocalFilePreview still guards
    // binary/large files when readFileText/readFileDataUrl returns metadata.
    previewKind: isHtml ? 'html' : isImage ? 'image' : 'text',
    source: raw,
    url: pathToFileUrl(path)
  }
}

async function enrichPreviewTarget(target: PreviewTarget | null): Promise<PreviewTarget | null> {
  if (!isDesktopFsRemoteMode() || !target || target.kind !== 'file' || target.previewKind === 'image') {
    return target
  }

  try {
    const result = await readDesktopFileText(target.path || target.source)
    const resolvedPath = result.path || target.path

    return {
      ...target,
      binary: result.binary,
      byteSize: result.byteSize,
      language: result.language || target.language,
      large: (result.byteSize ?? 0) > 512 * 1024,
      mimeType: result.mimeType,
      path: resolvedPath,
      url: resolvedPath ? pathToFileUrl(resolvedPath) : target.url
    }
  } catch {
    return target
  }
}

export async function normalizeOrLocalPreviewTarget(
  rawTarget: string,
  cwd?: string | null
): Promise<PreviewTarget | null> {
  if (!isDesktopFsRemoteMode()) {
    try {
      const normalized = await window.hermesDesktop?.normalizePreviewTarget?.(rawTarget, cwd || undefined)

      if (normalized) {
        return enrichPreviewTarget(normalized)
      }
    } catch {
      // Running Electron may still have the old HTML-only preview IPC. Fall
      // through to renderer-side classification so text/images still open.
    }
  }

  return enrichPreviewTarget(localPreviewTarget(rawTarget, cwd))
}
