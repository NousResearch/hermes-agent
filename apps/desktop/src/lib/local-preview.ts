import { isDesktopFsRemoteMode, readDesktopFileText } from '@/lib/desktop-fs'
import { artifactExtension } from '@/lib/generated-artifacts'
import type { PreviewTarget } from '@/store/preview'

const HTML_EXTENSIONS = new Set(['.htm', '.html'])
const IMAGE_EXTENSIONS = new Set(['.bmp', '.gif', '.jpeg', '.jpg', '.png', '.svg', '.webp'])
const BINARY_EXTENSIONS = new Set([
  '.doc',
  '.docx',
  '.gz',
  '.mov',
  '.mp3',
  '.mp4',
  '.pdf',
  '.tar',
  '.wav',
  '.xls',
  '.xlsx',
  '.zip'
])

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

function joinPath(base: string, rel: string) {
  if (!base) {
    return rel
  }

  return `${base.replace(/\/+$/, '')}/${rel.replace(/^\.?\//, '')}`
}

function isAbsoluteFileTarget(value: string): boolean {
  return /^file:\/\//i.test(value) || value.startsWith('/') || /^[A-Za-z]:[\\/]/.test(value)
}

function pathToFileUrl(path: string) {
  const normalizedPath = path.replace(/\\/g, '/')
  const encoded = normalizedPath
    .split('/')
    .map(part => (part.endsWith(':') ? part : encodeURIComponent(part)))
    .join('/')

  return `file://${encoded.startsWith('/') ? encoded : `/${encoded}`}`
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
      path = decodeURIComponent(new URL(raw).pathname)
    } catch {
      path = raw.replace(/^file:\/\//i, '')
    }
  } else if (!isAbsoluteFileTarget(raw) && cwd) {
    path = joinPath(cwd, raw)
  }

  const ext = artifactExtension(path)
  const isHtml = HTML_EXTENSIONS.has(ext)
  const isImage = IMAGE_EXTENSIONS.has(ext)
  const isBinary = BINARY_EXTENSIONS.has(ext)

  return {
    binary: isBinary || undefined,
    kind: 'file',
    label: basename(path),
    language: LANGUAGE_BY_EXT[ext] || 'text',
    path,
    previewKind: isHtml ? 'html' : isImage ? 'image' : isBinary ? 'binary' : 'text',
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

    return {
      ...target,
      binary: result.binary,
      byteSize: result.byteSize,
      language: result.language || target.language,
      large: (result.byteSize ?? 0) > 512 * 1024,
      mimeType: result.mimeType
    }
  } catch {
    return target
  }
}

export async function normalizeOrLocalPreviewTarget(
  rawTarget: string,
  cwd?: string | null
): Promise<PreviewTarget | null> {
  try {
    const normalized = await window.hermesDesktop?.normalizePreviewTarget?.(rawTarget, cwd || undefined)

    if (normalized) {
      return enrichPreviewTarget(normalized)
    }
  } catch {
    // Running Electron may still have the old HTML-only preview IPC. Fall
    // through to renderer-side local classification so text/images still open.
  }

  return enrichPreviewTarget(localPreviewTarget(rawTarget, cwd))
}
