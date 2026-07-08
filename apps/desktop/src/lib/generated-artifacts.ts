const MARKDOWN_IMAGE_RE = /!\[[^\]]*]\(([^)\s]+)\)/g
const MARKDOWN_LINK_RE = /\[[^\]]+]\(([^)\s]+)\)/g
const URL_RE = /https?:\/\/[^\s<>"')]+/g
const POSIX_PATH_RE = /(^|[\s("'`])((?:\/|~\/|\.\.?\/)[^\s"'`<>]+(?:\.[a-z0-9]{1,10})?)/gi
const WINDOWS_PATH_RE = /(^|[\s("'`])([A-Za-z]:[\\/][^\s"'`<>]+(?:\.[a-z0-9]{1,10})?)/gi
const FILE_URL_RE = /file:\/\/[^\s<>"')]+/gi
const DATA_IMAGE_RE = /data:image\/[a-z0-9.+-]+;base64,[a-z0-9+/=]+/gi
const KEY_HINT_RE = /(path|file|url|image|artifact|output|download|result|target|preview|href|link)/i

export const GENERATED_ARTIFACT_EXTENSIONS = new Set([
  '.bmp',
  '.csv',
  '.doc',
  '.docx',
  '.gif',
  '.gz',
  '.htm',
  '.html',
  '.jpeg',
  '.jpg',
  '.json',
  '.md',
  '.mov',
  '.mp3',
  '.mp4',
  '.pdf',
  '.png',
  '.svg',
  '.tar',
  '.txt',
  '.wav',
  '.webp',
  '.xls',
  '.xlsx',
  '.zip'
])

function normalizeArtifactValue(value: string): string {
  return value.trim().replace(/[),.;]+$/, '')
}

export function artifactExtension(value: string): string {
  const clean = value.split(/[?#]/, 1)[0] || value
  const idx = clean.lastIndexOf('.')

  return idx >= 0 ? clean.slice(idx).toLowerCase() : ''
}

export function looksLikeLocalArtifactPath(value: string): boolean {
  return (
    /^file:\/\//i.test(value) ||
    /^(?:\/|~\/|\.{1,2}\/).+/i.test(value) ||
    /^[A-Za-z]:[\\/].+/.test(value)
  )
}

function looksLikePreviewUrl(value: string): boolean {
  return (
    /^https?:\/\/(?:localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\])/i.test(value) ||
    GENERATED_ARTIFACT_EXTENSIONS.has(artifactExtension(value))
  )
}

export function isGeneratedArtifactTarget(value: string): boolean {
  const normalized = normalizeArtifactValue(value)

  if (!normalized) {
    return false
  }

  if (/^data:image\//i.test(normalized)) {
    return true
  }

  if (/^https?:\/\//i.test(normalized)) {
    return looksLikePreviewUrl(normalized)
  }

  if (!looksLikeLocalArtifactPath(normalized)) {
    return false
  }

  return GENERATED_ARTIFACT_EXTENSIONS.has(artifactExtension(normalized))
}

function collectStringValues(
  value: unknown,
  keyPath: string,
  collector: (value: string, keyPath: string) => void
): void {
  if (typeof value === 'string') {
    collector(value, keyPath)

    return
  }

  if (Array.isArray(value)) {
    value.forEach((entry, index) => collectStringValues(entry, `${keyPath}.${index}`, collector))

    return
  }

  if (!value || typeof value !== 'object') {
    return
  }

  for (const [key, child] of Object.entries(value as Record<string, unknown>)) {
    collectStringValues(child, keyPath ? `${keyPath}.${key}` : key, collector)
  }
}

function parseMaybeJson(value: string): unknown {
  if (!value.trim()) {
    return null
  }

  try {
    return JSON.parse(value)
  } catch {
    return null
  }
}

function pushUnique(targets: string[], seen: Set<string>, value: string): void {
  const normalized = normalizeArtifactValue(value)

  if (!isGeneratedArtifactTarget(normalized) || seen.has(normalized)) {
    return
  }

  seen.add(normalized)
  targets.push(normalized)
}

export function extractGeneratedArtifactTargetsFromText(text: string): string[] {
  const targets: string[] = []
  const seen = new Set<string>()

  for (const match of text.matchAll(MARKDOWN_IMAGE_RE)) {
    pushUnique(targets, seen, match[1] || '')
  }

  for (const match of text.matchAll(MARKDOWN_LINK_RE)) {
    const start = match.index ?? 0

    if (start > 0 && text[start - 1] === '!') {
      continue
    }

    pushUnique(targets, seen, match[1] || '')
  }

  for (const match of text.matchAll(URL_RE)) {
    pushUnique(targets, seen, match[0] || '')
  }

  for (const match of text.matchAll(FILE_URL_RE)) {
    pushUnique(targets, seen, match[0] || '')
  }

  for (const match of text.matchAll(DATA_IMAGE_RE)) {
    pushUnique(targets, seen, match[0] || '')
  }

  for (const match of text.matchAll(POSIX_PATH_RE)) {
    pushUnique(targets, seen, match[2] || '')
  }

  for (const match of text.matchAll(WINDOWS_PATH_RE)) {
    pushUnique(targets, seen, match[2] || '')
  }

  return targets
}

export function extractGeneratedArtifactTargetsFromToolPayload(payload: unknown): string[] {
  const targets: string[] = []
  const seen = new Set<string>()

  collectStringValues(payload, '', (value, keyPath) => {
    const normalized = normalizeArtifactValue(value)

    if (!normalized) {
      return
    }

    if (KEY_HINT_RE.test(keyPath) || looksLikeLocalArtifactPath(normalized) || /^https?:\/\//i.test(normalized)) {
      pushUnique(targets, seen, normalized)
    }

    for (const target of extractGeneratedArtifactTargetsFromText(normalized)) {
      pushUnique(targets, seen, target)
    }

    const parsed = parseMaybeJson(normalized)

    if (parsed !== null) {
      for (const target of extractGeneratedArtifactTargetsFromToolPayload(parsed)) {
        pushUnique(targets, seen, target)
      }
    }
  })

  return targets
}
