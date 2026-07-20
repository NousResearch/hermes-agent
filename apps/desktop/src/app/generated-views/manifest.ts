import { SANDBOXED_HTML_MAX_BYTES } from '@/app/chat/right-rail/sandboxed-html-approval'

export const GENERATED_VIEW_MANIFEST_MAX_BYTES = 32 * 1024
export const GENERATED_VIEW_HTML_MAX_BYTES = SANDBOXED_HTML_MAX_BYTES
export const GENERATED_VIEW_ID_PATTERN = /^[a-z0-9][a-z0-9_-]{0,63}$/

export const GENERATED_VIEW_CAPABILITIES = ['theme:read', 'state:persist'] as const
export const GENERATED_VIEW_BINDINGS = ['hermes:status', 'hermes:usage-30d'] as const

export type GeneratedViewCapability = (typeof GENERATED_VIEW_CAPABILITIES)[number]
export type GeneratedViewBinding = (typeof GENERATED_VIEW_BINDINGS)[number]

export interface GeneratedViewManifest {
  version: 1
  id: string
  title: string
  entry: string
  capabilities: GeneratedViewCapability[]
  bindings: GeneratedViewBinding[]
}

const MANIFEST_KEYS = new Set(['version', 'id', 'title', 'entry', 'capabilities', 'bindings'])

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value))
}

function requireString(record: Record<string, unknown>, key: string): string {
  const value = record[key]

  if (typeof value !== 'string') {
    throw new Error(`view.json ${key} must be a string`)
  }

  return value
}

function validateStringList<T extends string>(
  value: unknown,
  key: string,
  allowed: readonly T[],
  optional = false
): T[] {
  if (value === undefined && optional) {
    return []
  }

  if (!Array.isArray(value)) {
    throw new Error(`view.json ${key} must be an array`)
  }

  const seen = new Set<T>()

  for (const item of value) {
    if (typeof item !== 'string' || !allowed.includes(item as T)) {
      throw new Error(`view.json ${key} contains unsupported value: ${String(item)}`)
    }

    seen.add(item as T)
  }

  return [...seen]
}

/** A manifest entry is a relative HTML path using portable `/` separators. */
export function validateGeneratedViewEntry(entry: string): string {
  if (!entry || entry.includes('\\') || entry.startsWith('/') || /^[A-Za-z]:/.test(entry)) {
    throw new Error('view.json entry must be a relative path')
  }

  const segments = entry.split('/')

  if (
    segments.some(
      segment =>
        !segment || segment === '.' || segment === '..' || [...segment].some(char => char.charCodeAt(0) <= 0x1f)
    )
  ) {
    throw new Error('view.json entry escapes its generated-view directory')
  }

  if (!entry.toLowerCase().endsWith('.html')) {
    throw new Error('view.json entry must point to an HTML file')
  }

  return segments.join('/')
}

export function validateGeneratedViewManifest(value: unknown, expectedId?: string): GeneratedViewManifest {
  if (!isRecord(value)) {
    throw new Error('view.json must be an object')
  }

  for (const key of Object.keys(value)) {
    if (!MANIFEST_KEYS.has(key)) {
      throw new Error(`view.json ${key} is not allowed`)
    }
  }

  if (value.version !== 1) {
    throw new Error('view.json version must be 1')
  }

  const id = requireString(value, 'id')

  if (!GENERATED_VIEW_ID_PATTERN.test(id)) {
    throw new Error('view.json id is invalid')
  }

  if (expectedId !== undefined && id !== expectedId) {
    throw new Error('view.json id must match its directory')
  }

  const title = requireString(value, 'title').trim()

  if (!title || title.length > 80) {
    throw new Error('view.json title must be 1-80 characters')
  }

  const entry = validateGeneratedViewEntry(requireString(value, 'entry'))
  const capabilities = validateStringList(value.capabilities, 'capabilities', GENERATED_VIEW_CAPABILITIES)
  const bindings = validateStringList(value.bindings, 'bindings', GENERATED_VIEW_BINDINGS, true)

  return { version: 1, id, title, entry, capabilities, bindings }
}

export function parseGeneratedViewManifest(source: string, expectedId?: string): GeneratedViewManifest {
  if (new TextEncoder().encode(source).byteLength > GENERATED_VIEW_MANIFEST_MAX_BYTES) {
    throw new Error(`view.json exceeds ${GENERATED_VIEW_MANIFEST_MAX_BYTES} bytes`)
  }

  let parsed: unknown

  try {
    parsed = JSON.parse(source)
  } catch {
    throw new Error('view.json is not valid JSON')
  }

  return validateGeneratedViewManifest(parsed, expectedId)
}

function portable(path: string): string {
  return path.replace(/\\/g, '/').replace(/\/+$/, '')
}

function windowsPath(path: string): boolean {
  return /^[A-Za-z]:\//.test(path)
}

/** Join a validated entry to its directory without importing node:path into the renderer. */
export function generatedViewEntryPath(directory: string, entry: string): string {
  return `${portable(directory)}/${validateGeneratedViewEntry(entry)}`
}

/** Canonical containment check for both POSIX and Windows paths returned by the fs facade. */
export function generatedViewPathIsContained(directory: string, candidate: string): boolean {
  const dir = portable(directory)
  const path = portable(candidate)
  const [comparableDir, comparablePath] = windowsPath(dir) ? [dir.toLowerCase(), path.toLowerCase()] : [dir, path]

  return comparablePath.startsWith(`${comparableDir}/`)
}

/** Stable approval input: manifest authority and authored bytes change the same digest. */
export function generatedViewApprovalSource(manifest: GeneratedViewManifest, html: string): string {
  return `${JSON.stringify({
    version: manifest.version,
    id: manifest.id,
    title: manifest.title,
    entry: manifest.entry,
    capabilities: manifest.capabilities,
    bindings: manifest.bindings
  })}\u0000${html}`
}
