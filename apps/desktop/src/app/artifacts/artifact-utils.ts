import { mediaTagValues } from '@/lib/chat-messages/parts'
import { isArtifactFilePath, mediaExternalUrl, mediaPathFromMarkdownHref, resolveMediaDisplaySrc } from '@/lib/media'
import type { SessionInfo, SessionMessage, SessionMessagesResponse } from '@/types/hermes'

export type ArtifactKind = 'image' | 'file' | 'link'
export type ArtifactFilter = 'all' | ArtifactKind
export const ARTIFACT_FILTERS: readonly ArtifactFilter[] = ['all', 'image', 'file', 'link']

export interface ArtifactRecord {
  id: string
  kind: ArtifactKind
  value: string
  href: string
  label: string
  sessionId: string
  profile?: string
  sessionTitle: string
  timestamp: number
}

export interface ArtifactLoadFailure {
  error: unknown
  session: SessionInfo
  /**
   * Which half of the per-session work failed. The load spans TWO steps — a
   * transcript READ and the artifact COLLECTION over what came back — and a
   * single catch block used to report both as "could not be read". A collect
   * failure means the transcript arrived fine and only the artifact scan
   * choked, which is a different defect with a different fix.
   */
  phase: 'read' | 'collect'
}

export interface ArtifactLoadResult {
  artifacts: ArtifactRecord[]
  failures: ArtifactLoadFailure[]
}

const ARTIFACT_MESSAGE_PAGE_SIZE = 100
const MAX_ARTIFACT_MESSAGE_PAGE_JSON_CHARS = 32_000_000

const MARKDOWN_IMAGE_RE = /!\[([^\]]*)\]\(([^)\s]+)\)/g
const MARKDOWN_LINK_RE = /\[([^\]]+)\]\(([^)\s]+)\)/g
const URL_RE = /https?:\/\/[^\s<>"')`]+/g
const PATH_RE = /(^|[\s("'`])((?:\/|~[\\/]|\.\.?[\\/]|\\\\)[^\s"'`<>]+(?:\.[a-z0-9]{1,8})?)/gi
const WINDOWS_PATH_RE = /(^|[\s("'`])([A-Za-z]:[\\/][^\s"'`<>]+(?:\.[a-z0-9]{1,8})?)/gi
const IMAGE_EXT_RE = /\.(?:png|jpe?g|gif|webp|svg|bmp)(?:\?.*)?$/i

const FILE_EXT_RE =
  /\.(?:png|jpe?g|gif|webp|svg|bmp|pdf|txt|json|md|csv|xlsx?|docx?|pptx?|html|zip|tar|gz|avi|flac|m4a|mkv|mp3|ogg|opus|wav|webm|mp4|mov)(?:\?.*)?$/i

const MAX_UNIX_SECONDS = 10_000_000_000

const ARTIFACT_PRODUCER_TOOL_RE =
  /(?:^|_)(?:creat(?:e|ion)|download|export|generat(?:e|ion)|render|save|speech|tts|write)(?:_|$)/i

const STRONG_TOOL_ARTIFACT_KEY_RE =
  /^(?:artifact_(?:file|image|path|url)|files?_(?:created|modified|written)|generated_(?:file|image|path|url)|media_tag|output_(?:file|path|url)|result_(?:file|path|url)|saved_to|screenshot_path)$/i

const PRODUCER_TOOL_ARTIFACT_KEY_RE =
  /^(?:artifact(?:s|_(?:file|image|path|url))?|attachment(?:s|_(?:file|image|path|url))?|download(?:s|_(?:file|path|url))?|(?:audio|image|video)(?:_(?:file|path|url))?|file_path|local_path|media(?:_(?:file|path|url))?|path)$/i

const SCREENSHOT_PATH_RE = /Screenshot path:\s*([^\r\n<>]+)/gi

// A pushValue callback plus whether the value is an explicit delivery the
// author asserted as an artifact (a raw `MEDIA:` tag), as opposed to a path
// scraped heuristically out of prose or a tool payload.
type PushValue = (value: string, explicit?: boolean) => void

// pip's own traffic, not session output (#52972): terminal output logs every
// download (`Downloading https://<index>/packages/…/pkg-1.0-py3-none-any.whl`,
// plus its `.whl.metadata`) and cached files live under pip's cache dir.
const PIP_CACHE_DIR_RE = /\/(?:\.cache\/pip|library\/caches\/pip|appdata\/local\/pip\/cache)\//
const WHEEL_RE = /\.whl(?:\.metadata)?$/
const PACKAGE_INDEX_DIST_RE = /\/packages\/.+\.(?:tar\.gz|tar\.bz2|zip|egg)(?:\.metadata)?$/

function artifactPathForFiltering(value: string): string {
  if (/^(?:https?|file):\/\//i.test(value)) {
    try {
      return decodeURIComponent(new URL(value).pathname).toLowerCase()
    } catch {
      // Malformed URL: fall through to plain string normalization.
    }
  }

  return value.replace(/\\/g, '/').toLowerCase()
}

function isPythonPackageDownload(value: string): boolean {
  const path = artifactPathForFiltering(value)

  return (
    PIP_CACHE_DIR_RE.test(path) ||
    WHEEL_RE.test(path) ||
    (/^https?:\/\//i.test(value) && PACKAGE_INDEX_DIST_RE.test(path))
  )
}

/**
 * Placeholder paths that appear in PROSE about how to deliver a file (the
 * system prompt's own "write MEDIA:/absolute/path/to/file" example, and the
 * tutorial paths around it). They are documentation, not delivered files, but
 * they carry a real-looking absolute path and a MEDIA tag — so the artifact
 * index used to list them as downloadable artifacts.
 *
 * Matched literally: these exact strings are what the prompt templates emit.
 * A real user path that merely CONTAINS "absolute/path/to" is not one of them.
 */
const PLACEHOLDER_ARTIFACT_PATHS: readonly string[] = [
  '/absolute/path/to/file',
  '/absolute/path/to/file.png',
  '/path/to/file',
  '/path/to/image.png'
]

function isPlaceholderArtifactPath(value: string): boolean {
  const normalized = value.replace(/\\/g, '/').replace(/\/+$/, '').toLowerCase()

  return PLACEHOLDER_ARTIFACT_PATHS.some(candidate => normalized === candidate.toLowerCase())
}

function looksLikeArtifact(value: string, explicit = false): boolean {
  if (isPlaceholderArtifactPath(value)) {
    return false
  }

  if (!explicit && isPythonPackageDownload(value)) {
    return false
  }

  if (/^(?:https?:\/\/|data:image\/)/.test(value)) {
    return true
  }

  if (!looksLikePathOrUrl(value)) {
    return false
  }

  // An explicitly delivered file is an artifact by definition even when its
  // extension is unknown — it should be listed as an opaque `file` entry
  // rather than silently vanish. Extensionless bare paths scraped from prose
  // stay excluded.
  if (explicit) {
    return true
  }

  return IMAGE_EXT_RE.test(value) || FILE_EXT_RE.test(value)
}

function artifactSessionTitle(session: SessionInfo): string {
  return session.title?.trim() || session.preview?.trim() || 'Untitled session'
}

function normalizeValue(value: string): string {
  let trimmed = value.trim()

  for (let i = 0; i < 3; i += 1) {
    const quote = trimmed[0]

    if (quote && quote === trimmed.at(-1) && ['"', "'", '`'].includes(quote)) {
      trimmed = trimmed.slice(1, -1).trim()

      continue
    }

    break
  }

  return trimmed.replace(/[`*]+$/g, '').replace(/[),.;]+$/, '')
}

// Chat renders file refs as `[label](#media:<encoded path>)`. Decode before
// classification so the Artifacts page keeps the path, not the href.
function decodeMediaHrefValue(value: string): string {
  return mediaPathFromMarkdownHref(value) ?? value
}

function unquoteMediaValue(value: string): string {
  let trimmed = value.trim()
  const quote = trimmed[0]

  if (quote && quote === trimmed.at(-1) && ['"', "'", '`'].includes(quote)) {
    return trimmed.slice(1, -1)
  }

  trimmed = trimmed.replace(/[`"'*_]{1,3}$/, '')

  return trimmed
}

function collectMediaValues(text: string, pushValue: PushValue): void {
  for (const value of mediaTagValues(text)) {
    pushValue(unquoteMediaValue(value), true)
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

function untrustedToolPayload(value: string): null | string {
  const trimmed = value.trim()
  const openTag = trimmed.match(/^<untrusted_tool_result\b[^>]*>\s*/)

  if (!openTag) {
    return null
  }

  const closeIndex = trimmed.lastIndexOf('</untrusted_tool_result>')

  if (closeIndex <= openTag[0].length) {
    return null
  }

  const wrapped = trimmed.slice(openTag[0].length, closeIndex).trim()
  const payloadStart = wrapped.indexOf('\n\n')

  return (payloadStart === -1 ? wrapped : wrapped.slice(payloadStart + 2)).trim()
}

function parseToolPayloads(text: string): unknown[] {
  const payloads: unknown[] = []

  for (const candidate of [text, untrustedToolPayload(text)]) {
    if (!candidate) {
      continue
    }

    const parsed = parseMaybeJson(candidate)

    if (parsed !== null) {
      payloads.push(parsed)
    }
  }

  return payloads
}

function isWindowsPath(value: string): boolean {
  return /^[A-Za-z]:[\\/]/.test(value) || value.startsWith('\\\\')
}

function looksLikePathOrUrl(value: string): boolean {
  return (
    value.startsWith('http://') ||
    value.startsWith('https://') ||
    value.startsWith('data:image/') ||
    isArtifactFilePath(value)
  )
}

function artifactKind(value: string): ArtifactKind {
  if (value.startsWith('data:image/') || IMAGE_EXT_RE.test(value)) {
    return 'image'
  }

  if (isArtifactFilePath(value)) {
    return 'file'
  }

  return 'link'
}

function artifactHref(value: string): string {
  if (value.startsWith('http://') || value.startsWith('https://') || value.startsWith('data:')) {
    return value
  }

  if (value.startsWith('file://') || value.startsWith('/') || isWindowsPath(value)) {
    return mediaExternalUrl(value)
  }

  return value
}

export async function artifactImageSrc(value: string): Promise<string> {
  // Delegate the whole local/remote ladder to the shared media resolver:
  // inline (http/data) stays as-is, remote gateway goes through the
  // authenticated fs bridge, local desktop through the Electron
  // readFileDataUrl, and bare non-path link values fall through untouched.
  // Reimplementing that ladder here would drift from resolveMediaDisplaySrc
  // and regress one of its legs (#83380).
  return resolveMediaDisplaySrc(value)
}

function artifactLabel(value: string): string {
  try {
    const url = new URL(value)
    const item = url.pathname.split('/').filter(Boolean).pop()

    return item || value
  } catch {
    const parts = value.split(/[\\/]/).filter(Boolean)

    return parts.pop() || value
  }
}

function normalizeArtifactTimestamp(timestamp: null | number | undefined): null | number {
  if (typeof timestamp !== 'number' || !Number.isFinite(timestamp) || timestamp <= 0) {
    return null
  }

  // Persisted session timestamps use Unix seconds. Values above the maximum
  // plausible Unix-seconds range are already milliseconds and stay unchanged.
  return timestamp < MAX_UNIX_SECONDS ? timestamp * 1000 : timestamp
}

function artifactTimestamp(message: SessionMessage, session: SessionInfo): number {
  return (
    normalizeArtifactTimestamp(message.timestamp) ??
    normalizeArtifactTimestamp(session.last_active) ??
    normalizeArtifactTimestamp(session.started_at) ??
    Date.now()
  )
}

function messageText(message: SessionMessage): string {
  if (typeof message.content === 'string' && message.content.trim()) {
    return message.content
  }

  if (typeof message.text === 'string' && message.text.trim()) {
    return message.text
  }

  if (typeof message.context === 'string' && message.context.trim()) {
    return message.context
  }

  return ''
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

function collectArtifactsFromText(text: string, pushValue: PushValue): void {
  collectMediaValues(text, pushValue)

  for (const match of text.matchAll(MARKDOWN_IMAGE_RE)) {
    pushValue(match[2] || '')
  }

  for (const match of text.matchAll(MARKDOWN_LINK_RE)) {
    const start = match.index ?? 0

    if (start > 0 && text[start - 1] === '!') {
      continue
    }

    const value = decodeMediaHrefValue(match[2] || '')

    if (looksLikeArtifact(value)) {
      pushValue(value)
    }
  }

  for (const match of text.matchAll(URL_RE)) {
    const value = match[0] || ''

    if (looksLikeArtifact(value)) {
      pushValue(value)
    }
  }

  for (const match of text.matchAll(PATH_RE)) {
    pushValue(match[2] || '')
  }

  for (const match of text.matchAll(WINDOWS_PATH_RE)) {
    pushValue(match[2] || '')
  }
}

function toolName(message: SessionMessage): string {
  return (message.tool_name || message.name || '').trim().toLowerCase()
}

function isArtifactProducerTool(name: string): boolean {
  // `bfl_flux3_*` tools were removed from the core toolset, but sessions
  // recorded while they existed still carry their tool messages — keep
  // matching so those artifacts stay visible in history.
  return ARTIFACT_PRODUCER_TOOL_RE.test(name) || name.startsWith('bfl_flux3_')
}

function isTerminalTool(name: string): boolean {
  return name === 'terminal'
}

// Shell-style tools report produced files as free text under generic keys
// (`output` / `stdout` / `path`). Their values are scanned as prose (MEDIA
// tags, markdown links, URLs, absolute paths) instead of being treated as a
// single path value.
const SHELL_OUTPUT_KEY_RE = /^(?:output|stdout|path)$/i

function explicitToolArtifactKey(keyPath: string, producerTool: boolean): boolean {
  return keyPath
    .split('.')
    .filter(segment => segment && !/^\d+$/.test(segment))
    .some(
      segment =>
        STRONG_TOOL_ARTIFACT_KEY_RE.test(segment) || (producerTool && PRODUCER_TOOL_ARTIFACT_KEY_RE.test(segment))
    )
}

function structuredToolPayload(message: SessionMessage): null | unknown {
  const content = message.content

  if (!content || typeof content !== 'object') {
    return null
  }

  if (!Array.isArray(content) && (content as Record<string, unknown>)._multimodal === true) {
    return (content as Record<string, unknown>).meta || null
  }

  return content
}

function collectArtifactsFromMessage(message: SessionMessage, pushValue: PushValue): void {
  const text = messageText(message)

  if (message.role === 'assistant' && text) {
    collectArtifactsFromText(text, pushValue)

    return
  }

  if (message.role !== 'tool') {
    return
  }

  const name = toolName(message)
  const producerTool = isArtifactProducerTool(name)
  const terminalTool = isTerminalTool(name)

  if (text && (producerTool || terminalTool)) {
    collectArtifactsFromText(text, pushValue)
  }

  if (name === 'browser_vision' && text) {
    for (const match of text.matchAll(SCREENSHOT_PATH_RE)) {
      pushValue(match[1] || '')
    }
  }

  const payloads = parseToolPayloads(text)
  const structured = structuredToolPayload(message)

  if (structured) {
    payloads.push(structured)
  }

  for (const parsed of payloads) {
    collectStringValues(parsed, 'tool_result', (value, keyPath) => {
      // Drop bare numeric array indices from the key path *intentionally*:
      // array-of-results payloads (e.g. `outputs.0.output`) must match via
      // their non-index segments, and with no index the shell-output/explicit
      // key tests match the last real segment. Do NOT switch this to
      // exact-key matching — it would silently stop indexing those shapes.
      const segments = keyPath.split('.').filter(segment => segment && !/^\d+$/.test(segment))

      const shellOutput = terminalTool && segments.some(segment => SHELL_OUTPUT_KEY_RE.test(segment))

      if (!shellOutput && !explicitToolArtifactKey(keyPath, producerTool)) {
        return
      }

      if (shellOutput) {
        // A shell result is free text: scan it for MEDIA tags, markdown
        // references, URLs and absolute paths rather than treating the
        // whole value as one path.
        //
        // False-positive budget: noisy stdout (`curl -v`, build logs) is
        // kept from flooding the panel because every candidate is filtered
        // through `looksLikeArtifact`, which requires a file/image extension
        // (IMAGE_EXT_RE / FILE_EXT_RE) or an explicit http(s)/data: scheme —
        // a bare error URL or un-extensioned path fails. Local file display
        // then resolves existence through the media ladder
        // (`artifactImageSrc` → `resolveMediaDisplaySrc`), so a candidate
        // whose file no longer exists is resolved to its fallback rather
        // than surfaced as a broken artifact.
        if (value) {
          collectArtifactsFromText(value, pushValue)
        }

        return
      }

      collectMediaValues(value, pushValue)

      const normalized = normalizeValue(decodeMediaHrefValue(value))

      if (normalized && looksLikeArtifact(normalized)) {
        pushValue(normalized)
      }
    })
  }
}

export function collectArtifactsForSession(session: SessionInfo, messages: SessionMessage[]): ArtifactRecord[] {
  const found = new Map<string, ArtifactRecord>()
  const title = artifactSessionTitle(session)

  for (const message of messages) {
    if (message.role !== 'assistant' && message.role !== 'tool') {
      continue
    }

    collectArtifactsFromMessage(message, (candidate, explicit = false) => {
      const value = normalizeValue(decodeMediaHrefValue(candidate))

      if (!value || !looksLikeArtifact(value, explicit)) {
        return
      }

      const key = `${session.id}:${value}`

      if (found.has(key)) {
        return
      }

      found.set(key, {
        id: key,
        kind: artifactKind(value),
        value,
        href: artifactHref(value),
        label: artifactLabel(value),
        sessionId: session.id,
        profile: session.profile,
        sessionTitle: title,
        timestamp: artifactTimestamp(message, session)
      })
    })
  }

  return Array.from(found.values())
}

/**
 * The backend that OWNS a session, as a request scope.
 *
 * The sessions list merges every connected backend's rows, so a row can belong
 * to a host other than the one the panel happens to be talking to. A row that
 * names its owning connection carries it in `connection_id`; an untagged row
 * belongs to the ambient backend that supplied the list.
 *
 * Reading a foreign session's transcript from the wrong backend is a 404 —
 * the transcript is simply not in that host's store. Passing the bare profile
 * name therefore made every cross-host session in the list fail its read, and
 * the panel reported those as unreadable sessions.
 */
export function artifactSessionScope(session: SessionInfo): { connectionId?: string; profile?: string } {
  const connectionId = (session.connection_id ?? '').trim()
  const profile = (session.profile ?? '').trim()

  // Both halves are required: a row tagged with a connection but no profile
  // cannot be routed (the owner route demands both), and the ambient backend
  // serves it.
  if (connectionId && profile) {
    return { connectionId, profile }
  }

  return { profile: session.profile }
}

export async function loadArtifactsForSessions(
  sessions: SessionInfo[],
  loadPage: (
    session: SessionInfo,
    page: { limit: number; offset: number }
  ) => Promise<Pick<SessionMessagesResponse, 'messages' | 'pagination'>>,
  options: { maxPageJsonChars?: number } = {}
): Promise<ArtifactLoadResult> {
  const artifacts: ArtifactRecord[] = []
  const failures: ArtifactLoadFailure[] = []
  const maxPageJsonChars = options.maxPageJsonChars ?? MAX_ARTIFACT_MESSAGE_PAGE_JSON_CHARS

  // Keep only one transcript page resident at a time. Recent sessions can each
  // be tens of megabytes, so retaining complete transcripts exhausts the
  // Desktop renderer even when transport requests are paginated.
  for (const session of sessions) {
    const sessionArtifacts = new Map<string, ArtifactRecord>()
    let offset = 0
    let readError: unknown = null
    let collectError: unknown = null

    // The READ phase: fetch the transcript page by page. A failure here means
    // the transcript itself is unavailable — transport, auth, or a store error.
    try {
      while (true) {
        const page = await loadPage(session, { limit: ARTIFACT_MESSAGE_PAGE_SIZE, offset })
        const pageJsonChars = (JSON.stringify(page.messages) ?? '').length

        if (pageJsonChars > maxPageJsonChars) {
          throw new Error(
            'Session transcript page exceeds the Desktop safe-load limit; use the Web Dashboard export for this session.'
          )
        }

        // The COLLECT phase: scan the page just read. It gets its own catch so
        // a scan error is not reported as an unreadable session — and it must
        // not abort the READ phase, which still has pages to fetch.
        try {
          for (const artifact of collectArtifactsForSession(session, page.messages)) {
            if (!sessionArtifacts.has(artifact.id)) {
              sessionArtifacts.set(artifact.id, artifact)
            }
          }
        } catch (error) {
          collectError = error
          break
        }

        // Legacy backends ignore pagination and return the full transcript.
        if (!page.pagination || page.messages.length === 0 || page.messages.length < page.pagination.limit) {
          break
        }

        offset += page.messages.length
      }
    } catch (error) {
      readError = error
    }

    if (readError !== null) {
      failures.push({ error: readError, session, phase: 'read' })
    } else if (collectError !== null) {
      failures.push({ error: collectError, session, phase: 'collect' })
    } else {
      artifacts.push(...sessionArtifacts.values())
    }
  }

  return { artifacts, failures }
}
