import { readDesktopFileDataUrl } from '@/lib/desktop-fs'
import { filePathFromMediaPath, isRemoteGateway, mediaExternalUrl } from '@/lib/media'
import type { SessionInfo, SessionMessage } from '@/types/hermes'

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
  sessionTitle: string
  timestamp: number
}

export interface ArtifactLoadFailure {
  error: unknown
  session: SessionInfo
}

export interface ArtifactLoadResult {
  artifacts: ArtifactRecord[]
  failures: ArtifactLoadFailure[]
}

const MARKDOWN_IMAGE_RE = /!\[([^\]]*)\]\(([^)\s]+)\)/g
const MARKDOWN_LINK_RE = /\[([^\]]+)\]\(([^)\s]+)\)/g
const MEDIA_RE = /[`"']?MEDIA:\s*(`[^`\n]+`|"[^"\n]+"|'[^'\n]+'|\S+)[`"']?/g
const URL_RE = /https?:\/\/[^\s<>"')]+/g
const PATH_RE = /(^|[\s("'`])((?:\/|~\/|\.\.?\/)[^\s"'`<>]+(?:\.[a-z0-9]{1,8})?)/gi
const WINDOWS_PATH_RE = /(^|[\s("'`])([A-Za-z]:[\\/][^\s"'`<>]+(?:\.[a-z0-9]{1,8})?)/gi
const IMAGE_EXT_RE = /\.(?:png|jpe?g|gif|webp|svg|bmp)(?:\?.*)?$/i

const FILE_EXT_RE =
  /\.(?:png|jpe?g|gif|webp|svg|bmp|pdf|txt|json|md|csv|zip|tar|gz|avi|flac|m4a|mkv|mp3|ogg|opus|wav|webm|mp4|mov)(?:\?.*)?$/i

const MAX_UNIX_SECONDS = 10_000_000_000

const ARTIFACT_PRODUCER_TOOL_RE =
  /(?:^|_)(?:creat(?:e|ion)|download|export|generat(?:e|ion)|render|save|speech|tts|write)(?:_|$)/i

const STRONG_TOOL_ARTIFACT_KEY_RE =
  /^(?:artifact_(?:file|image|path|url)|files?_(?:created|modified|written)|generated_(?:file|image|path|url)|media_tag|output_(?:file|path|url)|result_(?:file|path|url)|saved_to|screenshot_path)$/i

const PRODUCER_TOOL_ARTIFACT_KEY_RE =
  /^(?:artifact(?:s|_(?:file|image|path|url))?|attachment(?:s|_(?:file|image|path|url))?|download(?:s|_(?:file|path|url))?|(?:audio|image|video)(?:_(?:file|path|url))?|file_path|local_path|media(?:_(?:file|path|url))?|path)$/i

const SCREENSHOT_PATH_RE = /Screenshot path:\s*([^\r\n<>]+)/gi

function artifactSessionTitle(session: SessionInfo): string {
  return session.title?.trim() || session.preview?.trim() || 'Untitled session'
}

function normalizeValue(value: string): string {
  return value.trim().replace(/[),.;]+$/, '')
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

function collectMediaValues(text: string, pushValue: (value: string) => void): void {
  for (const match of text.matchAll(MEDIA_RE)) {
    pushValue(unquoteMediaValue(match[1] || ''))
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
    value.startsWith('file://') ||
    value.startsWith('data:image/') ||
    value.startsWith('/') ||
    value.startsWith('./') ||
    value.startsWith('../') ||
    value.startsWith('~/') ||
    isWindowsPath(value)
  )
}

function looksLikeArtifact(value: string): boolean {
  if (/^(?:https?:\/\/|data:image\/)/.test(value)) {
    return true
  }

  return looksLikePathOrUrl(value) && (IMAGE_EXT_RE.test(value) || FILE_EXT_RE.test(value))
}

function artifactKind(value: string): ArtifactKind {
  if (value.startsWith('data:image/') || IMAGE_EXT_RE.test(value)) {
    return 'image'
  }

  if (
    value.startsWith('/') ||
    value.startsWith('./') ||
    value.startsWith('../') ||
    value.startsWith('~/') ||
    value.startsWith('file://') ||
    isWindowsPath(value)
  ) {
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

export async function artifactImageSrc(value: string, href = artifactHref(value)): Promise<string> {
  if (/^(?:https?|data):/i.test(value)) {
    return href
  }

  if (typeof window !== 'undefined' && window.hermesDesktop && isRemoteGateway()) {
    return readDesktopFileDataUrl(filePathFromMediaPath(value))
  }

  return href
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

function collectArtifactsFromText(text: string, pushValue: (value: string) => void): void {
  collectMediaValues(text, pushValue)

  for (const match of text.matchAll(MARKDOWN_IMAGE_RE)) {
    pushValue(match[2] || '')
  }

  for (const match of text.matchAll(MARKDOWN_LINK_RE)) {
    const start = match.index ?? 0

    if (start > 0 && text[start - 1] === '!') {
      continue
    }

    const value = match[2] || ''

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
  return ARTIFACT_PRODUCER_TOOL_RE.test(name) || name.startsWith('bfl_flux3_')
}

function explicitToolArtifactKey(keyPath: string, producerTool: boolean): boolean {
  return keyPath
    .split('.')
    .filter(segment => segment && !/^\d+$/.test(segment))
    .some(
      segment => STRONG_TOOL_ARTIFACT_KEY_RE.test(segment) || (producerTool && PRODUCER_TOOL_ARTIFACT_KEY_RE.test(segment))
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

function collectArtifactsFromMessage(message: SessionMessage, pushValue: (value: string) => void): void {
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

  if (text && producerTool) {
    collectMediaValues(text, pushValue)
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
      if (!explicitToolArtifactKey(keyPath, producerTool)) {
        return
      }

      collectMediaValues(value, pushValue)

      const normalized = normalizeValue(value)

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

    collectArtifactsFromMessage(message, candidate => {
      const value = normalizeValue(candidate)

      if (!value || !looksLikeArtifact(value)) {
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
        sessionTitle: title,
        timestamp: artifactTimestamp(message, session)
      })
    })
  }

  return Array.from(found.values())
}

export async function loadArtifactsForSessions(
  sessions: SessionInfo[],
  loadMessages: (session: SessionInfo) => Promise<SessionMessage[]>
): Promise<ArtifactLoadResult> {
  const artifacts: ArtifactRecord[] = []
  const failures: ArtifactLoadFailure[] = []

  // Keep only one transcript resident at a time. Recent sessions can each be
  // tens of megabytes, so loading the whole page concurrently can exhaust both
  // the Desktop renderer and a remote dashboard backend.
  for (const session of sessions) {
    try {
      const messages = await loadMessages(session)
      artifacts.push(...collectArtifactsForSession(session, messages))
    } catch (error) {
      failures.push({ error, session })
    }
  }

  return { artifacts, failures }
}
