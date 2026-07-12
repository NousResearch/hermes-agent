import { readDesktopFileDataUrl } from '@/lib/desktop-fs'
import { filePathFromMediaPath, isRemoteGateway, mediaExternalUrl } from '@/lib/media'
import type { SessionInfo, SessionMessage } from '@/types/hermes'

export type ArtifactKind = 'image' | 'file' | 'link'
export type ArtifactFilter = 'all' | ArtifactKind
export const ARTIFACT_FILTERS: readonly ArtifactFilter[] = ['all', 'image', 'file', 'link']

export interface ArtifactRecord {
  id: string
  imageCandidate: boolean
  kind: ArtifactKind
  value: string
  href: string
  label: string
  /** True only when a trusted tool result proves the image path is a generated output. */
  previewable: boolean
  profile: string
  sessionId: string
  sessionTitle: string
  timestamp: number
}

const MARKDOWN_IMAGE_RE = /!\[([^\]]*)\]\(([^)\s]+)\)/g
const MARKDOWN_LINK_RE = /\[([^\]]+)\]\(([^)\s]+)\)/g
const URL_RE = /https?:\/\/[^\s<>"')]+/g
const PATH_RE = /(^|[\s("'`])((?:\/|~\/|\.\.?\/)[^\s"'`<>]+(?:\.[a-z0-9]{1,8})?)/gi
const IMAGE_EXT_RE = /\.(?:avif|png|jpe?g|gif|webp|svg|bmp)(?:[?#].*)?$/i

const FILE_EXT_RE =
  /\.(?:avif|png|jpe?g|gif|webp|svg|bmp|pdf|txt|json|md|csv|ya?ml|toml|ini|tsx?|jsx?|html?|css|s[ac]ss|sh|bash|zsh|py|rb|rs|go|java|c|cc|cpp|h|hpp|conf|log|lock|sql|xml|docx?|xlsx?|pptx?|zip|tar|gz|tgz|7z|rar|mp3|wav|m4a|ogg|flac|opus|mp4|webm|mov|mkv|avi)(?:[?#].*)?$/i

const AMBIGUOUS_WEB_ROOT_RE = /^\/(?:assets|images|media|static|uploads)\//i
const KEY_HINT_RE = /(path|file|url|image|artifact|output|download|result|target)/i
const ROUTE_TEMPLATE_RE = /(?:\$\{|\{[^}]*\}|\/:\w)/
const TRUSTED_IMAGE_RESULT_TOOLS = new Set(['image_generate'])
const TRUSTED_IMAGE_RESULT_KEY_RE = /(?:^|\.)(?:agent_visible_image|host_image|image)$/i

function artifactSessionTitle(session: SessionInfo): string {
  return session.title?.trim() || session.preview?.trim() || 'Untitled session'
}

function normalizeValue(value: string): string {
  const firstLine = value.trim().split(/(?:\\n|[\r\n])/, 1)[0] || ''

  return firstLine.replace(/[),.;]+$/, '')
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

function looksLikePathOrUrl(value: string): boolean {
  return (
    value.startsWith('http://') ||
    value.startsWith('https://') ||
    value.startsWith('file://') ||
    value.startsWith('data:image/') ||
    value.startsWith('/') ||
    value.startsWith('./') ||
    value.startsWith('../') ||
    value.startsWith('~/')
  )
}

function looksLikeArtifact(value: string): boolean {
  if (/^(?:https?:\/\/|data:image\/)/.test(value)) {
    return true
  }

  if (looksLikePathOrUrl(value) && (IMAGE_EXT_RE.test(value) || FILE_EXT_RE.test(value))) {
    return true
  }

  if ((value.startsWith('/') || value.startsWith('file://')) && !ROUTE_TEMPLATE_RE.test(value)) {
    const path = value.startsWith('file://') ? filePathFromMediaPath(value) : value
    const basename = path.split(/[?#]/, 1)[0]?.split('/').filter(Boolean).pop() || ''

    return basename.includes('.') && basename !== '.' && basename !== '..'
  }

  return false
}

function artifactKind(value: string, imageCandidate = false): ArtifactKind {
  if (imageCandidate) {
    return 'image'
  }

  if (value.startsWith('data:image/') || IMAGE_EXT_RE.test(value)) {
    return 'image'
  }

  if (
    value.startsWith('/') ||
    value.startsWith('./') ||
    value.startsWith('../') ||
    value.startsWith('~/') ||
    value.startsWith('file://')
  ) {
    return 'file'
  }

  if (/^https?:\/\//i.test(value) && FILE_EXT_RE.test(value)) {
    return 'file'
  }

  return 'link'
}

function artifactHref(value: string): string {
  if (value.startsWith('http://') || value.startsWith('https://')) {
    return value
  }

  if (
    value.startsWith('data:') ||
    value.startsWith('./') ||
    value.startsWith('../') ||
    value.startsWith('~/') ||
    AMBIGUOUS_WEB_ROOT_RE.test(value)
  ) {
    return ''
  }

  if (value.startsWith('file://')) {
    return value
  }

  if (value.startsWith('/')) {
    return `file://${value}`
  }

  return value
}

export async function artifactImageSrc(
  value: string,
  href = artifactHref(value),
  previewable = false,
  profile?: string,
  allowNetwork = false
): Promise<string> {
  // Historical prose has no output provenance. Web/data sources are already
  // renderable; filesystem paths load only when a trusted tool result opts in.
  if (AMBIGUOUS_WEB_ROOT_RE.test(value) && !previewable) {
    return ''
  }

  if (/^data:image\//i.test(value)) {
    return value
  }

  if (/^https?:/i.test(value)) {
    return previewable || allowNetwork ? href : ''
  }

  if (!previewable) {
    return ''
  }

  if (typeof window !== 'undefined' && window.hermesDesktop && (profile !== undefined || isRemoteGateway())) {
    return readDesktopFileDataUrl(filePathFromMediaPath(value), profile)
  }

  return href || mediaExternalUrl(value)
}

function artifactLabel(value: string, labelHint?: string): string {
  if (/^data:image\//i.test(value)) {
    const normalizedHint = labelHint?.trim().replace(/\s+/g, ' ').slice(0, 120)

    if (normalizedHint) {
      return normalizedHint
    }

    return 'data:image'
  }

  try {
    const url = new URL(value)
    const item = url.pathname.split('/').filter(Boolean).pop()

    return item || value
  } catch {
    const parts = value.split(/[\\/]/).filter(Boolean)

    return parts.pop() || value
  }
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

function collectArtifactsFromText(
  text: string,
  pushValue: (value: string, labelHint?: string, previewable?: boolean, imageCandidate?: boolean) => void
): void {
  for (const match of text.matchAll(MARKDOWN_IMAGE_RE)) {
    pushValue(match[2] || '', match[1] || '', false, true)
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
}

function collectArtifactsFromMessage(
  message: SessionMessage,
  pushValue: (value: string, labelHint?: string, previewable?: boolean, imageCandidate?: boolean) => void
): void {
  const text = messageText(message)

  if (text) {
    collectArtifactsFromText(text, pushValue)
  }

  if (message.role !== 'tool' && !Array.isArray(message.tool_calls)) {
    return
  }

  if (Array.isArray(message.tool_calls)) {
    for (const call of message.tool_calls) {
      collectStringValues(call, 'tool_call', (value, keyPath) => {
        const normalized = normalizeValue(value)

        if (!normalized) {
          return
        }

        if (KEY_HINT_RE.test(keyPath) && (looksLikePathOrUrl(normalized) || FILE_EXT_RE.test(normalized))) {
          pushValue(normalized)
        }
      })
    }
  }

  const parsed = parseMaybeJson(text)

  if (parsed !== null) {
    const parsedRecord = parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : null
    const toolName = (message.tool_name || message.name || '').trim().toLowerCase()

    const trustedToolResult =
      message.role === 'tool' &&
      TRUSTED_IMAGE_RESULT_TOOLS.has(toolName) &&
      (parsedRecord as Record<string, unknown> | null)?.success === true &&
      !Object.prototype.hasOwnProperty.call(parsedRecord, 'error')

    collectStringValues(parsed, 'tool_result', (value, keyPath) => {
      const normalized = normalizeValue(value)

      if (!normalized) {
        return
      }

      if ((KEY_HINT_RE.test(keyPath) || looksLikePathOrUrl(normalized)) && looksLikeArtifact(normalized)) {
        const trustedImage = trustedToolResult && TRUSTED_IMAGE_RESULT_KEY_RE.test(keyPath)

        pushValue(normalized, undefined, trustedImage, trustedImage)
      }
    })
  }
}

function boundedArtifactIdentity(value: string): string {
  if (!/^data:image\//i.test(value)) {
    return value
  }

  let hash = 2166136261

  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index)
    hash = Math.imul(hash, 16777619)
  }

  return `data:image:${value.length}:${(hash >>> 0).toString(16)}`
}

export function collectArtifactsForSession(session: SessionInfo, messages: SessionMessage[]): ArtifactRecord[] {
  const found = new Map<string, ArtifactRecord>()
  const profile = session.profile?.trim() || 'default'
  const title = artifactSessionTitle(session)

  for (const message of messages) {
    if (message.role !== 'assistant' && message.role !== 'tool') {
      continue
    }

    collectArtifactsFromMessage(message, (candidate, labelHint, previewable = false, imageCandidate = false) => {
      const value = normalizeValue(candidate)

      if (!value || !looksLikeArtifact(value)) {
        return
      }

      const identity = boundedArtifactIdentity(value)
      const key = JSON.stringify([profile, session.id, identity])
      const timestamp = message.timestamp || session.last_active || session.started_at || Date.now()
      const label = artifactLabel(value, labelHint)
      const existing = found.get(key)

      if (existing) {
        const nextTimestamp = Math.max(existing.timestamp, timestamp)
        const nextLabel = existing.label === 'data:image' && label !== 'data:image' ? label : existing.label
        const nextPreviewable = existing.previewable || previewable
        const nextImageCandidate = existing.imageCandidate || imageCandidate
        const nextKind = nextImageCandidate ? 'image' : existing.kind

        if (
          nextTimestamp !== existing.timestamp ||
          nextLabel !== existing.label ||
          nextPreviewable !== existing.previewable ||
          nextImageCandidate !== existing.imageCandidate ||
          nextKind !== existing.kind
        ) {
          found.set(key, {
            ...existing,
            imageCandidate: nextImageCandidate,
            kind: nextKind,
            label: nextLabel,
            previewable: nextPreviewable,
            timestamp: nextTimestamp
          })
        }

        return
      }

      found.set(key, {
        id: key,
        imageCandidate,
        kind: artifactKind(value, imageCandidate),
        value,
        href: artifactHref(value),
        label,
        previewable,
        profile,
        sessionId: session.id,
        sessionTitle: title,
        timestamp
      })
    })
  }

  return Array.from(found.values())
}
