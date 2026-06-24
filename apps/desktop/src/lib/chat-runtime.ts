import type { ThreadMessage } from '@assistant-ui/react'

import type { QuickModelOption } from '@/app/chat/composer/types'
import type { ClientSessionState, CommandDispatchResponse } from '@/app/types'
import { formatRefValue } from '@/components/assistant-ui/directive-text'
import { type ChatMessage, type ChatMessagePart, chatMessageText, textPart } from '@/lib/chat-messages'
import type { ComposerAttachment } from '@/store/composer'
import type { ModelOptionsResponse, SessionInfo } from '@/types/hermes'

export const SLASH_COMMAND_RE = /^\/[^\s/]*(?:\s|$)/
export const BUILTIN_PERSONALITIES = [
  'helpful',
  'concise',
  'technical',
  'creative',
  'teacher',
  'kawaii',
  'catgirl',
  'pirate',
  'shakespeare',
  'surfer',
  'noir',
  'uwu',
  'philosopher',
  'hype'
]

const THINKING_STATUS_PREFIX_RE =
  /^\s*(?:(?:[^\s.]{1,16})\s+)?(?:processing|thinking|reasoning|analyzing|pondering|contemplating|musing|cogitating|ruminating|deliberating|mulling|reflecting|computing|synthesizing|formulating|brainstorming)\.\.\.\s*/i

const EMPTY_THINKING_PLACEHOLDER_RE =
  /\b(?:current rewritten thinking|next thinking to process|provide the thinking content|don't see any .*thinking)\b/i

export function createClientSessionState(
  storedSessionId: string | null = null,
  messages: ChatMessage[] = []
): ClientSessionState {
  return {
    storedSessionId,
    messages,
    branch: '',
    cwd: '',
    model: '',
    provider: '',
    reasoningEffort: '',
    serviceTier: '',
    fast: false,
    yolo: false,
    personality: '',
    busy: false,
    awaitingResponse: false,
    streamId: null,
    sawAssistantPayload: false,
    pendingBranchGroup: null,
    interrupted: false,
    needsInput: false,
    turnStartedAt: null
  }
}

export function sessionTitle(session: SessionInfo): string {
  return session.title?.trim() || session.preview?.trim() || 'Untitled session'
}

export function coerceGatewayText(value: unknown): string {
  if (typeof value === 'string') {
    return value
  }

  if (value === null || value === undefined) {
    return ''
  }

  if (Array.isArray(value)) {
    return value
      .map(item => {
        if (typeof item === 'string') {
          return item
        }

        if (item && typeof item === 'object') {
          const row = item as Record<string, unknown>

          if (typeof row.text === 'string') {
            return row.text
          }

          if (typeof row.output_text === 'string') {
            return row.output_text
          }
        }

        return ''
      })
      .join('')
  }

  if (typeof value === 'object') {
    const row = value as Record<string, unknown>

    if (typeof row.text === 'string') {
      return row.text
    }

    if (typeof row.output_text === 'string') {
      return row.output_text
    }

    try {
      return JSON.stringify(value)
    } catch {
      return ''
    }
  }

  return String(value)
}

/**
 * Normalize a reasoning/thinking text payload from the gateway.
 *
 * Only the leading status prefix (e.g. "Hermes is thinking...") and the
 * obvious placeholder echoes are stripped. We deliberately do NOT trim
 * the delta — reasoning streams as small chunks (often individual tokens
 * with leading or trailing spaces), and trimming each chunk before
 * concatenation collapses adjacent words together. Whitespace between
 * tokens belongs to the data, not chrome.
 */
export function coerceThinkingText(value: unknown): string {
  const raw = coerceGatewayText(value).replace(THINKING_STATUS_PREFIX_RE, '')

  return EMPTY_THINKING_PLACEHOLDER_RE.test(raw) ? '' : raw
}

export function isImageGenerationTool(name?: string): boolean {
  return name === 'image_generate'
}

export function contextPath(path: string, cwd: string): string {
  if (!cwd) {
    return path
  }

  const normalizedCwd = cwd.endsWith('/') ? cwd : `${cwd}/`

  return path.startsWith(normalizedCwd) ? path.slice(normalizedCwd.length) : path
}

export function attachmentId(kind: ComposerAttachment['kind'], value: string): string {
  return `${kind}:${value}`
}

export function pathLabel(path: string): string {
  return path.split(/[\\/]/).filter(Boolean).pop() || path
}

export function attachmentDisplayText(attachment: ComposerAttachment): string | null {
  if (attachment.kind === 'terminal' && attachment.detail) {
    return `\`\`\`terminal\n${attachment.detail.trim()}\n\`\`\``
  }

  if (attachment.refText) {
    return attachment.refText
  }

  if (attachment.kind === 'image') {
    const id = attachment.detail || attachment.path || attachment.label

    return id ? `@image:${formatRefValue(id)}` : null
  }

  return null
}

/**
 * Display ref for the optimistic (in-flight) user bubble.
 *
 * Images prefer their in-hand base64 preview (a `data:` URL) over a file path.
 * `DirectiveContent` runs `extractEmbeddedImages` first, so a raw `data:` URL
 * renders as an inline thumbnail with zero network. An `@image:<localpath>` ref
 * would instead route through `/api/media`, which in remote mode 403s ("Path
 * outside media roots") on a local path the gateway can't read yet — flashing a
 * fallback chip until submit uploads the bytes. The preview also survives the
 * post-sync rewrite (bytes go to the agent via the attached-image pipeline, not
 * this display ref), so the thumbnail stays stable instead of remounting.
 *
 * Everything else (files, folders, terminals, post-sync `@file:` refs) falls
 * through to `attachmentDisplayText`.
 */
export function optimisticAttachmentRef(attachment: ComposerAttachment): string | null {
  if (attachment.kind === 'image' && attachment.previewUrl?.startsWith('data:')) {
    return attachment.previewUrl
  }

  return attachmentDisplayText(attachment)
}

export function personalityNamesFromConfig(config: unknown): string[] {
  const root = config && typeof config === 'object' ? (config as Record<string, unknown>) : {}
  const agent = root.agent && typeof root.agent === 'object' ? (root.agent as Record<string, unknown>) : {}
  const personalities = agent.personalities

  return personalities && typeof personalities === 'object' && !Array.isArray(personalities)
    ? Object.keys(personalities as Record<string, unknown>)
    : []
}

export function normalizePersonalityValue(value: string): string {
  const trimmed = value.trim().toLowerCase()

  return !trimmed || trimmed === 'default' || trimmed === 'none' ? '' : trimmed
}

export function parseSlashCommand(command: string) {
  const match = command.replace(/^\/+/, '').match(/^(\S+)\s*(.*)$/)

  return match ? { name: match[1], arg: match[2].trim() } : { name: '', arg: '' }
}

export function parseCommandDispatch(raw: unknown): CommandDispatchResponse | null {
  if (!raw || typeof raw !== 'object') {
    return null
  }

  const row = raw as Record<string, unknown>
  const str = (value: unknown) => (typeof value === 'string' ? value : undefined)

  switch (row.type) {
    case 'exec':

    case 'plugin':
      return { type: row.type, output: str(row.output) }

    case 'alias':
      return typeof row.target === 'string' ? { type: 'alias', target: row.target } : null

    case 'skill':
      return typeof row.name === 'string' ? { type: 'skill', name: row.name, message: str(row.message) } : null

    case 'send':
      return typeof row.message === 'string' ? { type: 'send', message: row.message, notice: str(row.notice) } : null

    case 'prefill':
      return typeof row.message === 'string'
        ? { type: 'prefill', message: row.message, notice: str(row.notice) }
        : null

    default:
      return null
  }
}

export function quickModelOptions(
  data: ModelOptionsResponse | undefined,
  currentProvider: string,
  currentModel: string
): QuickModelOption[] {
  const seen = new Set<string>()
  const options: QuickModelOption[] = []

  const providers = [...(data?.providers ?? [])].sort((a, b) => {
    if (a.slug === currentProvider) {
      return -1
    }

    if (b.slug === currentProvider) {
      return 1
    }

    if (a.is_current) {
      return -1
    }

    if (b.is_current) {
      return 1
    }

    return 0
  })

  const add = (provider: string, providerName: string, model: string) => {
    const key = `${provider}:${model}`

    if (!model || seen.has(key)) {
      return
    }

    seen.add(key)
    options.push({ provider, providerName, model })
  }

  if (currentProvider && currentModel) {
    add(currentProvider, currentProvider, currentModel)
  }

  for (const provider of providers) {
    const models = [...(provider.models ?? [])].sort((a, b) => {
      if (provider.slug === currentProvider && a === currentModel) {
        return -1
      }

      if (provider.slug === currentProvider && b === currentModel) {
        return 1
      }

      return 0
    })

    for (const model of models) {
      add(provider.slug, provider.name, model)
    }

    if (options.length >= 8) {
      break
    }
  }

  return options.slice(0, 8)
}

export function toRuntimeMessage(message: ChatMessage): ThreadMessage {
  const role =
    message.role === 'user' || message.role === 'assistant' || message.role === 'system' ? message.role : 'assistant'

  const createdAt = message.timestamp
    ? new Date(message.timestamp * 1000)
    : new Date(Number(message.id.match(/\d+/)?.[0]) || Date.now())

  if (role === 'user') {
    return {
      id: message.id,
      role,
      content: message.parts.filter((part): part is Extract<ChatMessagePart, { type: 'text' }> => part.type === 'text'),
      attachments: [],
      createdAt,
      metadata: {
        custom: {
          attachmentRefs: message.attachmentRefs ?? [],
          ...(message.document ? { document: message.document, documentVersion: message.documentVersion } : {})
        }
      }
    } as ThreadMessage
  }

  if (role === 'system') {
    const text = chatMessageText(message)

    return {
      id: message.id,
      role,
      content: [textPart(text)],
      createdAt,
      metadata: { custom: {} }
    } as ThreadMessage
  }

  return {
    id: message.id,
    role,
    content: message.parts as Extract<ThreadMessage, { role: 'assistant' }>['content'],
    createdAt,
    status: message.error
      ? { type: 'incomplete', reason: 'error', error: message.error }
      : message.pending
        ? { type: 'running' }
        : { type: 'complete', reason: 'stop' },
    metadata: {
      unstable_state: null,
      unstable_annotations: [],
      unstable_data: [],
      steps: [],
      custom: {}
    }
  } as ThreadMessage
}
