import { atom } from 'nanostores'

export interface MessageReplyTarget {
  messageId: string
  quote: string
}

const REPLY_QUOTE_MAX_CHARS = 220
const REPLY_BUTTON_MIN_CHARS = 300

export const $messageRepliesEnabled = atom(false)
export const $messageReplyTarget = atom<MessageReplyTarget | null>(null)

export function messageRepliesEnabledFromConfig(config: unknown): boolean {
  const desktop = config && typeof config === 'object' ? (config as { desktop?: unknown }).desktop : null
  const replies = desktop && typeof desktop === 'object' ? (desktop as { message_replies?: unknown }).message_replies : null

  return Boolean(replies && typeof replies === 'object' && (replies as { enabled?: unknown }).enabled === true)
}

export function messageRepliesEnabledForProfile(profile: string | null | undefined): boolean {
  return (profile ?? '').trim() === 'life-advisor'
}

export function messageTextCanBeRepliedTo(text: string): boolean {
  return text.trim().length >= REPLY_BUTTON_MIN_CHARS
}

export function quotePreview(text: string, maxChars = REPLY_QUOTE_MAX_CHARS): string {
  const normalized = text.replace(/\s+/g, ' ').trim()

  if (normalized.length <= maxChars) {
    return normalized
  }

  return `${normalized.slice(0, Math.max(0, maxChars - 1)).trimEnd()}…`
}

export function startMessageReply(target: MessageReplyTarget): void {
  $messageReplyTarget.set({ messageId: target.messageId, quote: quotePreview(target.quote) })
}

export function clearMessageReply(): void {
  $messageReplyTarget.set(null)
}

const REPLY_START = '[Replying to this earlier Hermes message]'
const REPLY_END = '[/Replying to this earlier Hermes message]'

export interface ParsedMessageReply {
  body: string
  quote: string | null
}

export function withReplyContext(text: string, target: MessageReplyTarget | null): string {
  if (!target) {
    return text
  }

  const body = text.trim()
  const quote = quotePreview(target.quote)

  return `${REPLY_START}\n${quote}\n${REPLY_END}\n\n${body}`
}

export function parseMessageReply(text: string): ParsedMessageReply {
  const trimmed = text.trimStart()

  if (!trimmed.startsWith(REPLY_START)) {
    return { body: text, quote: null }
  }

  const afterStart = trimmed.slice(REPLY_START.length).replace(/^\n/, '')
  const endIndex = afterStart.indexOf(REPLY_END)

  if (endIndex === -1) {
    return { body: text, quote: null }
  }

  const quote = afterStart.slice(0, endIndex).trim()
  const body = afterStart.slice(endIndex + REPLY_END.length).replace(/^\s*\n/, '').trimStart()

  return { body, quote: quote || null }
}
