import { $composerQuotes, setComposerQuote } from '@/store/composer'

import { requestComposerInsert } from './focus'
import { quoteRefValue } from './rich-editor'

interface ReplyComposerInsert {
  (text: string, options: { mode: 'block'; target: 'main' }): void
}

const REPLY_LABEL_MAX_LENGTH = 40

function replyLabel(messageText: string) {
  const collapsed = messageText.replace(/[`"']/g, '').replace(/\s+/g, ' ').trim()

  if (!collapsed) {
    return 'quote'
  }

  return collapsed.length > REPLY_LABEL_MAX_LENGTH
    ? `${collapsed.slice(0, REPLY_LABEL_MAX_LENGTH).trimEnd()}…`
    : collapsed
}

function allocateReplyLabel(messageText: string) {
  const base = replyLabel(messageText)
  const quotes = $composerQuotes.get()

  if (!Object.hasOwn(quotes, base)) {
    return base
  }

  let suffix = 2

  while (Object.hasOwn(quotes, `${base} (${suffix})`)) {
    suffix += 1
  }

  return `${base} (${suffix})`
}

/** Format a complete chat message as one continuous Markdown blockquote. */
export function quoteMessageForReply(messageText: string): string {
  if (!messageText.trim()) {
    return ''
  }

  return messageText
    .replace(/\r\n?/g, '\n')
    .split('\n')
    .map(line => `> ${line}`)
    .join('\n')
}

/** Quote a whole message and send it through the composer's external-insert bus. */
export function insertMessageReply(messageText: string, insert: ReplyComposerInsert = requestComposerInsert): boolean {
  const quoted = quoteMessageForReply(messageText)

  if (!quoted) {
    return false
  }

  const label = allocateReplyLabel(messageText)

  setComposerQuote(label, quoted)
  insert(`@quote:${quoteRefValue(label)}`, { mode: 'block', target: 'main' })

  return true
}
