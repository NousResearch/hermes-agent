import { encodeComposerQuote } from '@/lib/composer-quote'

import { type ComposerTarget, requestComposerInsert } from './focus'

interface ReplyComposerInsert {
  (text: string, options: { mode: 'block'; target: ComposerTarget | 'active' }): void
}

interface ReplyComposerOptions {
  insert?: ReplyComposerInsert
  target?: ComposerTarget | 'active'
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
export function insertMessageReply(
  messageText: string,
  { insert = requestComposerInsert, target = 'active' }: ReplyComposerOptions = {}
): boolean {
  const quoted = quoteMessageForReply(messageText)

  if (!quoted) {
    return false
  }

  const payload = encodeComposerQuote({ body: quoted, label: replyLabel(messageText) })

  insert(`@quote:\`${payload}\``, { mode: 'block', target })

  return true
}
