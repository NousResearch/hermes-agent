import { requestComposerInsert } from './focus'

interface ReplyComposerInsert {
  (text: string, options: { mode: 'block'; target: 'main' }): void
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

  insert(quoted, { mode: 'block', target: 'main' })

  return true
}
