/**
 * A reply is a markdown quote under the name that wrote it. The page stays a
 * plain file an agent can read, while "who wrote this" survives as something
 * the editor can style apart from the user's own writing — a bare "name:" line
 * could not, since the user may type one themselves.
 */
export function replyBlock(handle: string, text: string, at: Date): string {
  const stamp = at.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })

  const body = text
    .trim()
    .split('\n')
    .map(line => (line ? `> ${line}` : '>'))
    .join('\n')

  return `> ${handle.replace(/^@/, '')} · ${stamp}\n${body}`
}

/**
 * Where the reply goes: right after the question it answers. The question is
 * found again at write time — the user keeps typing while an agent thinks, so
 * an offset captured at send time would land in the wrong place. A question
 * that is gone (edited away) puts its reply at the end of the page rather than
 * dropping it.
 */
export function replyInsertion(doc: string, question: string, block: string): { from: number; insert: string } {
  const at = doc.indexOf(question)
  const from = at === -1 ? doc.length : at + question.length
  // Whatever the user has written below must not end up glued to the last
  // quoted line — but text already a blank line away is left as it is.
  const rest = doc.slice(from)
  const trailing = rest.trim() && !rest.startsWith('\n\n') ? '\n\n' : ''

  return { from, insert: `\n\n${block}${trailing}` }
}
