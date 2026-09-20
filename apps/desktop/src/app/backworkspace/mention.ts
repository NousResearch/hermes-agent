export interface MentionToken {
  /** Document offset of the `@`. */
  from: number
  /** What has been typed after the `@` (no leading `@`). */
  query: string
  /** The caret — the end of the token. */
  to: number
}

// A mention starts a word: at the start of a line or after whitespace, so an
// email address (`a@b.com`) never opens the list. Letters and digits are
// matched by Unicode class, not `\w`, so a renamed bot in any script completes.
const MENTION_AT_CARET = /(^|\s)@([\p{L}\p{N}._-]*)$/u

/** The `@name` being typed at `caret`, or null when the caret is not in one. */
export function mentionTokenAt(doc: string, caret: number): MentionToken | null {
  const lineStart = doc.lastIndexOf('\n', Math.max(0, caret - 1)) + 1
  const match = MENTION_AT_CARET.exec(doc.slice(lineStart, caret))

  if (!match) {
    return null
  }

  return { from: lineStart + match.index + match[1].length, query: match[2], to: caret }
}

/** The change that turns the typed `@…` into `insert` followed by a space. */
export function mentionInsertion(token: MentionToken, insert: string) {
  const text = `${insert} `

  return {
    changes: { from: token.from, insert: text, to: token.to },
    selection: { anchor: token.from + text.length }
  }
}

export interface Paragraph {
  from: number
  text: string
  to: number
}

/** The block of text around `pos`, bounded by blank lines — what a question is. */
export function paragraphAt(doc: string, pos: number): Paragraph {
  const before = doc.lastIndexOf('\n\n', Math.max(0, pos - 1))
  const after = doc.indexOf('\n\n', pos)
  const from = before === -1 ? 0 : before + 2
  const to = after === -1 ? doc.length : after

  return { from, text: doc.slice(from, to).trim(), to }
}

const MENTION_IN_TEXT = /(?:^|\s)(@[\p{L}\p{N}._-]+)/u

/** The first `@name` in `text`, or null when it mentions nobody. */
export function firstMentionIn(text: string): null | string {
  return MENTION_IN_TEXT.exec(text)?.[1] ?? null
}
