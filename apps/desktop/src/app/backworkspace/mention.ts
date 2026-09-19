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
