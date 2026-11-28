/**
 * The redesign's renames as search aliases: the classic word someone still types, the place it
 * names now. The ⌘K palette offers the destination for any of these words, so "sessions" finds
 * Chats. One table: add a word here, never a condition at a call site.
 */

export type RenamedDestination = 'agents' | 'chats' | 'library' | 'machines' | 'routines'

export interface OldNameAlias {
  /** The classic word's code: content-free, it is what metrics record. */
  code: 'artifacts' | 'cron' | 'gateways' | 'profiles' | 'sessions'
  destination: RenamedDestination
  /** Lowercase English forms, the code's own word first. A locale's classic word is added by the caller. */
  words: readonly string[]
}

export const OLD_NAME_ALIASES: readonly OldNameAlias[] = [
  { code: 'sessions', destination: 'chats', words: ['sessions', 'session'] },
  { code: 'profiles', destination: 'agents', words: ['profiles', 'profile'] },
  { code: 'artifacts', destination: 'library', words: ['artifacts', 'artifact'] },
  {
    code: 'cron',
    destination: 'routines',
    words: ['cron', 'cron jobs', 'cron job', 'crons', 'scheduled jobs', 'scheduled job']
  },
  { code: 'gateways', destination: 'machines', words: ['gateways', 'gateway'] }
]

/** Shorter fragments match too much ("pro" is also providers); three letters already single out a word. */
const MIN_PREFIX = 3

export interface OldNameMatch {
  alias: OldNameAlias
  /** The query is one of the alias's words, not just the start of one. */
  exact: boolean
}

const normalize = (text: string) => text.trim().toLowerCase().replace(/\s+/g, ' ')

/** The alias a query is typing, if any: a whole word, or the start of one past {@link MIN_PREFIX}.
 *  `localWords` adds a locale's own classic word per code. */
export function matchOldName(
  query: string,
  localWords: Partial<Record<OldNameAlias['code'], string>> = {}
): null | OldNameMatch {
  const needle = normalize(query)

  if (needle.length < MIN_PREFIX) {
    return null
  }

  const candidates = OLD_NAME_ALIASES.map(alias => {
    const local = localWords[alias.code]

    return { alias, words: local ? [...alias.words, normalize(local)] : alias.words }
  })

  // A whole word anywhere beats a prefix of an earlier row's word.
  const exact = candidates.find(({ words }) => words.includes(needle))

  if (exact) {
    return { alias: exact.alias, exact: true }
  }

  const partial = candidates.find(({ words }) => words.some(word => word.startsWith(needle)))

  return partial ? { alias: partial.alias, exact: false } : null
}
