export type VocabularyTone = 'danger' | 'neutral' | 'notice' | 'unknown'

export type VocabularyKey =
  | 'facetDestructive'
  | 'facetRead'
  | 'facetUnclassified'
  | 'facetWrite'
  | 'hintCreate'
  | 'hintDelete'
  | 'hintDestructive'
  | 'hintIdempotent'
  | 'hintOpenWorld'
  | 'hintReadOnly'
  | 'hintUpdate'

export interface VocabularyEntry {
  key: VocabularyKey
  tone: VocabularyTone
}

export interface VocabularyTag {
  key: VocabularyKey | null
  raw: string
  tone: VocabularyTone
}

export interface VocabularyCopy {
  label: string
  long: string
}

export type VocabularyStrings = Record<VocabularyKey, VocabularyCopy>

export const VOCABULARY = {
  createHint: { key: 'hintCreate', tone: 'notice' },
  deleteHint: { key: 'hintDelete', tone: 'danger' },
  destructive: { key: 'facetDestructive', tone: 'danger' },
  destructiveHint: { key: 'hintDestructive', tone: 'danger' },
  idempotentHint: { key: 'hintIdempotent', tone: 'neutral' },
  openWorldHint: { key: 'hintOpenWorld', tone: 'notice' },
  read: { key: 'facetRead', tone: 'neutral' },
  readOnlyHint: { key: 'hintReadOnly', tone: 'neutral' },
  unclassified: { key: 'facetUnclassified', tone: 'unknown' },
  updateHint: { key: 'hintUpdate', tone: 'neutral' },
  write: { key: 'facetWrite', tone: 'notice' }
} satisfies Record<string, VocabularyEntry>

function vocabularyEntry(raw: string): undefined | VocabularyEntry {
  if (!Object.hasOwn(VOCABULARY, raw)) {
    return undefined
  }

  // SAFETY: guarded by `Object.hasOwn` on the line above.
  return VOCABULARY[raw as keyof typeof VOCABULARY]
}

export const FACET_ORDER: readonly (keyof typeof VOCABULARY)[] = ['read', 'write', 'destructive', 'unclassified']

export const HINT_ORDER: readonly (keyof typeof VOCABULARY)[] = [
  'readOnlyHint',
  'createHint',
  'updateHint',
  'deleteHint',
  'destructiveHint',
  'idempotentHint',
  'openWorldHint'
]

const MAX_LABEL = 8

export function shortenUnknown(raw: string): string {
  const stem = raw.trim().replace(/Hint$/, '')
  const word = stem.split(/[\s_\-.:/]+/).find(part => part.length > 0) ?? ''
  const head = /^[a-z]+|^[A-Z][a-z]*/.exec(word)?.[0] ?? ''

  if (head.length === 0) {
    return ''
  }

  return (head.charAt(0).toUpperCase() + head.slice(1)).slice(0, MAX_LABEL)
}

export function vocabularyTag(raw: string): VocabularyTag {
  const entry = vocabularyEntry(raw)

  if (entry) {
    return { key: entry.key, raw, tone: entry.tone }
  }

  return { key: shortenUnknown(raw).length === 0 ? 'facetUnclassified' : null, raw, tone: 'unknown' }
}

const FACET_SAYS = { destructive: 'destructiveHint' } satisfies Record<string, string>

const facetSays = (facet: string): string | undefined => (facet === 'destructive' ? FACET_SAYS.destructive : undefined)

export function hintTags(hints: readonly string[], facet?: string): VocabularyTag[] {
  const said = facet === undefined ? undefined : facetSays(facet)
  const seen = new Set(hints.filter(hint => hint !== said))
  const known = HINT_ORDER.filter(hint => seen.has(hint))
  const rest = [...seen].filter(hint => !(hint in VOCABULARY))

  return [...known, ...rest].map(vocabularyTag)
}

export function tagCopy(tag: VocabularyTag, strings: VocabularyStrings): VocabularyCopy {
  return tag.key ? strings[tag.key] : { label: shortenUnknown(tag.raw), long: tag.raw }
}
