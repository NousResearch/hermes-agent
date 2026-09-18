// One map from every wire value to the word we show.
//
// The provider sends four facets and seven behaviour hints, all as bare strings,
// and it may add an eighth tomorrow. Every surface that renders a facet or a hint
// reads this file, so a tag is the same word in the tool row, the disclosure and
// the dialog header — and an unrecognised value degrades to a readable tag
// instead of a blank one or a crash.
//
// The map holds i18n KEYS, not English. `tagCopy` resolves a tag against
// `t.connectorsPage.vocabulary`; an unknown value carries its own shortened text
// instead of a key, because there is no translation for a word we have never
// seen.

/** How loud the tag reads. Components map these onto the badge variants. */
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

/** A resolved tag. Exactly one of `key` / `label` is set: a known value carries a
 *  translatable key, an unknown one carries the shortened raw value. */
export interface VocabularyTag {
  key: VocabularyKey | null
  /** The raw wire value, for the unknown branch and for `data-` attributes. */
  raw: string
  /** Set only when `key` is null. Never blank. */
  shortLabel: string | null
  tone: VocabularyTone
}

export interface VocabularyCopy {
  /** Eight characters at most, so seven tags fit one wrapping line at 390px. */
  label: string
  /** One plain sentence, for the disclosure and the tooltip. */
  long: string
}

export type VocabularyStrings = Record<VocabularyKey, VocabularyCopy>

/** The four facets the provider classifies a tool with. */
export const FACET_VOCABULARY = {
  destructive: { key: 'facetDestructive', tone: 'danger' },
  read: { key: 'facetRead', tone: 'neutral' },
  unclassified: { key: 'facetUnclassified', tone: 'unknown' },
  write: { key: 'facetWrite', tone: 'notice' }
} satisfies Record<string, VocabularyEntry>

/** The seven behaviour hints. They overlap the facets on purpose — a facet is the
 *  provider's verdict, a hint is what the tool declared about itself. */
export const HINT_VOCABULARY = {
  createHint: { key: 'hintCreate', tone: 'notice' },
  deleteHint: { key: 'hintDelete', tone: 'danger' },
  destructiveHint: { key: 'hintDestructive', tone: 'danger' },
  idempotentHint: { key: 'hintIdempotent', tone: 'neutral' },
  openWorldHint: { key: 'hintOpenWorld', tone: 'notice' },
  readOnlyHint: { key: 'hintReadOnly', tone: 'neutral' },
  updateHint: { key: 'hintUpdate', tone: 'neutral' }
} satisfies Record<string, VocabularyEntry>

/** The order facet chips and the facet summary render in: least to most costly. */
export const FACET_ORDER = ['read', 'write', 'destructive', 'unclassified'] as const

/** The order hint chips render in. Hints the connector does not use are dropped. */
export const HINT_ORDER = [
  'readOnlyHint',
  'createHint',
  'updateHint',
  'deleteHint',
  'destructiveHint',
  'idempotentHint',
  'openWorldHint'
] as const

const MAX_LABEL = 8

/** A word for a value we ship no vocabulary for. `search_repositories_v2` reads
 *  as `Search`, `mutateHint` as `Mutate`. Truncated hard at eight characters so
 *  an unknown tag can never widen the row it sits in. */
export function shortenUnknown(raw: string): string {
  const stem = raw.trim().replace(/Hint$/, '')
  const word = stem.split(/[\s_\-.:/]+/).find(part => part.length > 0) ?? ''
  // camelCase and PascalCase both arrive from providers; take the leading run.
  // Anything that does not start with a plain letter has no readable word in it,
  // and the caller turns an empty answer into the translated "Unknown" tag.
  const head = /^[a-z]+|^[A-Z][a-z]*/.exec(word)?.[0] ?? ''

  if (head.length === 0) {
    return ''
  }

  return (head.charAt(0).toUpperCase() + head.slice(1)).slice(0, MAX_LABEL)
}

function tagFor(raw: string, table: Record<string, VocabularyEntry>, unknownKey: VocabularyKey): VocabularyTag {
  const entry = table[raw]

  if (entry) {
    return { key: entry.key, raw, shortLabel: null, tone: entry.tone }
  }

  const shortLabel = shortenUnknown(raw)

  // A value so mangled it shortens to nothing still needs a word, and "Unknown"
  // is exactly what it means. Falling through to the key keeps it translatable.
  if (shortLabel.length === 0) {
    return { key: unknownKey, raw, shortLabel: null, tone: 'unknown' }
  }

  return { key: null, raw, shortLabel, tone: 'unknown' }
}

export function facetTag(facet: string): VocabularyTag {
  return tagFor(facet, FACET_VOCABULARY, 'facetUnclassified')
}

export function hintTag(hint: string): VocabularyTag {
  return tagFor(hint, HINT_VOCABULARY, 'facetUnclassified')
}

/** Every hint a tool declares, in the fixed order, with unknown values kept at
 *  the end so a new provider value never reshuffles the familiar tags. */
export function hintTags(hints: readonly string[]): VocabularyTag[] {
  const seen = new Set(hints)
  const known = HINT_ORDER.filter(hint => seen.has(hint))
  const rest = hints.filter(hint => !(hint in HINT_VOCABULARY))

  return [...known, ...rest].map(hintTag)
}

/** The words for one tag. Unknown values borrow their own shortened text for the
 *  label and their raw value for the sentence, so nothing is ever blank. */
export function tagCopy(tag: VocabularyTag, strings: VocabularyStrings): VocabularyCopy {
  if (tag.key) {
    return strings[tag.key]
  }

  return { label: tag.shortLabel ?? tag.raw, long: tag.raw }
}
