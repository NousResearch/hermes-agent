import { type Codec, Codecs, persistentAtom } from '@/lib/persisted'

// Per-view sort direction for the Capabilities lists — persisted so each tab
// remembers most/least-used across navigations and restarts.
export const $skillsSortDesc = persistentAtom('hermes.desktop.capabilities.skillsSortDesc', true, Codecs.bool)
export const $toolsetsSortDesc = persistentAtom('hermes.desktop.capabilities.toolsetsSortDesc', true, Codecs.bool)

// Skills-tab provenance filter — persisted so a "show me what Hermes learned"
// view survives navigation and restarts. 'all' shows everything; any other
// value keeps only rows whose `provenance` matches (null rows belong to All).
// Renderer-owned presentation state: nothing else can change it, so it lives
// with the other list prefs rather than in backend truth.
export type SkillsProvenanceFilter = 'all' | null | 'agent' | 'bundled' | 'hub'

const PROVENANCE_FILTER_VALUES = ['all', 'agent', 'bundled', 'hub'] as const

// Validates on read: a stale or hand-edited stored value falls back to All
// instead of leaking an arbitrary string into the filter comparison.
const provenanceFilterCodec: Codec<SkillsProvenanceFilter> = {
  decode: raw =>
    (PROVENANCE_FILTER_VALUES as readonly string[]).includes(raw) ? (raw as SkillsProvenanceFilter) : 'all',
  encode: value => value
}

export const $skillsProvenanceFilter = persistentAtom<SkillsProvenanceFilter>(
  'hermes.desktop.capabilities.skillsProvenanceFilter',
  'all',
  provenanceFilterCodec
)
