// Pure, framework-free helpers for the Provider Manager's right-pane model
// list: alphabetical letter grouping, capability inference, and sorting.
// Unit-tested in model-list-utils.test.ts without React.

import { displayModelName } from '@/lib/model-status-label'
import type { ModelFamily } from '@/store/model-visibility'

export type CapabilityFilter = 'vision' | 'multimodal' | 'reasoning' | 'fast'

/** Stable display order for capability chips/badges. */
export const CAPABILITY_ORDER: CapabilityFilter[] = ['vision', 'multimodal', 'reasoning', 'fast']

/**
 * Distinct active-state colors per capability so the filter bar reads at a
 * glance. Classes are Tailwind utility strings (kept here, in the pure helper,
 * so the component stays presentational and the palette is unit-testable).
 */
export const CAPABILITY_ACTIVE_CLASS: Record<CapabilityFilter, string> = {
  // Stronger, theme-aware colors: /30 background + darker text for readable
  // contrast on the app's light theme (the previous /20 + light text was too faint).
  vision: 'bg-violet-500/30 text-violet-700 border border-violet-500/40',
  multimodal: 'bg-cyan-500/30 text-cyan-700 border border-cyan-500/40',
  reasoning: 'bg-amber-500/30 text-amber-700 border border-amber-500/40',
  fast: 'bg-emerald-500/30 text-emerald-700 border border-emerald-500/40'
}

/**
 * Resting (unselected) filter-chip colors. Same hue family as
 * CAPABILITY_ACTIVE_CLASS so a chip always reads as its capability — a light
 * tint at rest, the full badge color when active. Kept beside the active map so
 * the badges and the filter chips share one source of truth.
 */
export const CAPABILITY_CHIP_CLASS: Record<CapabilityFilter, string> = {
  vision: 'bg-violet-500/10 text-violet-600 border border-violet-500/25 hover:bg-violet-500/20',
  multimodal: 'bg-cyan-500/10 text-cyan-600 border border-cyan-500/25 hover:bg-cyan-500/20',
  reasoning: 'bg-amber-500/10 text-amber-600 border border-amber-500/25 hover:bg-amber-500/20',
  fast: 'bg-emerald-500/10 text-emerald-600 border border-emerald-500/25 hover:bg-emerald-500/20'
}

export interface ModelLetterGroup {
  letter: string
  families: ModelFamily[]
}

/**
 * Group collapsed model families into alphabetical sections by the first
 * letter of their display name. Digits and symbols fall under "#". Family
 * order within a letter is preserved (callers pass already-sorted families).
 * Returns an empty array when there are no families.
 */
export function groupModelsByLetter(families: readonly ModelFamily[]): ModelLetterGroup[] {
  if (families.length === 0) {
    return []
  }

  const buckets = new Map<string, ModelFamily[]>()

  for (const family of families) {
    const first = displayModelName(family.id).trim().charAt(0).toUpperCase()
    const letter = /[A-Z]/.test(first) ? first : '#'
    const bucket = buckets.get(letter) ?? []
    bucket.push(family)
    buckets.set(letter, bucket)
  }

  return [...buckets.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([letter, fams]) => ({ letter, families: fams }))
}

export interface ModelCapabilities {
  vision: boolean
  multimodal: boolean
  reasoning: boolean
  fast: boolean
}

/**
 * Infer capability flags for a model from its id + display name.
 *
 * `reasoning` and `fast` come from the backend `capabilities` map when
 * present; `vision`/`multimodal` are NOT exposed by the catalog type, so they
 * are heuristically inferred from well-known id/name patterns. The heuristic
 * is intentionally conservative and documented; if the backend later supplies
 * these fields, swap the source here with no UI change.
 */
export function inferModelCapabilities(
  modelId: string,
  displayName: string,
  backend?: { reasoning?: boolean; fast?: boolean }
): ModelCapabilities {
  const id = modelId.toLowerCase()
  const name = displayName.toLowerCase()

  const vision = /vision|image|gpt-4o|gpt-4\.1|claude-3|claude-opus|claude-sonnet|gemini|llava|qwen-vl|pixtral|mistral-(large|small).*(vision)?/i.test(
    `${id} ${name}`
  )
  const multimodal = vision || /multimodal|audio|tts|whisper|transcribe|omni/i.test(`${id} ${name}`)

  return {
    vision,
    multimodal,
    reasoning: backend?.reasoning ?? /reasoning|think|r1|o1|o3|o4|deepseek-reasoner|qwq/i.test(`${id} ${name}`),
    fast: backend?.fast ?? /-fast$/i.test(id)
  }
}

export type ModelSortMode = 'activeFirst' | 'az' | 'za'

/**
 * Sort collapsed families for display. `activeFirst` floats visible models
 * above hidden ones, each subgroup alphabetical. `az`/`za` are pure
 * alphabetical by display name. Stable: equal keys keep their input order.
 */
export function sortModelFamilies(
  families: readonly ModelFamily[],
  mode: ModelSortMode,
  isVisible: (modelId: string) => boolean
): ModelFamily[] {
  const sorted = [...families]

  if (mode === 'az' || mode === 'za') {
    sorted.sort((a, b) => {
      const cmp = displayModelName(a.id).localeCompare(displayModelName(b.id))
      return mode === 'za' ? -cmp : cmp
    })
    return sorted
  }

  // activeFirst: visible before hidden, each group alphabetical.
  const visible: ModelFamily[] = []
  const hidden: ModelFamily[] = []

  for (const family of families) {
    ;(isVisible(family.id) ? visible : hidden).push(family)
  }

  const byName = (list: ModelFamily[]) =>
    [...list].sort((a, b) => displayModelName(a.id).localeCompare(displayModelName(b.id)))

  return [...byName(visible), ...byName(hidden)]
}

/**
 * Label for the discover/update button in the model list header.
 * Empty list → "Discover models" (first-time probe); non-empty → "Update list"
 * (refresh an existing catalog). Pure so it's trivially unit-testable.
 */
export function discoverButtonLabel(
  modelCount: number,
  copy: { discoverModels: string; updateList: string }
): string {
  return modelCount === 0 ? copy.discoverModels : copy.updateList
}
