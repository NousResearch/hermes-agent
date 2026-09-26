/**
 * Layout presets — the FancyZones treatment.
 *
 * A preset is a CONTRIBUTION (`area: 'layouts'`, `data: LayoutNode`): the app
 * registers its bundled presets as `source: 'core'`, plugins register theirs
 * exactly the same way, and user-saved presets round-trip through localStorage
 * and re-register as `source: 'user'`. The picker (renderer.tsx) reads one
 * uniform list via `useContributions('layouts')`.
 */

import { registry } from '@/contrib/registry'
import { readJson, writeJson, writeKey } from '@/lib/storage'
import { asLayoutIntent, type Tiered } from '@/store/interface-mode'

import { allPaneIds, findGroupOfPane, isLayoutNode, type LayoutNode } from './model'
import { $dismissedPanes, $hiddenTreePanes, $layoutTree, applyTree, markActivePreset } from './store'

export const LAYOUTS_AREA = 'layouts'

/**
 * A preset is the tree plus what the tree cannot say. `resting` names the
 * panes it places but leaves CLOSED — every other pane it places opens on
 * apply, so a preset states what is on screen. `tier` names the one mode
 * whose shelf carries it. Both stay off `data`, which every consumer reads as
 * a bare `LayoutNode`.
 */
export interface LayoutPresetSpec extends Tiered {
  id: string
  order: number
  resting?: readonly string[]
  title: string
  tree: LayoutNode
}

const specs = new Map<string, Pick<LayoutPresetSpec, 'resting' | 'tier'>>()

export function registerBundledPresets(bundled: readonly LayoutPresetSpec[]) {
  for (const spec of bundled) {
    rememberSpec(spec.id, spec)
  }

  return registry.registerMany(
    bundled.map(({ id, order, title, tree }) => ({ id, area: LAYOUTS_AREA, title, order, data: tree }))
  )
}

function rememberSpec(id: string, { resting, tier }: Pick<LayoutPresetSpec, 'resting' | 'tier'>) {
  specs.set(id, { resting, tier })
}

/** User decks have no tier, so every shelf carries them. */
export const layoutPresetTier = (id: string) => specs.get(id)?.tier

const NO_RESTING: ReadonlySet<string> = new Set()

export const layoutPresetResting = (id: string): ReadonlySet<string> => {
  const resting = specs.get(id)?.resting

  return resting ? new Set(resting) : NO_RESTING
}

// v2: v1 presets predate semantic placement (see store.ts) — retire them.
const USER_KEY = 'hermes.desktop.layoutPresets.v2'

writeKey('hermes.desktop.layoutPresets.v1', null)

interface StoredPreset {
  name: string
  resting?: string[]
  tree: LayoutNode
}

const userDisposers = new Map<string, () => void>()

function loadUserPresets(): Record<string, StoredPreset> {
  const parsed = readJson<Record<string, StoredPreset>>(USER_KEY) ?? {}
  const out: Record<string, StoredPreset> = {}

  for (const [id, preset] of Object.entries(parsed)) {
    if (preset && typeof preset.name === 'string' && isLayoutNode(preset.tree)) {
      out[id] = preset
      rememberSpec(id, preset)
    }
  }

  return out
}

function persistUserPresets(presets: Record<string, StoredPreset>) {
  writeJson(USER_KEY, presets)
}

function registerUserPreset(id: string, preset: StoredPreset) {
  userDisposers.get(id)?.()
  userDisposers.set(
    id,
    registry.register({ id, area: LAYOUTS_AREA, source: 'user', title: preset.name, data: preset.tree })
  )
}

// Register persisted user presets at module load.
const userPresets = loadUserPresets()

for (const [id, preset] of Object.entries(userPresets)) {
  registerUserPreset(id, preset)
}

/** Save any tree as a named user preset (and make it active). A deck saved
 *  from the live layout remembers which of its panes were closed, so applying
 *  it later restores what was on screen, not just where things sat. */
export function saveLayoutPresetTree(name: string, tree: LayoutNode, resting: readonly string[] = []): string | null {
  const trimmed = name.trim()

  if (!tree || !trimmed) {
    return null
  }

  const id = `user-${
    trimmed
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '') || Date.now().toString(36)
  }`

  userPresets[id] = { name: trimmed, tree, resting: [...resting] }
  persistUserPresets(userPresets)
  rememberSpec(id, userPresets[id])
  registerUserPreset(id, userPresets[id])
  markActivePreset(id)

  return id
}

/** Save the CURRENT tree as a named user preset (and make it active). A pane
 *  rests when it is closed — hidden, dismissed or folded to its rail — not
 *  when it merely sits behind a sibling tab. */
export function saveCurrentLayoutAs(name: string) {
  const tree = $layoutTree.get()

  if (tree) {
    const hidden = $hiddenTreePanes.get()
    const dismissed = $dismissedPanes.get()
    const rests = (id: string) => hidden.has(id) || dismissed.has(id) || Boolean(findGroupOfPane(tree, id)?.minimized)

    saveLayoutPresetTree(name, tree, allPaneIds(tree).filter(rests))
  }
}

export function deleteUserPreset(id: string) {
  if (!(id in userPresets)) {
    return
  }

  delete userPresets[id]
  specs.delete(id)
  persistUserPresets(userPresets)
  userDisposers.get(id)?.()
  userDisposers.delete(id)
}

export const isUserPreset = (id: string) => id in userPresets

/** Apply a preset's tree (deep-cloned so live edits never mutate the preset),
 *  opening what it places and resting what it says to. In Simple the mode
 *  already decides what rests, so the open/close side of a layout pick yields
 *  to it instead of surfacing shadowed panes for the session. */
export function applyLayoutPreset(id: string, tree: LayoutNode) {
  asLayoutIntent(() => applyTree(structuredClone(tree), id, specs.get(id)?.resting))
}
