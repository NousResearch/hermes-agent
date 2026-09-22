import { atom, computed, type ReadableAtom, type WritableAtom } from 'nanostores'

import { type Codec, persistentAtom } from '@/lib/persisted'
import type { SidebarRowMeta } from '@/store/layout'
import type { ToolViewMode } from '@/store/tool-view'

// Interface mode: does this window show the machinery, or just the
// conversation? Two answers. ADVANCED is the app as it has always been — every
// preference the user set, honoured. SIMPLE is chat-first: the standing
// developer instrumentation (statusbar, profile rail, terminal / files / review
// panes, technical tool payloads, the artifacts / scheduled-jobs rows) rests
// out of the way. Mode changes what is SHOWN, never what Hermes can do: every
// route still answers ⌘K, every pane still answers its keybind and the agent's
// `focus_pane`.
//
// A mode is a RESOLVER INPUT, not a preset that writes preferences. Three
// layers, one precedence, one place:
//
//   effective(surface) = sessionReveal ?? policy[mode] ?? userPreference
//
// `policy.advanced` is empty, so Advanced falls through to the user's own
// atoms by construction and nothing is ever written on an existing install.
// Simple SHADOWS a preference rather than overwriting it, so switching back
// restores the exact prior setup with nothing to snapshot or repair. A toggle
// pressed while a surface is shadowed (⌃` in Simple) lands in the session
// layer — the terminal appears now, and the next launch is Simple's resting
// state again — so a mode is a default, not a lock, and no session reveal can
// pollute a preference the user set in the other mode.
//
// Consumers never learn the word "mode". A preference store wraps its own atom
// in `modeBound` and exports the EFFECTIVE atom under the name everything
// already reads; list surfaces (titlebar tools, sidebar nav rows) tag items
// with a `tier` and run one `shownInMode` filter at the site they already
// filter. Both keep the policy table below the single owner of the answer.

export type InterfaceMode = 'advanced' | 'simple'

/** Picker order — the quiet option first. */
export const INTERFACE_MODES: readonly InterfaceMode[] = ['simple', 'advanced']

export const DEFAULT_INTERFACE_MODE: InterfaceMode = 'advanced'

const INTERFACE_MODE_STORAGE_KEY = 'hermes.desktop.interfaceMode.v1'

// Advanced is the ABSENCE of a mode: encoded as "no key" so a user who never
// touched the picker never gains a record, and clearing the key means Advanced.
const modeCodec: Codec<InterfaceMode> = {
  decode: raw => (raw === 'simple' ? 'simple' : DEFAULT_INTERFACE_MODE),
  encode: mode => (mode === DEFAULT_INTERFACE_MODE ? null : mode)
}

export const $interfaceMode = persistentAtom<InterfaceMode>(
  INTERFACE_MODE_STORAGE_KEY,
  DEFAULT_INTERFACE_MODE,
  modeCodec
)

export function setInterfaceMode(mode: InterfaceMode) {
  $interfaceMode.set(mode)
}

/** The ⌘K row and the rebindable `view.toggleSimpleMode` action. */
export function toggleSimpleMode() {
  setInterfaceMode($interfaceMode.get() === 'simple' ? 'advanced' : 'simple')
}

// ---------------------------------------------------------------------------
// Policy: every preference a mode may shadow, typed by what it holds.
// ---------------------------------------------------------------------------

/** What a policy may look at when its answer depends on the install. */
export interface ModeContext {
  profileCount: number
}

export interface ModePolicy {
  fileBrowserOpen: boolean
  hideCodeDiffs: boolean
  profileRailVisible: boolean
  reasoningCollapsedByDefault: boolean
  reviewOpen: boolean
  sidebarRowMeta: SidebarRowMeta[]
  statusbarVisible: boolean
  terminalOpen: boolean
  toolViewMode: ToolViewMode
}

export type ModePolicyKey = keyof ModePolicy

type PolicyEntry<K extends ModePolicyKey> = ((context: ModeContext) => ModePolicy[K]) | ModePolicy[K]

type PolicyTable = { readonly [K in ModePolicyKey]?: PolicyEntry<K> }

const SIMPLE_POLICY: PolicyTable = {
  // Hide-style panes rest closed; their keybinds and the agent still reveal them.
  fileBrowserOpen: false,
  // Inline diffs are the review pane's job; the changed-files summary stays.
  hideCodeDiffs: true,
  // Hide a door only when there is somewhere else to go: with the statusbar
  // (and its fallback profile dropdown) gone, a second profile makes the rail
  // the only way to switch, so it stays for multi-profile installs.
  profileRailVisible: context => context.profileCount > 1,
  // A quiet "Thought for Ns" row instead of a live reasoning stream.
  reasoningCollapsedByDefault: true,
  reviewOpen: false,
  // What was said and when — cost, tokens, PR and profile chips are readouts.
  sidebarRowMeta: ['preview', 'updated'],
  statusbarVisible: false,
  terminalOpen: false,
  // Product summaries; the technical payload view is the instrumentation itself.
  toolViewMode: 'product'
}

// Advanced is deliberately `{}`: anything added here would stop being the
// user's own preference the moment they opened the picker.
const POLICY: Record<InterfaceMode, PolicyTable> = {
  advanced: {},
  simple: SIMPLE_POLICY
}

/** Does this mode have an opinion about the surface at all? */
const shadows = (key: ModePolicyKey, mode: InterfaceMode) => key in POLICY[mode]

// The install facts a policy may consult. Fed by the app shell (profile roster),
// kept off this module's imports so preference stores can depend on it without
// dragging the session graph in.
export const $modeContext = atom<ModeContext>({ profileCount: 1 })

export function setModeContext(patch: Partial<ModeContext>) {
  $modeContext.set({ ...$modeContext.get(), ...patch })
}

// ---------------------------------------------------------------------------
// Session layer: reveals made while a surface is shadowed. In memory only, so
// a restart returns to the mode's resting state; cleared on mode change so a
// reveal cannot leak into the other mode.
// ---------------------------------------------------------------------------

const $modeReveals = atom<Partial<ModePolicy>>({})

$interfaceMode.listen(() => $modeReveals.set({}))

function resolvePolicy<K extends ModePolicyKey>(
  key: K,
  mode: InterfaceMode,
  context: ModeContext
): ModePolicy[K] | undefined {
  const entry = POLICY[mode][key] as PolicyEntry<K> | undefined

  return typeof entry === 'function' ? entry(context) : entry
}

/**
 * Wrap a preference in the mode resolver. Reads give the EFFECTIVE value;
 * writes route to the session layer while the surface is shadowed and to the
 * preference otherwise, so every existing toggle keeps working unchanged.
 */
export function modeBound<K extends ModePolicyKey>(
  key: K,
  $preference: ReadableAtom<ModePolicy[K]>,
  setPreference: (value: ModePolicy[K]) => void
): WritableAtom<ModePolicy[K]> {
  const $effective = computed(
    [$interfaceMode, $modeReveals, $modeContext, $preference],
    (mode, reveals, context, preference): ModePolicy[K] => {
      const policy = resolvePolicy(key, mode, context)

      if (policy === undefined) {
        return preference
      }

      return key in reveals ? (reveals[key] as ModePolicy[K]) : policy
    }
  )

  // A computed store updates itself through its own `set`, so the routed write
  // cannot live on it. Mirror it into a plain atom and route that atom's `set`.
  const $bound = atom<ModePolicy[K]>($effective.get())
  const mirror = $bound.set

  $effective.subscribe(value => mirror(value as ModePolicy[K]))

  $bound.set = value => {
    if (!shadows(key, $interfaceMode.get())) {
      setPreference(value)
    } else if (!layoutIntent) {
      $modeReveals.set({ ...$modeReveals.get(), [key]: value })
    }
  }

  return $bound
}

// A layout pick opens and closes the panes it places through these same
// setters. That is LAYOUT intent, not "show me this": while a surface is
// shadowed the mode already decides what rests, so inside `asLayoutIntent`
// a shadowed write is dropped rather than turned into a session reveal — and
// Simple still never writes a preference.
let layoutIntent = false

export function asLayoutIntent(run: () => void) {
  layoutIntent = true

  try {
    run()
  } finally {
    layoutIntent = false
  }
}

const shadowedCache = new Map<ModePolicyKey, ReadableAtom<boolean>>()

/** Is the surface's value the mode's, not the user's? Settings rows read it to
 *  say who set their value. Cached per key so `useStore` keeps one subscription. */
export function $modeShadowed(key: ModePolicyKey): ReadableAtom<boolean> {
  let cached = shadowedCache.get(key)

  if (!cached) {
    cached = computed($interfaceMode, mode => shadows(key, mode))
    shadowedCache.set(key, cached)
  }

  return cached
}

// ---------------------------------------------------------------------------
// Tiers: list surfaces declare which items are instrumentation.
// ---------------------------------------------------------------------------

/** A tier names the one mode an item belongs to; an item with no tier shows in every mode. */
export type InterfaceTier = InterfaceMode

export interface Tiered {
  tier?: InterfaceTier
}

/** Predicate for the filter a list already runs: `items.filter(shownInMode(mode))`. */
export function shownInMode(mode: InterfaceMode): (item: Tiered) => boolean {
  return item => item.tier === undefined || item.tier === mode
}

/** The one derived boolean for surfaces that are a single element, not a list. */
export const $showsAdvancedChrome: ReadableAtom<boolean> = computed($interfaceMode, mode => mode === 'advanced')
