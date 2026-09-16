import { atom } from 'nanostores'

import { chatSurfaceRoot } from '@/app/chat/surface-vars'
import { persistBoolean, persistString, storedBoolean, storedString } from '@/lib/storage'

const POPOUT_STORAGE_KEY = 'hermes.desktop.composerPopout.window.v1'
const POPOUT_GESTURES_ENABLED_STORAGE_KEY = 'hermes.desktop.composerPopout.gesturesEnabled'
const ZONES_STORAGE_KEY = 'hermes.desktop.composerPopout.zones.v1'
const LEGACY_ENABLED_KEY = 'hermes.desktop.composerPopout.enabled'
const LEGACY_POSITION_KEY = 'hermes.desktop.composerPopout.position'

/** Viewport-relative bottom/right insets keep the resting corner stable as a draft grows. */
export interface PopoutPosition {
  bottom: number
  right: number
}

export interface ComposerPopoutState {
  poppedOut: boolean
  position: PopoutPosition
}

export const POPOUT_WIDTH_REM = 19.5
export const POPOUT_ESTIMATED_HEIGHT = 56
const DEFAULT_POSITION: PopoutPosition = { bottom: 24, right: 24 }
const EDGE_MARGIN = 8
const gesturesEnabledAtLoad = storedBoolean(POPOUT_GESTURES_ENABLED_STORAGE_KEY, true)

const isPosition = (value: unknown): value is PopoutPosition => {
  const position = value as Partial<PopoutPosition> | null

  return Number.isFinite(position?.bottom) && Number.isFinite(position?.right)
}

function readState(key: string): unknown {
  try {
    return JSON.parse(storedString(key) || 'null')
  } catch {
    return null
  }
}

function isState(value: unknown): value is ComposerPopoutState {
  const state = value as Partial<ComposerPopoutState> | null

  return typeof state?.poppedOut === 'boolean' && isPosition(state.position)
}

function load(): ComposerPopoutState {
  const saved = readState(POPOUT_STORAGE_KEY)

  if (isState(saved)) {
    return { ...saved, poppedOut: gesturesEnabledAtLoad && saved.poppedOut }
  }

  // Upgrade existing floats without resurrecting a stale singleton preference
  // after the user has already expressed per-zone intent.
  const zones = readState(ZONES_STORAGE_KEY)

  if (zones && typeof zones === 'object') {
    const states = Object.values(zones).filter(isState)
    const state = states.find(zone => zone.poppedOut) ?? states[0]

    return {
      poppedOut: gesturesEnabledAtLoad && Boolean(state?.poppedOut),
      position: state?.position ?? DEFAULT_POSITION
    }
  }

  const position = readState(LEGACY_POSITION_KEY)

  return {
    poppedOut: gesturesEnabledAtLoad && storedBoolean(LEGACY_ENABLED_KEY, false),
    position: isPosition(position) ? position : DEFAULT_POSITION
  }
}

/** One floating placement per window; drafts and runtimes remain session-owned. */
export const $composerPopout = atom<ComposerPopoutState>(load())
export const $composerPopoutGesturesEnabled = atom(gesturesEnabledAtLoad)

const persist = () => persistString(POPOUT_STORAGE_KEY, JSON.stringify($composerPopout.get()))

export function setComposerPoppedOut(poppedOut: boolean) {
  const state = $composerPopout.get()

  if (state.poppedOut !== poppedOut) {
    $composerPopout.set({ ...state, poppedOut })
  }

  persist()
}

export function setComposerPopoutGesturesEnabled(value: boolean) {
  $composerPopoutGesturesEnabled.set(value)
  persistBoolean(POPOUT_GESTURES_ENABLED_STORAGE_KEY, value)

  if (!value) {
    setComposerPoppedOut(false)
  }
}

export interface PopoutSize {
  height: number
  width: number
}

export interface PopoutBounds {
  bottom: number
  left: number
  right: number
  top: number
}

interface SetPositionOptions {
  area?: PopoutBounds
  persist?: boolean
  size?: PopoutSize
}

const clampRange = (value: number, lo: number, hi: number) => Math.min(Math.max(value, lo), Math.max(lo, hi))
const rootFontSize = () => parseFloat(getComputedStyle(document.documentElement).fontSize) || 16

/** The dock target stays in the owning pane even though the float spans the window. */
export function readPopoutBounds(composer: Element | null): PopoutBounds | undefined {
  const el = chatSurfaceRoot(composer)?.querySelector('[data-slot="composer-bounds"]')

  if (!el) {
    return undefined
  }

  const { bottom, height, left, right, top, width } = el.getBoundingClientRect()

  return width > 0 && height > 0 ? { bottom, left, right, top } : undefined
}

/** Clamp the whole box, not just its anchor. Never write a hidden tab's measurement into shared intent. */
export function clampPopoutPosition(
  { bottom, right }: PopoutPosition,
  size?: PopoutSize,
  area?: PopoutBounds
): PopoutPosition {
  const width = size?.width || POPOUT_WIDTH_REM * rootFontSize()
  const height = size?.height || POPOUT_ESTIMATED_HEIGHT
  const { innerHeight: vh, innerWidth: vw } = window
  const a = area ?? { bottom: vh, left: 0, right: vw, top: 0 }

  return {
    bottom: clampRange(bottom, vh - a.bottom + EDGE_MARGIN, vh - a.top - height - EDGE_MARGIN),
    right: clampRange(right, vw - a.right + EDGE_MARGIN, vw - a.left - width - EDGE_MARGIN)
  }
}

/** Drag updates stay in memory; only the release writes storage. */
export function setComposerPopoutPosition(
  position: PopoutPosition,
  { area, persist: shouldPersist, size }: SetPositionOptions = {}
): PopoutPosition {
  const next = clampPopoutPosition(position, size, area)
  const state = $composerPopout.get()

  if (state.position.bottom !== next.bottom || state.position.right !== next.right) {
    $composerPopout.set({ ...state, position: next })
  }

  if (shouldPersist) {
    persist()
  }

  return next
}
