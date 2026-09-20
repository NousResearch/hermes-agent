// The Hermes IDE shell's own layout state — region sizes and visibility.
//
// Deliberately NOT the app's pane-shell tree: the IDE is a fixed four-region
// anatomy (explorer | editor + browser | chat), so its layout is a small
// persisted record rather than an editable tree. The key is window-scoped by
// name — the IDE window owns this, never the primary.

import { type Codec, persistentAtom } from '@/lib/persisted'

export interface IdeLayoutState {
  browserHeight: number
  browserOpen: boolean
  chatOpen: boolean
  chatWidth: number
  explorerOpen: boolean
  explorerWidth: number
}

export const IDE_EXPLORER_MIN_WIDTH = 180
export const IDE_EXPLORER_MAX_WIDTH = 480
export const IDE_CHAT_MIN_WIDTH = 300
export const IDE_CHAT_MAX_WIDTH = 720
export const IDE_BROWSER_MIN_HEIGHT = 140
export const IDE_BROWSER_MAX_HEIGHT = 600

export const IDE_LAYOUT_DEFAULTS: IdeLayoutState = {
  browserHeight: 260,
  browserOpen: true,
  chatOpen: true,
  chatWidth: 400,
  explorerOpen: true,
  explorerWidth: 260
}

export const IDE_LAYOUT_STORAGE_KEY = 'hermes.desktop.ideLayout.v1'

function clamp(value: number, min: number, max: number) {
  // A drag delta can never be non-finite, but a corrupt persisted record must
  // not become NaN downstream — fall to the floor instead of propagating.
  return Number.isFinite(value) ? Math.min(Math.max(value, min), max) : min
}

function finiteOr(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

function decodeState(raw: string): IdeLayoutState {
  try {
    const parsed = JSON.parse(raw) as null | Partial<IdeLayoutState> | undefined

    if (!parsed || typeof parsed !== 'object') {
      return IDE_LAYOUT_DEFAULTS
    }

    return {
      browserHeight: clamp(
        finiteOr(parsed.browserHeight, IDE_LAYOUT_DEFAULTS.browserHeight),
        IDE_BROWSER_MIN_HEIGHT,
        IDE_BROWSER_MAX_HEIGHT
      ),
      browserOpen: parsed.browserOpen !== false,
      chatOpen: parsed.chatOpen !== false,
      chatWidth: clamp(finiteOr(parsed.chatWidth, IDE_LAYOUT_DEFAULTS.chatWidth), IDE_CHAT_MIN_WIDTH, IDE_CHAT_MAX_WIDTH),
      explorerOpen: parsed.explorerOpen !== false,
      explorerWidth: clamp(
        finiteOr(parsed.explorerWidth, IDE_LAYOUT_DEFAULTS.explorerWidth),
        IDE_EXPLORER_MIN_WIDTH,
        IDE_EXPLORER_MAX_WIDTH
      )
    }
  } catch {
    return IDE_LAYOUT_DEFAULTS
  }
}

const codec: Codec<IdeLayoutState> = {
  decode: decodeState,
  encode: value => JSON.stringify(value)
}

export const $ideLayout = persistentAtom<IdeLayoutState>(IDE_LAYOUT_STORAGE_KEY, IDE_LAYOUT_DEFAULTS, codec)

export function setIdeBrowserHeight(height: number) {
  const state = $ideLayout.get()

  $ideLayout.set({ ...state, browserHeight: clamp(height, IDE_BROWSER_MIN_HEIGHT, IDE_BROWSER_MAX_HEIGHT) })
}

export function setIdeChatWidth(width: number) {
  const state = $ideLayout.get()

  $ideLayout.set({ ...state, chatWidth: clamp(width, IDE_CHAT_MIN_WIDTH, IDE_CHAT_MAX_WIDTH) })
}

export function setIdeExplorerWidth(width: number) {
  const state = $ideLayout.get()

  $ideLayout.set({ ...state, explorerWidth: clamp(width, IDE_EXPLORER_MIN_WIDTH, IDE_EXPLORER_MAX_WIDTH) })
}

export function toggleIdeBrowser() {
  const state = $ideLayout.get()

  $ideLayout.set({ ...state, browserOpen: !state.browserOpen })
}

export function toggleIdeChat() {
  const state = $ideLayout.get()

  $ideLayout.set({ ...state, chatOpen: !state.chatOpen })
}

export function toggleIdeExplorer() {
  const state = $ideLayout.get()

  $ideLayout.set({ ...state, explorerOpen: !state.explorerOpen })
}
