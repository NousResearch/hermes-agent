export const PREVIEW_RAIL_MIN_WIDTH = '18rem'
// Keep the docked rail itself bounded like an editor side panel. Responsive
// presets size the inner preview frame, not this PaneShell track, so a stale
// persisted desktop/ultrawide width cannot let the preview consume the chat.
export const PREVIEW_RAIL_MAX_WIDTH = '42rem'

const INTRINSIC = `clamp(${PREVIEW_RAIL_MIN_WIDTH}, 36vw, 32rem)`

// Track for <Pane id="preview">. Folds the intrinsic clamp with a min-floor
// against --chat-min-width so the default preview never squeezes chat. Manual
// drag / responsive presets can intentionally widen beyond this default up to
// PREVIEW_RAIL_MAX_WIDTH.
export const PREVIEW_RAIL_PANE_WIDTH = `min(${INTRINSIC}, max(0rem, calc(100vw - var(--pane-chat-sidebar-width) - var(--pane-file-browser-width, 0rem) - var(--chat-min-width))))`

export const PREVIEW_RATIO_MIN_WIDTH_PX = 288
export const PREVIEW_RATIO_MAX_VIEWPORT_FRACTION = 0.9

export const PREVIEW_RATIO_PRESETS = [
  { id: 'fold', label: 'Fold', ratioLabel: '6:5', ratio: 6 / 5 },
  { id: 'iphone', label: 'iPhone', ratioLabel: '9:16', ratio: 9 / 16 },
  { id: 'desktop', label: 'Desktop', ratioLabel: '16:9', ratio: 16 / 9 },
  { id: 'ultrawide', label: 'Ultrawide', ratioLabel: '21:9', ratio: 21 / 9 }
] as const

export type PreviewRatioPreset = (typeof PREVIEW_RATIO_PRESETS)[number]

export function maxPreviewRatioWidth(innerWidth = typeof window === 'undefined' ? 1280 : window.innerWidth): number {
  return Math.max(PREVIEW_RATIO_MIN_WIDTH_PX, Math.round(innerWidth * PREVIEW_RATIO_MAX_VIEWPORT_FRACTION))
}

export function previewWidthForRatio({
  height,
  maxWidth = maxPreviewRatioWidth(),
  minWidth = PREVIEW_RATIO_MIN_WIDTH_PX,
  ratio
}: {
  height: number
  maxWidth?: number
  minWidth?: number
  ratio: number
}): number {
  const safeHeight = Number.isFinite(height) && height > 0 ? height : 720
  const desired = Math.round(safeHeight * ratio)
  const hi = Math.max(minWidth, maxWidth)

  return Math.round(Math.min(hi, Math.max(minWidth, desired)))
}
