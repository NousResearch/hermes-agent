import type { HermesConnection } from '@/global'

export const TITLEBAR_HEIGHT = 34
export const MACOS_TRAFFIC_LIGHTS_HEIGHT = 14
/** Titlebar tool hit target (both axes). */
export const TITLEBAR_CONTROL_SIZE = 24
/** Codicon glyph box in titlebar clusters — optical match to traffic-light row. */
export const TITLEBAR_ICON_SIZE = 13.9
export const TITLEBAR_ICON_BADGE_SCALE = 0.65
export const TITLEBAR_CONTROL_OFFSET_X = 74
export const TITLEBAR_CONTROL_HEIGHT = TITLEBAR_CONTROL_SIZE
export const TITLEBAR_CONTROLS_TOP = (TITLEBAR_HEIGHT - TITLEBAR_CONTROL_HEIGHT) / 2

/** Inline font-size for titlebar Codicons — beats unlayered codicon.css `font: 16px`. */
export function titlebarIconSizeCss(scale = 1): string {
  return `${TITLEBAR_ICON_SIZE * scale}px`
}

export const TITLEBAR_FALLBACK_WINDOW_BUTTON_X = 24
// Edge inset used when no left-side native controls take up that space —
// Windows/Linux (native overlay is on the right) and macOS fullscreen
// (traffic lights are hidden). Matches the right-cluster's 0.75rem padding.
export const TITLEBAR_EDGE_INSET = 14

// macOS Tahoe = Darwin 25+. Keep in sync with electron/titlebar-overlay-width.ts.
export const MACOS_TAHOE_DARWIN_MAJOR = 25

// macOS traffic-light row only: nudge the left toolbar cluster down to sit on
// the same optical center as the native buttons on pre-Tahoe macOS. null
// windowButtonPosition means Windows/Linux, macOS fullscreen, or Tahoe.
export const TITLEBAR_MAC_TRAFFIC_LIGHTS_Y_NUDGE = 'calc(var(--spacing) * 0.9)'

export interface TitlebarChromeContext {
  darwinMajor?: number
  isFullscreen?: boolean
  windowButtonPosition?: HermesConnection['windowButtonPosition']
}

export function titlebarControlsYNudge({
  darwinMajor = 0,
  isFullscreen = false,
  windowButtonPosition
}: TitlebarChromeContext = {}): string {
  if (isFullscreen || windowButtonPosition === null || darwinMajor >= MACOS_TAHOE_DARWIN_MAJOR) {
    return '0px'
  }

  return TITLEBAR_MAC_TRAFFIC_LIGHTS_Y_NUDGE
}

/** Right-cluster inset — WCO width when present; macOS fullscreen matches left edge inset. */
export function titlebarToolsRightCss(
  nativeOverlayWidth: number,
  { darwinMajor = 0, isFullscreen = false }: Pick<TitlebarChromeContext, 'darwinMajor' | 'isFullscreen'> = {},
  extraWidth = 0
): string {
  if (nativeOverlayWidth > 0) {
    return `${nativeOverlayWidth + extraWidth}px`
  }

  if (isFullscreen && darwinMajor > 0) {
    return extraWidth > 0 ? `calc(${TITLEBAR_EDGE_INSET}px + ${extraWidth}px)` : `${TITLEBAR_EDGE_INSET}px`
  }

  return extraWidth > 0 ? `calc(0.75rem + ${extraWidth}px)` : '0.75rem'
}

/** Buttons Electron's window-controls overlay carries (min/max/close). */
export const NATIVE_CAPTION_BUTTON_COUNT = 3

// Third-party caption buttons (DisplayFusion, Actual Window Manager, AutoHotkey
// hooks) are drawn into the same caption band but are NOT reported by Chromium's
// window-controls overlay, so the app would otherwise lay its own right-hand
// tools underneath them. On Windows such a button measures ~0.72 of a native
// one; expressing it as a ratio of the MEASURED overlay keeps the reservation
// right across DPI and UI scale without reading devicePixelRatio.
export const TITLEBAR_EXTERNAL_BUTTON_RATIO = 0.72

/** Fallback width (CSS px) per external button when the overlay is unavailable. */
export const TITLEBAR_EXTERNAL_BUTTON_FALLBACK_WIDTH = 33

/** Width (CSS px) to reserve for `count` third-party caption buttons. */
export function titlebarExternalButtonsWidth(overlayWidth: number, count: number): number {
  if (count <= 0) {
    return 0
  }

  const perButton =
    overlayWidth > 0
      ? (overlayWidth / NATIVE_CAPTION_BUTTON_COUNT) * TITLEBAR_EXTERNAL_BUTTON_RATIO
      : TITLEBAR_EXTERNAL_BUTTON_FALLBACK_WIDTH

  return Math.round(perButton * count)
}

// Titlebar palette only. All sizing/radius/cursor/centering come from the
// shared <Button size="icon-titlebar"> (used polymorphically via asChild) —
// Button is the single source of button styling.
export const titlebarButtonClass =
  'text-muted-foreground/85 hover:bg-(--ui-control-hover-background) hover:text-foreground'

/** Shared flex shell for left/right/pane titlebar tool rows — no gap; 24px buttons abut. */
export const titlebarToolClusterClass =
  'fixed z-70 flex flex-row items-center pointer-events-auto select-none [-webkit-app-region:no-drag]'

/** Width reserved for N abutting titlebar tool buttons. */
export function titlebarToolsWidthCss(toolCount: number): string {
  return `calc(${toolCount} * var(--titlebar-control-size))`
}

/**
 * Left inset that clears the left-hand tool cluster, so header content (the
 * session title) is never laid out underneath it. Only needed when no pane
 * occupies the window's left edge — when one does, the cluster sits over that
 * pane and the header needs nothing.
 */
export function titlebarContentInsetCss(controlsLeft: number, leftToolCount: number): string {
  return `calc(${controlsLeft}px + ${titlebarToolsWidthCss(leftToolCount)} + 0.75rem)`
}

// pl-/pr- rather than px-/pr-: `px-` and `pr-` both declare padding-right, so
// which one wins depends on generated-CSS order, not class order. The left
// content inset and the right tool reservation must not compete.
export const titlebarHeaderBaseClass =
  'pointer-events-none relative z-3 flex h-(--titlebar-height) w-full min-w-0 shrink-0 items-center justify-start gap-3 overflow-hidden border-b border-(--ui-stroke-tertiary) bg-(--ui-chat-surface-background) pl-[max(0.75rem,var(--titlebar-content-inset,0rem))] pr-[calc(var(--titlebar-tools-right,0.75rem)+var(--titlebar-tools-width,0px)+0.75rem)]'

// Title row inside the header — must stay in the flex truncate chain.
export const titlebarHeaderTitleClass = 'min-w-0 flex-1 overflow-hidden'

export const titlebarHeaderShadowClass =
  "after:pointer-events-none after:absolute after:left-0 after:right-0 after:top-full after:h-4 after:bg-linear-to-b after:from-(--ui-chat-surface-background) after:to-transparent after:content-['']"

export function titlebarControlsPosition(
  windowButtonPosition: HermesConnection['windowButtonPosition'] | undefined,
  isFullscreen = false
) {
  const top = Math.max(0, TITLEBAR_CONTROLS_TOP)

  // No left-side native controls to dodge:
  //   - Windows/Linux: native min/max/close render on the right via titleBarOverlay.
  //   - macOS fullscreen: traffic lights are hidden.
  // In both cases, pin the cluster to the edge with a small inset.
  if (windowButtonPosition === null || isFullscreen) {
    return { left: TITLEBAR_EDGE_INSET, top }
  }

  return {
    left: (windowButtonPosition?.x ?? TITLEBAR_FALLBACK_WINDOW_BUTTON_X) + TITLEBAR_CONTROL_OFFSET_X,
    top
  }
}
