import type { HermesConnection } from '@/global'

export const TITLEBAR_HEIGHT = 34
/** Browser touch chrome: the band (and the controls flush in it) are 44px
 *  touch targets, not Electron's compact band. */
export const TOUCH_TITLEBAR_HEIGHT = 44
/** The redesign's band: Figma's 60px header (906:19463). Electron seats its
 *  native controls in the same band (electron/window-chrome.ts). */
export const PORTAL_TITLEBAR_HEIGHT = 60
/** The redesign's compact band in the browser (a compact window: menu, title, new session; a short
 *  one, where height is the transcript's). A coarse pointer takes the touch band instead. */
export const PORTAL_COMPACT_TITLEBAR_HEIGHT = 48

export interface TitlebarBandInput {
  portal: boolean
  /** A coarse pointer on the browser host (`$touchPointer`): native Desktop never reports one. */
  touch: boolean
  /** The window is in the compact width class, or short (app/redesign/breakpoints.ts). */
  compact?: boolean
  /** The browser-hosted Desktop. Native windows keep the 60px band at every width, because Electron
   *  seats the traffic lights and caption buttons in it from the shell choice alone. */
  browserHost?: boolean
}

/** The window's titlebar band height: classic 34 (touch 44); the redesign 60, and on a compact browser
 *  window 48 (touch 44) so the band does not eat a phone's height. */
export function titlebarBandHeight({ portal, touch, compact = false, browserHost = false }: TitlebarBandInput): number {
  if (!portal) {
    return touch ? TOUCH_TITLEBAR_HEIGHT : TITLEBAR_HEIGHT
  }

  if (compact && browserHost) {
    return touch ? TOUCH_TITLEBAR_HEIGHT : PORTAL_COMPACT_TITLEBAR_HEIGHT
  }

  return PORTAL_TITLEBAR_HEIGHT
}

/** Permanent native drag target beside a top-edge tab strip. */
export const TITLEBAR_DRAG_HANDLE_WIDTH = 48
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

export const TITLEBAR_FALLBACK_WINDOW_BUTTON_X = 12
// Preserve other platforms' first paint until native window state arrives.
const TITLEBAR_NON_MAC_FALLBACK_WINDOW_BUTTON_X = 24
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
  { darwinMajor = 0, isFullscreen = false }: Pick<TitlebarChromeContext, 'darwinMajor' | 'isFullscreen'> = {}
): string {
  if (nativeOverlayWidth > 0) {
    return `${nativeOverlayWidth}px`
  }

  if (isFullscreen && darwinMajor > 0) {
    return `${TITLEBAR_EDGE_INSET}px`
  }

  return '0.75rem'
}

// Titlebar palette only. All sizing/radius/cursor/centering come from the
// shared <Button size="icon-titlebar"> (used polymorphically via asChild) —
// Button is the single source of button styling.
export const titlebarButtonClass =
  'text-muted-foreground/85 hover:bg-(--ui-control-hover-background) hover:text-foreground'

/** Shared flex shell for left/right/pane titlebar tool rows. Classic leaves
 *  `--titlebar-control-gap` unset, so its 24px buttons abut; the redesign
 *  spaces its 30px chrome buttons. */
export const titlebarToolClusterClass =
  'fixed z-70 flex flex-row items-center gap-(--titlebar-control-gap) pointer-events-auto select-none [-webkit-app-region:no-drag]'

/** Width reserved for N titlebar tool buttons, each with its trailing gap (so
 *  a cluster laid beside another keeps the same spacing between them). */
export function titlebarToolsWidthCss(toolCount: number): string {
  return `calc(${toolCount} * (var(--titlebar-control-size) + var(--titlebar-control-gap, 0px)))`
}

export const titlebarHeaderBaseClass =
  'pointer-events-none relative z-3 flex h-(--titlebar-height) w-full min-w-0 shrink-0 items-center justify-start gap-3 overflow-hidden border-b border-(--ui-stroke-tertiary) bg-(--ui-chat-surface-background) px-[max(0.75rem,var(--titlebar-content-inset,0rem))] pr-[calc(var(--titlebar-tools-right,0.75rem)+var(--titlebar-tools-width,0px)+0.75rem)]'

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

  const isMac = typeof navigator !== 'undefined' && /mac/i.test(navigator.platform)
  const fallbackX = isMac ? TITLEBAR_FALLBACK_WINDOW_BUTTON_X : TITLEBAR_NON_MAC_FALLBACK_WINDOW_BUTTON_X

  return {
    left: (windowButtonPosition?.x ?? fallbackX) + TITLEBAR_CONTROL_OFFSET_X,
    top
  }
}

/** Rebind panel measurements after titlebar nodes or their positions change. */
export const TITLEBAR_CHROME_CHANGED_EVENT = 'hermes:titlebar-chrome-changed'

/** Width of the redesign's left rail column (Figma 906:19409). */
export const PORTAL_RAIL_WIDTH = 60
/** Band controls clear macOS traffic lights (in the rail column): Figma's breadcrumb starts its
 *  Back/Forward group 90px past the rail (navigation 16292:15284, x = 150). */
export const PORTAL_BAND_LEAD_MAC = PORTAL_RAIL_WIDTH + 90
/** With no native lights on the left (Windows/Linux and the browser report no position, macOS
 *  fullscreen hides them), controls align with the sidebar and page-nav text inset (x = 80). */
export const PORTAL_BAND_LEAD_PLAIN = PORTAL_RAIL_WIDTH + 20

/** Whether native window controls occupy the top of the rail column: macOS traffic lights, unless
 *  fullscreen hides them. Windows/Linux (their caption buttons sit at the right) and the browser
 *  report no position. A connection not yet reported falls back by platform, as
 *  titlebarControlsPosition does. */
export function windowLightsInRail(
  windowButtonPosition: HermesConnection['windowButtonPosition'] | undefined,
  isFullscreen = false,
  isMac = typeof navigator !== 'undefined' && /mac/i.test(navigator.platform)
): boolean {
  if (isFullscreen || windowButtonPosition === null) {
    return false
  }

  return Boolean(windowButtonPosition) || isMac
}

/** Where the redesign band's left control cluster starts, in window px. Every left-cluster control
 *  (page Back, sidebar toggle, …) starts here via `--portal-band-start`; nothing positions itself
 *  per page. A compact window has no rail column, so the cluster moves in by the rail's width. */
export function portalBandLeadingInset(
  windowButtonPosition: HermesConnection['windowButtonPosition'] | undefined,
  isFullscreen = false,
  isMac = typeof navigator !== 'undefined' && /mac/i.test(navigator.platform),
  compact = false
): number {
  const rail = compact ? PORTAL_RAIL_WIDTH : 0

  return (
    (windowLightsInRail(windowButtonPosition, isFullscreen, isMac) ? PORTAL_BAND_LEAD_MAC : PORTAL_BAND_LEAD_PLAIN) -
    rail
  )
}
