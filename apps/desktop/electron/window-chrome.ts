// The titlebar band each app window's native chrome is laid out for. The
// renderer draws the band (src/app/shell/titlebar.ts, same numbers); main
// places the macOS traffic lights and sizes the Windows/Linux caption overlay
// to match it, from the shell choice in the machine's shared UI preferences
// (ui-preferences.ts) — read at window creation so the first paint is right,
// and re-applied when the choice changes.

/** Shared-preferences key of the redesign opt-out (src/store/shell-variant.ts). */
export const SHELL_CHOICE_KEY = 'hermes.desktop.agentShell.v1'

/** Classic: Electron's compact band. */
export const CLASSIC_TITLEBAR_HEIGHT = 34
/** Redesign: Figma's 60px header (906:19463). */
export const PORTAL_TITLEBAR_HEIGHT = 60
/** Native traffic-light cluster height on macOS. */
export const MACOS_TRAFFIC_LIGHTS_HEIGHT = 14

/** Windows that render the app shell follow the shell choice; the popped-out
 *  Browser keeps the classic layout in every mode (as the renderer does). */
export type WindowChromeKind = 'browser' | 'chat'

/** The redesign is on unless the user rolled it back (`'false'`), exactly as
 *  the renderer's `$portalPreferred` decodes the same key. */
export function portalShellPreferred(values: Readonly<Record<string, string>> | null | undefined): boolean {
  return values?.[SHELL_CHOICE_KEY] !== 'false'
}

export function windowTitlebarHeight({ kind, portal }: { kind: WindowChromeKind; portal: boolean }): number {
  return portal && kind === 'chat' ? PORTAL_TITLEBAR_HEIGHT : CLASSIC_TITLEBAR_HEIGHT
}

/** Traffic lights vertically centred in the band. The redesign seats them in
 *  its 60px rail column at Figma's inset (906:19415: first light's left edge
 *  ≈ x 23, centre y ≈ 30). */
export function windowButtonPosition(titlebarHeight: number): { x: number; y: number } {
  return {
    x: titlebarHeight === PORTAL_TITLEBAR_HEIGHT ? 23 : 12,
    y: Math.round(titlebarHeight / 2 - MACOS_TRAFFIC_LIGHTS_HEIGHT / 2)
  }
}
