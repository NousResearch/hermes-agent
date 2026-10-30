/**
 * What the renderer's host can do, by name.
 *
 * The same renderer runs in Electron (native Desktop) and in a browser tab the
 * gateway serves (the Webapp). Call sites ask for the capability they gate,
 * never for the host, so a reader sees why a branch exists and a later change
 * to one capability does not ride along on every other host check.
 *
 * Each capability belongs to exactly one host. The host is read on every call
 * (not once at import): the browser bridge writes the host marker after some
 * modules have evaluated, and `isBrowserHostedDesktop` already answers both
 * before and after it.
 */

import { isBrowserHostedDesktop } from '@/lib/platform'

type Host = 'browser' | 'native'

export const HOST_CAPABILITIES = Object.freeze({
  /** Mouse back/forward buttons are the app's to handle; a browser tab keeps its own history. */
  appMouseHistoryButtons: 'native',
  /** The browser opens its own context menu, which the app menu must cancel or let through. */
  browserContextMenu: 'browser',
  /** Shortcuts whose native chord the browser takes first (new tab, close tab) default to another chord. */
  browserShortcutDefaults: 'browser',
  /** Writes go through Electron's clipboard IPC before `navigator.clipboard`. */
  clipboardIpc: 'native',
  /** Gateway connection management (switch, connect, recover) lives on this host. */
  gatewayManagement: 'native',
  /** Links may open in the in-app browser pane instead of the OS browser. */
  inAppLinks: 'native',
  /** A composer press keeps editor focus (iOS blurs before submit runs). */
  keepFocusOnPress: 'browser',
  /** Cut, Copy, Paste and spelling rows come from Electron main. */
  mainProcessEditVerbs: 'native',
  /** Narrow overlays get titlebar chrome: Hide names the zone, the pane pads to the titlebar. */
  narrowOverlayChrome: 'browser',
  /** The default project folder is picked with the OS dialog. */
  nativeFolderPicker: 'native',
  /** Host files are read through the Electron bridge, not the gateway's REST routes. */
  nativeHostFiles: 'native',
  /** Main can enumerate the OS windows around this one (read_window_below); a browser tab cannot. */
  osWindowEnumeration: 'native',
  /** Preferences are stored per serving origin (and per profile) in this tab. */
  originScopedPrefs: 'browser',
  /** The profile rail shows its squares (drag, hold-to-recolor) instead of always condensing. */
  profileRailSquares: 'native',
  /** A failed terminal spawn arms a retry (the shell runs on the server). */
  retryTerminalSpawn: 'browser',
  /** The roster names profiles this one gateway does not serve. */
  servedProfileRoster: 'browser',
  /** The connection is the serving origin's, and its primary is the profile it launched as. */
  servingOriginConnection: 'browser',
  /** Code fences size to their longest line and scroll sideways inside the block. */
  sidewaysCodeFences: 'browser',
  /** The tab title mirrors session status. */
  tabTitleStatus: 'browser',
  /** The startup profile is the tab's own `?profile=`, so every pick is remembered. */
  tabStartupProfile: 'browser',
  /** A tap outside a revealed narrow overlay closes it. */
  tapOutsideDismiss: 'browser',
  /** Controls and the titlebar grow to touch size (44px). */
  touchSizedControls: 'browser',
  /** A session token may ride a URL (plugin socket, file download link). */
  urlTokenAuth: 'native',
  /** This machine can serve the Webapp (Settings → Web access). */
  webAccessHost: 'native',
  /** User-facing copy names the Webapp. */
  webappCopy: 'browser',
  /** The preview pane can mount an Electron `<webview>`; otherwise it uses an iframe. */
  webviewPreview: 'native',
  /** The window has a native material and opacity surface (glass, translucency). */
  windowMaterial: 'native'
} as const satisfies Record<string, Host>)

export type HostCapability = keyof typeof HOST_CAPABILITIES

export function hostHas(capability: HostCapability): boolean {
  return (HOST_CAPABILITIES[capability] === 'browser') === isBrowserHostedDesktop()
}
