/**
 * Which input model the HUD can safely use on this system.
 *
 * The HUD's default is per-element click-through: the window ignores the mouse
 * and turns solid only where the renderer's hit test finds a control under the
 * cursor. That design rests on `setIgnoreMouseEvents(false)` actually handing
 * the input region back. On X11 it does not — once a window has ignored the
 * mouse the X server keeps hit-testing straight through it, and no later call
 * restores it. Ignore is a one-way door there, so the only policy that works
 * is never to walk through it: keep the HUD a normal solid window.
 *
 * That trade is a real loss — a solid HUD swallows clicks in the faded band
 * above the composer — so it is scoped to the Ozone backend where restore is
 * known to be broken, not to every Linux desktop.
 *
 * The windowing backend, not the login session type, is what decides this.
 * Electron 20+ defaults `ozone-platform-hint` to `auto`, which prefers a
 * native Wayland surface on a Wayland session. That is why HUD drag via
 * `setBounds` is a no-op for those users (#82851): they are already a Wayland
 * client. Native Wayland keeps click-through (main's cursor poll re-arms it).
 * X11 and explicit XWayland (`--ozone-platform=x11` / `ozone_platform_hint:
 * x11`) take the solid path.
 */

export type HudInputPolicy = 'click-through' | 'solid'

type Backend = 'wayland' | 'x11'

/**
 * The Ozone platform this process was asked for, or null when nothing asked.
 *
 * `--ozone-platform` names a backend outright and `--ozone-platform-hint`
 * (equivalently `ELECTRON_OZONE_PLATFORM_HINT`) asks for one, with `auto`
 * meaning "Wayland if the session is Wayland". The explicit switch wins over
 * the hint, and a repeated switch resolves last-one-wins, matching Chromium.
 */
function requestedOzonePlatform(env: NodeJS.ProcessEnv, argv: readonly string[]): null | string {
  let explicit: null | string = null
  let hint: null | string = null

  for (const arg of argv) {
    const match = /^--ozone-platform(-hint)?=(.+)$/.exec(arg)

    if (!match) {
      continue
    }

    if (match[1]) {
      hint = match[2].toLowerCase()
    } else {
      explicit = match[2].toLowerCase()
    }
  }

  return explicit ?? hint ?? env.ELECTRON_OZONE_PLATFORM_HINT?.toLowerCase() ?? null
}

function sessionIsWayland(env: NodeJS.ProcessEnv): boolean {
  return env.XDG_SESSION_TYPE === 'wayland' || (Boolean(env.WAYLAND_DISPLAY) && !env.DISPLAY)
}

/**
 * The backend Electron will actually use on Linux.
 *
 * Unset / `auto` follows the session: Wayland sessions get a native Wayland
 * surface (Electron 20+ default), X11 sessions stay on X11. An explicit
 * `--ozone-platform=x11` (or `desktop.ozone_platform_hint: x11`) is the
 * COSMIC always-on-top escape hatch and lands on the solid HUD path.
 */
function linuxBackend(env: NodeJS.ProcessEnv, argv: readonly string[]): Backend {
  const requested = requestedOzonePlatform(env, argv)

  if (requested === 'x11') {
    return 'x11'
  }

  if (requested === 'wayland') {
    return 'wayland'
  }

  return sessionIsWayland(env) ? 'wayland' : 'x11'
}

/**
 * The input model the HUD should use.
 *
 * macOS and Windows keep the click-through design: `setIgnoreMouseEvents(true,
 * { forward: true })` is `@platform darwin,win32`, so the renderer goes on
 * seeing the cursor while the window ignores it and can re-arm whenever the
 * pointer comes back to the bar.
 */
export function hudInputPolicy(platform: string, env: NodeJS.ProcessEnv, argv: readonly string[]): HudInputPolicy {
  if (platform !== 'linux') {
    return 'click-through'
  }

  return linuxBackend(env, argv) === 'x11' ? 'solid' : 'click-through'
}

/**
 * Whether the renderer needs a native compositor drag region for the HUD.
 *
 * Native Wayland cannot honour app-driven window positions, so it needs
 * `-webkit-app-region: drag`. X11 can use the renderer's `moveBy` path and
 * must not install that region: it consumes the pointer stream needed by the
 * renderer's immediate Ctrl+primary-button drag.
 */
export function hudUsesNativeDrag(platform: string, env: NodeJS.ProcessEnv, argv: readonly string[]): boolean {
  return platform === 'linux' && hudInputPolicy(platform, env, argv) === 'click-through'
}

/**
 * Whether a renderer-driven HUD move needs the X11 workspace-transfer bridge.
 *
 * While the pointer is grabbed the window is made temporarily visible on all
 * desktops. Releasing the grab clears that flag; Chromium's X11 backend then
 * reads `_NET_CURRENT_DESKTOP` and sends `_NET_WM_DESKTOP` for that desktop,
 * which gives KWin the same "hold window + switch desktop" behaviour as a
 * native titlebar drag. Native Wayland already delegates movement to the
 * compositor and must stay out of this path.
 */
export function hudUsesWorkspaceTransfer(
  platform: string,
  env: NodeJS.ProcessEnv,
  argv: readonly string[]
): boolean {
  return platform === 'linux' && hudInputPolicy(platform, env, argv) === 'solid'
}
