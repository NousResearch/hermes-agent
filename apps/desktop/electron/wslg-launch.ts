import { detectRemoteDisplay, isWslEnvironment } from './bootstrap-platform'

// Ozone is selected before application JavaScript. Never appendSwitch here:
// that leaves the browser on X11 while GPU children receive Wayland.
export function wslgLaunchArgs(
  argv: readonly string[],
  env: NodeJS.ProcessEnv,
  platform: NodeJS.Platform,
  isWsl = isWslEnvironment(env, platform),
  electronFlags: readonly string[] = []
): string[] | null {
  const displayEnv = { ...env, HERMES_DESKTOP_DISABLE_GPU: undefined }

  // Remote/forwarded displays stay on the software-rendering path. A local
  // Wayland session — native Linux or WSLg — otherwise defaults to XWayland
  // unless the platform is on argv before application JavaScript.
  if (platform !== 'linux' || detectRemoteDisplay({ env: displayEnv, platform })) {
    return null
  }

  const nativeWayland = env.XDG_SESSION_TYPE === 'wayland' || Boolean(env.WAYLAND_DISPLAY)
  const wslgWayland = Boolean(isWsl && env.WAYLAND_DISPLAY)

  if (!nativeWayland && !wslgWayland) {
    return null
  }

  if (hasOzonePlatform(argv)) {
    return null
  }

  // desktop.electron_flags are not always on argv yet (.desktop entry, source
  // launches). An explicit platform there wins over the automatic default, and
  // must be on the relaunch command line or appendSwitch applies it too late.
  const flagged = explicitOzonePlatform(electronFlags)

  if (flagged !== null) {
    return flagged ? [...argv, `--ozone-platform=${flagged}`] : null
  }

  const hintArg = findOzoneHint(argv) ?? findOzoneHint(electronFlags)
  const hint = hintArg ?? env.ELECTRON_OZONE_PLATFORM_HINT
  const backend = hint === 'x11' ? 'x11' : 'wayland'

  return [...argv, `--ozone-platform=${backend}`]
}

function hasOzonePlatform(args: readonly string[]): boolean {
  return args.some(arg => arg === '--ozone-platform' || arg.startsWith('--ozone-platform='))
}

function explicitOzonePlatform(args: readonly string[]): string | null {
  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i] ?? ''

    if (arg === '--ozone-platform') {
      const next = args[i + 1]

      return next && !next.startsWith('-') ? next : ''
    }

    if (arg.startsWith('--ozone-platform=')) {
      return arg.slice('--ozone-platform='.length)
    }
  }

  return null
}

function findOzoneHint(args: readonly string[]): string | undefined {
  const hintArg = args.findLast(arg => arg.startsWith('--ozone-platform-hint='))

  return hintArg?.slice('--ozone-platform-hint='.length)
}
