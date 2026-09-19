import { detectRemoteDisplay, isWslEnvironment } from './bootstrap-platform'

const AUTO_WAYLAND_FLAG = '--hermes-wslg-auto-wayland'
const X11_FALLBACK_FLAG = '--hermes-wslg-x11-fallback'
export const WSLG_X11_FALLBACK_EXIT_CODE = 76

function ozoneHint(argv: readonly string[], env: NodeJS.ProcessEnv): { value: string | undefined; explicit: boolean } {
  for (let index = argv.length - 1; index >= 0; index -= 1) {
    const arg = argv[index]

    if (arg.startsWith('--ozone-platform-hint=')) {
      return { value: arg.slice('--ozone-platform-hint='.length), explicit: true }
    }

    if (arg === '--ozone-platform-hint') {
      return { value: argv[index + 1], explicit: true }
    }
  }

  return env.ELECTRON_OZONE_PLATFORM_HINT
    ? { value: env.ELECTRON_OZONE_PLATFORM_HINT, explicit: true }
    : { value: undefined, explicit: false }
}

function hasOzonePlatform(argv: readonly string[]): boolean {
  return argv.some(arg => arg === '--ozone-platform' || arg.startsWith('--ozone-platform='))
}

// Ozone is selected before application JavaScript. Never appendSwitch here:
// that leaves the browser on X11 while GPU children receive Wayland.
export function wslgLaunchArgs(
  argv: readonly string[],
  env: NodeJS.ProcessEnv,
  platform: NodeJS.Platform,
  isWsl = isWslEnvironment(env, platform)
): string[] | null {
  const displayEnv = { ...env, HERMES_DESKTOP_DISABLE_GPU: undefined }

  if (platform !== 'linux' || !isWsl || !env.WAYLAND_DISPLAY || detectRemoteDisplay({ env: displayEnv, platform })) {
    return null
  }

  if (hasOzonePlatform(argv)) {
    return null
  }

  const hint = ozoneHint(argv, env)
  const backend = hint.value === 'x11' ? 'x11' : 'wayland'

  return backend === 'wayland' && !hint.explicit
    ? [...argv, `--ozone-platform=${backend}`, AUTO_WAYLAND_FLAG]
    : [...argv, `--ozone-platform=${backend}`]
}

/**
 * Re-exec once on X11 when the automatically selected WSLg Wayland renderer
 * cannot launch. Explicit ozone choices never receive the auto marker, so the
 * fallback cannot override a user or config hint.
 */
export function wslgX11FallbackArgs(argv: readonly string[]): string[] | null {
  if (!argv.includes(AUTO_WAYLAND_FLAG) || argv.includes(X11_FALLBACK_FLAG) || !hasOzonePlatform(argv)) {
    return null
  }

  const next: string[] = []

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index]

    if (arg === AUTO_WAYLAND_FLAG || arg === '--ozone-platform') {
      if (arg === '--ozone-platform') {
        index += 1
      }

      continue
    }

    if (arg.startsWith('--ozone-platform=')) {
      continue
    }

    next.push(arg)
  }

  return [...next, '--ozone-platform=x11', X11_FALLBACK_FLAG]
}

export function shouldFallbackWslgRenderer(details: { reason?: string; exitCode?: number | string | undefined }): boolean {
  return details.reason === 'launch-failed' && String(details.exitCode) === '1002'
}
