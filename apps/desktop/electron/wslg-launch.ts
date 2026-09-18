import { detectRemoteDisplay, isWslEnvironment } from './bootstrap-platform'

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

  if (argv.some(arg => arg === '--ozone-platform' || arg.startsWith('--ozone-platform='))) {
    return null
  }

  const hintArg = argv.findLast(arg => arg.startsWith('--ozone-platform-hint='))
  const hint = hintArg?.split('=')[1] ?? env.ELECTRON_OZONE_PLATFORM_HINT
  const backend = hint === 'x11' ? 'x11' : 'wayland'

  return [...argv, `--ozone-platform=${backend}`]
}
