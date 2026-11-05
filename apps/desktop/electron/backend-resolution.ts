/**
 * HERMES_DESKTOP_IGNORE_EXISTING=1 (`hermes desktop --ignore-existing`) keeps
 * Desktop off installed runtimes (ACTIVE_HERMES_ROOT, or the install a
 * published user-bin launcher reports). Backend resolution
 * then falls through to bootstrap-needed, which shows the first-run choice
 * (connect a remote, or install) instead of starting a local serve.
 *
 * The bundled payload, HERMES_DESKTOP_HERMES_ROOT, an unpackaged checkout and
 * HERMES_DESKTOP_HERMES resolve before this rung and are not affected.
 *
 * A runtime this launch installed is always used. Skipping it would send the
 * post-install re-resolve (and every later resolve) back to the installer.
 *
 * A light (remote-only) artifact has no local runtime by definition, so the
 * flag is always in effect there: an installation already present on the
 * machine must never be adopted.
 */
import { INSTALL_STAMP, type InstallStamp } from './install-stamp'

export function createInstalledRuntimeGate(
  env: NodeJS.ProcessEnv,
  log: (message: string) => void,
  stamp: Readonly<InstallStamp> | null = INSTALL_STAMP
) {
  let installedThisLaunch = false
  const remoteOnly = stamp?.payload === 'light'

  return {
    /** The ACTIVE_HERMES_ROOT rung: probe the installed runtime unless the flag skips it. */
    async resolve<T>(root: string, probe: () => Promise<T | null>): Promise<T | null> {
      if (!installedThisLaunch && (remoteOnly || env.HERMES_DESKTOP_IGNORE_EXISTING === '1')) {
        log(
          remoteOnly
            ? `[bootstrap] remote-only artifact; skipping the installed runtime at ${root}`
            : `[bootstrap] HERMES_DESKTOP_IGNORE_EXISTING=1; skipping the installed runtime at ${root}`
        )

        return null
      }

      return probe()
    },

    /** Re-resolve after a successful bootstrap; the new runtime is exempt from the flag. */
    afterInstall<T>(resolveBackend: () => Promise<T>): Promise<T> {
      installedThisLaunch = true

      return resolveBackend()
    }
  }
}
