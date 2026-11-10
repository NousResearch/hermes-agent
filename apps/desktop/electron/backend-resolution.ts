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
 */
export function createInstalledRuntimeGate(env: NodeJS.ProcessEnv, log: (message: string) => void) {
  let installedThisLaunch = false

  // The last runtime this launch proved usable. A single failed probe must not
  // retract that proof: the probe is a bounded health check (15s), and a resolve
  // runs again on every gateway restart, exactly when the box is busiest. Both
  // recorded misfires (desktop.log 2026-11-07 11:49:45, 2026-11-08 01:00:50)
  // demoted a healthy install to "no usable Hermes install" -> first-run setup
  // on the strength of one timed-out probe.
  let proved: { root: string; runtime: unknown } | null = null

  const skipsInstalledRuntimes = () =>
    env.HERMES_DESKTOP_IGNORE_EXISTING === '1' && !installedThisLaunch

  return {
    /** The ACTIVE_HERMES_ROOT rung: probe the installed runtime unless the flag skips it. */
    async resolve<T>(root: string, probe: () => Promise<T | null>): Promise<T | null> {
      if (skipsInstalledRuntimes()) {
        log(`[bootstrap] HERMES_DESKTOP_IGNORE_EXISTING=1; skipping the installed runtime at ${root}`)

        return null
      }

      const usable: T | null = await probe()

      if (usable) {
        proved = { root, runtime: usable }

        return usable
      }

      if (proved && proved.root === root) {
        log(
          `[bootstrap] the --version probe for ${root} failed this time, but this launch already ` +
            'proved it usable; keeping it instead of first-run setup'
        )

        return proved.runtime as T
      }

      return null
    },

    /** Re-resolve after a successful bootstrap; the new runtime is exempt from the flag. */
    afterInstall<T>(resolveBackend: () => Promise<T>): Promise<T> {
      installedThisLaunch = true

      return resolveBackend()
    },

    /** True when the operator asked for the first-run choice; a present install must not override it. */
    skipsInstalledRuntimes
  }
}
