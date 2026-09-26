import fs from 'node:fs'
import path from 'node:path'

import { sourceDeclaresServe } from './backend-command'
import { execProbe, isTimeoutError, PROBE_TIMEOUT_MS } from './backend-probes'

interface ServeCandidate {
  command?: string | null
  root?: string
  args?: string[]
  env?: Record<string, string>
  shell?: boolean
  label?: string
}

type BackendSubcommand = 'serve' | 'dashboard' | null

// Does the resolved runtime understand the `serve` subcommand? The desktop
// spawns `hermes serve`; some older runtimes only have `dashboard`, so main.ts
// uses the legacy `dashboard --no-open` form after verifying it exists.
//
// Fast path: read the runtime's own dashboard.py (instant, covers managed
// installs, dev checkouts, and the Windows venv). Fallback: probe the CLI once
// (covers a bare `hermes` resolved from PATH with no known source root). Result
// is cached per resolved runtime after a command succeeds. Failed checks are
// evicted so a repaired runtime can be retried without restarting Desktop.
//
// One cache per desktop runtime context; source inspection precedes a CLI probe.
export function createBackendServeSupportResolver(hermesHome: string, rememberLog: (message: string) => void) {
  const cache = new Map<string, Promise<BackendSubcommand>>()

  return async function backendSubcommand(backend: ServeCandidate): Promise<BackendSubcommand> {
    if (!backend || !backend.command) {
      return null
    }

    const key = `${backend.command}::${backend.root || ''}`

    if (cache.has(key)) {
      return cache.get(key)!
    }

    const pending = (async () => {
      let supportsServe: boolean | null = null

      if (backend.root) {
        try {
          const src = await fs.promises.readFile(
            path.join(backend.root, 'hermes_cli', 'subcommands', 'dashboard.py'),
            'utf8'
          )

          supportsServe = sourceDeclaresServe(src)
        } catch {
          supportsServe = null // source unreadable — fall through to the probe
        }
      }

      const prefix = backend.args && backend.args[0] === '-m' ? backend.args.slice(0, 2) : []

      const probeOptions = {
        cwd: backend.root || undefined,
        env: { ...process.env, HERMES_HOME: hermesHome, ...(backend.env || {}) },
        timeout: PROBE_TIMEOUT_MS,
        stdio: 'ignore' as const,
        shell: Boolean(backend.shell),
        windowsHide: true
      }

      if (supportsServe === null) {
        try {
          // Same cold-Windows Python-startup class as the runtime probes
          // (#61764/#72632/#72707): `serve --help` imports at least as much as
          // `hermes --version` (~10.5s measured cold), and a false negative here
          // must not be cached as a missing command. Share the probe budget.
          await execProbe(backend.command, [...prefix, 'serve', '--help'], probeOptions)
          supportsServe = true
        } catch (err) {
          if (isTimeoutError(err)) {
            if (cache.get(key) === pending) {
              cache.delete(key)
            }

            rememberLog(`[backend] \`serve\` probe timed out for ${backend.label || key}`)

            return null
          }

          supportsServe = false
        }
      }

      if (supportsServe) {
        rememberLog(`[backend] \`serve\` supported for ${backend.label || key}`)

        return 'serve'
      }

      // An older runtime may have dashboard but not serve. A version-only CLI
      // has neither, so never turn a failed serve check into an invalid spawn.
      try {
        await execProbe(backend.command, [...prefix, 'dashboard', '--help'], probeOptions)
        rememberLog(`[backend] \`serve\` unsupported → routing via legacy \`dashboard\` for ${backend.label || key}`)

        return 'dashboard'
      } catch (err) {
        if (cache.get(key) === pending) {
          cache.delete(key)
        }

        rememberLog(
          `[backend] could not verify \`serve\` or \`dashboard\` for ${backend.label || key}` +
            (isTimeoutError(err) ? ' (dashboard probe timed out)' : '')
        )

        return null
      }
    })()

    // Publish the promise before yielding; late results never overwrite a newer entry.
    cache.set(key, pending)

    return pending
  }
}
