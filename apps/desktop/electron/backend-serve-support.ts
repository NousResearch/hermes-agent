import fs from 'node:fs'
import path from 'node:path'

import { sourceDeclaresServe } from './backend-command'
import { execProbe, PROBE_TIMEOUT_MS } from './backend-probes'

interface ServeCandidate {
  command?: string | null
  root?: string
  args?: string[]
  env?: Record<string, string>
  shell?: boolean
  label?: string
}

// One cache per desktop runtime context; source inspection precedes a CLI probe.
export function createBackendServeSupportResolver(hermesHome: string, rememberLog: (message: string) => void) {
  const cache = new Map<string, Promise<boolean>>()

  return async function backendSupportsServe(backend: ServeCandidate): Promise<boolean> {
    if (!backend || !backend.command) {
      return true
    }

    const key = `${backend.command}::${backend.root || ''}`

    if (cache.has(key)) {
      return cache.get(key)!
    }

    const pending = (async () => {
      let supported: boolean | null = null

      if (backend.root) {
        try {
          const src = fs.readFileSync(path.join(backend.root, 'hermes_cli', 'subcommands', 'dashboard.py'), 'utf8')
          supported = sourceDeclaresServe(src)
        } catch {
          supported = null // source unreadable — fall through to the probe
        }
      }

      if (supported === null) {
        try {
          const prefix = backend.args && backend.args[0] === '-m' ? backend.args.slice(0, 2) : []
          // Same cold-Windows Python-startup class as the runtime probes
          // (#61764/#72632/#72707): `serve --help` imports at least as much as
          // `hermes --version` (~10.5s measured cold), and a false negative here
          // is cached for the process lifetime, silently routing a modern
          // runtime through the legacy `dashboard` form. Share the probe budget
          // and its timeout-only retry instead of a thinner local bound.
          await execProbe(backend.command, [...prefix, 'serve', '--help'], {
            cwd: backend.root || undefined,
            env: { ...process.env, HERMES_HOME: hermesHome, ...(backend.env || {}) },
            timeout: PROBE_TIMEOUT_MS,
            stdio: 'ignore',
            // `.cmd`/`.bat` shim backends carry shell: true in their descriptor
            // (see resolveHermesBackend step 4); execFileSync of a .cmd without
            // shell throws EINVAL on modern Node, which the catch below would
            // mis-cache as "serve unsupported" for the process lifetime.
            shell: Boolean(backend.shell),
            windowsHide: true
          })
          supported = true
        } catch {
          supported = false
        }
      }

      rememberLog(
        `[backend] \`serve\` ${supported ? 'supported' : 'unsupported → routing via legacy `dashboard`'} for ${backend.label || key}`
      )

      return supported
    })()

    // Publish the promise before yielding; late results never overwrite a newer entry.
    cache.set(key, pending)

    return pending
  }
}
