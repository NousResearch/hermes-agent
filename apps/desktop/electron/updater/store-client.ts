import { APPINSTALLER_CHECK_TIMEOUT_MS, runAppInstallerChecker } from '../appinstaller-checker'

import { type StoreResult, StoreStrategy, type StoreStrategyDeps } from './store'

export interface StoreClientDeps extends Omit<StoreStrategyDeps, 'run'> {
  python: string
  script: string
  sitePackages: string
  env: NodeJS.ProcessEnv
  windowHandle: () => Buffer | null
  run?: typeof runAppInstallerChecker
}

function parseResult(code: number, stdout: string): StoreResult {
  const unknown = (error: string): StoreResult => ({ available: null, packages: [], ok: false, error })
  let value: unknown

  try {
    value = JSON.parse(stdout)
  } catch {
    return unknown(`Microsoft Store returned invalid JSON (exit ${code})`)
  }

  if (!value || typeof value !== 'object') {
    return unknown('Microsoft Store returned no result')
  }

  const row = value as Partial<StoreResult>

  if (
    (code !== 0 && code !== 2) ||
    row.ok !== true ||
    typeof row.available !== 'boolean' ||
    !Array.isArray(row.packages) ||
    !row.packages.every(item => typeof item === 'string')
  ) {
    return unknown(typeof row.error === 'string' ? row.error : `Microsoft Store request failed (exit ${code})`)
  }

  return { ok: true, available: row.available, packages: row.packages }
}

export function createStoreStrategy(deps: StoreClientDeps): StoreStrategy {
  const run = deps.run ?? runAppInstallerChecker

  return new StoreStrategy({
    ...deps,
    run: async mode => {
      const env: NodeJS.ProcessEnv = { ...deps.env, PYTHONPATH: deps.sitePackages }
      delete env.PYTHONHOME
      delete env.VIRTUAL_ENV
      delete env.PYTHONEXECUTABLE
      delete env.__PYVENV_LAUNCHER__
      const handle = deps.windowHandle()

      if (mode !== 'check' && !handle) {
        throw new Error('A desktop window is required for Microsoft Store consent')
      }

      const args = ['--mode', mode]

      if (handle) {
        const hwnd = handle.length === 8 ? handle.readBigUInt64LE().toString() : handle.readUInt32LE().toString()
        args.push('--hwnd', hwnd)
      }

      const result = await run(deps.python, deps.script, {
        args,
        env,
        timeoutMs: mode === 'check' ? APPINSTALLER_CHECK_TIMEOUT_MS : 1_830_000,
        waitForExit: mode !== 'check'
      })

      if (mode !== 'check' && result.code !== 0) {
        const failed = parseResult(result.code, result.stdout)

        return { ...failed, ok: false, error: failed.error || `Microsoft Store ${mode} did not complete` }
      }

      return parseResult(result.code, result.stdout)
    }
  })
}
