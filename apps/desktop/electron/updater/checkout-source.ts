import { execFile } from 'node:child_process'
import { promisify } from 'node:util'

import { buildDesktopBackendEnv } from '../backend-env'
import { hiddenWindowsChildOptions } from '../windows-child-options'

export interface SourceUpdate {
  channel: 'main' | 'stable' | 'canary'
  latestTag?: string
  targetSha?: string
  error?: string
  message?: string
}

export interface SourceUpdateProbe {
  python: string | null
  git: string
  updateRoot: string
  hermesHome: string
}

const execute: typeof execFile.__promisify__ = promisify(execFile)

export function sourceUpdateEnvironment(updateRoot: string, hermesHome: string): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = {
    ...process.env,
    ...buildDesktopBackendEnv(),
    HERMES_HOME: hermesHome,
    HERMES_INSTALL_ROOT: updateRoot
  }

  delete env.HERMES_RUNTIME_DIR

  return env
}

/** Python owns config identity and publication checks; this bridge only transports them. */
export async function readSourceUpdate(probe: SourceUpdateProbe): Promise<SourceUpdate> {
  if (!probe.python) {
    throw new Error('No Python interpreter is available to check the source update channel.')
  }

  const result: { stdout: string; stderr: string } = await execute(
    probe.python,
    ['-m', 'hermes_cli.source_releases', '--install-root', probe.updateRoot, '--git', probe.git],
    hiddenWindowsChildOptions({
      cwd: probe.updateRoot,
      env: sourceUpdateEnvironment(probe.updateRoot, probe.hermesHome),
      encoding: 'utf8',
      timeout: 360000,
      maxBuffer: 1024 * 1024
    })
  )

  const selection: SourceUpdate = JSON.parse(result.stdout) as SourceUpdate

  if (!['main', 'stable', 'canary'].includes(selection.channel)) {
    throw new Error('The source update check returned an invalid channel.')
  }

  if (
    selection.channel !== 'main' &&
    !selection.error &&
    (!selection.latestTag || !/^[0-9a-f]{40}$/.test(selection.targetSha ?? ''))
  ) {
    throw new Error('The source update check returned no release commit.')
  }

  return selection
}
