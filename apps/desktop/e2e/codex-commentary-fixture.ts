/** Seeding helper adapted from royalaid's #79727, using only a disposable home
 * and an explicit/repository interpreter, never an installed user's Hermes venv. */
import { execFile } from 'node:child_process'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { promisify } from 'node:util'

import { buildAppEnv, type Sandbox } from './fixtures'

const REPO_ROOT = path.resolve(import.meta.dirname, '../../..')
const run = promisify(execFile)

export function commentaryAppEnv(sandbox: Sandbox): Record<string, string> {
  const suffix = process.platform === 'win32' ? ['Scripts', 'python.exe'] : ['bin', 'python']
  const python =
    process.env.HERMES_DESKTOP_PYTHON ??
    ['.venv', 'venv'].map(dir => path.join(REPO_ROOT, dir, ...suffix)).find(candidate => fs.existsSync(candidate))
  if (!python) throw new Error('Create the repository venv or provide HERMES_DESKTOP_PYTHON for this E2E')
  return buildAppEnv(sandbox, { HERMES_DESKTOP_PYTHON: python, PYTHONPATH: REPO_ROOT })
}

export async function seedCodexCommentarySession(sandbox: Sandbox, sessionId: string): Promise<void> {
  const env = commentaryAppEnv(sandbox)
  await run(
    env.HERMES_DESKTOP_PYTHON,
    [
      path.join(import.meta.dirname, 'seed-codex-commentary-session.py'),
      '--hermes-home',
      sandbox.hermesHome,
      '--state-db',
      path.join(sandbox.hermesHome, 'state.db'),
      '--session-id',
      sessionId
    ],
    { cwd: REPO_ROOT, env, timeout: 60_000 }
  )
}
