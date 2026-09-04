/**
 * Stage the pm payload into build/agent-payload for a BUNDLED desktop
 * build. Runs `hermes pm bundle` with the repo's own venv interpreter —
 * the payload carries the repo snapshot (committed state), the tool
 * store + facts, and a relocatable venv on the pinned interpreter.
 *
 * Plain `npm run dist` (no payload) still works: before-build.mjs writes
 * an external:true stub and the app resolves a runtime at first launch.
 */
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

const desktopRoot = path.join(import.meta.dirname, '..')
const repoRoot = path.join(desktopRoot, '..', '..')
const payloadDir = path.join(desktopRoot, 'build', 'agent-payload')

function venvPython() {
  for (const venv of ['.venv', 'venv']) {
    for (const rel of [
      ['Scripts', 'python.exe'],
      ['bin', 'python']
    ]) {
      const candidate = path.join(repoRoot, venv, ...rel)

      if (fs.existsSync(candidate)) {
        return candidate
      }
    }
  }

  return null
}

const python = venvPython()

if (!python) {
  console.error('stage-payload: no repo venv found — run `uv sync` in the repo root first')
  process.exit(1)
}

console.log(`stage-payload: ${python} -m pm.cli bundle --out ${payloadDir}`)
execFileSync(python, ['-m', 'pm.cli', 'bundle', '--out', payloadDir], {
  cwd: repoRoot,
  stdio: 'inherit',
  env: { ...process.env, PYTHONUTF8: '1' }
})
