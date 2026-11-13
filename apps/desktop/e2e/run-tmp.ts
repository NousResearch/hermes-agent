/**
 * One temp root per Playwright run, removed when the runner exits.
 *
 * Playwright (artifact and stack-trace dirs), the Electron app, its backend
 * and every sandbox mkdtemp under os.tmpdir(), and a run that fails or times
 * out in setup leaves all of it behind (a clean run of two specs left eight
 * `playwright-tracing-*` dirs). Workers and the apps they launch inherit
 * TMPDIR, so everything lands here; a SIGKILLed run leaves one dir, not dozens.
 *
 * The root also names the run's backends (`HERMES_E2E_RUN_TMP` in their
 * environment): the exit hook stops them before removing it, and the next
 * runner stops the ones a killed run left (./run-tmp-reaper). Canonical, so a
 * backend's resolved HERMES_HOME still lies under it behind a symlinked tmpdir.
 *
 * Config modules load in the runner and again in each worker; only the first
 * (runner) load creates the root.
 */
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { reapAbandonedRuns, reapRunBackends, recordRunOwner } from './run-tmp-reaper'

if (!process.env.HERMES_E2E_RUN_TMP) {
  for (const abandoned of reapAbandonedRuns()) {
    console.warn(`[e2e] stopped the backends a killed run left under ${abandoned}`)
  }

  const root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'pw-')))
  recordRunOwner(root)
  process.env.HERMES_E2E_RUN_TMP = root
  process.env.TMPDIR = root
  process.env.TEMP = root
  process.env.TMP = root
  process.on('exit', () => {
    reapRunBackends(candidate => candidate === root)
    fs.rmSync(root, { recursive: true, force: true })
  })
}
