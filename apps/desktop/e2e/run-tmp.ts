/**
 * One temp root per Playwright run, removed when the runner exits.
 *
 * Playwright (artifact and stack-trace dirs), the Electron app, its backend
 * and every sandbox mkdtemp under os.tmpdir(), and a run that fails or times
 * out in setup leaves all of it behind (a clean run of two specs left eight
 * `playwright-tracing-*` dirs). Workers and the apps they launch inherit
 * TMPDIR, so everything lands here; a SIGKILLed run leaves one dir, not dozens.
 *
 * Config modules load in the runner and again in each worker; only the first
 * (runner) load creates the root.
 */
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

if (!process.env.HERMES_E2E_RUN_TMP) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'pw-'))
  process.env.HERMES_E2E_RUN_TMP = root
  process.env.TMPDIR = root
  process.env.TEMP = root
  process.env.TMP = root
  process.on('exit', () => fs.rmSync(root, { recursive: true, force: true }))
}
