import * as fs from 'node:fs'
import * as path from 'node:path'

import { seedInstall, UPDATE_ROOT } from './harness'

/**
 * Refuse to run where an update could reach the developer's own Hermes. The
 * cells run `hermes update --gateway`, which stops gateways it can see; CI
 * runners have none, and locally the suite must run in its own PID namespace
 * (e.g. `bwrap --dev-bind / / --unshare-pid --proc /proc ...` with the user
 * bus hidden), where PID 1 is not the host's init.
 */
function assertIsolated(): void {
  if (process.env.CI === 'true' || process.env.HERMES_E2E_UPDATE_ISOLATED === '1') {
    return
  }

  let init = ''

  try {
    init = fs.readFileSync('/proc/1/comm', 'utf8').trim()
  } catch {
    // no procfs: not Linux, the suite does not run here anyway
  }

  if (init === 'systemd' || init === 'init') {
    throw new Error(
      'The Desktop update suite runs real `hermes update --gateway` hand-offs. Run it on CI, or locally inside ' +
        'its own PID namespace with the user bus hidden (PID 1 here is the host init).'
    )
  }
}

/**
 * Build the real install every spec starts from. HERMES_E2E_UPDATE_REUSE=1
 * keeps an existing one (local iteration only: it must have been built from
 * the current HEAD, which is what the install clones).
 */
export default function globalSetup(): void {
  assertIsolated()

  if (process.env.HERMES_E2E_UPDATE_REUSE === '1' && fs.existsSync(path.join(UPDATE_ROOT, 'install.json'))) {
    return
  }

  const started = Date.now()
  const facts = seedInstall()
  console.log(
    `[update-e2e] seeded install at ${facts.checkout} (${facts.headSha}) in ${Math.round((Date.now() - started) / 1000)} s`
  )
}
