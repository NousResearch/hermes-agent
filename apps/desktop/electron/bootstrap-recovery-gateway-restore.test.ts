// handOffWindowsBootstrapRecovery() is entangled with real Electron app init
// (app.quit, process spawning, module-load side effects), so — following the
// same convention as backend-dial-claim.test.ts — this reads main.ts's own
// source and asserts the expected call pattern in a windowed slice, rather
// than importing and invoking the function directly.

import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

const here = path.dirname(fileURLToPath(import.meta.url))
const mainSource = fs.readFileSync(path.join(here, 'main.ts'), 'utf8').replace(/\r\n/g, '\n')

describe('handOffWindowsBootstrapRecovery gateway restore on failed hand-off (#70337 sibling gap)', () => {
  it('restores the gateways stopped by releaseBackendLockForUpdate when the hand-off is not viable', () => {
    const anchor = mainSource.indexOf(
      '[bootstrap] recovery hand-off not viable, staying alive:'
    )

    expect(anchor).toBeGreaterThan(-1)

    const body = mainSource.slice(anchor, anchor + 500)

    // releaseBackendLockForUpdate() (called earlier in this function) took
    // every profile's gateway down via `gateway stop --all` on Windows.
    // applyUpdates()'s four abort paths all restore via
    // startGatewaysAfterUpdateAbort() — this sibling branch must too, or a
    // failed recovery hand-off strands every profile's gateway stopped.
    expect(body).toContain('IS_WINDOWS')
    expect(body).toContain('startGatewaysAfterUpdateAbort(venvHermesShimPath(updateRoot))')
    expect(body).toContain('return false')
  })

  it('the restore call sits inside handOffWindowsBootstrapRecovery, not a different function', () => {
    const fnStart = mainSource.indexOf('async function handOffWindowsBootstrapRecovery(reason) {')
    const anchor = mainSource.indexOf(
      '[bootstrap] recovery hand-off not viable, staying alive:'
    )

    expect(fnStart).toBeGreaterThan(-1)
    expect(anchor).toBeGreaterThan(fnStart)
    // Sanity bound: the failure branch must be well within this one function
    // (it's the last branch before the success path, ~90 lines in), not
    // accidentally matched inside some unrelated function far down the file.
    expect(anchor - fnStart).toBeLessThan(4500)
  })
})
