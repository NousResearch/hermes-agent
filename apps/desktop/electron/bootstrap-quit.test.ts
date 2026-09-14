import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { runBootstrap } from './bootstrap-runner'

for (const boundary of ['resolution', 'manifest'] as const) {
  test.skipIf(process.platform === 'win32')(`quit during ${boundary} cancels bootstrap before stages`, async () => {
    const home = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-bootstrap-quit-'))
    const controller = new AbortController()
    const marker = path.join(home, 'manifest-started')
    let manifestPid: number | undefined

    fs.mkdirSync(path.join(home, 'scripts'))
    fs.writeFileSync(
      path.join(home, 'scripts/install.sh'),
      `#!/bin/bash\nprintf started > "$HERMES_HOME/manifest-started"\nprintf 'manifest-pid=%s\\n' "$$"\nwhile :; do :; done\n`
    )

    try {
      const result = await runBootstrap({
        installStamp: null,
        activeRoot: path.join(home, 'agent'),
        sourceRepoRoot: home,
        hermesHome: home,
        abortSignal: controller.signal,
        onEvent: event => {
          if (boundary === 'resolution' && event.line?.includes('using local')) {
            controller.abort()
          }

          const match = event.line?.match(/^manifest-pid=(\d+)$/)

          if (match) {
            manifestPid = Number(match[1])
            controller.abort()
          }
        }
      })

      assert.equal(result.ok, false)
      assert.equal(result.cancelled, true)
      assert.equal(fs.existsSync(marker), boundary === 'manifest')

      if (manifestPid) {
        assert.throws(() => process.kill(manifestPid!, 0))
      }
    } finally {
      controller.abort()

      if (manifestPid) {
        try {
          process.kill(manifestPid, 'SIGKILL')
        } catch {
          /* already exited */
        }
      }

      fs.rmSync(home, { recursive: true, force: true })
    }
  })
}
