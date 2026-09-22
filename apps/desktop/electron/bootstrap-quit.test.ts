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

// Mirrors install.sh's --stage protocol: the stage body runs in a subshell and
// its long steps (git clone, uv pip install) are children that inherit the
// stdout pipe.
function writeStageInstaller(home: string, stageBody: string) {
  fs.mkdirSync(path.join(home, 'scripts'))
  fs.writeFileSync(
    path.join(home, 'scripts/install.sh'),
    [
      '#!/bin/bash',
      'if [ "$1" = "--manifest" ]; then',
      `  echo '{"stages":[{"name":"python-deps"}]}'`,
      '  exit 0',
      'fi',
      `( ${stageBody} )`,
      `echo '{"ok":true,"stage":"python-deps"}'`,
      ''
    ].join('\n')
  )
}

const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))

async function waitForExit(pid: number, timeoutMs: number) {
  const deadline = Date.now() + timeoutMs

  while (Date.now() < deadline) {
    try {
      process.kill(pid, 0)
    } catch {
      return true
    }

    await delay(25)
  }

  return false
}

test.skipIf(process.platform === 'win32')(
  "cancel during a stage kills the stage's whole tree, including a step that ignores SIGTERM",
  async () => {
    const home = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-bootstrap-stage-cancel-'))
    const controller = new AbortController()
    const pids: Record<string, number> = {}
    let markAborted: () => void
    const aborted = new Promise<void>(resolve => (markAborted = resolve))

    writeStageInstaller(
      home,
      String.raw`sh -c "echo quick-pid=\$\$; exec sleep 30" & sh -c "trap '' TERM; echo stubborn-pid=\$\$; exec sleep 30"; wait`
    )

    const run = runBootstrap({
      installStamp: null,
      activeRoot: path.join(home, 'agent'),
      sourceRepoRoot: home,
      hermesHome: home,
      abortSignal: controller.signal,
      onEvent: event => {
        const match = event.line?.match(/^(quick|stubborn)-pid=(\d+)$/)

        if (match) {
          pids[match[1]] = Number(match[2])

          if (pids.quick && pids.stubborn) {
            controller.abort()
            markAborted()
          }
        }
      }
    })

    try {
      await Promise.race([aborted, run])
      assert.ok(pids.quick && pids.stubborn, 'the stage children never started')

      // SIGTERM reaches the quick step at once; the stubborn one only goes on
      // the SIGKILL after the runner's grace period.
      assert.ok(await waitForExit(pids.quick, 1_000), `stage child ${pids.quick} survived the cancel`)

      const outcome = await Promise.race([run, delay(6_000).then(() => 'pending' as const)])
      assert.notEqual(outcome, 'pending', 'runBootstrap still pending 6 s after cancel')
      assert.ok(await waitForExit(pids.stubborn, 1_000), `stage child ${pids.stubborn} survived the cancel`)
    } finally {
      for (const pid of Object.values(pids)) {
        try {
          process.kill(pid, 'SIGKILL')
        } catch {
          /* already exited */
        }
      }

      await run
      fs.rmSync(home, { recursive: true, force: true })
    }
  },
  15_000
)

test.skipIf(process.platform === 'win32')(
  'a cancel during a stage resolves as cancelled, not as that stage failing',
  async () => {
    const home = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-bootstrap-stage-cancel-'))
    const controller = new AbortController()

    writeStageInstaller(home, 'sleep 10')

    try {
      const result = await runBootstrap({
        installStamp: null,
        activeRoot: path.join(home, 'agent'),
        sourceRepoRoot: home,
        hermesHome: home,
        abortSignal: controller.signal,
        onEvent: event => {
          if (event.type === 'stage' && event.state === 'running') {
            controller.abort()
          }
        }
      })

      assert.deepEqual(result, { ok: false, cancelled: true })
    } finally {
      fs.rmSync(home, { recursive: true, force: true })
    }
  }
)
