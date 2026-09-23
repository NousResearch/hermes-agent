import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { createUpdateHandoffRuntime } from './update-handoff-runtime'

test('a CLI install without a handoff script keeps the branch-pinned manual path and clears admission', async () => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-update-runtime-'))

  try {
    let inFlight = false
    const progress: any[] = []

    const runtime = createUpdateHandoffRuntime({
      hermesHome: home,
      isWindows: true,
      isMac: false,
      isPackaged: false,
      fileExists: () => false,
      resolveUpdateRoot: () => home,
      runGit: async () => {
        assert.equal(inFlight, true)

        return { code: 0, stdout: 'feature/rollout\n', stderr: '' }
      },
      resolveHealedBranch: async (_root: string, branch: string) => branch,
      getUpdateInFlight: () => inFlight,
      setUpdateInFlight: (value: boolean) => {
        inFlight = value
      },
      emitUpdateProgress: (payload: any) => progress.push(payload),
      rememberLog: () => {}
    } as any)

    const result = await runtime.applyUpdates()

    assert.deepEqual(result, {
      ok: true,
      manual: true,
      command: 'hermes update --branch feature/rollout',
      hermesRoot: home
    })
    assert.deepEqual(progress, [{ stage: 'manual', message: result.command, percent: null }])
    assert.equal(inFlight, false)
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
})

test('POSIX release gate leaves backend processes untouched', async () => {
  const runtime = createUpdateHandoffRuntime({ isWindows: false } as any)

  assert.deepEqual(await runtime.releaseBackendLock('unused', 'updates'), { unlocked: true })
})
