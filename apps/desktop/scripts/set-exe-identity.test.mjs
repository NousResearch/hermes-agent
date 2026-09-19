import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { test } from 'vitest'

import { stampExeIdentity } from './set-exe-identity.mjs'

function makeDesktopRoot() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-exe-identity-'))
  fs.mkdirSync(path.join(root, 'assets'))
  fs.writeFileSync(path.join(root, 'assets', 'icon.ico'), 'icon')
  const exe = path.join(root, 'Hermes.exe')
  fs.writeFileSync(exe, 'exe')
  return { exe, root }
}

test('retries transient rcedit commit failures with bounded backoff', async () => {
  const { exe, root } = makeDesktopRoot()
  const delays = []
  let attempts = 0

  try {
    await stampExeIdentity(exe, root, {
      rcedit: async () => {
        attempts += 1
        if (attempts < 3) throw new Error('Unable to commit changes')
      },
      sleep: async delay => delays.push(delay)
    })

    assert.equal(attempts, 3)
    assert.deepEqual(delays, [500, 1000])
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('stops retrying after the bounded rcedit retry budget is exhausted', async () => {
  const { exe, root } = makeDesktopRoot()
  let attempts = 0

  try {
    await assert.rejects(
      stampExeIdentity(exe, root, {
        rcedit: async () => {
          attempts += 1
          throw new Error('Unable to commit changes')
        },
        sleep: async () => {}
      }),
      /Unable to commit changes/
    )
    assert.equal(attempts, 4)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})
