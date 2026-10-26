import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { handoffResultPath, type ReleaseHolder } from './handoff-result'
import { releaseHolderConsentPath, releaseHolderRetry } from './release-holder-consent'
import { reportHandoffResult } from './update-handoff-report'

const SOGOU: ReleaseHolder = {
  pid: 4242,
  name: 'SogouCloud.exe',
  exe: 'D:\\Program Files (x86)\\SogouInput\\16.7.0.4912\\SogouCloud.exe'
}

function writeHeldResult(home: string) {
  fs.writeFileSync(
    handoffResultPath(home),
    JSON.stringify({
      ok: true,
      exit_code: 0,
      manual: true,
      message: 'the Desktop app could not be replaced because SogouCloud.exe (pid 4242) is using its folder.',
      branch: 'main',
      warnings: ['build: the Desktop app folder is in use by SogouCloud.exe (pid 4242)'],
      release_holders: [SOGOU],
      finished_at: Math.floor(Date.now() / 1000)
    })
  )
}

async function reportWithAnswer(home: string, response: number) {
  const boxes: any[] = []
  const consented: ReleaseHolder[][] = []
  reportHandoffResult({
    hermesHome: home,
    expectedStartedAt: null,
    expectedRunId: null,
    log: () => undefined,
    dialog: {
      showMessageBox: async (options: any) => {
        boxes.push(options)

        return { response, checkboxChecked: false }
      }
    },
    shell: { showItemInFolder: () => undefined },
    openUpdates: () => undefined,
    closeHoldersAndRetry: holders => consented.push(holders)
  })
  await new Promise(resolve => setTimeout(resolve, 0))

  return { boxes, consented }
}

test('#101789: a swap blocked by a program outside Hermes names it and closes it only on an explicit click', async () => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'release-holder-'))

  writeHeldResult(home)
  const dismissed = await reportWithAnswer(home, 1)
  assert.equal(dismissed.boxes.length, 1)
  assert.match(dismissed.boxes[0].message, /SogouCloud\.exe is blocking the Desktop app update/)
  assert.match(dismissed.boxes[0].detail, /pid 4242/)
  assert.deepEqual(dismissed.boxes[0].buttons, ['Close SogouCloud.exe and retry', 'Not now'])
  assert.equal(dismissed.boxes[0].cancelId, 1, 'closing the dialog never consents')
  assert.deepEqual(dismissed.consented, [], 'dismissing must not close the program')

  writeHeldResult(home)
  const accepted = await reportWithAnswer(home, 0)
  assert.deepEqual(accepted.consented, [[SOGOU]])
})

test('the consent names the programs for the next swap, then the update runs again', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'release-holder-root-'))
  fs.mkdirSync(path.join(root, 'apps', 'desktop', 'release'), { recursive: true })
  let applied = 0

  releaseHolderRetry({
    isWindows: true,
    resolveUpdateRoot: () => root,
    applyUpdates: async () => {
      applied += 1

      return { ok: true }
    },
    log: () => undefined
  })!([SOGOU])
  await new Promise(resolve => setTimeout(resolve, 0))

  const consent = JSON.parse(fs.readFileSync(releaseHolderConsentPath(root), 'utf8'))
  assert.deepEqual(consent.exes, [SOGOU.exe])
  assert.ok(Math.abs(consent.written_at - Date.now() / 1000) < 60)
  assert.equal(applied, 1)
})
