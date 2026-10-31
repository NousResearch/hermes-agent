import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { handoffResultPath, type ReleaseHolder } from './handoff-result'
import { bindReleaseHolderConsent, releaseHolderConsentPath, releaseHolderRetry } from './release-holder-consent'
import { reportHandoffResult } from './update-handoff-report'

const SOGOU: ReleaseHolder = {
  pid: 4242,
  startTime: 1761470000.125,
  name: 'SogouCloud.exe',
  exe: 'D:\\Program Files (x86)\\SogouInput\\16.7.0.4912\\SogouCloud.exe'
}

// A protected process: Windows would not reveal its program path, so it cannot be closed from here.
const PROTECTED: ReleaseHolder = { pid: 5151, startTime: 1761470001.5, name: 'AVHelper.exe', exe: '' }

function writeHeldResult(home: string, holders: ReleaseHolder[]) {
  fs.writeFileSync(
    handoffResultPath(home),
    JSON.stringify({
      ok: true,
      exit_code: 0,
      manual: true,
      message: 'the Desktop app could not be replaced because a program outside Hermes is using its folder.',
      branch: 'main',
      warnings: ['build: the Desktop app folder is in use'],
      // windows.ps1's wire shape: the start time stays the CLI's text
      release_holders: holders.map(({ startTime, ...holder }) => ({ ...holder, start_time: startTime.toFixed(3) })),
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

function updateRoot(): string {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'release-holder-root-'))
  fs.mkdirSync(path.join(root, 'apps', 'desktop', 'release'), { recursive: true })

  return root
}

/** The dialog's click, with an apply that hands off as the Windows script path does (or not). */
async function retry(root: string, holders: ReleaseHolder[], outcome: { ok: boolean; handedOff?: boolean }) {
  releaseHolderRetry({
    isWindows: true,
    resolveUpdateRoot: () => root,
    applyUpdates: async () => {
      bindReleaseHolderConsent(root, 'desktop-retry-1', () => undefined) // checkout.ts, once the hand-off run id exists

      return outcome
    },
    log: () => undefined
  })!(holders)
  await new Promise(resolve => setTimeout(resolve, 0))
}

test('#101789: a swap blocked by a program outside Hermes names it and closes it only on an explicit click', async () => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'release-holder-'))

  writeHeldResult(home, [SOGOU])
  const dismissed = await reportWithAnswer(home, 1)
  assert.equal(dismissed.boxes.length, 1)
  assert.match(dismissed.boxes[0].message, /SogouCloud\.exe is blocking the Desktop app update/)
  assert.match(dismissed.boxes[0].detail, /pid 4242/)
  assert.deepEqual(dismissed.boxes[0].buttons, ['Close SogouCloud.exe and retry', 'Not now'])
  assert.equal(dismissed.boxes[0].cancelId, 1, 'closing the dialog never consents')
  assert.deepEqual(dismissed.consented, [], 'dismissing must not close the program')

  writeHeldResult(home, [SOGOU])
  const accepted = await reportWithAnswer(home, 0)
  assert.deepEqual(accepted.consented, [[SOGOU]])
})

test('review N3: a holder that cannot be closed from here is named for the user, with no Close button', async () => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'release-holder-'))

  writeHeldResult(home, [PROTECTED])
  const alone = await reportWithAnswer(home, 0)
  assert.deepEqual(alone.boxes[0].buttons, ['OK'])
  assert.match(alone.boxes[0].detail, /AVHelper\.exe \(pid 5151\)\n {3}Close this one yourself/)
  assert.deepEqual(alone.consented, [], 'no retry that would only show the same dialog again')

  writeHeldResult(home, [SOGOU, PROTECTED])
  const mixed = await reportWithAnswer(home, 0)
  assert.deepEqual(mixed.boxes[0].buttons, ['Close SogouCloud.exe and retry', 'Not now'])
  assert.deepEqual(mixed.consented, [[SOGOU]])
})

test('review N1: the consent is bound to the retry run and the exact processes the user saw', async () => {
  const root = updateRoot()

  await retry(root, [SOGOU, PROTECTED], { ok: true, handedOff: true })

  const consent = JSON.parse(fs.readFileSync(releaseHolderConsentPath(root), 'utf8'))
  assert.equal(consent.run_id, 'desktop-retry-1')
  assert.deepEqual(consent.holders, [{ pid: 4242, start_time: 1761470000.125, exe: SOGOU.exe }])
})

test('review N1: a retry that does not hand off leaves no consent, and a later update records none', async () => {
  for (const outcome of [{ ok: false }, { ok: true }]) {
    const root = updateRoot()

    await retry(root, [SOGOU], outcome)
    assert.equal(fs.existsSync(releaseHolderConsentPath(root)), false, JSON.stringify(outcome))

    // The next apply, started from the Updates page and not from the dialog, consents to nothing.
    bindReleaseHolderConsent(root, 'manual-update-2', () => undefined)
    assert.equal(fs.existsSync(releaseHolderConsentPath(root)), false, JSON.stringify(outcome))
  }
})
