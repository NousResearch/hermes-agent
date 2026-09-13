import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  checkAppInstallerUpdate,
  triggerAppInstallerUpdate,
  win32AppInstallerFeedPath
} from './app-updater'

// ── feed hosting + paths ────────────────────────────────────────────

test('win32 App Installer feed paths are per-channel and per-variant', () => {
  assert.equal(win32AppInstallerFeedPath('stable', false), 'win32/stable/')
  assert.equal(win32AppInstallerFeedPath('canary', false), 'win32/canary/')
  assert.equal(win32AppInstallerFeedPath('stable', true), 'win32/light/stable/')
  assert.equal(win32AppInstallerFeedPath('canary', true), 'win32/light/canary/')
})

// ── win32 arm (OS App Installer checker + trigger) ─────────────────

function fakePayloadRunner(stdout: string, code = 0) {
  const calls: string[] = []

  return {
    runner: {
      python: 'C:\\payload\\tools\\cpython\\python.exe',
      script: 'C:\\payload\\scripts\\check-appinstaller-update.py',
      run: async (python: string, script: string) => {
        calls.push(`${python} ${script}`)

        return { code, stdout }
      }
    } as any,
    calls
  }
}

test('win32 check reports available when the OS says a newer package exists', async () => {
  const { runner } = fakePayloadRunner(JSON.stringify({ available: true, availability: 'Available' }))
  const result = await checkAppInstallerUpdate(runner)

  assert.equal(result.available, true)
  assert.equal(result.availability, 'Available')
})

test('win32 check reports not-available cleanly', async () => {
  const { runner } = fakePayloadRunner(JSON.stringify({ available: false, availability: 'NoApplicableUpdate' }))
  const result = await checkAppInstallerUpdate(runner)

  assert.equal(result.available, false)
})

test('win32 check surfaces an unknown verdict (missing winrt) without crashing', async () => {
  const { runner } = fakePayloadRunner(JSON.stringify({ available: null, error: 'winrt import failed' }), 1)
  const result = await checkAppInstallerUpdate(runner)

  assert.equal(result.available, null)
  assert.match(result.error || '', /winrt import failed/)
})

test('win32 trigger prepares a local descriptor before teardown and file activation', async () => {
  const calls: string[] = []

  const installer = {
    prepare: async (url: string) => { calls.push(`download:${url}`);

 return 'update.appinstaller' },
    open: async (file: string) => { calls.push(`open:${file}`);

 return '' }
  }

  const result = await triggerAppInstallerUpdate(
    'https://updates.example.com/', 'stable', false, installer,
    () => void calls.push('teardown')
  )

  assert.equal(result.ok, true)
  assert.deepEqual(calls, [
    'download:https://updates.example.com/win32/stable/stable.appinstaller',
    'teardown', 'open:update.appinstaller'
  ])
})

test('win32 trigger refuses failed downloads before teardown and surfaces file-open errors', async () => {
  const teardown = () => { throw new Error('unexpected teardown') }
  await assert.rejects(triggerAppInstallerUpdate('https://updates.example.com', 'canary', true, {
    prepare: async url => { assert.match(url, /win32\/light\/canary\/canary.appinstaller$/); throw new Error('download failed') },
    open: async () => ''
  }, teardown), /download failed/)
  await assert.rejects(triggerAppInstallerUpdate('https://updates.example.com', 'stable', false, {
    prepare: async () => 'update.appinstaller', open: async () => 'No file association'
  }), /No file association/)
})
