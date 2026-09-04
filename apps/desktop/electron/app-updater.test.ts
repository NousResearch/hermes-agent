import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  checkAppInstallerUpdate,
  PLACEHOLDER_FEED_BASE_URL,
  triggerAppInstallerUpdate,
  win32AppInstallerFeedPath
} from './app-updater'

// ── feed hosting + paths ────────────────────────────────────────────

test('the feed base URL placeholder is a documented dead end', () => {
  // Overridden by `updates.desktop_feed_base_url` (read inline in main.ts);
  // the placeholder itself must never resolve to a real host.
  assert.equal(PLACEHOLDER_FEED_BASE_URL, 'https://updates.invalid/hermes-desktop')
})

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

test('win32 trigger opens ms-appinstaller with the channel .appinstaller after teardown', async () => {
  const calls: string[] = []

  const shell = {
    openExternal: async (url: string) => void calls.push(`open:${url}`)
  }

  const result = await triggerAppInstallerUpdate(
    'https://updates.example.com/',
    'stable',
    false,
    shell as any,
    () => void calls.push('teardown')
  )

  assert.equal(result.ok, true)
  assert.deepEqual(calls, [
    'teardown',
    'open:ms-appinstaller:?source=https%3A%2F%2Fupdates.example.com%2Fwin32%2Fstable%2Fstable.appinstaller'
  ])
})

test('win32 trigger light+canary targets the light canary feed dir', async () => {
  const calls: string[] = []

  const shell = {
    openExternal: async (url: string) => void calls.push(url)
  }

  await triggerAppInstallerUpdate('https://updates.example.com', 'canary', true, shell as any)

  assert.equal(calls[0], 'ms-appinstaller:?source=https%3A%2F%2Fupdates.example.com%2Fwin32%2Flight%2Fcanary%2Fcanary.appinstaller')
})
