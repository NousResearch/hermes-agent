import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterEach, beforeEach, describe, test, vi } from 'vitest'

import { resolveWatchdogPrewarmedBackend } from './watchdog-backend'

describe('resolveWatchdogPrewarmedBackend', () => {
  let tmpDir = ''
  let previousLocalAppData: string | undefined

  beforeEach(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-wd-'))
    previousLocalAppData = process.env.LOCALAPPDATA
    process.env.LOCALAPPDATA = tmpDir
  })

  afterEach(() => {
    if (previousLocalAppData === undefined) {
      delete process.env.LOCALAPPDATA
    } else {
      process.env.LOCALAPPDATA = previousLocalAppData
    }

    vi.unstubAllGlobals()
    fs.rmSync(tmpDir, { recursive: true, force: true })
  })

  test('returns null when manifest is missing', async () => {
    assert.equal(await resolveWatchdogPrewarmedBackend(), null)
  })

  test('returns connection when manifest probes healthy', async () => {
    const manifestDir = path.join(tmpDir, 'HermesWatchdog')
    fs.mkdirSync(manifestDir, { recursive: true })
    fs.writeFileSync(
      path.join(manifestDir, 'desktop-backend.json'),
      JSON.stringify({
        baseUrl: 'http://127.0.0.1:54321',
        token: 'abc',
        port: 54321,
        hermesRoot: 'C:\\repo',
        managed: true
      })
    )

    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({ ok: true }))
    )

    const got = await resolveWatchdogPrewarmedBackend({ hermesRoot: 'C:\\repo' })

    assert.ok(got)
    assert.equal(got?.baseUrl, 'http://127.0.0.1:54321')
    assert.equal(got?.token, 'abc')
    assert.equal(got?.source, 'watchdog')
  })

  test('rejects manifest when hermes root mismatches explicit override', async () => {
    const manifestDir = path.join(tmpDir, 'HermesWatchdog')
    fs.mkdirSync(manifestDir, { recursive: true })
    fs.writeFileSync(
      path.join(manifestDir, 'desktop-backend.json'),
      JSON.stringify({
        baseUrl: 'http://127.0.0.1:54321',
        token: 'abc',
        port: 54321,
        hermesRoot: 'C:\\other'
      })
    )

    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({ ok: true }))
    )

    assert.equal(await resolveWatchdogPrewarmedBackend({ hermesRoot: 'C:\\repo' }), null)
  })
})
