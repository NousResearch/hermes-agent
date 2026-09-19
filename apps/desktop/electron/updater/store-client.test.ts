import { describe, expect, it } from 'vitest'

import { createStoreStrategy } from './store-client'

it('the production runner passes modes, full HWND and isolated payload imports', async () => {
  const calls: { python: string; args: readonly string[]; env?: NodeJS.ProcessEnv; timeout?: number; waitForExit?: boolean }[] = []
  const handle = Buffer.alloc(8)
  handle.writeBigUInt64LE(0x1234567887654321n)

  const strategy = createStoreStrategy({
    python: 'packaged-python.exe',
    script: 'check-store-update.py',
    sitePackages: 'payload-deps',
    env: { PYTHONHOME: 'foreign', VIRTUAL_ENV: 'foreign', PYTHONPATH: 'foreign' },
    windowHandle: () => handle,
    appVersion: '1.0.0',
    teardown: async () => {},
    restore: async () => {},
    quit: () => {},
    emitProgress: () => {},
    registerPendingRelaunch: async () => ({ automatic: true, cancel: async () => {} }),
    run: async (python, script, deps) => {
      calls.push({ python, args: [script, ...(deps.args ?? [])], env: deps.env, timeout: deps.timeoutMs, waitForExit: deps.waitForExit })

      return {
        code: deps.args?.includes('check') ? 2 : 0,
        stdout: JSON.stringify({ ok: true, available: true, packages: ['package-v2'] })
      }
    }
  })

  await strategy.check()
  await strategy.apply()
  expect(calls.map(call => call.args)).toEqual(
    ['check', 'download', 'install'].map(mode => [
      'check-store-update.py',
      '--mode',
      mode,
      '--hwnd',
      '1311768467139281697'
    ])
  )
  expect(calls.every(call => call.python === 'packaged-python.exe')).toBe(true)
  expect(
    calls.every(call => call.env?.PYTHONPATH === 'payload-deps' && !call.env.PYTHONHOME && !call.env.VIRTUAL_ENV)
  ).toBe(true)
  expect(calls[1].timeout).toBeGreaterThan(calls[0].timeout!)
  expect(calls.map(call => call.waitForExit)).toEqual([false, true, true])
})

describe('Store runner refuses unusable replies', () => {
  it.each([
    { code: 1, stdout: '{"ok":true,"available":false}' },
    { code: 0, stdout: '{"ok":true,"available":null}' },
    { code: 0, stdout: 'not-json' }
  ])('does not turn $stdout into a current-version verdict', async response => {
    const strategy = createStoreStrategy({
      python: 'python',
      script: 'checker',
      sitePackages: 'deps',
      env: {},
      windowHandle: () => Buffer.alloc(8, 1),
      appVersion: '1',
      teardown: async () => {},
      restore: async () => {},
      quit: () => {},
      emitProgress: () => {},
      registerPendingRelaunch: async () => ({ automatic: true, cancel: async () => {} }),
      run: async () => response
    })

    const status = await strategy.check()
    expect(status.error).toBeTruthy()
    expect(status.updateAvailable).toBe(false)
    await expect(strategy.apply()).rejects.toThrow()
  })
})
