import { describe, expect, it, vi } from 'vitest'

import { preUpdateBackupEnabled, readPreUpdateBackupEnabled } from './pre-update-backup-config'

describe('preUpdateBackupEnabled', () => {
  it.each([false, null, 'off', 'false', 'none', 'disabled', ' DISABLED '])(
    'disables the desktop backup for the Python updater off alias %j',
    value => {
      expect(preUpdateBackupEnabled(value)).toBe(false)
    }
  )

  it.each([true, 'quick', 'full', 'zip', 'true', 'unexpected', 0, {}, undefined])(
    'keeps the safety backup for %j',
    value => {
      expect(preUpdateBackupEnabled(value)).toBe(true)
    }
  )
})

describe('readPreUpdateBackupEnabled', () => {
  it.each([
    ['managed false overriding user true', 'false', false],
    ['managed true overriding user false', 'true', true],
    ['an environment-expanded off alias', '"off"', false]
  ])('uses the effective config result for %s', async (_name, stdout, expected) => {
    const run = vi.fn().mockResolvedValue({ stdout })

    const runtime = {
      command: '/runtime/python',
      args: ['-m', 'hermes_cli.main', 'config', 'get', 'updates.pre_update_backup', '--json'],
      env: { PYTHONPATH: '/runtime/hermes' }
    }

    await expect(readPreUpdateBackupEnabled(runtime, '/profiles/active', run)).resolves.toBe(expected)
    expect(run).toHaveBeenCalledWith(
      runtime.command,
      runtime.args,
      expect.objectContaining({
        env: expect.objectContaining({
          HERMES_HOME: '/profiles/active',
          PYTHONPATH: '/runtime/hermes'
        })
      })
    )
  })

  it('fails safe when runtime resolution fails', async () => {
    await expect(
      readPreUpdateBackupEnabled(Promise.reject(new Error('resolver failed')), '/profiles/active', vi.fn())
    ).resolves.toBe(true)
  })

  it.each([
    ['runtime failure', vi.fn().mockRejectedValue(new Error('probe failed'))],
    ['malformed output', vi.fn().mockResolvedValue({ stdout: 'not-json' })]
  ])('fails safe for %s', async (_name, run) => {
    await expect(
      readPreUpdateBackupEnabled({ command: '/runtime/hermes', args: ['config', 'get'] }, '/profiles/active', run)
    ).resolves.toBe(true)
  })
})
