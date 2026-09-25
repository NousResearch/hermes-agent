import { describe, expect, it, vi } from 'vitest'

import { readUpdatesCheckEnabled } from './update-check-config'

describe('readUpdatesCheckEnabled', () => {
  it.each([
    ['explicit false', 'false', false],
    ['string false alias', '"false"', false],
    ['explicit true', 'true', true],
    ['unset key (null)', 'null', true],
    ['unexpected truthy value', '"weekly"', true]
  ])('maps the config result for %s', async (_name, stdout, expected) => {
    const run = vi.fn().mockResolvedValue({ stdout })

    const runtime = {
      command: '/runtime/hermes',
      args: ['-p', 'default', 'config', 'get', 'updates.check', '--json'],
      env: { HERMES_RUNTIME_DIR: '/runtime/tools' }
    }

    await expect(readUpdatesCheckEnabled(runtime, run)).resolves.toBe(expected)
    expect(run).toHaveBeenCalledWith(
      runtime.command,
      runtime.args,
      expect.objectContaining({
        env: expect.objectContaining({
          HERMES_RUNTIME_DIR: '/runtime/tools'
        })
      })
    )
  })

  it('fails open when runtime resolution fails', async () => {
    await expect(
      readUpdatesCheckEnabled(Promise.reject(new Error('resolver failed')), vi.fn())
    ).resolves.toBe(true)
  })

  it.each([
    ['runtime failure', vi.fn().mockRejectedValue(new Error('probe failed'))],
    ['malformed output', vi.fn().mockResolvedValue({ stdout: 'not-json' })],
    ['missing runtime command', vi.fn(), { args: ['config', 'get'] }]
  ])('fails open for %s', async (_name, run, runtime) => {
    await expect(readUpdatesCheckEnabled(runtime ?? { command: '/runtime/hermes', args: [] }, run)).resolves.toBe(
      true
    )
  })
})
