import { afterEach, describe, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'
import { $confirmRequest, settleConfirm } from '@/store/confirm'

import { runPluginUpdate } from './plugin-update'

const p = en.skills.plugins
const tern = { catalog_name: undefined, name: 'hermes-tern' }

const caution = {
  ok: false,
  scan_blocked: true,
  scan_verdict: 'caution',
  scan_findings: [{ category: 'execution', description: 'eval', file: 'helper.py', line: 1, severity: 'high' }]
}

afterEach(() => settleConfirm(false))

describe('runPluginUpdate scan override', () => {
  it('shows a caution finding, and only "Update anyway" retries the update with force', async () => {
    const request = vi.fn().mockResolvedValueOnce(caution).mockResolvedValueOnce({ ok: true, unchanged: false })

    const done = runPluginUpdate(request, tern, null, p)

    await vi.waitFor(() => expect($confirmRequest.get()?.title).toBe(p.updateScanTitle('hermes-tern')))
    expect($confirmRequest.get()?.description).toContain('HIGH  execution  helper.py:1')
    expect($confirmRequest.get()?.destructive).toBe(true)
    expect(request).toHaveBeenCalledTimes(1)

    settleConfirm(true)

    expect(await done).toEqual({ kind: 'applied' })
    expect(request).toHaveBeenNthCalledWith(
      2,
      'plugins.manage',
      expect.objectContaining({ action: 'update', force: true, name: 'hermes-tern' }),
      expect.any(Number)
    )
  })

  it('cancel leaves the plugin alone, and a dangerous verdict never offers the override', async () => {
    const cancelled = vi.fn().mockResolvedValue(caution)
    const done = runPluginUpdate(cancelled, tern, null, p)

    await vi.waitFor(() => expect($confirmRequest.get()).not.toBeNull())
    settleConfirm(false)
    expect((await done).kind).toBe('scan')
    expect(cancelled).toHaveBeenCalledTimes(1)

    const dangerous = vi.fn().mockResolvedValue({ ...caution, scan_verdict: 'dangerous' })

    expect((await runPluginUpdate(dangerous, tern, null, p)).kind).toBe('scan')
    expect($confirmRequest.get()).toBeNull()
    expect(dangerous).toHaveBeenCalledTimes(1)
  })
})
