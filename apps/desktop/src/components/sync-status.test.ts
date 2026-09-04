import { describe, expect, test } from 'vitest'

import type { DesktopSyncReceipt } from '@/global'

import { deriveSyncStatusSummary } from './sync-status'

describe('deriveSyncStatusSummary', () => {
  test('null receipt degrades to nothing-to-show', () => {
    const summary = deriveSyncStatusSummary(null)
    expect(summary.headline).toBeNull()
    expect(summary.level).toBe('ok')
    expect(summary.disabledPlugins).toEqual([])
  })

  test('empty receipt degrades the same', () => {
    const summary = deriveSyncStatusSummary({})
    expect(summary.headline).toBeNull()
  })

  test('failed rebuild is the top signal (error level)', () => {
    const receipt: DesktopSyncReceipt = {
      kind: 'sync',
      outcome: 'failed',
      venv_rebuild: { ok: false, reason: 'uv sync exited 1: unsatisfiable' }
    }

    const summary = deriveSyncStatusSummary(receipt)
    expect(summary.level).toBe('error')
    expect(summary.headline).toContain('rebuild failed')
  })

  test('bisect disables surface as warn with reasons', () => {
    const receipt: DesktopSyncReceipt = {
      kind: 'sync',
      outcome: 'bisected',
      plugin_bisect: [
        { plugin: 'bad-plug', action: 'disabled', reason: 'conflicts with core pin' },
        { plugin: 'kept-plug', action: 'kept', reason: '' }
      ]
    }

    const summary = deriveSyncStatusSummary(receipt)
    expect(summary.level).toBe('warn')
    expect(summary.disabledPlugins).toEqual([
      { plugin: 'bad-plug', reason: 'conflicts with core pin' }
    ])
    expect(summary.headline).toContain('disabled')
  })

  test('needs-fixing beats updates in priority', () => {
    const receipt: DesktopSyncReceipt = {
      kind: 'plugin-check',
      outcome: 'updates-available',
      plugin_checks: [
        {
          name: 'plug',
          needs_fixing: 'update_url mismatch: saved vs manifest'
        },
        { name: 'other', update_available: true, current: '1.0', latest: '2.0' }
      ]
    }

    const summary = deriveSyncStatusSummary(receipt)
    expect(summary.level).toBe('warn')
    expect(summary.headline).toContain('update-url')
    expect(summary.needsFixing).toHaveLength(1)
    expect(summary.updatesAvailable).toEqual([
      { name: 'other', current: '1.0', latest: '2.0' }
    ])
  })

  test('plain updates-available is info, not warn', () => {
    const receipt: DesktopSyncReceipt = {
      kind: 'plugin-check',
      outcome: 'updates-available',
      plugin_checks: [{ name: 'plug', update_available: true, current: '1.0', latest: '1.1' }]
    }

    const summary = deriveSyncStatusSummary(receipt)
    expect(summary.level).toBe('info')
    expect(summary.headline).toContain('updates available')
  })

  test('unknown update_available (null) is not an update', () => {
    const receipt: DesktopSyncReceipt = {
      kind: 'plugin-check',
      plugin_checks: [{ name: 'mystery', update_available: null, reason: 'not on PyPI' }]
    }

    const summary = deriveSyncStatusSummary(receipt)
    expect(summary.updatesAvailable).toEqual([])
    expect(summary.headline).toBeNull()
  })
})
