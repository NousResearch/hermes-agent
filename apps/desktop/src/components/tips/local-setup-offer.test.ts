/**
 * The campaign offer against the live stores: eligibility fetch, the showTip
 * wiring (button included), retirement, the reshow clock, and the cursor
 * guard that keeps a campaign showing from restarting the rotation's walk.
 */

import { cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const getLocalModelsStatus = vi.fn()
const getLocalCatalog = vi.fn()

vi.mock('@/hermes', () => ({
  getLocalCatalog: (...args: unknown[]) => getLocalCatalog(...args),
  getLocalModelsStatus: (...args: unknown[]) => getLocalModelsStatus(...args)
}))

import { en } from '@/i18n/en'
import { LOCAL_SETUP_TIP_ID } from '@/lib/tips/local-cta'
import { $localModelsEnabled } from '@/store/local-models-flag'
import { $connection } from '@/store/session'
import { $activeTip, $lastTipId, $retiredTips, $tipShownAt } from '@/store/tips'

import { offerLocalSetupTip, resetLocalSetupOfferCache } from './local-setup-offer'

function primeEligibleBackend() {
  getLocalModelsStatus.mockResolvedValue({ models: [], runtime_installed: false })
  getLocalCatalog.mockResolvedValue({ models: [{ fits: true, id: 'qwen3.8-27b' }] })
}

async function flushFetch() {
  await Promise.resolve()
  await Promise.resolve()
  await Promise.resolve()
}

describe('offerLocalSetupTip', () => {
  beforeEach(() => {
    resetLocalSetupOfferCache()
    // The campaign ships behind --local like every local-models surface.
    $localModelsEnabled.set(true)
    $activeTip.set(null)
    $retiredTips.set([])
    $tipShownAt.set({})
    $lastTipId.set(null)
    $connection.set({ mode: 'local' } as never)
    getLocalModelsStatus.mockReset()
    getLocalCatalog.mockReset()
  })

  afterEach(() => {
    cleanup()
  })

  it('holds the first quiet moment while the read flies, then shows on the next', async () => {
    primeEligibleBackend()

    const openLocalModels = vi.fn()

    // First offer: fetch in flight — the moment is HELD (true, so the
    // rotation's walk cannot take it and arm the cooldown ahead of the
    // campaign), but nothing is on screen yet.
    expect(offerLocalSetupTip(en.tips, openLocalModels)).toBe(true)
    expect($activeTip.get()).toBeNull()
    await flushFetch()

    // Second offer: cached yes — bubble goes up with the CTA wired.
    expect(offerLocalSetupTip(en.tips, openLocalModels)).toBe(true)

    const tip = $activeTip.get()

    expect(tip?.tipId).toBe(LOCAL_SETUP_TIP_ID)
    expect(tip?.action?.label).toBe(en.tips.items['local-setup'].action)

    tip?.action?.onSelect()
    expect(openLocalModels).toHaveBeenCalledTimes(1)
    // The CTA closes the bubble on its way to the pane.
    expect($activeTip.get()).toBeNull()
  })

  it('never restarts the rotation walk: the campaign id stays out of the cursor', async () => {
    primeEligibleBackend()
    $lastTipId.set('cron')

    offerLocalSetupTip(en.tips, vi.fn())
    await flushFetch()
    offerLocalSetupTip(en.tips, vi.fn())

    expect($activeTip.get()?.tipId).toBe(LOCAL_SETUP_TIP_ID)
    expect($lastTipId.get()).toBe('cron')
  })

  it('stays quiet on an ineligible machine without refetching', async () => {
    getLocalModelsStatus.mockResolvedValue({ models: [{ id: 'staged' }], runtime_installed: true })
    getLocalCatalog.mockResolvedValue({ models: [{ fits: true, id: 'qwen3.8-27b' }] })

    offerLocalSetupTip(en.tips, vi.fn())
    await flushFetch()

    expect(offerLocalSetupTip(en.tips, vi.fn())).toBe(false)
    expect($activeTip.get()).toBeNull()
    expect(getLocalModelsStatus).toHaveBeenCalledTimes(1)
  })

  it('honors the ✕ forever and the ignored-bubble clock for a week', async () => {
    primeEligibleBackend()

    $retiredTips.set([LOCAL_SETUP_TIP_ID])
    expect(offerLocalSetupTip(en.tips, vi.fn())).toBe(false)
    expect(getLocalModelsStatus).not.toHaveBeenCalled()

    $retiredTips.set([])
    $tipShownAt.set({ [LOCAL_SETUP_TIP_ID]: Date.now() - 60_000 })
    expect(offerLocalSetupTip(en.tips, vi.fn())).toBe(false)
    expect(getLocalModelsStatus).not.toHaveBeenCalled()
  })

  it('asks nothing of a remote backend', () => {
    $connection.set({ mode: 'remote' } as never)

    expect(offerLocalSetupTip(en.tips, vi.fn())).toBe(false)
    expect(getLocalModelsStatus).not.toHaveBeenCalled()
  })

  it('never runs without the --local launch flag (strict), even on an eligible machine', () => {
    $localModelsEnabled.set(false)

    expect(offerLocalSetupTip(en.tips, vi.fn())).toBe(false)
    // Declined before any read: no fetch, no held moment, no cooldown spent.
    expect(getLocalModelsStatus).not.toHaveBeenCalled()
    expect($activeTip.get()).toBeNull()
  })

  it('a failed read stands down for the session instead of retrying', async () => {
    getLocalModelsStatus.mockRejectedValue(new Error('backend gone'))
    getLocalCatalog.mockRejectedValue(new Error('backend gone'))

    offerLocalSetupTip(en.tips, vi.fn())
    await flushFetch()

    expect(offerLocalSetupTip(en.tips, vi.fn())).toBe(false)
    expect(getLocalModelsStatus).toHaveBeenCalledTimes(1)
  })
})
