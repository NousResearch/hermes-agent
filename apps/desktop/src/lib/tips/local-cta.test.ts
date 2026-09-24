/**
 * The local-setup campaign's policy, tested as the pure decisions they are:
 * who qualifies (eligibility), when the bubble may return (the clock), and
 * that the campaign cannot corrupt the rotation's cursor.
 */

import { describe, expect, it } from 'vitest'

import { TIP_CATALOG } from '@/lib/tips/catalog'
import { LOCAL_SETUP_RESHOW_MS, LOCAL_SETUP_TIP_ID, localSetupDue, localSetupEligible } from '@/lib/tips/local-cta'
import type { LocalCatalogModel, LocalModelsStatus } from '@/types/hermes'

function status(overrides: Partial<LocalModelsStatus> = {}): LocalModelsStatus {
  return {
    active_model_id: null,
    loading: {},
    models: [],
    models_dir: '',
    placement: null,
    runtime_installed: false,
    server_running: false,
    ...overrides
  } as LocalModelsStatus
}

function fittingModel(): LocalCatalogModel {
  return { fits: true, id: 'qwen3.8-27b' } as LocalCatalogModel
}

describe('localSetupEligible', () => {
  it('offers setup to a local backend with a fitting model and nothing staged', () => {
    expect(localSetupEligible('local', status(), [fittingModel()])).toBe(true)
  })

  it('never promises local privacy on a remote or cloud backend', () => {
    for (const mode of ['remote', 'cloud', 'ssh', null]) {
      expect(localSetupEligible(mode, status(), [fittingModel()])).toBe(false)
    }
  })

  it('stays quiet when no catalog model fits the machine', () => {
    expect(localSetupEligible('local', status(), [{ fits: false } as LocalCatalogModel])).toBe(false)
    expect(localSetupEligible('local', status(), [])).toBe(false)
  })

  it('retires itself once the machine is set up', () => {
    const setUp = status({
      models: [{ id: 'qwen3.8-27b' }] as LocalModelsStatus['models'],
      runtime_installed: true
    })

    expect(localSetupEligible('local', setUp, [fittingModel()])).toBe(false)
  })

  it('still offers when the runtime exists but no model is staged', () => {
    expect(localSetupEligible('local', status({ runtime_installed: true }), [fittingModel()])).toBe(true)
  })

  it('answers false while state is still loading', () => {
    expect(localSetupEligible('local', null, [fittingModel()])).toBe(false)
    expect(localSetupEligible('local', status(), null)).toBe(false)
  })
})

describe('localSetupDue', () => {
  it('is due when never shown', () => {
    expect(localSetupDue(Date.now(), undefined)).toBe(true)
  })

  it('holds for a week after an ignored showing, then returns', () => {
    const now = Date.now()

    expect(localSetupDue(now, now - LOCAL_SETUP_RESHOW_MS + 1000)).toBe(false)
    expect(localSetupDue(now, now - LOCAL_SETUP_RESHOW_MS)).toBe(true)
  })
})

describe('campaign identity', () => {
  it('lives outside the rotation catalog — the walk must never land on it', () => {
    // TipId already excludes the campaign id at the type level; this guards
    // the runtime data against someone re-adding it as a catalog entry.
    const ids: readonly string[] = TIP_CATALOG.map(def => def.id)

    expect(ids).not.toContain(LOCAL_SETUP_TIP_ID)
  })
})
