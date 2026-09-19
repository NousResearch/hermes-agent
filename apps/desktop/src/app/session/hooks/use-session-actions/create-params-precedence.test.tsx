import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as ProfileModule from '@/store/profile'
import {
  $currentModel,
  getCurrentModelSource,
  resetExplicitManualPickForTests,
  setCurrentModel,
  setCurrentModelSource,
  setCurrentProvider,
  setCurrentReasoningEffort
} from '@/store/session'
import { computeLastUsedSelection, getCurrentComposerScope, noteModelSelectionInUse } from '@/store/session'

// The params builder touches two store/profile helpers that dial the gateway
// registry — the model/provider precedence under test never depends on them.
vi.mock('@/store/profile', async importOriginal => {
  const actual = await importOriginal<typeof ProfileModule>()

  return {
    ...actual,
    ensureGatewayProfile: vi.fn(async () => undefined),
    resolveNewChatOwnerRoute: vi.fn(() => null)
  }
})

import { desktopSessionCreateParams } from './index'

/**
 * v16c — the create-params precedence, asserted directly at the
 * linearization point New Session actually uses:
 *
 *   manual composer  >  scope-matched sticky  >  backend default (no override)
 *
 * The repro this pins: a confirmed pick on a focused TILE updates the sticky
 * (never the primary composer). The next New Session — opened with the
 * composer still on the boot default — must be created on the STICKY's pair,
 * not the stale composer.
 */
describe('desktopSessionCreateParams model precedence (v16c)', () => {
  beforeEach(() => {
    setCurrentModel('glm-5.3')
    setCurrentProvider('zai')
    setCurrentModelSource('default')
    setCurrentReasoningEffort('high', 'draft-init')
    window.localStorage.removeItem('hermes.desktop.composer.last-model')
    window.localStorage.removeItem('hermes.desktop.composer.last-provider')
    window.localStorage.removeItem('hermes.desktop.composer.last-scope')
  })

  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.removeItem('hermes.desktop.composer.last-model')
    window.localStorage.removeItem('hermes.desktop.composer.last-provider')
    window.localStorage.removeItem('hermes.desktop.composer.last-scope')
  })

  it('source=default + sticky Flash válido (scope casante) → create em Flash', async () => {
    // The repro: tile pick confirmed, sticky = Flash, primary composer still
    // shows the boot default (glm-5.3) with source 'default'.
    noteModelSelectionInUse('glm-5.3-flash', 'zai', getCurrentComposerScope())

    const params = await desktopSessionCreateParams('')

    expect(params.model).toBe('glm-5.3-flash')
    expect(params.provider).toBe('zai')
  })

  it('manual composer vence o sticky', async () => {
    // A manual pick on the composer is the user's CURRENT choice — it beats
    // whatever ran last.
    setCurrentModel('grok-4.5')
    setCurrentProvider('xai')
    setCurrentModelSource('manual')
    noteModelSelectionInUse('glm-5.3-flash', 'zai', getCurrentComposerScope())

    const params = await desktopSessionCreateParams('')

    expect(params.model).toBe('grok-4.5')
    expect(params.provider).toBe('xai')
  })

  it('sticky de scope incompatível não viaja', async () => {
    // v14b rule: a sticky minted under another gateway scope legitimately
    // loses to that context's own default — no model/provider override.
    noteModelSelectionInUse('glm-5.3-flash', 'zai', 'conn:other-box::default')

    const params = await desktopSessionCreateParams('')

    expect(params.model).toBeUndefined()
    expect(params.provider).toBeUndefined()
  })

  it('sticky vazio não viaja (primeiro uso)', async () => {
    const params = await desktopSessionCreateParams('')

    expect(params.model).toBeUndefined()
    expect(params.provider).toBeUndefined()
  })

  it('sticky atualiza no meio do caminho é o que o create lê (janela do repro)', async () => {
    // The exact repro ordering: composer resolved at boot (default), the
    // sticky lands LATER (tile confirmation) — the create snapshot must read
    // the sticky AT CREATE TIME, which is what the fresh draft runs on.
    expect(getCurrentModelSource()).toBe('default')
    noteModelSelectionInUse('glm-5.3-flash', 'zai', getCurrentComposerScope())

    const params = await desktopSessionCreateParams('')

    expect(computeLastUsedSelection().model).toBe('glm-5.3-flash')
    expect(params.model).toBe('glm-5.3-flash')
  })

  it('v17: composer sticky_seed não é override — sticky fresco vence no create', async () => {
    // Boot restored the composer from the sticky (source sticky_seed). A tile
    // pick then moved the sticky on. The create must use the FRESHER sticky,
    // not the stale seed sitting on the composer.
    noteModelSelectionInUse('glm-5.3', 'zai', getCurrentComposerScope()) // what boot seeded from
    setCurrentModelSource('sticky_seed')
    noteModelSelectionInUse('glm-5.3-flash', 'zai', getCurrentComposerScope()) // tile confirmed later

    const params = await desktopSessionCreateParams('')

    expect(params.model).toBe('glm-5.3-flash')
    expect(params.provider).toBe('zai')
  })

  it('v17: manual LEGADO cujo par == sticky é rebaixado a sticky_seed (migração)', async () => {
    // Pre-v17 boots wrote the sticky-derived composer pair with source
    // 'manual'. Reading it back must demote it (pair matches the sticky =
    // restored seed, not a pick) so it cannot beat a fresher sticky.
    setCurrentModel('glm-5.3')
    setCurrentProvider('zai')
    setCurrentModelSource('manual')
    noteModelSelectionInUse('glm-5.3', 'zai', getCurrentComposerScope()) // sticky == composer pair (the legacy boot state)

    // The migration fired on read: provenance is now sticky_seed.
    expect(getCurrentModelSource()).toBe('sticky_seed')

    // …and the sticky then moved on (tile confirmation) — create follows it.
    noteModelSelectionInUse('glm-5.3-flash', 'zai', getCurrentComposerScope())

    const params = await desktopSessionCreateParams('')

    expect(params.model).toBe('glm-5.3-flash')
  })

  it('v17: manual REAL (par divergente do sticky) NÃO é rebaixado', async () => {
    // A pair the sticky cannot explain is a genuine user pick — the migration
    // must leave it alone and it still wins the create precedence.
    setCurrentModel('grok-4.5')
    setCurrentProvider('xai')
    setCurrentModelSource('manual')
    noteModelSelectionInUse('glm-5.3-flash', 'zai', getCurrentComposerScope())

    expect(getCurrentModelSource()).toBe('manual')

    const params = await desktopSessionCreateParams('')

    expect(params.model).toBe('grok-4.5')
    expect(params.provider).toBe('xai')
  })

  it('v17 RESTART: manual legado persistido + sticky igual → boot migra p/ sticky_seed e o create segue um sticky futuro', async () => {
    // The restart/migration repro — the most delicate part of v17, because
    // explicitManualPickInThisUiSession resets on every UI boot. Persisted
    // state before the boot (what a pre-v17 renderer left on disk):
    //   source=manual, composer=glm-5.3, sticky=glm-5.3
    // A NEW renderer reads it back:
    //   → the getter migrates manual → sticky_seed (pair == sticky)
    // Then the user picks Flash on a TILE:
    //   → sticky becomes Flash; the primary composer stays glm-5.3/sticky_seed
    // A New Session is created:
    //   → create must use Flash (no manual override exists to beat it).
    //
    // The fresh-UI-boot flag is simulated by resetExplicitManualPickForTests()
    // (a real restart reloads the module with the flag false).
    resetExplicitManualPickForTests()

    // Pre-boot persisted state, written directly to storage (not via helpers
    // that would trip the in-session pick flag).
    window.localStorage.setItem('hermes.desktop.composer.model', 'glm-5.3')
    window.localStorage.setItem('hermes.desktop.composer.provider', 'zai')
    window.localStorage.setItem('hermes.desktop.composer.model-source', 'manual')
    window.localStorage.setItem('hermes.desktop.composer.last-model', 'glm-5.3')
    window.localStorage.setItem('hermes.desktop.composer.last-provider', 'zai')
    window.localStorage.setItem('hermes.desktop.composer.last-scope', getCurrentComposerScope())

    // Mirror the persisted state onto the live atoms (what boot hydration does).
    setCurrentModel('glm-5.3')
    setCurrentProvider('zai')

    // The NEW renderer reads provenance: migration must have demoted it.
    expect(getCurrentModelSource()).toBe('sticky_seed')

    // Tile pick confirms Flash — the sticky moves on, the primary composer
    // deliberately does not (v16c rule).
    noteModelSelectionInUse('glm-5.3-flash', 'zai', getCurrentComposerScope())

    // Primary composer still shows the old pair with the seed provenance.
    expect($currentModel.get()).toBe('glm-5.3')
    expect(getCurrentModelSource()).toBe('sticky_seed')

    // New Session: create reads the FRESHER sticky, not the stale seed.
    const params = await desktopSessionCreateParams('')

    expect(params.model).toBe('glm-5.3-flash')
    expect(params.provider).toBe('zai')
  })
})
