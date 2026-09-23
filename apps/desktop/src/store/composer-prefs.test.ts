import { afterEach, describe, expect, it } from 'vitest'

import { $composerEnterSends, applyComposerPrefsFromConfig } from './composer-prefs'

afterEach(() => $composerEnterSends.set(true))

describe('composer prefs from Hermes config', () => {
  it('applies explicit boolean values exactly', () => {
    applyComposerPrefsFromConfig({ desktop: { composer: { enter_sends: false } } })

    expect($composerEnterSends.get()).toBe(false)

    applyComposerPrefsFromConfig({ desktop: { composer: { enter_sends: true } } })

    expect($composerEnterSends.get()).toBe(true)
  })

  it('preserves the backward-compatible Enter-to-send default for missing or invalid values', () => {
    $composerEnterSends.set(false)
    applyComposerPrefsFromConfig({ desktop: { composer: { enter_sends: 'false' } } })

    expect($composerEnterSends.get()).toBe(true)

    $composerEnterSends.set(false)
    applyComposerPrefsFromConfig({})

    expect($composerEnterSends.get()).toBe(true)
  })
})
