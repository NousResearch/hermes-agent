import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import {
  $composerEditorPreferences,
  composerEditorPreferencesFromConfig,
  setComposerEditorPreferencesFromConfig
} from './composer-preferences'

const originalLanguage = navigator.language

function setNavigatorLanguage(language: string) {
  Object.defineProperty(navigator, 'language', { configurable: true, value: language })
}

describe('composer editor preferences', () => {
  beforeEach(() => {
    setNavigatorLanguage('fr-CA')
    setComposerEditorPreferencesFromConfig({})
  })

  afterEach(() => {
    setNavigatorLanguage(originalLanguage)
    setComposerEditorPreferencesFromConfig({})
  })

  it('defaults spellcheck on and follows the system language', () => {
    expect(composerEditorPreferencesFromConfig({})).toEqual({
      language: 'fr-CA',
      spellcheck: true
    })
  })

  it('honors explicit desktop editor spellcheck and language settings', () => {
    expect(
      composerEditorPreferencesFromConfig({
        desktop: { editor: { language: 'en-GB', spellcheck: false } }
      })
    ).toEqual({
      language: 'en-GB',
      spellcheck: false
    })
  })

  it('normalizes blank language back to the system language', () => {
    expect(
      composerEditorPreferencesFromConfig({
        desktop: { editor: { language: '   ', spellcheck: true } }
      })
    ).toEqual({
      language: 'fr-CA',
      spellcheck: true
    })
  })

  it('updates the shared atom from config refreshes', () => {
    setComposerEditorPreferencesFromConfig({
      desktop: { editor: { language: 'de-DE', spellcheck: false } }
    })

    expect($composerEditorPreferences.get()).toEqual({
      language: 'de-DE',
      spellcheck: false
    })
  })
})
