import { describe, expect, it } from 'vitest'

import {
  $composerSpellcheck,
  $composerSpellcheckLanguage,
  normalizeSpellcheckLanguage,
  setComposerEditorFromConfig
} from './composer-editor'

describe('normalizeSpellcheckLanguage', () => {
  it('accepts well-formed BCP-47 tags', () => {
    expect(normalizeSpellcheckLanguage('en-US')).toBe('en-US')
    expect(normalizeSpellcheckLanguage('de')).toBe('de')
    expect(normalizeSpellcheckLanguage('zh-Hant-TW')).toBe('zh-Hant-TW')
  })

  it('trims surrounding whitespace', () => {
    expect(normalizeSpellcheckLanguage('  en-GB  ')).toBe('en-GB')
  })

  it('rejects malformed or non-string values', () => {
    expect(normalizeSpellcheckLanguage('en_US')).toBe('')
    expect(normalizeSpellcheckLanguage('e')).toBe('')
    expect(normalizeSpellcheckLanguage('english')).toBe('')
    expect(normalizeSpellcheckLanguage('')).toBe('')
    expect(normalizeSpellcheckLanguage(undefined)).toBe('')
    expect(normalizeSpellcheckLanguage(42)).toBe('')
  })
})

describe('setComposerEditorFromConfig', () => {
  it('defaults spellcheck off so code/paths are not flagged (#44415)', () => {
    setComposerEditorFromConfig(undefined)
    expect($composerSpellcheck.get()).toBe(false)
    setComposerEditorFromConfig({})
    expect($composerSpellcheck.get()).toBe(false)
  })

  it('enables spellcheck only on an explicit true', () => {
    setComposerEditorFromConfig({ spellcheck: true, language: 'en-US' })
    expect($composerSpellcheck.get()).toBe(true)
    expect($composerSpellcheckLanguage.get()).toBe('en-US')
  })

  it('treats a truthy non-boolean as off (only literal true opts in)', () => {
    setComposerEditorFromConfig({ spellcheck: 'yes' })
    expect($composerSpellcheck.get()).toBe(false)
  })

  it('normalizes the language and ignores a malformed tag', () => {
    setComposerEditorFromConfig({ language: '  de-DE ' })
    expect($composerSpellcheckLanguage.get()).toBe('de-DE')
    setComposerEditorFromConfig({ language: 'not a tag' })
    expect($composerSpellcheckLanguage.get()).toBe('')
  })

  it('survives a non-object config value', () => {
    setComposerEditorFromConfig('nonsense')
    expect($composerSpellcheck.get()).toBe(false)
    expect($composerSpellcheckLanguage.get()).toBe('')
  })
})
