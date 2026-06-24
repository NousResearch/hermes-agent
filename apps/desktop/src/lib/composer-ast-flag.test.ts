import { afterEach, describe, expect, it, vi } from 'vitest'

import { COMPOSER_AST_STORAGE_KEY, isComposerAstEnabled } from './composer-ast-flag'

afterEach(() => {
  vi.unstubAllEnvs()
  localStorage.clear()
})

describe('isComposerAstEnabled', () => {
  it('defaults to true when no override is set', () => {
    expect(isComposerAstEnabled()).toBe(true)
  })

  it('is true when VITE_HERMES_COMPOSER_AST is "true"', () => {
    vi.stubEnv('VITE_HERMES_COMPOSER_AST', 'true')
    expect(isComposerAstEnabled()).toBe(true)
  })

  it('is false when VITE_HERMES_COMPOSER_AST is "false"', () => {
    vi.stubEnv('VITE_HERMES_COMPOSER_AST', 'false')
    expect(isComposerAstEnabled()).toBe(false)
  })

  it('is true when the localStorage override is "1"', () => {
    localStorage.setItem(COMPOSER_AST_STORAGE_KEY, '1')
    expect(isComposerAstEnabled()).toBe(true)
  })

  it('is false when the localStorage override is "0"', () => {
    localStorage.setItem(COMPOSER_AST_STORAGE_KEY, '0')
    expect(isComposerAstEnabled()).toBe(false)
  })
})
