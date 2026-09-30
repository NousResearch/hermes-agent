// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $composerContextSuggestions } from '@/store/composer-context-suggestions'

import { ContextSuggestionsSetting } from './context-suggestions-setting'

const mocks = vi.hoisted(() => ({
  cache: vi.fn(),
  loadedConfig: {} as Record<string, unknown>,
  save: vi.fn(),
  writeScope: { connectionId: 'connection-a', profile: 'default' }
}))

vi.mock('@/hermes', () => ({
  saveHermesConfig: (config: Record<string, unknown>, scope?: unknown) =>
    mocks.save(config, scope) ?? Promise.resolve({ ok: true })
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      settings: {
        appearance: {
          contextSuggestionsTitle: 'Suggest context files',
          contextSuggestionsDesc: 'Show file and folder suggestions above the composer while typing @.'
        },
        config: { autosaveFailed: 'Autosave failed' }
      }
    }
  })
}))

vi.mock('@/store/notifications', () => ({
  notifyError: vi.fn()
}))

vi.mock('../hooks/use-config-record', () => ({
  setHermesConfigCache: (config: Record<string, unknown>) => mocks.cache(config),
  useHermesConfigRecord: () => ({
    data: mocks.loadedConfig,
    writeScope: mocks.writeScope
  })
}))

afterEach(cleanup)

function flip() {
  fireEvent.click(screen.getByRole('switch'))
}

describe('ContextSuggestionsSetting', () => {
  beforeEach(() => {
    mocks.cache.mockClear()
    mocks.save.mockClear()
    mocks.loadedConfig = { desktop: { composer: { context_suggestions: true } } }
    $composerContextSuggestions.set(true)
  })

  it('reflects the config value and writes a sparse patch on flip', () => {
    render(<ContextSuggestionsSetting />)

    expect(screen.getByRole('switch').getAttribute('aria-checked')).toBe('true')

    flip()

    expect(mocks.save).toHaveBeenCalledWith(
      { desktop: { composer: { context_suggestions: false } } },
      mocks.writeScope
    )
    expect(mocks.cache).toHaveBeenCalledWith({
      desktop: { composer: { context_suggestions: false } }
    })
  })

  it('defaults to on when the key is absent and clears the renderer store on flip off', () => {
    mocks.loadedConfig = {}
    render(<ContextSuggestionsSetting />)

    expect(screen.getByRole('switch').getAttribute('aria-checked')).toBe('true')

    flip()

    // The renderer store the composer gates on follows the flip immediately.
    expect($composerContextSuggestions.get()).toBe(false)
  })
})
