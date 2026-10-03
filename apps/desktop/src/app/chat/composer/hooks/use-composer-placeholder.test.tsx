import { cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'
import { ko } from '@/i18n/ko'
import { $perSessionBrowse, browseBackward, isBrowsingHistory } from '@/store/composer-input-history'

import { useComposerPlaceholder } from './use-composer-placeholder'

let activeLocale: 'en' | 'ko' = 'en'

vi.mock('@/i18n', () => ({
  useI18n: () => ({ locale: activeLocale, t: activeLocale === 'ko' ? ko : en })
}))
vi.mock('@/store/session', () => ({ setSessionPickerOpen: vi.fn() }))

interface PlaceholderProps {
  disabled: boolean
  reconnecting: boolean
  sessionId: null | string
}

const ready: PlaceholderProps = { disabled: false, reconnecting: false, sessionId: 'existing' }

describe('useComposerPlaceholder locale and conversation changes', () => {
  beforeEach(() => {
    activeLocale = 'en'
    $perSessionBrowse.set({})
    vi.spyOn(Math, 'random').mockReturnValue(0.3)
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    $perSessionBrowse.set({})
  })

  it('updates the mounted hint with its locale while preserving history, choice, and connection-state copy', () => {
    const { result, rerender } = renderHook(props => useComposerPlaceholder(props), { initialProps: ready })
    const english = result.current
    const randomCalls = vi.mocked(Math.random).mock.calls.length

    expect(en.composer.followUpPlaceholders).toContain(english)
    browseBackward('existing', 'draft to preserve', ['previous input'])
    activeLocale = 'ko'
    rerender(ready)

    expect(ko.composer.followUpPlaceholders).toContain(result.current)
    expect(isBrowsingHistory('existing')).toBe(true)
    const korean = result.current
    rerender(ready)
    expect(result.current).toBe(korean)
    expect(vi.mocked(Math.random).mock.calls.length).toBe(randomCalls)

    rerender({ ...ready, disabled: true })
    expect(result.current).toBe(ko.composer.placeholderStarting)
    rerender({ ...ready, disabled: true, reconnecting: true })
    expect(result.current).toBe(ko.composer.placeholderReconnecting)
    rerender(ready)
    expect(result.current).toBe(korean)

    activeLocale = 'en'
    rerender(ready)
    expect(result.current).toBe(english)
    expect(vi.mocked(Math.random).mock.calls.length).toBe(randomCalls)
  })

  it('keeps a starter through draft persistence and selects anew only for a different conversation', () => {
    const fresh: PlaceholderProps = { ...ready, sessionId: null }
    const persisted: PlaceholderProps = { ...ready, sessionId: 'saved-draft' }
    const { result, rerender } = renderHook(props => useComposerPlaceholder(props), { initialProps: fresh })
    const starter = result.current
    const randomCalls = vi.mocked(Math.random).mock.calls.length

    expect(en.composer.newSessionPlaceholders).toContain(starter)
    rerender(persisted)
    expect(result.current).toBe(starter)
    expect(vi.mocked(Math.random).mock.calls.length).toBe(randomCalls)

    activeLocale = 'ko'
    rerender(persisted)
    expect(ko.composer.newSessionPlaceholders).toContain(result.current)
    expect(vi.mocked(Math.random).mock.calls.length).toBe(randomCalls)

    browseBackward('saved-draft', 'draft', ['previous input'])
    rerender({ ...ready, sessionId: 'another-conversation' })
    expect(ko.composer.followUpPlaceholders).toContain(result.current)
    expect(isBrowsingHistory('saved-draft')).toBe(false)
    expect(vi.mocked(Math.random).mock.calls.length).toBe(randomCalls + 1)

    rerender(fresh)
    expect(ko.composer.newSessionPlaceholders).toContain(result.current)
    expect(vi.mocked(Math.random).mock.calls.length).toBe(randomCalls + 2)
    const nextStarter = result.current
    rerender(fresh)
    expect(result.current).toBe(nextStarter)
    expect(vi.mocked(Math.random).mock.calls.length).toBe(randomCalls + 2)
  })
})
