// @vitest-environment jsdom
import { renderHook } from '@testing-library/react'
import { type ReactNode } from 'react'
import { afterEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $voiceLiveGrokStatus, $voiceLiveStatus } from '@/store/voice-live'

import { useVoiceEngineName } from './voice-engine-rows'

function wrapper({ children }: { children: ReactNode }) {
  return <I18nProvider configClient={null} initialLocale="en">{children}</I18nProvider>
}

afterEach(() => {
  $voiceLiveStatus.set(null)
  $voiceLiveGrokStatus.set(null)
})

describe('useVoiceEngineName', () => {
  it('is null until the backend has answered', () => {
    const { result } = renderHook(() => useVoiceEngineName(), { wrapper })

    expect(result.current).toBeNull()
  })

  it('names the chained engine when the backend selected it', () => {
    $voiceLiveStatus.set({ available: false, mode: 'chained', model: '', reason: null, voice: '' })

    const { result } = renderHook(() => useVoiceEngineName(), { wrapper })

    expect(result.current).toBe('speech-to-text')
  })

  it('names GPT-Live when selected', () => {
    $voiceLiveStatus.set({ available: true, mode: 'gpt-live', model: 'gpt-live-1', reason: null, voice: 'marin' })

    const { result } = renderHook(() => useVoiceEngineName(), { wrapper })

    expect(result.current).toBe('GPT-Live')
  })

  it('names Grok Live when selected — the new third engine', () => {
    $voiceLiveStatus.set({ available: true, mode: 'grok-live', model: 'grok-voice-latest', reason: null, voice: 'eve' })

    const { result } = renderHook(() => useVoiceEngineName(), { wrapper })

    expect(result.current).toBe('Grok Live')
  })
})
