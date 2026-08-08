import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { StrictMode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { queryClient } from '@/lib/query-client'
import { $activeGatewayProfile } from '@/store/profile'
import type { HermesConfigRecord } from '@/types/hermes'

const getHermesConfigRecord = vi.fn()
const getHermesConfigSchema = vi.fn()
const saveHermesConfig = vi.fn()
const getElevenLabsVoices = vi.fn()

vi.mock('@/hermes', () => ({
  getHermesConfigRecord: () => getHermesConfigRecord(),
  getHermesConfigSchema: () => getHermesConfigSchema(),
  saveHermesConfig: (config: unknown) => saveHermesConfig(config),
  getElevenLabsVoices: () => getElevenLabsVoices(),
  // Pulled in via useOnProfileSwitch → @/store/profile.
  getProfiles: async () => ({ profiles: [] }),
  setApiRequestProfile: () => {},
  STARTUP_REQUEST_TIMEOUT_MS: 1000
}))

const voiceConfig = (voice: string): HermesConfigRecord => ({
  tts: { openai: { model: 'gpt-4o-mini-tts', voice } }
})

// The debounce is 550ms; anything comfortably past it proves the timer either
// fired (autosave path) or was cancelled (profile-switch path).
const AUTOSAVE_WINDOW_MS = 700

const settle = (ms: number) =>
  act(async () => {
    await new Promise(resolve => setTimeout(resolve, ms))
  })

beforeEach(() => {
  // Radix popover / cmdk (the free-input combobox) need these on open; jsdom
  // ships neither (mirrors toolset-config-panel.test.tsx).
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
  vi.stubGlobal(
    'ResizeObserver',
    class {
      observe() {}
      unobserve() {}
      disconnect() {}
    }
  )

  getHermesConfigRecord.mockImplementation(async () => voiceConfig('alloy'))
  getHermesConfigSchema.mockResolvedValue({ fields: {} })
  saveHermesConfig.mockResolvedValue({ ok: true })
  getElevenLabsVoices.mockResolvedValue({ available: false, voices: [] })
})

afterEach(() => {
  cleanup()
  queryClient.clear()
  $activeGatewayProfile.set('default')
  vi.clearAllMocks()
  vi.unstubAllGlobals()
})

async function renderVoicePanel() {
  const { VoiceProviderFields } = await import('./voice-provider-fields')

  return render(
    // StrictMode is load-bearing: the app runs under it, and the profile-switch
    // hook regression this file guards only reproduces with Strict Mode's
    // second effect pass.
    <StrictMode>
      <QueryClientProvider client={queryClient}>
        <VoiceProviderFields providerKey="openai" section="tts" />
      </QueryClientProvider>
    </StrictMode>
  )
}

describe('VoiceProviderFields profile switches', () => {
  it('never saves profile A’s draft after a switch, and reseeds profile B’s record', async () => {
    await renderVoicePanel()
    const voiceInput = await screen.findByDisplayValue('alloy')

    // Edit while on profile A — this schedules the 550ms debounced autosave.
    fireEvent.change(voiceInput, { target: { value: 'echo' } })
    expect(await screen.findByDisplayValue('echo')).toBeTruthy()

    // Switch to profile B before the debounce fires. The store-level boundary
    // hard-resets the shared record; the panel must drop the dirty draft and
    // cancel the pending save.
    getHermesConfigRecord.mockImplementation(async () => voiceConfig('nova'))

    act(() => {
      $activeGatewayProfile.set('coder')
    })

    // The empty draft reseeds from profile B's fresh fetch.
    expect(await screen.findByDisplayValue('nova')).toBeTruthy()

    // Let the (cancelled) debounce window elapse: profile A's edited record
    // must never have been PUT — that would write A's config into B.
    await settle(AUTOSAVE_WINDOW_MS)
    expect(saveHermesConfig).not.toHaveBeenCalled()
  })

  it('still autosaves an edit when no switch happens (cancel is not a kill-switch)', async () => {
    await renderVoicePanel()
    const voiceInput = await screen.findByDisplayValue('alloy')

    fireEvent.change(voiceInput, { target: { value: 'echo' } })
    await settle(AUTOSAVE_WINDOW_MS)

    expect(saveHermesConfig).toHaveBeenCalledTimes(1)
    expect(saveHermesConfig.mock.calls[0][0]).toEqual(voiceConfig('echo'))
  })
})
