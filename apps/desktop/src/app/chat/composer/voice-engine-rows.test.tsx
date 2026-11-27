import { VOICE_ENGINE_AREA } from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestProfile } from '@/api/client'
import { DropdownMenu, DropdownMenuContent } from '@/components/ui/dropdown-menu'
import { createPluginContext } from '@/contrib/plugin'
import type { LiveVoiceEngine } from '@/contrib/voice-engines'
import { en } from '@/i18n/en'
import { $voiceSignIn } from '@/store/voice-engine-sign-in'
import { $voiceLiveStatus } from '@/store/voice-live'

import { useVoiceEngineName, VOICE_ENGINE_SETTINGS_HREF, VoiceEngineRows } from './voice-engine-rows'

const mocks = vi.hoisted(() => ({
  listOAuthProviders: vi.fn(),
  requestRoute: vi.fn(),
  setVoiceChatMode: vi.fn(async () => null)
}))

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  listOAuthProviders: mocks.listOAuthProviders
}))

vi.mock('@/store/recovery-requests', () => ({ requestRoute: mocks.requestRoute }))

vi.mock('@/store/voice-live', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  setVoiceChatMode: mocks.setVoiceChatMode
}))

const c = en.composer
const unloads: Array<() => void> = []

function loadEngine(pluginId: string, id: string, engine: Partial<LiveVoiceEngine>) {
  const disposers: Array<() => void> = []
  const ctx = createPluginContext(pluginId, dispose => disposers.push(dispose))

  ctx.register({
    area: VOICE_ENGINE_AREA,
    data: {
      label: 'Fake Voice',
      start: async () => ({ stop: async () => undefined }),
      status: async () => ({ available: true }),
      ...engine
    },
    id
  })
  unloads.push(() => disposers.splice(0).forEach(dispose => dispose()))
}

function renderRows() {
  return render(
    <DropdownMenu open>
      <DropdownMenuContent>
        <VoiceEngineRows disabled={false} />
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

beforeEach(() => {
  $voiceSignIn.set({})
  $voiceLiveStatus.set({ available: true, mode: 'chained', model: 'gpt-live-1', reason: null, voice: 'marin' })
  mocks.listOAuthProviders.mockResolvedValue({
    providers: [
      {
        cli_command: '',
        docs_url: '',
        flow: 'device_code',
        id: 'xai-oauth',
        name: 'xAI Grok',
        status: { logged_in: false, token_preview: 'tok-SHOULD-NOT-RENDER' }
      }
    ]
  })
})

afterEach(() => {
  cleanup()
  unloads.splice(0).forEach(unload => unload())
  vi.clearAllMocks()
  $voiceLiveStatus.set(null)
})

describe('VoiceEngineRows with an extension engine', () => {
  it('lists the engine by its label next to the built-ins and persists its namespaced id', async () => {
    loadEngine('hermes-xai', 'grok', { description: 'full-duplex, delegates to Hermes', label: 'Grok Voice' })
    renderRows()

    expect(screen.getByText(c.voiceEngineChained)).toBeTruthy()
    expect(screen.getByText(c.voiceEngineLive)).toBeTruthy()
    const row = await screen.findByText('Grok Voice (full-duplex, delegates to Hermes)')
    expect(screen.queryByText('hermes-xai:grok')).toBeNull()

    fireEvent.click(row)

    await waitFor(() => expect(mocks.setVoiceChatMode).toHaveBeenCalledWith('hermes-xai:grok'))
  })

  it('shows the engine status reason and keeps an unavailable engine unpickable', async () => {
    loadEngine('hermes-xai', 'grok', {
      label: 'Grok Voice',
      status: async () => ({ available: false, reason: 'Set XAI_API_KEY or sign in with xAI' })
    })
    const { baseElement } = renderRows()

    expect(await screen.findByText('Set XAI_API_KEY or sign in with xAI')).toBeTruthy()
    const item = baseElement.querySelector('[data-voice-engine="hermes-xai:grok"]')
    expect(item?.getAttribute('aria-disabled')).toBe('true')
  })

  it("names the plugin and the active profile when the plugin gate refuses the engine's backend", async () => {
    setApiRequestProfile('work')
    loadEngine('hermes-xai', 'grok', {
      label: 'Grok Voice',
      status: async () => Promise.reject(new Error('404: {"detail":"Plugin not found"}'))
    })

    try {
      const { baseElement } = renderRows()

      expect(await screen.findByText(c.voiceEngineNotEnabled('hermes-xai', 'work'))).toBeTruthy()
      expect(screen.queryByText(/Plugin not found/)).toBeNull()
      const item = baseElement.querySelector('[data-voice-engine="hermes-xai:grok"]')
      expect(item?.getAttribute('aria-disabled')).toBe('true')
    } finally {
      setApiRequestProfile(null)
    }
  })

  it('offers "Sign in with …" for a logged-out provider and opens Settings ▸ Voice', async () => {
    loadEngine('hermes-xai', 'grok', { label: 'Grok Voice', signIn: { provider: 'xai-oauth' } })
    const { baseElement } = renderRows()

    const signIn = await screen.findByText(c.voiceEngineSignIn('xAI Grok'))
    expect(baseElement.textContent).not.toContain('tok-SHOULD-NOT-RENDER')

    fireEvent.click(signIn)

    expect(mocks.requestRoute).toHaveBeenCalledWith(VOICE_ENGINE_SETTINGS_HREF)
  })

  it('has no sign-in item once the provider is signed in', async () => {
    mocks.listOAuthProviders.mockResolvedValue({
      providers: [
        {
          cli_command: '',
          docs_url: '',
          flow: 'device_code',
          id: 'xai-oauth',
          name: 'xAI Grok',
          status: { logged_in: true }
        }
      ]
    })
    loadEngine('hermes-xai', 'grok', { label: 'Grok Voice', signIn: { provider: 'xai-oauth' } })
    renderRows()

    await waitFor(() => expect(mocks.listOAuthProviders).toHaveBeenCalled())
    await act(async () => undefined)

    expect(screen.queryByText(c.voiceEngineSignIn('xAI Grok'))).toBeNull()
  })

  it('checks the selected extension engine', async () => {
    loadEngine('hermes-xai', 'grok', { label: 'Grok Voice' })
    $voiceLiveStatus.set({ available: false, mode: 'hermes-xai:grok', model: '', reason: 'no key', voice: '' })
    const { baseElement } = renderRows()

    await screen.findByText('Grok Voice')
    const item = baseElement.querySelector('[data-voice-engine="hermes-xai:grok"]')
    expect(item?.getAttribute('aria-checked')).toBe('true')
  })
})

describe('useVoiceEngineName', () => {
  function Name() {
    return <span>{useVoiceEngineName()}</span>
  }

  it('names the selected extension engine, and chained when that plugin is gone', () => {
    loadEngine('hermes-xai', 'grok', { label: 'Grok Voice' })
    $voiceLiveStatus.set({ available: false, mode: 'hermes-xai:grok', model: '', reason: null, voice: '' })

    const { container } = render(<Name />)
    expect(container.textContent).toBe('Grok Voice')

    act(() => unloads.splice(0).forEach(unload => unload()))
    expect(container.textContent).toBe(c.voiceEngineChainedShort)
  })
})
