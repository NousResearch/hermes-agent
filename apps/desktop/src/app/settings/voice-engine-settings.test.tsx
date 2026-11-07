import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { atom } from 'nanostores'
import { createRef } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as ConfigApi from '@/api/config'
import { createPluginContext } from '@/contrib/plugin'
import { type LiveVoiceEngine, VOICE_ENGINE_AREA } from '@/contrib/voice-engines'
import { en } from '@/i18n/en'
import { $agentPlugins, $agentPluginsProfile, $agentPluginsStatus, type AgentPluginRow } from '@/store/agent-plugins'
import { $voiceSignIn } from '@/store/voice-engine-sign-in'

import type { ConfigSettings as ConfigSettingsType } from './config-settings'

// Settings ▸ Voice ▸ Conversation with an extension engine registered: the
// engine is a labelled Voice Chat Mode option, and, once selected, its
// provider sign-in row and the plugin's own settings keys render under it.
const api = vi.hoisted(() => ({
  cancelOAuthSession: vi.fn(async () => ({ ok: true })),
  disconnectOAuthProvider: vi.fn(async () => ({ ok: true, provider: 'xai-oauth' })),
  getElevenLabsVoices: vi.fn(async () => ({ available: false })),
  getHermesConfigRecord: vi.fn(),
  getHermesConfigSchema: vi.fn(),
  getProfiles: vi.fn(async () => ({ active: 'default', profiles: [] })),
  listOAuthProviders: vi.fn(),
  pollOAuthSession: vi.fn(),
  saveHermesConfig: vi.fn(async () => ({ ok: true })),
  startOAuthLogin: vi.fn()
}))

const confirmMock = vi.hoisted(() => vi.fn(async () => true))
const openExternal = vi.hoisted(() => vi.fn())

vi.mock('@/hermes', async () => ({
  ...(await vi.importActual<typeof ConfigApi>('@/api/config')),
  ...api,
  getHermesConfigRecord: (profile?: string) => api.getHermesConfigRecord(profile),
  getHermesConfigSchema: () => api.getHermesConfigSchema(),
  profileScopeKey: (scope?: unknown) => (typeof scope === 'string' && scope.trim()) || 'default',
  setApiRequestProfile: () => {}
}))

vi.mock('@/lib/long-poll', () => ({ restOfLongPollGap: async () => undefined }))
vi.mock('@/lib/external-link', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  openExternalLink: openExternal
}))
vi.mock('@/store/confirm', () => ({ confirm: confirmMock }))
vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({ useGatewayRequest: () => ({ requestGateway: vi.fn() }) }))
vi.mock('../hooks/use-on-profile-switch', () => ({ useOnProfileSwitch: () => {} }))

vi.mock('@/store/settings-scope', () => ({
  $settingsRequestProfile: atom<string | undefined>('default'),
  $settingsScopeEditsNonDefault: atom(false),
  $settingsScopeOverride: atom<null | string>(null),
  $settingsScopeProfile: atom<string>('default')
}))

vi.mock('@/store/projects', () => ({
  $worktreeRefreshToken: atom(0),
  repoDiscoveryPolicyFromConfig: () => ({ enabled: true, exclude_paths: [], roots: [] }),
  repoDiscoveryPolicySignature: (policy: unknown) => JSON.stringify(policy),
  scanAndRecordRepos: vi.fn().mockResolvedValue(undefined)
}))

let ConfigSettings: typeof ConfigSettingsType

beforeAll(async () => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
  ;({ ConfigSettings } = await import('./config-settings'))
}, 60_000)

const c = en.composer
const unloads: Array<() => void> = []

function loadGrok(overrides: Partial<LiveVoiceEngine> = {}) {
  const disposers: Array<() => void> = []
  const ctx = createPluginContext('hermes-xai', dispose => disposers.push(dispose))

  ctx.register({
    area: VOICE_ENGINE_AREA,
    data: {
      label: 'Grok Voice',
      settingsKeys: ['voice', 'instructions'],
      signIn: { provider: 'xai-oauth' },
      start: async () => ({ stop: async () => undefined }),
      status: async () => ({ available: true }),
      ...overrides
    },
    id: 'grok'
  })
  unloads.push(() => disposers.splice(0).forEach(dispose => dispose()))
}

const xaiRow = {
  description: 'xAI tools',
  enabled: true,
  install_dir: '/home/u/.hermes/plugins/hermes-xai',
  key: 'hermes-xai',
  name: 'hermes-xai',
  settings_schema: [
    { description: 'Grok voice name', key: 'voice', label: 'voice', required: false, type: 'string', value: 'Ara' },
    { description: 'Persona', key: 'instructions', label: 'instructions', required: false, type: 'string', value: '' },
    { description: 'Collections', key: 'collection_id', label: 'collection_id', required: false, type: 'string' }
  ]
} as unknown as AgentPluginRow

function provider(loggedIn: boolean) {
  return {
    providers: [
      {
        cli_command: '',
        disconnectable: true,
        docs_url: '',
        flow: 'device_code',
        id: 'xai-oauth',
        name: 'xAI Grok',
        status: { logged_in: loggedIn, token_preview: loggedIn ? 'tok-NEVER-SHOWN' : null }
      }
    ]
  }
}

function renderVoiceConversation(mode: string) {
  api.getHermesConfigRecord.mockResolvedValue({ voice: { voice_chat_mode: mode } })
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  return render(
    <MemoryRouter>
      <QueryClientProvider client={client}>
        <ConfigSettings activeSectionId="voice" importInputRef={createRef()} subpage="conversation" />
      </QueryClientProvider>
    </MemoryRouter>
  )
}

beforeEach(() => {
  $voiceSignIn.set({})
  $agentPlugins.set([xaiRow])
  $agentPluginsProfile.set('default')
  $agentPluginsStatus.set('ready')
  api.getHermesConfigSchema.mockResolvedValue({ fields: { 'voice.voice_chat_mode': { type: 'string' } } })
  api.listOAuthProviders.mockResolvedValue(provider(false))
})

afterEach(() => {
  cleanup()
  unloads.splice(0).forEach(unload => unload())
  vi.clearAllMocks()
})

describe('Settings ▸ Voice with an extension engine', () => {
  it('separates Duplex mode from its engine without changing persisted identifiers', async () => {
    loadGrok()
    renderVoiceConversation('gpt-live')

    const mode = await screen.findByRole('combobox', { name: 'Voice Chat Mode' })
    const engine = await screen.findByRole('combobox', { name: c.voiceEngine })

    expect(mode.textContent).toContain('Duplex')
    expect(engine.textContent).toContain('OpenAI')
    fireEvent.click(engine)
    fireEvent.click(await screen.findByRole('option', { name: 'Grok Voice' }))

    await waitFor(() =>
      expect(api.saveHermesConfig).toHaveBeenCalledWith({ voice: { voice_chat_mode: 'hermes-xai:grok' } }, 'default')
    )
    expect(mode.textContent).toContain('Duplex')
  })

  it('offers Auto and chosen response languages for the native OpenAI engine', async () => {
    api.getHermesConfigSchema.mockResolvedValue({
      fields: {
        'voice.voice_chat_mode': { type: 'string' },
        'voice.gpt_live.language': { type: 'string' }
      }
    })
    api.getHermesConfigRecord.mockResolvedValueOnce({
      voice: { voice_chat_mode: 'gpt-live', gpt_live: { language: 'auto' } }
    })
    renderVoiceConversation('gpt-live')

    const languageLabel = await screen.findByText('OpenAI Response Language')
    const languageRow = languageLabel.closest('div[id]') as HTMLElement

    fireEvent.click(within(languageRow).getByRole('combobox'))
    expect(await screen.findByRole('option', { name: 'Auto' })).toBeTruthy()
    fireEvent.click(await screen.findByRole('option', { name: 'French (fr)' }))

    await waitFor(() =>
      expect(api.saveHermesConfig).toHaveBeenCalledWith({ voice: { gpt_live: { language: 'fr' } } }, 'default')
    )
  })

  it('offers the engine by label, never its raw id', async () => {
    loadGrok()
    renderVoiceConversation('chained')

    fireEvent.click(await screen.findByRole('combobox', { name: 'Voice Chat Mode' }))
    fireEvent.click(await screen.findByRole('option', { name: 'Duplex' }))
    fireEvent.click(await screen.findByRole('combobox', { name: c.voiceEngine }))
    expect(await screen.findByRole('option', { name: 'Grok Voice' })).toBeTruthy()
    expect(screen.queryByText('hermes-xai:grok')).toBeNull()
  })

  it('names a saved engine whose plugin is gone instead of dropping the value', async () => {
    renderVoiceConversation('hermes-xai:grok')

    expect(await screen.findByText(c.voiceEngineMissing('hermes-xai:grok'))).toBeTruthy()
  })

  it('renders the sign-in row and only the settings keys the engine asked for', async () => {
    loadGrok()
    renderVoiceConversation('hermes-xai:grok')

    expect(await screen.findByText(c.voiceEngineNotSignedIn('xAI Grok'))).toBeTruthy()
    expect(api.listOAuthProviders).toHaveBeenCalledWith('default')
    expect(screen.getByLabelText('Voice')).toBeTruthy()
    expect(screen.getByLabelText('Instructions')).toBeTruthy()
    expect(screen.queryByLabelText('Collection ID')).toBeNull()
  })

  it('signs in through the device-code flow, showing the code, never a token', async () => {
    loadGrok()
    api.startOAuthLogin.mockResolvedValue({
      expires_in: 900,
      flow: 'device_code',
      poll_interval: 5,
      session_id: 's1',
      user_code: 'WXYZ-9876',
      verification_url: 'https://accounts.x.ai/device'
    })
    let approve: (value: unknown) => void = () => undefined
    api.pollOAuthSession.mockImplementation(() => new Promise(resolve => (approve = resolve)))
    const { baseElement } = renderVoiceConversation('hermes-xai:grok')

    fireEvent.click(await screen.findByRole('button', { name: c.voiceEngineSignInAction }))

    expect(await screen.findByText(c.voiceEngineSignInCode('WXYZ-9876'))).toBeTruthy()
    expect(openExternal).toHaveBeenCalledWith('https://accounts.x.ai/device')
    expect(api.startOAuthLogin).toHaveBeenCalledWith('xai-oauth', 'default')

    api.listOAuthProviders.mockResolvedValue(provider(true))
    await act(async () => approve({ session_id: 's1', status: 'approved' }))

    expect(await screen.findByText(c.voiceEngineSignedIn('xAI Grok'))).toBeTruthy()
    expect(baseElement.textContent).not.toContain('tok-NEVER-SHOWN')
  })

  it('cancels a pending sign-in', async () => {
    loadGrok()
    api.startOAuthLogin.mockResolvedValue({
      expires_in: 900,
      flow: 'device_code',
      poll_interval: 5,
      session_id: 's2',
      user_code: 'CODE-0000',
      verification_url: 'https://accounts.x.ai/device'
    })
    api.pollOAuthSession.mockImplementation(() => new Promise(() => undefined))
    renderVoiceConversation('hermes-xai:grok')

    fireEvent.click(await screen.findByRole('button', { name: c.voiceEngineSignInAction }))
    fireEvent.click(await screen.findByRole('button', { name: en.common.cancel }))

    await waitFor(() => expect(api.cancelOAuthSession).toHaveBeenCalledWith('s2', 'default'))
    expect(await screen.findByText(c.voiceEngineNotSignedIn('xAI Grok'))).toBeTruthy()
  })

  it('signs out only after a confirmation that names what it removes', async () => {
    loadGrok()
    api.listOAuthProviders.mockResolvedValue(provider(true))
    renderVoiceConversation('hermes-xai:grok')

    fireEvent.click(await screen.findByRole('button', { name: c.voiceEngineSignOutAction }))

    await waitFor(() => expect(api.disconnectOAuthProvider).toHaveBeenCalledWith('xai-oauth', 'default'))
    expect(confirmMock).toHaveBeenCalledWith(
      expect.objectContaining({ description: c.voiceEngineSignOutBody('xAI Grok'), destructive: true })
    )
  })

  it('keeps the sign-in when the confirmation is declined', async () => {
    loadGrok()
    confirmMock.mockResolvedValueOnce(false)
    api.listOAuthProviders.mockResolvedValue(provider(true))
    renderVoiceConversation('hermes-xai:grok')

    fireEvent.click(await screen.findByRole('button', { name: c.voiceEngineSignOutAction }))
    await act(async () => undefined)

    expect(api.disconnectOAuthProvider).not.toHaveBeenCalled()
  })
})
