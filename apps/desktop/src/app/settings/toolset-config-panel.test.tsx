import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render as rtlRender, screen, waitFor } from '@testing-library/react'
import type { ReactElement } from 'react'
import { MemoryRouter } from 'react-router-dom'
import type * as ReactRouterDom from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ToolsetConfig } from '@/types/hermes'

// EnvVarField navigates to Settings → Keys via useNavigate, so every render
// needs a router context. The navigate spy asserts the deep-link target.
const navigateSpy = vi.fn()

vi.mock('react-router-dom', async importOriginal => ({
  ...(await importOriginal<typeof ReactRouterDom>()),
  useNavigate: () => navigateSpy
}))

// The inline VoiceProviderFields reads the shared config record through React
// Query, so the panel needs a QueryClientProvider (fresh per render — cached
// config from one test must not leak into the next).
const render = (ui: ReactElement) =>
  rtlRender(
    <MemoryRouter>
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        {ui}
      </QueryClientProvider>
    </MemoryRouter>
  )

const getToolsetConfig = vi.fn()
const getToolsetModels = vi.fn()
const selectToolsetModel = vi.fn()
const selectToolsetProvider = vi.fn()
const setEnvVar = vi.fn()
const deleteEnvVar = vi.fn()
const revealEnvVar = vi.fn()
const runToolsetPostSetup = vi.fn()
const getActionStatus = vi.fn()
const startOAuthLogin = vi.fn()
const pollOAuthSession = vi.fn()
const getHermesConfigRecord = vi.fn()
const getHermesConfigSchema = vi.fn()
const saveHermesConfig = vi.fn()
const getElevenLabsVoices = vi.fn()

vi.mock('@/hermes', () => ({
  getToolsetConfig: (name: string) => getToolsetConfig(name),
  getToolsetModels: (name: string, provider?: string) => getToolsetModels(name, provider),
  selectToolsetModel: (name: string, model: string, provider?: string) => selectToolsetModel(name, model, provider),
  selectToolsetProvider: (name: string, provider: string, capability?: string) =>
    capability === undefined
      ? selectToolsetProvider(name, provider)
      : selectToolsetProvider(name, provider, capability),
  setEnvVar: (key: string, value: string) => setEnvVar(key, value),
  deleteEnvVar: (key: string) => deleteEnvVar(key),
  revealEnvVar: (key: string) => revealEnvVar(key),
  runToolsetPostSetup: (name: string, key: string) => runToolsetPostSetup(name, key),
  getActionStatus: (name: string, lines?: number) => getActionStatus(name, lines),
  startOAuthLogin: (providerId: string) => startOAuthLogin(providerId),
  pollOAuthSession: (providerId: string, sessionId: string) => pollOAuthSession(providerId, sessionId),
  getHermesConfigRecord: () => getHermesConfigRecord(),
  getHermesConfigSchema: () => getHermesConfigSchema(),
  saveHermesConfig: (config: unknown) => saveHermesConfig(config),
  getElevenLabsVoices: () => getElevenLabsVoices()
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

vi.mock('@/store/activity', () => ({
  upsertDesktopActionTask: vi.fn()
}))

function config(overrides: Partial<ToolsetConfig> = {}): ToolsetConfig {
  return {
    name: 'tts',
    has_category: true,
    active_provider: null,
    providers: [
      {
        name: 'Microsoft Edge TTS',
        badge: 'free',
        tag: 'No API key needed',
        env_vars: [],
        post_setup: null,
        requires_nous_auth: false,
        is_active: false
      },
      {
        name: 'ElevenLabs',
        badge: 'paid',
        tag: 'Most natural voices',
        env_vars: [
          { key: 'ELEVENLABS_API_KEY', prompt: 'ElevenLabs API key', url: 'https://x', default: null, is_set: false }
        ],
        post_setup: null,
        requires_nous_auth: false,
        is_active: false
      }
    ],
    ...overrides
  }
}

beforeEach(() => {
  // Radix menus/selects call these on open; jsdom implements neither, so the
  // dropdown never opens without the stubs (mirrors model-settings.test.tsx).
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()

  getToolsetConfig.mockResolvedValue(config())
  getToolsetModels.mockResolvedValue({
    name: 'tts',
    has_models: false,
    models: [],
    current: null,
    default: null
  })
  selectToolsetModel.mockResolvedValue({ ok: true, name: 'image_gen', model: 'z-image-turbo' })
  selectToolsetProvider.mockResolvedValue({ ok: true, name: 'tts', provider: 'ElevenLabs' })
  setEnvVar.mockResolvedValue({ ok: true })
  deleteEnvVar.mockResolvedValue({ ok: true })
  getHermesConfigRecord.mockResolvedValue({
    tts: {
      provider: 'edge',
      edge: { voice: 'en-US-AriaNeural' },
      openai: { model: 'gpt-4o-mini-tts', voice: 'alloy' },
      elevenlabs: { voice_id: 'pNInz6obpgDQGcFmaJgB', model_id: 'eleven_multilingual_v2' }
    }
  })
  getHermesConfigSchema.mockResolvedValue({ fields: {}, category_order: [] })
  saveHermesConfig.mockResolvedValue({ ok: true })
  getElevenLabsVoices.mockResolvedValue({ available: false, voices: [] })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('ToolsetConfigPanel', () => {
  it('renders inline voice/model fields for a TTS provider row carrying tts_provider', async () => {
    // The Capabilities gap: provider rows only showed API keys — voice/model
    // settings lived exclusively in Settings → Voice. Rows now carry the
    // backend's tts_provider key and the panel renders the same config
    // fields inline (here: OpenAI TTS Model + OpenAI Voice).
    getToolsetConfig.mockResolvedValue(
      config({
        active_provider: 'OpenAI TTS',
        providers: [
          {
            name: 'OpenAI TTS',
            badge: 'paid',
            tag: 'High quality voices',
            env_vars: [
              { key: 'VOICE_TOOLS_OPENAI_KEY', prompt: 'OpenAI API key', url: 'https://x', default: null, is_set: true }
            ],
            post_setup: null,
            requires_nous_auth: false,
            is_active: true,
            tts_provider: 'openai'
          }
        ]
      })
    )

    const { ToolsetConfigPanel } = await import('./toolset-config-panel')
    render(<ToolsetConfigPanel onConfiguredChange={vi.fn()} toolset="tts" />)

    expect(await screen.findByText('OpenAI TTS Model')).toBeTruthy()
    expect(screen.getByText('OpenAI Voice')).toBeTruthy()
    // Voice/model names are free-input comboboxes seeded with the current
    // config value — a custom voice ID must be typeable, not gated by a
    // closed Select.
    const voiceInput = screen.getByDisplayValue('alloy')
    fireEvent.change(voiceInput, { target: { value: 'marin' } })
    await waitFor(() => expect(saveHermesConfig).toHaveBeenCalled(), { timeout: 3000 })
    const saved = saveHermesConfig.mock.calls.at(-1)?.[0] as Record<string, Record<string, Record<string, string>>>
    expect(saved.tts.openai.voice).toBe('marin')
  })

  it('renders no inline voice fields for rows without tts_provider (older backend)', async () => {
    const { ToolsetConfigPanel } = await import('./toolset-config-panel')
    render(<ToolsetConfigPanel onConfiguredChange={vi.fn()} toolset="tts" />)

    await screen.findByText('Microsoft Edge TTS')
    expect(screen.queryByText('Edge Voice')).toBeNull()
    expect(screen.queryByText('OpenAI Voice')).toBeNull()
  })

  it('lists providers from the config endpoint', async () => {
    const { ToolsetConfigPanel } = await import('./toolset-config-panel')
    render(<ToolsetConfigPanel onConfiguredChange={vi.fn()} toolset="tts" />)

    expect(await screen.findByText('Microsoft Edge TTS')).toBeTruthy()
    expect(screen.getByText('ElevenLabs')).toBeTruthy()
    expect(getToolsetConfig).toHaveBeenCalledWith('tts')
  })

  it('selects a provider when clicked', async () => {
    const { ToolsetConfigPanel } = await import('./toolset-config-panel')
    render(<ToolsetConfigPanel onConfiguredChange={vi.fn()} toolset="tts" />)

    const elevenlabs = await screen.findByRole('button', { name: /ElevenLabs/ })
    fireEvent.click(elevenlabs)

    await waitFor(() => expect(selectToolsetProvider).toHaveBeenCalledWith('tts', 'ElevenLabs'))
  })

  it('serializes provider selection while a previous choice is pending', async () => {
    let resolveSelection: (value: { name: string; ok: boolean; provider: string }) => void = () => undefined
    selectToolsetProvider.mockImplementationOnce(
      () =>
        new Promise(resolve => {
          resolveSelection = resolve
        })
    )

    const { ToolsetConfigPanel } = await import('./toolset-config-panel')
    render(<ToolsetConfigPanel onConfiguredChange={vi.fn()} toolset="tts" />)

    const edge = await screen.findByRole('button', { name: /Microsoft Edge TTS/ })
    const elevenlabs = screen.getByRole('button', { name: /ElevenLabs/ })
    fireEvent.click(edge)

    await waitFor(() => expect(selectToolsetProvider).toHaveBeenCalledWith('tts', 'Microsoft Edge TTS'))
    expect(elevenlabs.hasAttribute('disabled')).toBe(true)
    fireEvent.click(elevenlabs)
    expect(selectToolsetProvider).toHaveBeenCalledTimes(1)

    resolveSelection({ name: 'tts', ok: true, provider: 'Microsoft Edge TTS' })
    await waitFor(() => expect(elevenlabs.hasAttribute('disabled')).toBe(false))
  })

  it('shows a backend model catalog for image_gen and persists a pick', async () => {
    getToolsetConfig.mockResolvedValue(
      config({
        name: 'image_gen',
        active_provider: 'FAL.ai',
        providers: [
          {
            name: 'FAL.ai',
            badge: 'paid',
            tag: 'Multi-model image generation',
            env_vars: [],
            post_setup: null,
            requires_nous_auth: false,
            is_active: true
          }
        ]
      })
    )
    getToolsetModels.mockResolvedValue({
      name: 'image_gen',
      has_models: true,
      provider: 'FAL.ai',
      plugin: 'fal',
      models: [
        { id: 'z-image-turbo', display: 'Z-Image Turbo', speed: 'fast', strengths: 'cheap drafts', price: '$0.005' },
        { id: 'flux-2-pro', display: 'FLUX 2 Pro', speed: 'slow', strengths: 'quality', price: '$0.05' }
      ],
      current: 'z-image-turbo',
      default: 'z-image-turbo'
    })

    const { ToolsetConfigPanel } = await import('./toolset-config-panel')
    render(<ToolsetConfigPanel onConfiguredChange={vi.fn()} toolset="image_gen" />)

    // Both catalog rows render with their picker metadata.
    expect(await screen.findByText('Z-Image Turbo')).toBeTruthy()
    expect(screen.getByText('FLUX 2 Pro')).toBeTruthy()
    expect(getToolsetModels).toHaveBeenCalledWith('image_gen', 'FAL.ai')

    // Picking a different model persists via the model endpoint.
    fireEvent.click(screen.getByRole('button', { name: /FLUX 2 Pro/ }))
    await waitFor(() => expect(selectToolsetModel).toHaveBeenCalledWith('image_gen', 'flux-2-pro', 'FAL.ai'))
  })

  it('does not fetch model catalogs for toolsets without them', async () => {
    const { ToolsetConfigPanel } = await import('./toolset-config-panel')
    render(<ToolsetConfigPanel onConfiguredChange={vi.fn()} toolset="tts" />)

    await screen.findByText('Microsoft Edge TTS')
    expect(getToolsetModels).not.toHaveBeenCalled()
  })

  it('saves an API key for a provider env var', async () => {
    const { ToolsetConfigPanel } = await import('./toolset-config-panel')
    render(<ToolsetConfigPanel onConfiguredChange={vi.fn()} toolset="tts" />)

    // Select the keyed provider so its env vars render.
    const elevenlabs = await screen.findByRole('button', { name: /ElevenLabs/ })
    fireEvent.click(elevenlabs)

    // Open the credential actions menu (Radix opens on pointerdown), then "Set".
    const trigger = await screen.findByRole('button', { name: /Actions for ELEVENLABS_API_KEY/ })
    fireEvent.pointerDown(trigger, { button: 0, ctrlKey: false, pointerType: 'mouse' })
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Set' }))

    const input = await screen.findByPlaceholderText('ElevenLabs API key')
    fireEvent.change(input, { target: { value: 'sk-test-123' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    await waitFor(() => expect(setEnvVar).toHaveBeenCalledWith('ELEVENLABS_API_KEY', 'sk-test-123'))
  })

  it('expands the active provider on load, not just the first configured one', async () => {
    // ElevenLabs is the active provider per config, even though the keyless
    // Edge TTS provider sorts first and is also "configured". The panel must
    // honor is_active and expand ElevenLabs (so its API-key field renders)
    // rather than defaulting to the first keyless provider. Regression test
    // for the GUI showing the wrong provider selected after relaunch.
    getToolsetConfig.mockResolvedValue(
      config({
        active_provider: 'ElevenLabs',
        providers: [
          {
            name: 'Microsoft Edge TTS',
            badge: 'free',
            tag: 'No API key needed',
            env_vars: [],
            post_setup: null,
            requires_nous_auth: false,
            is_active: false
          },
          {
            name: 'ElevenLabs',
            badge: 'paid',
            tag: 'Most natural voices',
            env_vars: [
              {
                key: 'ELEVENLABS_API_KEY',
                prompt: 'ElevenLabs API key',
                url: 'https://x',
                default: null,
                is_set: true
              }
            ],
            post_setup: null,
            requires_nous_auth: false,
            is_active: true
          }
        ]
      })
    )

    const { ToolsetConfigPanel } = await import('./toolset-config-panel')
    render(<ToolsetConfigPanel onConfiguredChange={vi.fn()} toolset="tts" />)

    // The active provider's env-var field only renders when it's the expanded
    // one — so finding it proves ElevenLabs (not Edge TTS) was auto-expanded.
    expect(await screen.findByText('ELEVENLABS_API_KEY')).toBeTruthy()
    // No provider selection was triggered — this is purely reflecting state.
    expect(selectToolsetProvider).not.toHaveBeenCalled()
  })

  it('runs a provider post-setup install hook and tails its log', async () => {
    // A browser-style toolset whose active provider declares a post_setup hook.
    getToolsetConfig.mockResolvedValue(
      config({
        name: 'browser',
        active_provider: 'Camofox',
        providers: [
          {
            name: 'Camofox',
            badge: 'local',
            tag: 'Stealth local browser',
            env_vars: [],
            post_setup: 'camofox',
            requires_nous_auth: false,
            is_active: true
          }
        ]
      })
    )
    runToolsetPostSetup.mockResolvedValue({ ok: true, pid: 4321, name: 'tools-post-setup', key: 'camofox' })
    // First poll: still running; second poll: finished cleanly.
    getActionStatus
      .mockResolvedValueOnce({
        exit_code: null,
        lines: ['Installing Camofox browser server...'],
        name: 'tools-post-setup',
        pid: 4321,
        running: true
      })
      .mockResolvedValue({
        exit_code: 0,
        lines: ['Installing Camofox browser server...', "Post-setup 'camofox' complete"],
        name: 'tools-post-setup',
        pid: 4321,
        running: false
      })

    const { ToolsetConfigPanel } = await import('./toolset-config-panel')
    render(<ToolsetConfigPanel onConfiguredChange={vi.fn()} toolset="browser" />)

    fireEvent.click(await screen.findByRole('button', { name: /Run setup/ }))

    await waitFor(() => expect(runToolsetPostSetup).toHaveBeenCalledWith('browser', 'camofox'))
    // The install log is tailed inline. The first poll fires after a 1200ms
    // delay (mirrors command-center's poll cadence), so allow >1200ms here.
    await waitFor(() => expect(getActionStatus).toHaveBeenCalledWith('tools-post-setup', 300), {
      timeout: 4000
    })
  })

  it('does not poll when the spawn endpoint reports ok:false', async () => {
    getToolsetConfig.mockResolvedValue(
      config({
        name: 'browser',
        active_provider: 'Camofox',
        providers: [
          {
            name: 'Camofox',
            badge: 'local',
            tag: 'Stealth local browser',
            env_vars: [],
            post_setup: 'camofox',
            requires_nous_auth: false,
            is_active: true
          }
        ]
      })
    )
    // Spawn failed server-side — must NOT proceed to poll a non-existent action.
    runToolsetPostSetup.mockResolvedValue({ ok: false, pid: 0, name: 'tools-post-setup' })

    const { ToolsetConfigPanel } = await import('./toolset-config-panel')
    render(<ToolsetConfigPanel onConfiguredChange={vi.fn()} toolset="browser" />)

    fireEvent.click(await screen.findByRole('button', { name: /Run setup/ }))

    await waitFor(() => expect(runToolsetPostSetup).toHaveBeenCalledWith('browser', 'camofox'))
    // Give the would-be first poll delay (1200ms) time to NOT fire.
    await new Promise(resolve => setTimeout(resolve, 1500))
    expect(getActionStatus).not.toHaveBeenCalled()
  })

  it('surfaces a non-zero exit code from the setup process', async () => {
    getToolsetConfig.mockResolvedValue(
      config({
        name: 'browser',
        active_provider: 'Camofox',
        providers: [
          {
            name: 'Camofox',
            badge: 'local',
            tag: 'Stealth local browser',
            env_vars: [],
            post_setup: 'camofox',
            requires_nous_auth: false,
            is_active: true
          }
        ]
      })
    )
    runToolsetPostSetup.mockResolvedValue({ ok: true, pid: 4321, name: 'tools-post-setup', key: 'camofox' })
    // Action finished but failed (non-zero exit).
    getActionStatus.mockResolvedValue({
      exit_code: 1,
      lines: ['Installing...', 'npm ERR! install failed'],
      name: 'tools-post-setup',
      pid: 4321,
      running: false
    })

    const { ToolsetConfigPanel } = await import('./toolset-config-panel')
    render(<ToolsetConfigPanel onConfiguredChange={vi.fn()} toolset="browser" />)

    fireEvent.click(await screen.findByRole('button', { name: /Run setup/ }))

    // The failing install log is still tailed and shown; exit_code:1 routes to
    // the error notify branch (asserted via the poll completing on a non-zero
    // status without throwing).
    await waitFor(() => expect(getActionStatus).toHaveBeenCalledWith('tools-post-setup', 300), {
      timeout: 4000
    })
    await waitFor(() => expect(screen.getByText(/npm ERR! install failed/)).toBeTruthy(), {
      timeout: 4000
    })
  })

  it('swaps the install hint for the installed one-liner when the provider is ready', async () => {
    // Server says the post_setup install is already satisfied (status ready) —
    // the "needs a one-time install" copy would contradict the Ready pill.
    getToolsetConfig.mockResolvedValue(
      config({
        name: 'browser',
        active_provider: 'Camofox',
        providers: [
          {
            name: 'Camofox',
            badge: 'local',
            tag: 'Stealth local browser',
            env_vars: [],
            post_setup: 'camofox',
            requires_nous_auth: false,
            is_active: true,
            status: 'ready'
          }
        ]
      })
    )

    const { ToolsetConfigPanel } = await import('./toolset-config-panel')
    render(<ToolsetConfigPanel onConfiguredChange={vi.fn()} toolset="browser" />)

    // Installed confirmation replaces the contradictory install prompt…
    expect(await screen.findByText(/Installed\. Re-run setup only if something is broken\./)).toBeTruthy()
    expect(screen.queryByText(/needs a one-time install/)).toBeNull()
    // …but a repair affordance stays available (c9's resting state renders
    // the low-key Re-run setup button instead of the primary CTA).
    expect(screen.getByRole('button', { name: /Re-run setup/ })).toBeTruthy()
  })

  describe('readiness pills', () => {
    it('renders the server status instead of assuming keyless rows are Ready', async () => {
      // The false-Ready bug: a logged-out Nous Subscription row and a
      // never-installed local TTS both have zero env vars — the old client
      // heuristic pilled every such row "Ready". The server now sends an
      // honest per-provider status; the pill must follow it.
      getToolsetConfig.mockResolvedValue(
        config({
          providers: [
            {
              name: 'Microsoft Edge TTS',
              badge: 'free',
              tag: 'No API key needed',
              env_vars: [],
              post_setup: null,
              requires_nous_auth: false,
              is_active: true,
              status: 'ready'
            },
            {
              name: 'Nous Subscription',
              badge: 'subscription',
              tag: 'Managed OpenAI TTS',
              env_vars: [],
              post_setup: null,
              requires_nous_auth: true,
              is_active: false,
              status: 'needs_auth'
            },
            {
              name: 'KittenTTS',
              badge: 'local · free',
              tag: 'Lightweight local ONNX TTS',
              env_vars: [],
              post_setup: 'kittentts',
              requires_nous_auth: false,
              is_active: false,
              status: 'needs_setup'
            }
          ]
        })
      )

      const { ToolsetConfigPanel } = await import('./toolset-config-panel')
      render(<ToolsetConfigPanel onConfiguredChange={vi.fn()} toolset="tts" />)

      await screen.findByText('Microsoft Edge TTS')
      // Exactly one Ready pill — the genuinely keyless Edge TTS row.
      expect(screen.getAllByText('Ready')).toHaveLength(1)
      expect(screen.getByText('Needs sign-in')).toBeTruthy()
      expect(screen.getByText('Needs setup')).toBeTruthy()
    })

    it('shows no Ready pill for a keyed provider the server marks needs_keys', async () => {
      getToolsetConfig.mockResolvedValue(
        config({
          providers: [
            {
              name: 'ElevenLabs',
              badge: 'paid',
              tag: 'Most natural voices',
              env_vars: [
                {
                  key: 'ELEVENLABS_API_KEY',
                  prompt: 'ElevenLabs API key',
                  url: 'https://x',
                  default: null,
                  is_set: false
                }
              ],
              post_setup: null,
              requires_nous_auth: false,
              is_active: false,
              status: 'needs_keys'
            }
          ]
        })
      )

      const { ToolsetConfigPanel } = await import('./toolset-config-panel')
      render(<ToolsetConfigPanel onConfiguredChange={vi.fn()} toolset="tts" />)

      await screen.findByText('ElevenLabs')
      expect(screen.queryByText('Ready')).toBeNull()
      // Missing keys are signalled by the env-var fields, not a warn pill.
      expect(screen.queryByText('Needs sign-in')).toBeNull()
      expect(screen.queryByText('Needs setup')).toBeNull()
    })

    it('falls back to the env-var heuristic when the backend sends no status', async () => {
      // Older backend (no `status` field): keyless rows keep the legacy
      // Ready pill, keyed-and-unset rows keep no pill. Narrow compat path —
      // desktop and backend update on separate clocks.
      const { ToolsetConfigPanel } = await import('./toolset-config-panel')
      render(<ToolsetConfigPanel onConfiguredChange={vi.fn()} toolset="tts" />)

      await screen.findByText('Microsoft Edge TTS')
      // Default config(): keyless Edge TTS (ready) + unset ElevenLabs (not).
      expect(screen.getAllByText('Ready')).toHaveLength(1)
      expect(screen.queryByText('Needs sign-in')).toBeNull()
      expect(screen.queryByText('Needs setup')).toBeNull()
    })

    it('flips a needs_keys provider to Ready locally after its key is saved', async () => {
      getToolsetConfig.mockResolvedValue(
        config({
          active_provider: 'ElevenLabs',
          providers: [
            {
              name: 'ElevenLabs',
              badge: 'paid',
              tag: 'Most natural voices',
              env_vars: [
                {
                  key: 'ELEVENLABS_API_KEY',
                  prompt: 'ElevenLabs API key',
                  url: 'https://x',
                  default: null,
                  is_set: false
                }
              ],
              post_setup: null,
              requires_nous_auth: false,
              is_active: true,
              status: 'needs_keys'
            }
          ]
        })
      )

      const { ToolsetConfigPanel } = await import('./toolset-config-panel')
      render(<ToolsetConfigPanel onConfiguredChange={vi.fn()} toolset="tts" />)

      expect(await screen.findByText('ELEVENLABS_API_KEY')).toBeTruthy()
      expect(screen.queryByText('Ready')).toBeNull()

      // Save a key: the pill must go Ready from the local envState patch even
      // though the (now stale) server status still says needs_keys.
      const trigger = await screen.findByRole('button', { name: /Actions for ELEVENLABS_API_KEY/ })
      fireEvent.pointerDown(trigger, { button: 0, ctrlKey: false, pointerType: 'mouse' })
      fireEvent.click(await screen.findByRole('menuitem', { name: 'Set' }))
      fireEvent.change(await screen.findByPlaceholderText('ElevenLabs API key'), { target: { value: 'sk-live' } })
      fireEvent.click(screen.getByRole('button', { name: 'Save' }))

      await waitFor(() => expect(screen.getByText('Ready')).toBeTruthy())
    })
  })

  describe('post-setup installed state', () => {
    it('renders Installed + Re-run setup instead of the primary CTA when the server says ready', async () => {
      // Regression (Windows 11 Capabilities journey): "Run setup" rendered
      // unconditionally, so an already-installed backend still showed the
      // primary install CTA and clicking it re-ran the whole npm/Chromium
      // install. status === 'ready' must flip to a resting Installed state.
      getToolsetConfig.mockResolvedValue(
        config({
          name: 'browser',
          active_provider: 'Local Browser',
          providers: [
            {
              name: 'Local Browser',
              badge: 'free',
              tag: 'Headless Chromium, no API key needed',
              env_vars: [],
              post_setup: 'agent_browser',
              requires_nous_auth: false,
              is_active: true,
              status: 'ready'
            }
          ]
        })
      )

      const { ToolsetConfigPanel } = await import('./toolset-config-panel')
      render(<ToolsetConfigPanel onConfiguredChange={vi.fn()} toolset="browser" />)

      await screen.findByText('Local Browser')
      expect(await screen.findByText('Installed')).toBeTruthy()
      expect(await screen.findByRole('button', { name: /Re-run setup/ })).toBeTruthy()
      expect(screen.queryByRole('button', { name: /^Run setup$/ })).toBeNull()
    })

    it('still runs the hook from the Re-run setup affordance', async () => {
      getToolsetConfig.mockResolvedValue(
        config({
          name: 'browser',
          active_provider: 'Local Browser',
          providers: [
            {
              name: 'Local Browser',
              badge: 'free',
              tag: 'Headless Chromium, no API key needed',
              env_vars: [],
              post_setup: 'agent_browser',
              requires_nous_auth: false,
              is_active: true,
              status: 'ready'
            }
          ]
        })
      )
      runToolsetPostSetup.mockResolvedValue({ ok: true, pid: 4321, name: 'tools-post-setup', key: 'agent_browser' })
      getActionStatus.mockResolvedValue({
        exit_code: 0,
        lines: ['agent-browser already installed, nothing to do'],
        name: 'tools-post-setup',
        pid: 4321,
        running: false
      })

      const { ToolsetConfigPanel } = await import('./toolset-config-panel')
      render(<ToolsetConfigPanel onConfiguredChange={vi.fn()} toolset="browser" />)

      fireEvent.click(await screen.findByRole('button', { name: /Re-run setup/ }))

      await waitFor(() => expect(runToolsetPostSetup).toHaveBeenCalledWith('browser', 'agent_browser'))
    })

    it('keeps the primary Run setup CTA when the server says needs_setup', async () => {
      getToolsetConfig.mockResolvedValue(
        config({
          name: 'browser',
          active_provider: 'Local Browser',
          providers: [
            {
              name: 'Local Browser',
              badge: 'free',
              tag: 'Headless Chromium, no API key needed',
              env_vars: [],
              post_setup: 'agent_browser',
              requires_nous_auth: false,
              is_active: true,
              status: 'needs_setup'
            }
          ]
        })
      )

      const { ToolsetConfigPanel } = await import('./toolset-config-panel')
      render(<ToolsetConfigPanel onConfiguredChange={vi.fn()} toolset="browser" />)

      await screen.findByText('Local Browser')
      // The Run setup CTA renders inside the expanded panel, which appears one
      // effect-driven re-render after the row itself — await it (getByRole
      // raced the auto-expand effect and flaked under the RQ provider).
      expect(await screen.findByRole('button', { name: /Run setup/ })).toBeTruthy()
      expect(screen.queryByText('Installed')).toBeNull()
    })
  })

  describe('managed Nous provider activation', () => {
    const nousBrowserConfig = () =>
      config({
        name: 'browser',
        active_provider: null,
        providers: [
          {
            name: 'Nous Subscription (Browser Use cloud)',
            badge: 'subscription',
            tag: 'Managed Browser Use billed to your subscription',
            env_vars: [],
            post_setup: 'agent_browser',
            requires_nous_auth: true,
            is_active: false,
            status: 'needs_auth'
          }
        ]
      })

    it('surfaces a sign-in notice when the PUT reports needs_nous_auth', async () => {
      // Regression (Windows 11 Capabilities journey): the GUI wrote
      // browser.cloud_provider but skipped the Portal entitlement handshake,
      // so the managed row silently never activated. The endpoint now
      // reports needs_nous_auth and the panel must surface a sign-in action
      // instead of the misleading "provider selected" success toast.
      const { notify } = await import('@/store/notifications')

      getToolsetConfig.mockResolvedValue(nousBrowserConfig())
      selectToolsetProvider.mockResolvedValue({
        ok: true,
        name: 'browser',
        provider: 'Nous Subscription (Browser Use cloud)',
        needs_nous_auth: true,
        feature: 'browser'
      })

      const { ToolsetConfigPanel } = await import('./toolset-config-panel')
      render(<ToolsetConfigPanel onConfiguredChange={vi.fn()} toolset="browser" />)

      fireEvent.click(await screen.findByRole('button', { name: /Nous Subscription/ }))

      await waitFor(() =>
        expect(selectToolsetProvider).toHaveBeenCalledWith('browser', 'Nous Subscription (Browser Use cloud)')
      )
      await waitFor(() =>
        expect(notify).toHaveBeenCalledWith(
          expect.objectContaining({
            kind: 'warning',
            action: expect.objectContaining({ label: expect.any(String) })
          })
        )
      )
      // No success toast — the row is not active yet.
      expect(notify).not.toHaveBeenCalledWith(expect.objectContaining({ kind: 'success' }))
    })

    it('drives the existing Nous OAuth device-code flow from the sign-in action and refetches', async () => {
      const { notify } = await import('@/store/notifications')

      getToolsetConfig.mockResolvedValue(nousBrowserConfig())
      selectToolsetProvider.mockResolvedValue({
        ok: true,
        name: 'browser',
        provider: 'Nous Subscription (Browser Use cloud)',
        needs_nous_auth: true,
        feature: 'browser'
      })
      startOAuthLogin.mockResolvedValue({
        flow: 'device_code',
        session_id: 'sess-1',
        user_code: 'NOUS-1234',
        verification_url: 'https://portal.nousresearch.com/device?user_code=NOUS-1234',
        poll_interval: 5,
        expires_in: 600
      })
      pollOAuthSession.mockResolvedValue({ session_id: 'sess-1', status: 'approved' })
      const openSpy = vi.spyOn(window, 'open').mockReturnValue(null)

      try {
        const { ToolsetConfigPanel } = await import('./toolset-config-panel')
        render(<ToolsetConfigPanel onConfiguredChange={vi.fn()} toolset="browser" />)

        fireEvent.click(await screen.findByRole('button', { name: /Nous Subscription/ }))

        // Grab the sign-in action off the warning notification and invoke it —
        // this is the affordance the toast renders as a button.
        await waitFor(() => expect(notify).toHaveBeenCalledWith(expect.objectContaining({ kind: 'warning' })))

        const warning = vi
          .mocked(notify)
          .mock.calls.map(call => call[0])
          .find(input => input.kind === 'warning')

        expect(warning?.action).toBeTruthy()
        getToolsetConfig.mockClear()
        warning!.action!.onClick()

        await waitFor(() => expect(startOAuthLogin).toHaveBeenCalledWith('nous'))
        expect(openSpy).toHaveBeenCalledWith(
          'https://portal.nousresearch.com/device?user_code=NOUS-1234',
          '_blank',
          'noopener,noreferrer'
        )
        // Approved poll → the panel refetches the config so status flips.
        await waitFor(() => expect(pollOAuthSession).toHaveBeenCalledWith('nous', 'sess-1'), { timeout: 8000 })
        await waitFor(() => expect(getToolsetConfig).toHaveBeenCalled(), { timeout: 8000 })
      } finally {
        openSpy.mockRestore()
      }
    }, 20000)

    it('shows the plain success toast when the managed row is already entitled', async () => {
      const { notify } = await import('@/store/notifications')

      getToolsetConfig.mockResolvedValue(nousBrowserConfig())
      selectToolsetProvider.mockResolvedValue({
        ok: true,
        name: 'browser',
        provider: 'Nous Subscription (Browser Use cloud)'
      })

      const { ToolsetConfigPanel } = await import('./toolset-config-panel')
      render(<ToolsetConfigPanel onConfiguredChange={vi.fn()} toolset="browser" />)

      fireEvent.click(await screen.findByRole('button', { name: /Nous Subscription/ }))

      await waitFor(() => expect(notify).toHaveBeenCalledWith(expect.objectContaining({ kind: 'success' })))
      expect(startOAuthLogin).not.toHaveBeenCalled()
    })
  })

  describe('API key deep link', () => {
    it('offers "Manage in API Keys" on a set key and navigates to Settings → Keys', async () => {
      getToolsetConfig.mockResolvedValue(
        config({
          active_provider: 'ElevenLabs',
          providers: [
            {
              name: 'ElevenLabs',
              badge: 'paid',
              tag: 'Most natural voices',
              env_vars: [
                {
                  key: 'ELEVENLABS_API_KEY',
                  prompt: 'ElevenLabs API key',
                  url: 'https://x',
                  default: null,
                  is_set: true
                }
              ],
              post_setup: null,
              requires_nous_auth: false,
              is_active: true,
              status: 'ready'
            }
          ]
        })
      )

      const { ToolsetConfigPanel } = await import('./toolset-config-panel')
      render(<ToolsetConfigPanel onConfiguredChange={vi.fn()} toolset="tts" />)

      const trigger = await screen.findByRole('button', { name: /Actions for ELEVENLABS_API_KEY/ })
      fireEvent.pointerDown(trigger, { button: 0, ctrlKey: false, pointerType: 'mouse' })
      fireEvent.click(await screen.findByRole('menuitem', { name: 'Manage in API Keys' }))

      await waitFor(() => expect(navigateSpy).toHaveBeenCalledWith('/settings?tab=keys&key=ELEVENLABS_API_KEY'))
    })

    it('hides "Manage in API Keys" while the key is unset', async () => {
      // Default config(): ElevenLabs key is not set. An unset key is managed
      // right here via Set — no point bouncing the user to another page.
      const { ToolsetConfigPanel } = await import('./toolset-config-panel')
      render(<ToolsetConfigPanel onConfiguredChange={vi.fn()} toolset="tts" />)

      // Expand the keyed provider so its env row renders. Wait for the
      // selection to commit: on a freshly loaded panel this races the default
      // provider initializer, and user intent must win that race.
      const elevenLabs = await screen.findByRole('button', { name: /ElevenLabs/ })
      fireEvent.click(elevenLabs)
      await waitFor(() => expect(elevenLabs.getAttribute('aria-pressed')).toBe('true'))

      const trigger = await screen.findByRole('button', { name: /Actions for ELEVENLABS_API_KEY/ })
      fireEvent.pointerDown(trigger, { button: 0, ctrlKey: false, pointerType: 'mouse' })

      await screen.findByRole('menuitem', { name: 'Set' })
      expect(screen.queryByRole('menuitem', { name: 'Manage in API Keys' })).toBeNull()
    })
  })

  describe('web capability split', () => {
    function webConfig(overrides: Partial<ToolsetConfig> = {}): ToolsetConfig {
      return {
        name: 'web',
        has_category: true,
        active_provider: 'SearXNG',
        active_search_backend: 'searxng',
        active_extract_backend: 'firecrawl',
        providers: [
          {
            name: 'SearXNG',
            badge: 'free · self-hosted',
            tag: 'Free metasearch',
            env_vars: [],
            post_setup: null,
            requires_nous_auth: false,
            is_active: true,
            status: 'ready',
            web_backend: 'searxng',
            capabilities: ['search']
          },
          {
            name: 'Firecrawl',
            badge: 'paid',
            tag: 'Full search + extract',
            env_vars: [],
            post_setup: null,
            requires_nous_auth: false,
            is_active: false,
            status: 'ready',
            web_backend: 'firecrawl',
            capabilities: ['search', 'extract']
          }
        ],
        ...overrides
      }
    }

    it('shows the resolved per-capability backends as badges', async () => {
      getToolsetConfig.mockResolvedValue(webConfig())

      const { ToolsetConfigPanel } = await import('./toolset-config-panel')
      render(<ToolsetConfigPanel onConfiguredChange={vi.fn()} toolset="web" />)

      expect(await screen.findByText('Search: searxng')).toBeTruthy()
      expect(screen.getByText('Extract: firecrawl')).toBeTruthy()
      // The row backing each capability gets an assignment pill.
      expect(screen.getByText('Search backend')).toBeTruthy()
      expect(screen.getByText('Extract backend')).toBeTruthy()
    })

    it('hides "Use for Extract" on a search-only provider and wires capability selection', async () => {
      getToolsetConfig.mockResolvedValue(webConfig())
      selectToolsetProvider.mockResolvedValue({ ok: true, name: 'web', provider: 'SearXNG', capability: 'search' })

      const { ToolsetConfigPanel } = await import('./toolset-config-panel')
      render(<ToolsetConfigPanel onConfiguredChange={vi.fn()} toolset="web" />)

      // Active/expanded provider is search-only SearXNG.
      await screen.findByText('Search: searxng')
      expect(await screen.findByRole('button', { name: 'Use for Search' })).toBeTruthy()
      expect(screen.queryByRole('button', { name: 'Use for Extract' })).toBeNull()

      // Expand Firecrawl (search + extract) and assign it as the search backend.
      fireEvent.click(screen.getByRole('button', { name: /Firecrawl/ }))
      const useForSearch = await screen.findByRole('button', { name: 'Use for Search' })
      fireEvent.click(useForSearch)

      await waitFor(() => expect(selectToolsetProvider).toHaveBeenCalledWith('web', 'Firecrawl', 'search'))
      // Badge tracks the local write without a refetch.
      await waitFor(() => expect(screen.getByText('Search: firecrawl')).toBeTruthy())
    })

    it('does not render capability chrome for non-web toolsets', async () => {
      const { ToolsetConfigPanel } = await import('./toolset-config-panel')
      render(<ToolsetConfigPanel onConfiguredChange={vi.fn()} toolset="tts" />)

      await screen.findByText('Microsoft Edge TTS')
      expect(screen.queryByText(/^Search: /)).toBeNull()
      expect(screen.queryByRole('button', { name: 'Use for Search' })).toBeNull()
    })
  })
})
