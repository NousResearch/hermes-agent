// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const getGlobalModelInfo = vi.fn()
const getGlobalModelOptions = vi.fn()
const getHermesConfigRecord = vi.fn()
const saveHermesConfig = vi.fn()
const setGlobalModel = vi.fn()
const broadcastDesktopStateChange = vi.fn()
const stores = vi.hoisted(() => ({ activeProfile: null as any }))

vi.mock('@/hermes', () => ({
  getGlobalModelInfo: () => getGlobalModelInfo(),
  getGlobalModelOptions: () => getGlobalModelOptions(),
  getHermesConfigRecord: () => getHermesConfigRecord(),
  saveHermesConfig: (config: unknown) => saveHermesConfig(config),
  setGlobalModel: (provider: string, model: string) => setGlobalModel(provider, model)
}))

vi.mock('@/lib/desktop-state-sync', () => ({
  broadcastDesktopStateChange: (domain: string, options?: unknown) => broadcastDesktopStateChange(domain, options)
}))

vi.mock('@/store/profile', async () => {
  const { atom } = await import('nanostores')
  stores.activeProfile = atom('default')

  return {
    $activeGatewayProfile: stores.activeProfile,
    normalizeProfileKey: (value: string) => value || 'default'
  }
})

import { QuickTab } from './quick-tab'

let config: Record<string, any>
let modelInfo: { model: string; provider: string }

beforeEach(() => {
  stores.activeProfile.set('work')
  config = {
    agent: { reasoning_effort: 'medium' },
    display: { show_reasoning: false },
    stt: { enabled: false, echo_transcripts: false, provider: 'local' },
    tts: { provider: 'edge' },
    voice: { auto_tts: false }
  }
  modelInfo = { provider: 'nous', model: 'hermes-4' }
  getGlobalModelInfo.mockImplementation(async () => ({ ...modelInfo }))
  getGlobalModelOptions.mockResolvedValue({
    providers: [
      { slug: 'nous', name: 'Nous', models: ['hermes-4'] },
      { slug: 'openrouter', name: 'OpenRouter', models: ['openai/gpt-5'] }
    ]
  })
  getHermesConfigRecord.mockImplementation(async () => structuredClone(config))
  saveHermesConfig.mockImplementation(async next => {
    config = structuredClone(next)

    return { ok: true }
  })
  setGlobalModel.mockImplementation(async (provider, model) => {
    modelInfo = { provider, model }

    return { ok: true, provider, model }
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('QuickTab Desktop integration', () => {
  it('writes every quick setting through the shared config API and notifies Desktop', async () => {
    render(<QuickTab />)
    await screen.findByRole('heading', { name: 'Default model' })

    fireEvent.change(screen.getByLabelText('Provider'), { target: { value: 'openrouter' } })
    fireEvent.change(screen.getByLabelText('Model'), { target: { value: 'openai/gpt-5' } })
    fireEvent.click(screen.getByRole('button', { name: 'Apply to Desktop' }))
    await waitFor(() => expect(setGlobalModel).toHaveBeenCalledWith('openrouter', 'openai/gpt-5'))
    await waitFor(() => expect(broadcastDesktopStateChange).toHaveBeenCalledWith('model', { profile: 'work' }))

    const changes: Array<[string, string | boolean]> = [
      ['Reasoning effort', 'ultra'],
      ['Show reasoning in chat', true],
      ['Speech-to-text enabled', true],
      ['Echo transcripts', true],
      ['Speech-to-text provider', 'groq'],
      ['Text-to-speech provider', 'openai'],
      ['Read replies aloud automatically', true]
    ]

    for (const [label, value] of changes) {
      const control = screen.getByLabelText(label)

      if (typeof value === 'boolean') {
        fireEvent.click(control)
      } else {
        fireEvent.change(control, { target: { value } })
      }

      await waitFor(() => expect(control.hasAttribute('disabled')).toBe(false))
    }

    expect(config).toMatchObject({
      agent: { reasoning_effort: 'ultra' },
      display: { show_reasoning: true },
      stt: { enabled: true, echo_transcripts: true, provider: 'groq' },
      tts: { provider: 'openai' },
      voice: { auto_tts: true }
    })
    expect(broadcastDesktopStateChange).toHaveBeenCalledWith('config', { profile: 'work' })
  })

  it('labels an empty model list instead of rendering a blank selector', async () => {
    modelInfo = { provider: 'nous', model: '' }
    getGlobalModelOptions.mockResolvedValue({
      providers: [{ slug: 'nous', name: 'Nous', models: [] }]
    })

    render(<QuickTab />)

    const option = await screen.findByRole('option', { name: 'No models available' })
    expect(option.hasAttribute('disabled')).toBe(true)
    expect(screen.getByLabelText('Model').hasAttribute('disabled')).toBe(true)
    expect(screen.getByRole('button', { name: 'Apply to Desktop' }).hasAttribute('disabled')).toBe(true)
  })

  it('selects the first configured model when Desktop has no default yet', async () => {
    modelInfo = { provider: '', model: '' }
    getGlobalModelOptions.mockResolvedValue({
      providers: [{ slug: 'copilot', name: 'GitHub Copilot', models: ['gpt-5.4'] }]
    })

    render(<QuickTab />)

    await screen.findByRole('heading', { name: 'Default model' })
    expect((screen.getByLabelText('Provider') as HTMLSelectElement).value).toBe('copilot')
    expect((screen.getByLabelText('Model') as HTMLSelectElement).value).toBe('gpt-5.4')
    expect(screen.getByRole('button', { name: 'Apply to Desktop' }).hasAttribute('disabled')).toBe(false)
  })

  it('preserves current future and plugin provider values in its selectors', async () => {
    config = {
      ...config,
      agent: { reasoning_effort: 'future-depth' },
      stt: { ...config.stt, provider: 'custom-stt' },
      tts: { provider: 'custom-tts' }
    }

    render(<QuickTab />)
    await screen.findByRole('heading', { name: 'Default model' })

    expect((screen.getByLabelText('Reasoning effort') as HTMLSelectElement).value).toBe('future-depth')
    expect((screen.getByLabelText('Speech-to-text provider') as HTMLSelectElement).value).toBe('custom-stt')
    expect((screen.getByLabelText('Text-to-speech provider') as HTMLSelectElement).value).toBe('custom-tts')
  })

  it('retries after a backend load failure', async () => {
    getGlobalModelInfo.mockRejectedValueOnce(new Error('Backend offline'))

    render(<QuickTab />)
    fireEvent.click(await screen.findByRole('button', { name: 'Retry' }))

    await screen.findByRole('heading', { name: 'Default model' })
    expect(getGlobalModelInfo).toHaveBeenCalledTimes(2)
  })
})
