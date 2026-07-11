import { useStore } from '@nanostores/react'
import * as React from 'react'

import {
  getGlobalModelInfo,
  getGlobalModelOptions,
  getHermesConfigRecord,
  saveHermesConfig,
  setGlobalModel
} from '@/hermes'
import { broadcastDesktopStateChange } from '@/lib/desktop-state-sync'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import type { HermesConfigRecord, ModelOptionProvider } from '@/types/hermes'

import { getNested, setNested } from '../lib/config-patch'

type LoadState = 'error' | 'loading' | 'ready'

const REASONING_OPTIONS = ['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'ultra']
const STT_PROVIDERS = ['local', 'groq', 'openai', 'mistral', 'xai', 'elevenlabs']

const TTS_PROVIDERS = [
  'edge',
  'openai',
  'elevenlabs',
  'xai',
  'minimax',
  'mistral',
  'gemini',
  'neutts',
  'kittentts',
  'piper'
]

function optionsWithCurrent(options: readonly string[], current: string): string[] {
  return current && !options.includes(current) ? [...options, current] : [...options]
}

export function QuickTab() {
  const activeProfile = normalizeProfileKey(useStore($activeGatewayProfile))
  const [state, setState] = React.useState<LoadState>('loading')
  const [error, setError] = React.useState('')
  const [busy, setBusy] = React.useState(false)
  const [provider, setProvider] = React.useState('')
  const [model, setModel] = React.useState('')
  const [providers, setProviders] = React.useState<ModelOptionProvider[]>([])
  const [config, setConfig] = React.useState<HermesConfigRecord | null>(null)
  const [status, setStatus] = React.useState('')

  const refresh = React.useCallback(async () => {
    setState('loading')
    setError('')

    try {
      const [info, options, record] = await Promise.all([
        getGlobalModelInfo(),
        getGlobalModelOptions(),
        getHermesConfigRecord()
      ])

      const availableProviders = options.providers || []

      const fallbackProvider =
        availableProviders.find(entry => (entry.models?.length ?? 0) > 0) || availableProviders[0]

      const nextProvider = info.provider || fallbackProvider?.slug || ''
      const providerOptions = availableProviders.find(entry => entry.slug === nextProvider)
      const nextModel = info.model || providerOptions?.models?.[0] || ''

      setProvider(nextProvider)
      setModel(nextModel)
      setProviders(availableProviders)
      setConfig(record)
      setState('ready')
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      setState('error')
    }
  }, [])

  React.useEffect(() => {
    void refresh()
  }, [refresh])

  const providerRow = providers.find(entry => entry.slug === provider)
  const models = providerRow?.models ?? []

  const sttEnabled = Boolean(getNested(config || {}, 'stt.enabled'))
  const sttProvider = String(getNested(config || {}, 'stt.provider') || 'local')
  const ttsProvider = String(getNested(config || {}, 'tts.provider') || 'edge')
  const reasoning = String(getNested(config || {}, 'agent.reasoning_effort') || 'xhigh')
  const showReasoning = Boolean(getNested(config || {}, 'display.show_reasoning'))
  const autoTts = Boolean(getNested(config || {}, 'voice.auto_tts'))
  const echoTranscripts = Boolean(getNested(config || {}, 'stt.echo_transcripts'))
  const reasoningOptions = optionsWithCurrent(REASONING_OPTIONS, reasoning)
  const sttProviderOptions = optionsWithCurrent(STT_PROVIDERS, sttProvider)
  const ttsProviderOptions = optionsWithCurrent(TTS_PROVIDERS, ttsProvider)

  const onSaveModel = async () => {
    if (!provider || !model) {
      return
    }

    setBusy(true)
    setStatus('')

    try {
      const result = await setGlobalModel(provider, model)
      setProvider(result.provider)
      setModel(result.model)
      // Read-after-write proof
      const proof = await getGlobalModelInfo()

      if (proof.provider !== result.provider || proof.model !== result.model) {
        throw new Error('Model write did not stick — re-check Desktop Settings → Model')
      }

      broadcastDesktopStateChange('model', { profile: activeProfile })
      setStatus(`Desktop model updated → ${result.provider} / ${result.model}`)
      await refresh()
    } catch (err) {
      setStatus(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  const patchConfig = async (path: string, value: unknown, label: string) => {
    if (!config) {
      return
    }

    setBusy(true)
    setStatus('')

    try {
      const next = setNested(config, path, value)
      await saveHermesConfig(next)
      const proof = await getHermesConfigRecord()
      const proved = getNested(proof, path)

      if (proved !== value && String(proved) !== String(value)) {
        throw new Error(`${label} write did not stick (got ${String(proved)})`)
      }

      setConfig(proof)
      broadcastDesktopStateChange('config', { profile: activeProfile })
      setStatus(`${label} updated in Desktop`)
    } catch (err) {
      setStatus(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  if (state === 'loading') {
    return (
      <div className="mbc-tab-panel mbc-stack" data-tab="quick">
        <section className="mbc-card">
          <p className="mbc-muted">Loading model + config from Desktop backend…</p>
        </section>
      </div>
    )
  }

  if (state === 'error') {
    return (
      <div className="mbc-tab-panel mbc-stack" data-tab="quick">
        <section className="mbc-card">
          <h3>Backend unreachable</h3>
          <p className="mbc-muted">{error || 'Could not load Desktop model/config APIs.'}</p>
          <button className="mbc-button" onClick={() => void refresh()} type="button">
            Retry
          </button>
        </section>
      </div>
    )
  }

  return (
    <div className="mbc-tab-panel mbc-stack" data-tab="quick">
      <section className="mbc-card">
        <h3>Default model</h3>
        <p className="mbc-muted">Default for new Desktop sessions.</p>
        <label className="mbc-field">
          <span>Provider</span>
          <select
            disabled={busy}
            onChange={event => {
              const nextProvider = event.target.value
              setProvider(nextProvider)
              const next = providers.find(entry => entry.slug === nextProvider)
              setModel(next?.models?.[0] || '')
            }}
            value={provider}
          >
            {providers.map(entry => (
              <option key={entry.slug} value={entry.slug}>
                {entry.name || entry.slug}
              </option>
            ))}
          </select>
        </label>
        <label className="mbc-field">
          <span>Model</span>
          <select disabled={busy || models.length === 0} onChange={event => setModel(event.target.value)} value={model}>
            {models.length === 0 ? (
              <option disabled value="">
                No models available
              </option>
            ) : (
              models.map(entry => (
                <option key={entry} value={entry}>
                  {entry}
                </option>
              ))
            )}
          </select>
        </label>
        <button
          className="mbc-button"
          disabled={busy || !provider || !model}
          onClick={() => void onSaveModel()}
          type="button"
        >
          Apply to Desktop
        </button>
      </section>

      <section className="mbc-card">
        <h3>Reasoning</h3>
        <label className="mbc-field">
          <span>Reasoning effort</span>
          <select
            disabled={busy}
            onChange={event => void patchConfig('agent.reasoning_effort', event.target.value, 'Reasoning')}
            value={reasoning}
          >
            {reasoningOptions.map(value => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>
        <label className="mbc-check">
          <input
            checked={showReasoning}
            disabled={busy}
            onChange={event => void patchConfig('display.show_reasoning', event.target.checked, 'Show reasoning')}
            type="checkbox"
          />
          <span>Show reasoning in chat</span>
        </label>
      </section>

      <section className="mbc-card">
        <h3>Speech-to-text</h3>
        <label className="mbc-check">
          <input
            checked={sttEnabled}
            disabled={busy}
            onChange={event => void patchConfig('stt.enabled', event.target.checked, 'STT enabled')}
            type="checkbox"
          />
          <span>Speech-to-text enabled</span>
        </label>
        <label className="mbc-check">
          <input
            checked={echoTranscripts}
            disabled={busy}
            onChange={event => void patchConfig('stt.echo_transcripts', event.target.checked, 'Echo transcripts')}
            type="checkbox"
          />
          <span>Echo transcripts</span>
        </label>
        <label className="mbc-field">
          <span>Speech-to-text provider</span>
          <select
            disabled={busy}
            onChange={event => void patchConfig('stt.provider', event.target.value, 'STT provider')}
            value={sttProvider}
          >
            {sttProviderOptions.map(value => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>
      </section>

      <section className="mbc-card">
        <h3>Text-to-speech</h3>
        <label className="mbc-field">
          <span>Text-to-speech provider</span>
          <select
            disabled={busy}
            onChange={event => void patchConfig('tts.provider', event.target.value, 'TTS provider')}
            value={ttsProvider}
          >
            {ttsProviderOptions.map(value => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>
        <label className="mbc-check">
          <input
            checked={autoTts}
            disabled={busy}
            onChange={event => void patchConfig('voice.auto_tts', event.target.checked, 'Auto TTS')}
            type="checkbox"
          />
          <span>Read replies aloud automatically</span>
        </label>
      </section>

      {status ? <p className="mbc-status">{status}</p> : null}
    </div>
  )
}
