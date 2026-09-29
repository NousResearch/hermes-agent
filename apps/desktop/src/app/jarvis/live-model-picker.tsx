import { useStore } from '@nanostores/react'
import { useState } from 'react'
import { useNavigate } from 'react-router'

import { getHermesConfigRecord, saveHermesConfigRecord } from '@/api/config'
import { useI18n } from '@/i18n'
import { $activeConnectionId } from '@/store/connections'
import { notifyError } from '@/store/notifications'
import { $activeGatewayProfile } from '@/store/profile'
import { $liveVoiceChoice, applyVoiceEngineFromConfig } from '@/store/voice-prefs'

import { SETTINGS_ROUTE } from '../routes'
import { setNested } from '../settings/helpers'

const MODELS = [
  { provider: 'gemini', model: 'gemini-3.8-live', label: 'Gemini 3.8 Live' },
  { provider: 'gemini', model: 'gemini-3.1-flash-live-preview', label: 'Gemini 3.1 Flash Live' },
  { provider: 'openai', model: 'gpt-realtime', label: 'OpenAI Realtime' }
] as const

export function LiveModelPicker({ connected }: { connected: boolean }) {
  const { locale } = useI18n()
  const live = useStore($liveVoiceChoice)
  const connectionId = useStore($activeConnectionId)
  const profile = useStore($activeGatewayProfile)
  const [saving, setSaving] = useState(false)
  const navigate = useNavigate()
  const value = `${live.provider}:${live.model}`
  const pl = locale === 'pl'

  const choose = async (selection: string) => {
    const model = MODELS.find(item => `${item.provider}:${item.model}` === selection)

    if (!model) {
      return
    }

    setSaving(true)

    const scope = { connectionId, profile }

    try {
      let config: Record<string, unknown> = { ...(await getHermesConfigRecord(scope)) }
      config = setNested(config, 'voice.engine', 'realtime')
      config = setNested(config, 'voice.realtime.provider', model.provider)
      config = setNested(
        config,
        model.provider === 'gemini' ? 'voice.realtime.gemini.model' : 'voice.realtime.model',
        model.model
      )
      const result = await saveHermesConfigRecord(config, scope)

      if (!result.ok) {
        throw new Error('Configuration was not saved')
      }

      if ($activeGatewayProfile.get() === profile && $activeConnectionId.get() === connectionId) {
        applyVoiceEngineFromConfig(config)
      }
    } catch (error) {
      notifyError(error, pl ? 'Nie zapisano modelu rozmowy' : 'Could not save voice model')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="mb-4 space-y-2">
      <label className="block text-xs font-medium text-(--ui-text-secondary)" htmlFor="czesiek-live-model">
        {pl ? 'Rozmowa Live · głos Cześka' : 'Live conversation · voice'}
      </label>
      <select
        className="jarvis-well w-full min-w-0 cursor-pointer px-3 py-2.5 text-sm text-(--ui-text-primary) outline-none focus-visible:outline focus-visible:outline-solid focus-visible:outline-2 focus-visible:outline-(--ui-accent)"
        disabled={!connected || saving}
        id="czesiek-live-model"
        onChange={event => void choose(event.target.value)}
        value={value}
      >
        {!MODELS.some(item => `${item.provider}:${item.model}` === value) && (
          <option value={value}>{live.model}</option>
        )}
        {MODELS.map(item => (
          <option key={item.model} value={`${item.provider}:${item.model}`}>
            {item.label}
          </option>
        ))}
      </select>
      <p className="text-xs leading-relaxed text-(--ui-text-tertiary)">
        {pl
          ? 'Zmiana działa od następnej rozmowy. Zadania mają osobny model poniżej.'
          : 'Applies to the next call. Tasks use the separate model below.'}
      </p>
      <button
        className="text-xs font-medium text-(--ui-accent) underline-offset-4 hover:underline"
        onClick={() => navigate(`${SETTINGS_ROUTE}?tab=voice`)}
        type="button"
      >
        {pl ? 'Klucze i ustawienia głosu' : 'Voice keys and settings'}
      </button>
    </div>
  )
}
