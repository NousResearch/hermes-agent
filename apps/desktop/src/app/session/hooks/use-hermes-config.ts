import { type MutableRefObject, useCallback, useEffect, useRef, useState } from 'react'

import { setTerminalFontFamilyFromConfig } from '@/app/right-sidebar/terminal/terminal-font'
import { getHermesConfig, getHermesConfigDefaults } from '@/hermes'
import { BUILTIN_PERSONALITIES, normalizePersonalityValue, personalityNamesFromConfig } from '@/lib/chat-runtime'
import { normalize } from '@/lib/text'
import { setDisplayTimestampsFromConfig } from '@/store/display-timestamps'
import {
  getComposerSelectionGeneration,
  getCurrentModelSource,
  setAvailablePersonalities,
  setCurrentFastMode,
  setCurrentPersonality,
  setCurrentReasoningEffort,
  setCurrentServiceTier,
  setDefaultReasoningEffort,
  setIntroPersonality
} from '@/store/session'
import {
  applyAutoSpeakFromConfig,
  applyThinkingSoundFromConfig,
  applyVoiceStopPhraseFromConfig
} from '@/store/voice-prefs'

const CONFIG_REFRESH_RETRY_MS = 2_000
const FAST_TIERS = new Set(['fast', 'priority', 'on'])

function recordingLimit(value: unknown) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return undefined
  }

  // Match the backend's contract: an explicit non-positive value disables the
  // automatic cap. `undefined` means the config is not available yet.
  return value > 0 ? value : null
}

/** config.yaml hands back whatever the user wrote — `reasoning_effort: false`
 *  (or `off`/`no`, which YAML also parses to boolean false) means thinking
 *  disabled, and a bare boolean must not throw on `.trim()`. */
function normalizeConfigEffort(value: unknown): string {
  if (value === false) {
    return 'none'
  }

  if (typeof value !== 'string') {
    return ''
  }

  const effort = normalize(value)

  return effort === 'false' || effort === 'disabled' ? 'none' : effort
}

interface HermesConfigOptions {
  activeSessionIdRef: MutableRefObject<string | null>
}

export function useHermesConfig({ activeSessionIdRef }: HermesConfigOptions) {
  const [voiceMaxRecordingSeconds, setVoiceMaxRecordingSeconds] = useState<number | null | undefined>(undefined)
  const [sttEnabled, setSttEnabled] = useState(true)
  const profileRefreshEpochRef = useRef(0)
  const configRetryTimerRef = useRef<number | null>(null)
  const refreshHermesConfigRef = useRef<(() => Promise<void>) | null>(null)

  const scheduleConfigRetry = useCallback(() => {
    if (configRetryTimerRef.current !== null) {
      return
    }

    configRetryTimerRef.current = window.setTimeout(() => {
      configRetryTimerRef.current = null
      void refreshHermesConfigRef.current?.()
    }, CONFIG_REFRESH_RETRY_MS)
  }, [])

  const refreshHermesConfig = useCallback(
    async (force = false, shouldPublish: () => boolean = () => true) => {
      if (force) {
        profileRefreshEpochRef.current += 1
      }

      const profileRefreshEpoch = profileRefreshEpochRef.current
      const selectionGeneration = getComposerSelectionGeneration()

      try {
        const [config, defaults] = await Promise.all([getHermesConfig(), getHermesConfigDefaults().catch(() => ({}))])

        if (configRetryTimerRef.current !== null) {
          window.clearTimeout(configRetryTimerRef.current)
          configRetryTimerRef.current = null
        }

        const canPublish = () => profileRefreshEpochRef.current === profileRefreshEpoch && shouldPublish()

        if (!canPublish()) {
          return
        }

        const personality = normalizePersonalityValue(
          typeof config.display?.personality === 'string' ? config.display.personality : ''
        )

        if (!canPublish()) {
          return
        }

        setIntroPersonality(personality)
        // Active sessions keep their per-session value; standalone falls back to config.
        setCurrentPersonality(prev => (activeSessionIdRef.current ? prev || personality : personality))
        setAvailablePersonalities([
          ...new Set([
            'none',
            ...BUILTIN_PERSONALITIES,
            ...personalityNamesFromConfig(defaults),
            ...personalityNamesFromConfig(config)
          ])
        ])

        const reasoning = normalizeConfigEffort(config.agent?.reasoning_effort)
        const tier = (config.agent?.service_tier ?? '').trim()

        // Publish the profile default regardless of whether the composer is
        // reseeded below: picker rows and preset application resolve "the
        // default" from here, so a manual model pick must not leave them
        // rendering/applying Hermes' built-in medium over the user's config.
        if (!canPublish()) {
          return
        }

        setDefaultReasoningEffort(reasoning)

        const shouldSeedComposer =
          !activeSessionIdRef.current &&
          getComposerSelectionGeneration() === selectionGeneration &&
          (force || getCurrentModelSource() !== 'manual')

        if (shouldSeedComposer) {
          if (!canPublish()) {
            return
          }

          setCurrentReasoningEffort(reasoning)
          setCurrentFastMode(FAST_TIERS.has(tier.toLowerCase()))
        }

        if (!canPublish()) {
          return
        }

        setCurrentServiceTier(prev => (activeSessionIdRef.current ? prev : tier))

        if (!canPublish()) {
          return
        }

        setVoiceMaxRecordingSeconds(recordingLimit(config.voice?.max_recording_seconds))
        setSttEnabled(config.stt?.enabled !== false)

        if (!canPublish()) {
          return
        }

        setDisplayTimestampsFromConfig(config.display?.timestamps)
        setTerminalFontFamilyFromConfig(config.terminal?.font_family)

        if (!canPublish()) {
          return
        }

        applyAutoSpeakFromConfig(config)
        applyVoiceStopPhraseFromConfig(config)
        applyThinkingSoundFromConfig(config)
      } catch {
        // Chat remains usable, but voice must not silently record with an
        // unrelated default limit. Retry after transient startup failures.
        scheduleConfigRetry()
      }
    },
    [activeSessionIdRef, scheduleConfigRetry]
  )

  useEffect(() => {
    refreshHermesConfigRef.current = refreshHermesConfig

    return () => {
      refreshHermesConfigRef.current = null
      if (configRetryTimerRef.current !== null) {
        window.clearTimeout(configRetryTimerRef.current)
        configRetryTimerRef.current = null
      }
    }
  }, [refreshHermesConfig])

  return { refreshHermesConfig, sttEnabled, voiceMaxRecordingSeconds }
}
