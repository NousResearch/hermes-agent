import { type MutableRefObject, useCallback, useState } from 'react'

import { getHermesConfig, getHermesConfigDefaults, getHermesConfigRecord, saveHermesConfig } from '@/hermes'
import { BUILTIN_PERSONALITIES, normalizePersonalityValue, personalityNamesFromConfig } from '@/lib/chat-runtime'
import {
  $currentCwd,
  setAvailablePersonalities,
  setCurrentCwd,
  setCurrentFastMode,
  setCurrentPersonality,
  setCurrentReasoningEffort,
  setCurrentServiceTier,
  setIntroPersonality
} from '@/store/session'

const DEFAULT_VOICE_SECONDS = 120
const FAST_TIERS = new Set(['fast', 'priority', 'on'])

function recordingLimit(value: unknown) {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : DEFAULT_VOICE_SECONDS
}

// Consumer default: reasoning blocks (推理过程块) are ON. The runtime ships
// display.show_reasoning=false and the gateway gates reasoning output on it
// server-side, but the 对话 settings page that exposed the toggle is gone from
// the consumer UI, so seed the value through the existing config persistence.
// After the first write the config reads back true and this is a no-op —
// nothing new is stored client-side.
async function ensureReasoningBlocksOnByDefault(): Promise<void> {
  const record = await getHermesConfigRecord()

  const display =
    record.display && typeof record.display === 'object' && !Array.isArray(record.display)
      ? (record.display as Record<string, unknown>)
      : {}

  if (display.show_reasoning === true) {
    return
  }

  await saveHermesConfig({ ...record, display: { ...display, show_reasoning: true } })
}

interface HermesConfigOptions {
  activeSessionIdRef: MutableRefObject<string | null>
  refreshProjectBranch: (cwd: string) => Promise<void>
}

export function useHermesConfig({ activeSessionIdRef, refreshProjectBranch }: HermesConfigOptions) {
  const [voiceMaxRecordingSeconds, setVoiceMaxRecordingSeconds] = useState(DEFAULT_VOICE_SECONDS)
  const [sttEnabled, setSttEnabled] = useState(true)

  const refreshHermesConfig = useCallback(async () => {
    try {
      const [config, defaults] = await Promise.all([getHermesConfig(), getHermesConfigDefaults().catch(() => ({}))])

      const personality = normalizePersonalityValue(
        typeof config.display?.personality === 'string' ? config.display.personality : ''
      )

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

      const cwd = (config.terminal?.cwd ?? '').trim()

      if (cwd && cwd !== '.') {
        setCurrentCwd(prev => prev || cwd)
        void refreshProjectBranch($currentCwd.get() || cwd)
      }

      const reasoning = (config.agent?.reasoning_effort ?? '').trim()
      const tier = (config.agent?.service_tier ?? '').trim()

      setCurrentReasoningEffort(prev => (activeSessionIdRef.current ? prev : reasoning))
      setCurrentServiceTier(prev => (activeSessionIdRef.current ? prev : tier))
      setCurrentFastMode(prev => (activeSessionIdRef.current ? prev : FAST_TIERS.has(tier.toLowerCase())))

      setVoiceMaxRecordingSeconds(recordingLimit(config.voice?.max_recording_seconds))
      setSttEnabled(config.stt?.enabled !== false)

      // Steady state (value already true) skips the extra round-trip entirely.
      if (config.display?.show_reasoning !== true) {
        void ensureReasoningBlocksOnByDefault().catch(() => undefined)
      }
    } catch {
      // Config is nice-to-have; chat still works without it.
    }
  }, [activeSessionIdRef, refreshProjectBranch])

  return { refreshHermesConfig, sttEnabled, voiceMaxRecordingSeconds }
}
