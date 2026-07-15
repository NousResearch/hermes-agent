import { type MutableRefObject, useCallback, useRef, useState } from 'react'

import { getEffectivePreventSleepConfig, getHermesConfig, getHermesConfigDefaults } from '@/hermes'
import { BUILTIN_PERSONALITIES, normalizePersonalityValue, personalityNamesFromConfig } from '@/lib/chat-runtime'
import { normalize } from '@/lib/text'
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
import { applyAutoSpeakFromConfig } from '@/store/voice-prefs'

const DEFAULT_VOICE_SECONDS = 120
const FAST_TIERS = new Set(['fast', 'priority', 'on'])

function recordingLimit(value: unknown) {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : DEFAULT_VOICE_SECONDS
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
  refreshProjectBranch: (cwd: string) => Promise<void>
}

export function useHermesConfig({ activeSessionIdRef, refreshProjectBranch }: HermesConfigOptions) {
  const [voiceMaxRecordingSeconds, setVoiceMaxRecordingSeconds] = useState(DEFAULT_VOICE_SECONDS)
  const [sttEnabled, setSttEnabled] = useState(true)
  const refreshGenerationRef = useRef(0)

  const refreshHermesConfig = useCallback(async () => {
    const generation = ++refreshGenerationRef.current

    try {
      const [config, defaults, preventSleep] = await Promise.all([
        getHermesConfig(),
        getHermesConfigDefaults().catch(() => ({})),
        getEffectivePreventSleepConfig().catch(() => null)
      ])

      // Startup, saves, reconnects, and profile switches can overlap. Only the
      // latest foreground-profile response may update local or machine state.
      if (generation !== refreshGenerationRef.current) {
        return
      }

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
        // Configured terminal.cwd beats a stale remembered workspace cwd
        // (#38855) — but never yank the workspace out from under an active
        // session; those keep their own cwd until the user detaches.
        setCurrentCwd(prev => (activeSessionIdRef.current ? prev : cwd))
        void refreshProjectBranch($currentCwd.get() || cwd)
      }

      const reasoning = normalizeConfigEffort(config.agent?.reasoning_effort)
      const tier = (config.agent?.service_tier ?? '').trim()

      setCurrentReasoningEffort(prev => (activeSessionIdRef.current ? prev : reasoning))
      setCurrentServiceTier(prev => (activeSessionIdRef.current ? prev : tier))
      setCurrentFastMode(prev => (activeSessionIdRef.current ? prev : FAST_TIERS.has(tier.toLowerCase())))

      setVoiceMaxRecordingSeconds(recordingLimit(config.voice?.max_recording_seconds))
      setSttEnabled(config.stt?.enabled !== false)
      applyAutoSpeakFromConfig(config)

      // This narrow endpoint uses the same Python resolver as Gateway. Older
      // backends may not expose it; in that case preserve the last-good native
      // blocker instead of falling back to duplicate renderer semantics.
      if (preventSleep) {
        await window.hermesDesktop?.refreshPowerSaveBlocker?.(preventSleep).catch(() => undefined)
      }
    } catch {
      // Config is nice-to-have; chat still works without it.
    }
  }, [activeSessionIdRef, refreshProjectBranch])

  return { refreshHermesConfig, sttEnabled, voiceMaxRecordingSeconds }
}
