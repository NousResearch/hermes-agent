import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { SETTINGS_ROUTE } from '@/app/routes'
import {
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  dropdownMenuRow
} from '@/components/ui/dropdown-menu'
import {
  liveVoiceEngineProfile,
  liveVoiceEngineStatus,
  type LiveVoiceEngineVerdict,
  type ResolvedLiveVoiceEngine,
  useLiveVoiceEngines
} from '@/contrib/voice-engines'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { isExtensionVoiceChatMode, type VoiceChatMode } from '@/lib/voice-live'
import { notifyError } from '@/store/notifications'
import { requestRoute } from '@/store/recovery-requests'
import { $voiceSignIn, refreshVoiceSignIn, signInKey } from '@/store/voice-engine-sign-in'
import { $voiceLiveStatus, selectedVoiceChatMode, setVoiceChatMode } from '@/store/voice-live'

/** Where an engine's sign-in and settings live (Settings ▸ Voice ▸ Conversation). */
export const VOICE_ENGINE_SETTINGS_HREF = `${SETTINGS_ROUTE}?tab=config:voice&page=conversation`

type EngineStatuses = Record<string, LiveVoiceEngineVerdict>

/** Each engine's `status()` and its provider's sign-in state, re-read every
 *  time the rows mount (a menu opening), never polled. */
function useEngineStatuses(engines: readonly ResolvedLiveVoiceEngine[]): EngineStatuses {
  const [statuses, setStatuses] = useState<EngineStatuses>({})

  useEffect(() => {
    let cancelled = false

    for (const engine of engines) {
      void liveVoiceEngineStatus(engine).then(status => {
        if (!cancelled) {
          setStatuses(current => ({ ...current, [engine.id]: status }))
        }
      })

      if (engine.signIn) {
        void refreshVoiceSignIn(engine.signIn.provider)
      }
    }

    return () => void (cancelled = true)
  }, [engines])

  return statuses
}

/**
 * Which engine the next voice conversation mounts: the chained
 * speech-to-text → Hermes → speech loop, GPT-Live delegating to Hermes, or a
 * live engine an extension registered on `VOICE_ENGINE_AREA`.
 *
 * Radio rows, not a toggle: the user is choosing between named things and
 * the checked row tells them which one the next press starts. Rendered inside
 * whichever menu the layout has room for (the folded voice menu, or the
 * right-click menu on the start button), so the same rows appear in both.
 * Hidden while the backend has not answered or predates the mode, so we never
 * offer a switch the gateway would refuse with 4002.
 */
export function VoiceEngineRows({ disabled }: { disabled: boolean }) {
  const { t } = useI18n()
  const c = t.composer
  const status = useStore($voiceLiveStatus)
  const engines = useLiveVoiceEngines()
  const statuses = useEngineStatuses(engines)
  const signIns = useStore($voiceSignIn)

  if (status === null) {
    return null
  }

  const liveAvailable = status.available
  const providerName = (provider: string) => signIns[signInKey(provider)]?.name ?? provider

  // Engines whose sign-in provider is known to be logged out get a "Sign in
  // with …" item; it opens Settings ▸ Voice, where the code and Cancel live.
  const loggedOut = engines.filter(
    engine => engine.signIn && signIns[signInKey(engine.signIn.provider)]?.loggedIn === false
  )

  return (
    <>
      <DropdownMenuLabel>{c.voiceEngine}</DropdownMenuLabel>
      <DropdownMenuRadioGroup
        onValueChange={value => {
          if (value !== 'chained' && value !== 'gpt-live' && !engines.some(engine => engine.id === value)) {
            return
          }

          triggerHaptic('open')
          setVoiceChatMode(value as VoiceChatMode).catch(error => notifyError(error, c.voiceEngineChangeFailed))
        }}
        value={selectedVoiceChatMode(status)}
      >
        <DropdownMenuRadioItem className={dropdownMenuRow} disabled={disabled} value="chained">
          {c.voiceEngineChained}
        </DropdownMenuRadioItem>
        <DropdownMenuRadioItem className={dropdownMenuRow} disabled={disabled || !liveAvailable} value="gpt-live">
          <span className="flex min-w-0 flex-col">
            <span>{c.voiceEngineLive}</span>
            {liveAvailable ? null : (
              <span className="text-muted-foreground truncate text-xs">
                {status.reason ?? c.voiceEngineLiveNeedsKey}
              </span>
            )}
          </span>
        </DropdownMenuRadioItem>
        {engines.map(engine => {
          const engineStatus = statuses[engine.id]
          // Pickable until status() answers; starting re-checks it anyway.
          const unavailable = engineStatus?.available === false

          return (
            <DropdownMenuRadioItem
              className={dropdownMenuRow}
              data-voice-engine={engine.id}
              disabled={disabled || unavailable}
              key={engine.id}
              value={engine.id}
            >
              <span className="flex min-w-0 flex-col">
                <span>{engine.description ? `${engine.label} (${engine.description})` : engine.label}</span>
                {unavailable ? (
                  <span className="text-muted-foreground truncate text-xs">
                    {engineStatus.notEnabled
                      ? c.voiceEngineNotEnabled(engine.pluginId, liveVoiceEngineProfile())
                      : (engineStatus.reason ?? c.voiceEngineUnavailable)}
                  </span>
                ) : null}
              </span>
            </DropdownMenuRadioItem>
          )
        })}
      </DropdownMenuRadioGroup>
      {loggedOut.map(engine => (
        <DropdownMenuItem
          className={dropdownMenuRow}
          data-voice-engine-sign-in={engine.id}
          disabled={disabled}
          key={`sign-in:${engine.id}`}
          onSelect={() => requestRoute(VOICE_ENGINE_SETTINGS_HREF)}
        >
          {c.voiceEngineSignIn(providerName(engine.signIn?.provider ?? ''))}
        </DropdownMenuItem>
      ))}
    </>
  )
}

/** Short engine name for tooltips, or null until the backend has answered. */
export function useVoiceEngineName(): null | string {
  const { t } = useI18n()
  const status = useStore($voiceLiveStatus)
  // Read from the live list (not the registry) so the name follows a plugin
  // loading or unloading.
  const engines = useLiveVoiceEngines()

  if (status === null) {
    return null
  }

  const mode = selectedVoiceChatMode(status)

  if (isExtensionVoiceChatMode(mode)) {
    return engines.find(engine => engine.id === mode)?.label ?? t.composer.voiceEngineChainedShort
  }

  return mode === 'gpt-live' ? t.composer.voiceEngineLiveShort : t.composer.voiceEngineChainedShort
}
