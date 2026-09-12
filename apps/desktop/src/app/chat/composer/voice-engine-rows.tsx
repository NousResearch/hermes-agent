import { useStore } from '@nanostores/react'

import { DropdownMenuLabel, DropdownMenuRadioGroup, DropdownMenuRadioItem, dropdownMenuRow } from '@/components/ui/dropdown-menu'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { notifyError } from '@/store/notifications'
import { $voiceLiveStatus, selectedVoiceChatMode, setVoiceChatMode } from '@/store/voice-live'

/**
 * Which engine the next voice conversation mounts: the chained
 * speech-to-text → Hermes → speech loop, or GPT-Live delegating to Hermes.
 *
 * Radio rows, not a toggle: the user is choosing between two named things and
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

  if (status === null) {
    return null
  }

  const liveAvailable = status.available

  return (
    <>
      <DropdownMenuLabel>{c.voiceEngine}</DropdownMenuLabel>
      <DropdownMenuRadioGroup
        onValueChange={value => {
          if (value !== 'chained' && value !== 'gpt-live') {
            return
          }

          triggerHaptic('open')
          setVoiceChatMode(value).catch(error => notifyError(error, c.voiceEngineChangeFailed))
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
              <span className="text-muted-foreground truncate text-xs">{status.reason ?? c.voiceEngineLiveNeedsKey}</span>
            )}
          </span>
        </DropdownMenuRadioItem>
      </DropdownMenuRadioGroup>
    </>
  )
}

/** Short engine name for tooltips, or null until the backend has answered. */
export function useVoiceEngineName(): null | string {
  const { t } = useI18n()
  const status = useStore($voiceLiveStatus)

  if (status === null) {
    return null
  }

  return selectedVoiceChatMode(status) === 'gpt-live' ? t.composer.voiceEngineLiveShort : t.composer.voiceEngineChainedShort
}
