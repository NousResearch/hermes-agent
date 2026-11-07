import { useStore } from '@nanostores/react'

import {
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  dropdownMenuRow
} from '@/components/ui/dropdown-menu'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { notifyError } from '@/store/notifications'
import { $voiceLiveGrokStatus, $voiceLiveStatus, selectedVoiceChatMode, setVoiceChatMode } from '@/store/voice-live'

/**
 * Which engine the next voice conversation mounts: the chained
 * speech-to-text → Hermes → speech loop, GPT-Live, or Grok-Live delegating
 * to Hermes.
 *
 * Radio rows, not a toggle: the user is choosing between named things and
 * the checked row tells them which one the next press starts. Rendered inside
 * whichever menu the layout has room for (the folded voice menu, or the
 * right-click menu on the start button), so the same rows appear in both.
 * Hidden until either backend status endpoint answers, so we never offer a
 * switch before this gateway's capabilities are known. The two live rows are
 * gated independently; an older backend may know only one of them.
 */
export function VoiceEngineRows({ disabled }: { disabled: boolean }) {
  const { t } = useI18n()
  const c = t.composer
  const status = useStore($voiceLiveStatus)
  const grokStatus = useStore($voiceLiveGrokStatus)

  if (status === null && grokStatus === null) {
    return null
  }

  const liveAvailable = status?.available ?? false
  const grokAvailable = grokStatus?.available ?? false
  const selected = selectedVoiceChatMode()

  return (
    <>
      <DropdownMenuLabel>{c.voiceEngine}</DropdownMenuLabel>
      <DropdownMenuRadioGroup
        onValueChange={value => {
          if (value !== 'chained' && value !== 'gpt-live' && value !== 'grok-live') {
            return
          }

          triggerHaptic('open')
          setVoiceChatMode(value).catch(error => notifyError(error, c.voiceEngineChangeFailed))
        }}
        value={selected}
      >
        <DropdownMenuRadioItem className={dropdownMenuRow} disabled={disabled} value="chained">
          {c.voiceEngineChained}
        </DropdownMenuRadioItem>
        <DropdownMenuRadioItem className={dropdownMenuRow} disabled={disabled || !liveAvailable} value="gpt-live">
          <span className="flex min-w-0 flex-col">
            <span>{c.voiceEngineLive}</span>
            {liveAvailable ? null : (
              <span className="text-muted-foreground truncate text-xs">
                {status?.reason ?? c.voiceEngineLiveNeedsKey}
              </span>
            )}
          </span>
        </DropdownMenuRadioItem>
        {grokStatus === null ? null : (
          <DropdownMenuRadioItem className={dropdownMenuRow} disabled={disabled || !grokAvailable} value="grok-live">
            <span className="flex min-w-0 flex-col">
              <span>{c.voiceEngineGrok}</span>
              {grokAvailable ? null : (
                <span className="text-muted-foreground truncate text-xs">
                  {grokStatus.reason ?? c.voiceEngineGrokNeedsKey}
                </span>
              )}
            </span>
          </DropdownMenuRadioItem>
        )}
      </DropdownMenuRadioGroup>
    </>
  )
}

/** Short engine name for tooltips, or null until the backend has answered. */
export function useVoiceEngineName(): null | string {
  const { t } = useI18n()
  const status = useStore($voiceLiveStatus)
  const grokStatus = useStore($voiceLiveGrokStatus)

  if (status === null && grokStatus === null) {
    return null
  }

  const selected = selectedVoiceChatMode()

  if (selected === 'gpt-live') {
    return t.composer.voiceEngineLiveShort
  }

  if (selected === 'grok-live') {
    return t.composer.voiceEngineGrokShort
  }

  return t.composer.voiceEngineChainedShort
}
