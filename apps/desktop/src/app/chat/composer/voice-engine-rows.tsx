import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import {
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  dropdownMenuRow
} from '@/components/ui/dropdown-menu'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Settings } from '@/lib/icons'
import { notifyError } from '@/store/notifications'
import {
  $geminiLiveKeyConfigured,
  $voiceLiveStatus,
  openGeminiLiveDialog,
  refreshGeminiLiveKeyStatus,
  selectedVoiceChatMode,
  setVoiceChatMode
} from '@/store/voice-live'

/**
 * Which engine the next voice conversation mounts: the chained
 * speech-to-text → Hermes → speech loop, GPT-Live delegating to Hermes,
 * or client-side Gemini Live directly on the laptop.
 *
 * Radio rows, not a toggle: the user is choosing between named engines and
 * the checked row tells them which one the next press starts. Rendered inside
 * whichever menu the layout has room for (the folded voice menu, or the
 * right-click menu on the start button), so the same rows appear in both.
 */
export function VoiceEngineRows({ disabled }: { disabled: boolean }) {
  const { t } = useI18n()
  const c = t.composer
  const status = useStore($voiceLiveStatus)
  const geminiKeyConfigured = useStore($geminiLiveKeyConfigured)

  useEffect(() => {
    if (geminiKeyConfigured === null) {
      void refreshGeminiLiveKeyStatus()
    }
  }, [geminiKeyConfigured])

  const mode = selectedVoiceChatMode(status)
  const liveAvailable = Boolean(status?.available)
  const isGeminiReady = geminiKeyConfigured ?? false

  return (
    <>
      <DropdownMenuLabel>{c.voiceEngine}</DropdownMenuLabel>
      <DropdownMenuRadioGroup
        onValueChange={value => {
          if (value !== 'chained' && value !== 'gpt-live' && value !== 'gemini-live') {
            return
          }

          triggerHaptic('open')
          setVoiceChatMode(value).catch(error => notifyError(error, c.voiceEngineChangeFailed))
        }}
        value={mode}
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
        <DropdownMenuRadioItem className={dropdownMenuRow} disabled={disabled} value="gemini-live">
          <span className="flex min-w-0 flex-col">
            <span className="flex items-center justify-between gap-2">
              <span>{c.voiceEngineGeminiLive}</span>
              <span className="text-muted-foreground text-[10px] uppercase font-mono tracking-wider">Laptop</span>
            </span>
            {isGeminiReady ? null : (
              <span className="text-muted-foreground truncate text-xs">
                {c.voiceEngineGeminiLiveNeedsKey}
              </span>
            )}
          </span>
        </DropdownMenuRadioItem>
      </DropdownMenuRadioGroup>

      <DropdownMenuItem
        className={dropdownMenuRow}
        onSelect={() => {
          triggerHaptic('open')
          openGeminiLiveDialog()
        }}
      >
        <Settings className="size-3.5 opacity-70" />
        <span>Configure Gemini Live...</span>
      </DropdownMenuItem>
    </>
  )
}

/** Short engine name for tooltips, or null until the backend has answered. */
export function useVoiceEngineName(): null | string {
  const { t } = useI18n()
  const status = useStore($voiceLiveStatus)

  const mode = selectedVoiceChatMode(status)
  if (mode === 'gemini-live') {
    return t.composer.voiceEngineGeminiLiveShort
  }
  if (mode === 'gpt-live') {
    return t.composer.voiceEngineLiveShort
  }
  return status === null ? null : t.composer.voiceEngineChainedShort
}
