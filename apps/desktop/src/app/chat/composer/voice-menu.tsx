import { useStore } from '@nanostores/react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  dropdownMenuRow,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { AudioLines, Ear, EarOff, iconSize, Loader2, Square, Volume2, VolumeX } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $wakeWord, toggleWakeWord } from '@/store/wake-word'

import { ACTIVE_ICON_BTN, GHOST_ICON_BTN } from './control-classes'
import type { ChatBarState, VoiceStatus } from './types'

export interface VoiceMenuProps {
  autoSpeak: boolean
  disabled: boolean
  state: ChatBarState
  voiceStatus: VoiceStatus
  onDictate: () => void
  onStartConversation: () => void
  onToggleAutoSpeak: () => void
}

/**
 * Every voice control behind one trigger: dictation, spoken replies, the
 * "hey hermes" wake word, and starting a full conversation.
 *
 * The bar carried four separate icon buttons — mic, speaker, ear, and the
 * voice-conversation primary — which is most of a Spotlight-width composer
 * spent on toggles that are set once and rarely touched. Collapsing them costs
 * one click on the rare change and gives the row back to the input.
 *
 * The trigger is not a static glyph: it REPORTS the loudest live voice state
 * (recording, transcribing, listening for the wake word, speaking replies), so
 * folding the controls away never hides the fact that something is listening.
 * A collapsed menu that looked idle while the mic was hot would be a worse
 * trade than the space it saves.
 */
export function VoiceMenu({
  autoSpeak,
  disabled,
  state,
  voiceStatus,
  onDictate,
  onStartConversation,
  onToggleAutoSpeak
}: VoiceMenuProps) {
  const { t } = useI18n()
  const c = t.composer
  const wake = useStore($wakeWord)

  const phrase = wake.phrase || 'hey hermes'
  const dictating = state.voice.active || voiceStatus !== 'idle'
  const wakeListening = wake.listening
  // Anything live keeps the trigger lit, so a folded menu can never look idle
  // while the mic is open.
  const active = dictating || wakeListening || autoSpeak

  const dictationLabel =
    voiceStatus === 'recording'
      ? c.stopDictation
      : voiceStatus === 'transcribing'
        ? c.transcribingDictation
        : c.voiceDictation

  const wakeLabel = wakeListening ? c.wakeWordListening(phrase) : c.wakeWordOff(phrase)
  const triggerLabel = dictating ? dictationLabel : wakeListening ? wakeLabel : c.voiceControls

  return (
    <DropdownMenu>
      <Tip label={wake.notice && !dictating ? `${triggerLabel} — ${wake.notice}` : triggerLabel}>
        <DropdownMenuTrigger asChild>
          <Button
            aria-label={triggerLabel}
            className={cn(GHOST_ICON_BTN, 'p-0', active && ACTIVE_ICON_BTN)}
            disabled={disabled}
            size="icon"
            type="button"
            variant="ghost"
          >
            {voiceStatus === 'recording' ? (
              <Square className={cn('fill-current', iconSize.xs)} />
            ) : voiceStatus === 'transcribing' ? (
              <Loader2 className={cn('animate-spin', iconSize.sm)} />
            ) : wakeListening ? (
              <Ear className={iconSize.sm} />
            ) : (
              <Codicon name="mic" size="0.875rem" />
            )}
          </Button>
        </DropdownMenuTrigger>
      </Tip>
      <DropdownMenuContent align="end" className="min-w-52">
        <DropdownMenuItem
          className={dropdownMenuRow}
          disabled={disabled}
          onSelect={() => {
            triggerHaptic('open')
            onStartConversation()
          }}
        >
          <AudioLines className={iconSize.sm} />
          {c.startVoice}
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        {/* Checkbox items, because all three are toggles the user is reading
            the CURRENT state of — the reason they were pressed-state buttons
            before. A plain row would fold that state away with the menu. */}
        <DropdownMenuCheckboxItem
          checked={dictating}
          className={dropdownMenuRow}
          disabled={disabled || !state.voice.enabled || voiceStatus === 'transcribing'}
          onSelect={event => {
            // Keep the menu open: dictation is a mode you watch, and closing
            // on select hides the recording state the trigger just entered.
            event.preventDefault()
            triggerHaptic(dictating ? 'close' : 'open')
            onDictate()
          }}
        >
          {dictationLabel}
        </DropdownMenuCheckboxItem>
        <DropdownMenuCheckboxItem
          checked={autoSpeak}
          className={dropdownMenuRow}
          disabled={disabled}
          onSelect={event => {
            event.preventDefault()
            triggerHaptic(autoSpeak ? 'close' : 'open')
            onToggleAutoSpeak()
          }}
        >
          {autoSpeak ? <Volume2 className={iconSize.sm} /> : <VolumeX className={iconSize.sm} />}
          {autoSpeak ? c.stopSpeakingReplies : c.speakReplies}
        </DropdownMenuCheckboxItem>
        <DropdownMenuCheckboxItem
          checked={wakeListening}
          className={dropdownMenuRow}
          disabled={disabled || wake.pending}
          onSelect={event => {
            event.preventDefault()
            triggerHaptic(wakeListening ? 'close' : 'open')
            void toggleWakeWord()
          }}
        >
          {wakeListening ? <Ear className={iconSize.sm} /> : <EarOff className={iconSize.sm} />}
          {wakeLabel}
        </DropdownMenuCheckboxItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
