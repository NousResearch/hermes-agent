import { useStore } from '@nanostores/react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Tip, TipKeybindLabel } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { AudioLines, Ear, EarOff, iconSize, Layers3, Loader2, Square, Volume2, VolumeX } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $hudMode, closeHud } from '@/store/hud'
import { $wakeWord, toggleWakeWord } from '@/store/wake-word'

import { ACTIVE_ICON_BTN, GHOST_ICON_BTN, PRIMARY_ICON_BTN } from './control-classes'
import type { ConversationStatus } from './hooks/use-voice-conversation'
import { ModelPill } from './model-pill'
import type { ChatBarState, VoiceStatus } from './types'
import { VoiceMenu } from './voice-menu'

// Re-exported: `context-menu.tsx` and other row neighbours have always reached
// for these here, and the row is where they read as belonging.
export { ACTIVE_ICON_BTN, GHOST_ICON_BTN, ICON_BTN, PRIMARY_ICON_BTN } from './control-classes'

interface ConversationProps {
  active: boolean
  level: number
  muted: boolean
  status: ConversationStatus
  onEnd: () => void
  onStart: () => void
  onStopTurn: () => void
  onToggleMute: () => void
}

export function ComposerControls({
  autoSpeak,
  busy,
  busyAction,
  canSubmit,
  compactModelPill = false,
  conversation,
  disabled,
  hasComposerPayload,
  state,
  voiceStatus,
  onDictate,
  onQueue,
  onToggleAutoSpeak
}: {
  autoSpeak: boolean
  busy: boolean
  busyAction: 'steer' | 'queue' | 'stop'
  canSubmit: boolean
  compactModelPill?: boolean
  conversation: ConversationProps
  disabled: boolean
  hasComposerPayload: boolean
  state: ChatBarState
  voiceStatus: VoiceStatus
  onDictate: () => void
  onQueue: () => void
  onToggleAutoSpeak: () => void
}) {
  const { t } = useI18n()
  const c = t.composer
  const hudMode = useStore($hudMode)

  if (conversation.active) {
    return <ConversationPill {...conversation} disabled={disabled} />
  }

  const showVoicePrimary = !busy && !hasComposerPayload
  // Steer is just send: a payload keeps the Send affordance mid-turn. Stop
  // only when the composer is empty and a turn is running.
  const showStop = busy && !hasComposerPayload
  const showQueueButton = busyAction !== 'stop' && hasComposerPayload

  return (
    <div className="ml-auto flex shrink-0 items-center gap-(--composer-control-gap)">
      <ModelPill compact={compactModelPill} disabled={disabled} model={state.model} />
      {/* The HUD is a Spotlight bar a few hundred pixels wide, so the four
          separate voice toggles fold into one menu there and leave the row to
          the input. The docked composer has the width and keeps them inline —
          same controls, same state, different budget. */}
      {hudMode ? (
        <VoiceMenu
          autoSpeak={autoSpeak}
          disabled={disabled}
          onDictate={onDictate}
          onStartConversation={conversation.onStart}
          onToggleAutoSpeak={onToggleAutoSpeak}
          state={state}
          voiceStatus={voiceStatus}
        />
      ) : (
        <>
          <DictationButton disabled={disabled} onToggle={onDictate} state={state.voice} status={voiceStatus} />
          <AutoSpeakButton active={autoSpeak} disabled={disabled} onToggle={onToggleAutoSpeak} />
          <WakeWordButton disabled={disabled} />
        </>
      )}
      {showQueueButton ? (
        <Tip label={<TipKeybindLabel actionId="composer.queue" text={c.queueMessage} />}>
          <Button
            aria-label={c.queueMessage}
            className={GHOST_ICON_BTN}
            disabled={disabled}
            onClick={onQueue}
            size="icon"
            type="button"
            variant="ghost"
          >
            <Layers3 className={iconSize.sm} />
          </Button>
        </Tip>
      ) : null}
      {showVoicePrimary ? (
        <Tip label={c.startVoice}>
          <Button
            aria-label={c.startVoice}
            className={PRIMARY_ICON_BTN}
            disabled={disabled}
            onClick={() => {
              triggerHaptic('open')
              conversation.onStart()
            }}
            size="icon"
            type="button"
          >
            <AudioLines className={iconSize.sm} />
          </Button>
        </Tip>
      ) : (
        <Tip
          label={
            showStop ? (
              <TipKeybindLabel actionId="composer.send" text={c.stop} />
            ) : (
              <TipKeybindLabel actionId="composer.send" text={c.send} />
            )
          }
        >
          <Button
            aria-label={showStop ? c.stop : c.send}
            className={PRIMARY_ICON_BTN}
            disabled={disabled || !canSubmit}
            type="submit"
          >
            {showStop ? (
              <span className="block size-2.5 rounded-[0.1875rem] bg-current" />
            ) : (
              <Codicon name="arrow-up" size="0.875rem" />
            )}
          </Button>
        </Tip>
      )}
      {/* The way out of HUD mode, riding the controls row rather than floating
          above the bar. The old chip lived in a 26px transparent strip reserved
          over the composer (--hud-chip-strip), which under glass is bare
          untinted material with a hidden button in it — a band of chrome above
          the surface, paid for in every state, for a control that is invisible
          until hovered. Here it costs no reserved space and sits with the other
          things you can press. */}
      {hudMode ? <ExitHudButton /> : null}
    </div>
  )
}

function ExitHudButton() {
  const { t } = useI18n()

  return (
    <Tip label={t.titlebar.exitHud}>
      <Button
        aria-label={t.titlebar.exitHud}
        className={cn(GHOST_ICON_BTN, 'p-0')}
        onClick={closeHud}
        size="icon"
        type="button"
        variant="ghost"
      >
        <Codicon name="screen-normal" size="0.875rem" />
      </Button>
    </Tip>
  )
}

function ConversationPill({
  disabled,
  level,
  muted,
  onEnd,
  onStopTurn,
  onToggleMute,
  status
}: ConversationProps & { disabled: boolean }) {
  const { t } = useI18n()
  const c = t.composer
  const speaking = status === 'speaking'
  const listening = status === 'listening' && !muted

  const label =
    status === 'speaking'
      ? c.speaking
      : status === 'transcribing'
        ? c.transcribing
        : status === 'thinking'
          ? c.thinking
          : muted
            ? c.muted
            : c.listening

  return (
    <div className="ml-auto flex shrink-0 items-center gap-(--composer-control-gap)">
      {/* Keep the ear visible during voice chat — shown paused, since the
          conversation holds the mic (the one time wake must not listen). */}
      <WakeWordButton disabled={disabled} pausedForVoice />
      <Tip label={muted ? c.unmuteMic : c.muteMic}>
        <Button
          aria-label={muted ? c.unmuteMic : c.muteMic}
          aria-pressed={muted}
          className={cn(GHOST_ICON_BTN, 'p-0', muted && 'bg-muted text-muted-foreground')}
          disabled={disabled}
          onClick={() => {
            triggerHaptic('selection')
            onToggleMute()
          }}
          size="icon"
          type="button"
          variant="ghost"
        >
          <Codicon name={muted ? 'mic-off' : 'mic'} size="1rem" />
        </Button>
      </Tip>
      {listening && (
        <Button
          aria-label={c.stopListening}
          className="h-(--composer-control-size) shrink-0 gap-1.5 rounded-full px-2.5 text-xs text-muted-foreground hover:bg-accent hover:text-foreground"
          disabled={disabled}
          onClick={() => {
            triggerHaptic('submit')
            onStopTurn()
          }}
          type="button"
          variant="ghost"
        >
          <Square className={cn('fill-current', iconSize.xs)} />
          <span>{c.stopShort}</span>
        </Button>
      )}
      <Button
        aria-label={c.endConversation}
        className="h-(--composer-control-size) gap-1.5 rounded-full bg-primary px-3 text-xs font-medium text-primary-foreground hover:bg-primary/90"
        disabled={disabled}
        onClick={() => {
          triggerHaptic('close')
          onEnd()
        }}
        type="button"
      >
        <ConversationIndicator level={level} listening={listening} speaking={speaking} />
        <span>{c.endShort}</span>
      </Button>
      <span className="sr-only" role="status">
        {label}
      </span>
    </div>
  )
}

function ConversationIndicator({
  level,
  listening,
  speaking
}: {
  level: number
  listening: boolean
  speaking: boolean
}) {
  if (speaking) {
    return <Loader2 className={cn('animate-spin', iconSize.xs)} />
  }

  const bars = [0.55, 0.85, 1, 0.85, 0.55]
  const normalized = Math.max(0, Math.min(level, 1))

  return (
    <span aria-hidden="true" className="flex h-3 items-center gap-0.5">
      {bars.map((weight, index) => {
        const height = listening ? 0.3 + Math.min(0.7, normalized * weight) : 0.3

        return <span className="w-0.5 rounded-full bg-current" key={index} style={{ height: `${height * 100}%` }} />
      })}
    </span>
  )
}

// Pure-TTS toggle: type normally, but have every assistant reply read aloud —
// no dictation, no full conversation loop. Filled/accent when on, mirroring the
// muted-mic pressed state above. Driven by (and persisted to) `voice.auto_tts`.
function AutoSpeakButton({ active, disabled, onToggle }: { active: boolean; disabled: boolean; onToggle: () => void }) {
  const { t } = useI18n()
  const c = t.composer
  const label = active ? c.stopSpeakingReplies : c.speakReplies

  return (
    <Tip label={label}>
      <Button
        aria-label={label}
        aria-pressed={active}
        className={cn(
          GHOST_ICON_BTN,
          'p-0',
          active && ACTIVE_ICON_BTN
        )}
        disabled={disabled}
        onClick={() => {
          triggerHaptic(active ? 'close' : 'open')
          onToggle()
        }}
        size="icon"
        type="button"
        variant="ghost"
      >
        {active ? <Volume2 className={iconSize.sm} /> : <VolumeX className={iconSize.sm} />}
      </Button>
    </Tip>
  )
}

// "Hey Hermes" wake-word toggle. ALWAYS rendered — the ear never hides. A
// user must always be able to click it to turn passive listening on; if the
// backend can't start (missing STT/TTS, deps still installing, no mic
// permission, etc.) the click surfaces the reason in the tooltip and the
// toggle stays off. States: listening (accent-highlighted), off (muted
// ear-off), and paused-for-voice (disabled while a voice conversation holds
// the mic — the one time wake genuinely must not listen). Backend refusals
// ({started:false, reason}) keep the toggle off and put the reason/hint in
// the tooltip.
function WakeWordButton({ disabled, pausedForVoice = false }: { disabled: boolean; pausedForVoice?: boolean }) {
  const { t } = useI18n()
  const c = t.composer
  const wake = useStore($wakeWord)

  const phrase = wake.phrase || 'hey hermes'

  const label = pausedForVoice
    ? c.wakeWordPausedVoice(phrase)
    : wake.listening
      ? c.wakeWordListening(phrase)
      : c.wakeWordOff(phrase)

  const tooltip = !pausedForVoice && wake.notice ? `${label} — ${wake.notice}` : label

  return (
    <Tip label={tooltip}>
      <Button
        aria-label={label}
        aria-pressed={wake.listening && !pausedForVoice}
        className={cn(
          GHOST_ICON_BTN,
          'p-0',
          wake.listening && !pausedForVoice && ACTIVE_ICON_BTN
        )}
        disabled={disabled || pausedForVoice || wake.pending}
        onClick={() => {
          triggerHaptic(wake.listening ? 'close' : 'open')
          void toggleWakeWord()
        }}
        size="icon"
        type="button"
        variant="ghost"
      >
        {wake.listening && !pausedForVoice ? <Ear className={iconSize.sm} /> : <EarOff className={iconSize.sm} />}
      </Button>
    </Tip>
  )
}

function DictationButton({
  disabled,
  state,
  status,
  onToggle
}: {
  disabled: boolean
  state: ChatBarState['voice']
  status: VoiceStatus
  onToggle: () => void
}) {
  const { t } = useI18n()
  const c = t.composer
  const active = state.active || status !== 'idle'

  const aria =
    status === 'recording' ? c.stopDictation : status === 'transcribing' ? c.transcribingDictation : c.voiceDictation

  return (
    <Tip label={aria}>
      <Button
        aria-label={aria}
        aria-pressed={active}
        className={cn(
          GHOST_ICON_BTN,
          'p-0',
          'data-[active=true]:bg-accent data-[active=true]:text-foreground',
          status === 'recording' && ACTIVE_ICON_BTN,
          status === 'transcribing' && 'bg-primary/10 text-primary'
        )}
        data-active={active}
        disabled={disabled || !state.enabled || status === 'transcribing'}
        onClick={() => {
          triggerHaptic(active ? 'close' : 'open')
          onToggle()
        }}
        size="icon"
        type="button"
        variant="ghost"
      >
        {status === 'recording' ? (
          <Square className={cn('fill-current', iconSize.xs)} />
        ) : status === 'transcribing' ? (
          <Loader2 className={cn('animate-spin', iconSize.sm)} />
        ) : (
          <Codicon name="mic" size="0.875rem" />
        )}
      </Button>
    </Tip>
  )
}
