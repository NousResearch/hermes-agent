import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { Input } from '@/components/ui/input'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import {
  $composerSendConfigPath,
  $composerSendPrefs,
  clampDoubleEnterMs,
  clampHoldMs,
  clampSendGraceMs,
  clampTypingIdleMs,
  type ComposerSendMode,
  DOUBLE_ENTER_MAX_MS,
  DOUBLE_ENTER_MIN_MS,
  HOLD_MAX_MS,
  HOLD_MIN_MS,
  SEND_GRACE_MAX_MS,
  SEND_GRACE_MIN_MS,
  type SendGraceScope,
  setComposerSendPrefs,
  TYPING_IDLE_MAX_MS,
  TYPING_IDLE_MIN_MS
} from '@/store/composer-send'
import { notify } from '@/store/notifications'

import { composerSendModeHint } from '../chat/composer/send-mode-hint'

import { ListRow } from './primitives'

interface MsFieldProps {
  clamp: (value: unknown) => number
  label: string
  max: number
  min: number
  onChange: (value: number) => void
  unit: string
  value: number
}

/**
 * One millisecond value: slider for the common case, number box for precision.
 *
 * The text is mirrored locally so a half-typed number ("4", on the way to
 * "450") is not clamped and persisted on every keystroke — it commits on blur
 * or Enter, and the slider stays clamped so its thumb never leaves the track.
 */
function MsField({ clamp, label, max, min, onChange, unit, value }: MsFieldProps) {
  const [draft, setDraft] = useState(String(value))

  useEffect(() => {
    setDraft(String(value))
  }, [value])

  const commit = (next: number | string) => {
    const clamped = clamp(next)

    setDraft(String(clamped))
    onChange(clamped)
  }

  return (
    <div className="flex items-center gap-3">
      <input
        aria-label={label}
        className="h-1.5 min-w-40 flex-1 cursor-pointer appearance-none rounded-full bg-(--ui-bg-tertiary)"
        max={max}
        min={min}
        onChange={event => commit(event.currentTarget.value)}
        step={10}
        style={{ accentColor: 'var(--dt-primary)' }}
        type="range"
        value={clamp(draft)}
      />
      <Input
        aria-label={label}
        className="w-20 text-right tabular-nums"
        max={max}
        min={min}
        onBlur={() => commit(draft)}
        onChange={event => setDraft(event.currentTarget.value)}
        onKeyDown={event => {
          if (event.key === 'Enter') {
            commit(draft)
          }
        }}
        suffix={unit}
        type="number"
        value={draft}
      />
    </div>
  )
}

/** Settings → Keyboards: how a draft gets committed, and what a guessed send
 *  costs before it goes.
 *
 *  This is the one composer shortcut that is a preference rather than a
 *  rebindable action — the modes disagree about what a bare Enter means, so
 *  they can't all be expressed as one combo. The rows below the control (the
 *  shortcuts map) follow the choice, and every numeric row names the JSON file
 *  main persists to, so the hand-edit path is discoverable rather than
 *  folklore. */
export function ComposerSendSettings() {
  const { t } = useI18n()
  const k = t.keybinds.composerSend
  const prefs = useStore($composerSendPrefs)
  const configPath = useStore($composerSendConfigPath)

  const fileHint = configPath ? k.fileHint(configPath) : undefined

  const commitMode = (mode: ComposerSendMode) => {
    triggerHaptic('selection')
    void setComposerSendPrefs({ mode })

    // Say what the switch did. The gesture is invisible until the user types,
    // and the placeholder hint only exists while the composer is empty — this
    // is the one moment they are looking at the setting and can be told.
    notify({
      kind: 'info',
      message: composerSendModeHint(mode, {
        chord: t.composer.placeholderSendChord,
        doubleTap: t.composer.placeholderSendDoubleTap,
        enterSends: t.composer.placeholderSendEnterSends,
        newline: t.composer.placeholderSendNewline,
        hold: t.composer.placeholderSendHold,
        pause: t.composer.placeholderSendPause
      }),
      title: k.title
    })
  }

  const commitGrace = (sendGrace: SendGraceScope) => {
    triggerHaptic('selection')
    void setComposerSendPrefs({ sendGrace })
  }

  return (
    <>
      <ListRow
        action={
          <SegmentedControl
            onChange={commitMode}
            options={[
              { id: 'enter', label: k.modeEnter },
              { id: 'double-enter', label: k.modeDoubleEnter },
              { id: 'pause', label: k.modePause },
              { id: 'hold', label: k.modeHold },
              { id: 'mod-enter', label: k.modeModEnter }
            ]}
            value={prefs.mode}
          />
        }
        description={k.description}
        title={k.title}
      />

      {/* `pause` measures its mid-flow presses against the same window as
          `double-enter`, so it needs the same knob. */}
      {(prefs.mode === 'double-enter' || prefs.mode === 'pause') && (
        <ListRow
          action={
            <MsField
              clamp={clampDoubleEnterMs}
              label={k.doubleTapTitle}
              max={DOUBLE_ENTER_MAX_MS}
              min={DOUBLE_ENTER_MIN_MS}
              onChange={doubleEnterMs => void setComposerSendPrefs({ doubleEnterMs })}
              unit={k.doubleTapUnit}
              value={prefs.doubleEnterMs}
            />
          }
          description={k.doubleTapDescription}
          hint={fileHint}
          title={k.doubleTapTitle}
        />
      )}

      {prefs.mode === 'hold' && (
        <ListRow
          action={
            <MsField
              clamp={clampHoldMs}
              label={k.holdMsTitle}
              max={HOLD_MAX_MS}
              min={HOLD_MIN_MS}
              onChange={holdMs => void setComposerSendPrefs({ holdMs })}
              unit={k.holdMsUnit}
              value={prefs.holdMs}
            />
          }
          description={k.holdMsDescription}
          hint={fileHint}
          title={k.holdMsTitle}
        />
      )}

      {prefs.mode === 'pause' && (
        <ListRow
          action={
            <MsField
              clamp={clampTypingIdleMs}
              label={k.typingIdleTitle}
              max={TYPING_IDLE_MAX_MS}
              min={TYPING_IDLE_MIN_MS}
              onChange={typingIdleMs => void setComposerSendPrefs({ typingIdleMs })}
              unit={k.typingIdleUnit}
              value={prefs.typingIdleMs}
            />
          }
          description={k.typingIdleDescription}
          hint={fileHint}
          title={k.typingIdleTitle}
        />
      )}

      {/* Unconditional: the grace window also covers the default `enter` mode,
          which is the setting people actually need undo-send for. */}
      <ListRow
        action={
          <SegmentedControl
            onChange={commitGrace}
            options={[
              { id: 'off', label: k.graceOff },
              { id: 'inferred', label: k.graceInferred },
              { id: 'all', label: k.graceAll }
            ]}
            value={prefs.sendGrace}
          />
        }
        description={k.graceDescription}
        title={k.graceTitle}
      />

      {prefs.sendGrace !== 'off' && (
        <ListRow
          action={
            <MsField
              clamp={clampSendGraceMs}
              label={k.graceMsTitle}
              max={SEND_GRACE_MAX_MS}
              min={SEND_GRACE_MIN_MS}
              onChange={sendGraceMs => void setComposerSendPrefs({ sendGraceMs })}
              unit={k.graceMsUnit}
              value={prefs.sendGraceMs}
            />
          }
          description={k.graceMsDescription}
          hint={fileHint}
          title={k.graceMsTitle}
        />
      )}
    </>
  )
}
