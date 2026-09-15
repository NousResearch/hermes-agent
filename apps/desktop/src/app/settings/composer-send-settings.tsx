import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { Switch } from '@/components/ui/switch'
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
  SEND_GRACE_REASONS,
  type SendGraceReason,
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

/** Settings → Keyboards: how a draft gets committed, and which sends wait.
 *
 *  Two things here are deliberately NOT modes: the long press is a switch that
 *  rides alongside whichever mode is chosen (a double-tap user should not have
 *  to give that up to get it), and the grace window is one checkbox per
 *  situation, because a second tap and a long press are things the user DID —
 *  only the send the app worked out on their behalf is worth holding by default.
 *  Every numeric row names the JSON file main persists to, so the hand-edit path
 *  is discoverable rather than folklore. */
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
        pause: t.composer.placeholderSendPause
      }),
      title: k.title
    })
  }

  const setHold = (sendOnHold: boolean) => {
    triggerHaptic('selection')
    void setComposerSendPrefs({ sendOnHold })
  }

  const toggleGrace = (reason: SendGraceReason, enabled: boolean) => {
    triggerHaptic('selection')

    const next = enabled
      ? SEND_GRACE_REASONS.filter(candidate => candidate === reason || prefs.sendGraceFor.includes(candidate))
      : prefs.sendGraceFor.filter(candidate => candidate !== reason)

    void setComposerSendPrefs({ sendGraceFor: next })
  }

  const reasonLabel: Record<SendGraceReason, string> = {
    doubleTap: k.graceReasonDoubleTap,
    enter: k.graceReasonEnter,
    hold: k.graceReasonHold,
    pause: k.graceReasonPause
  }

  const delayed = prefs.sendGraceFor.length

  const summary =
    delayed === 0
      ? k.graceNone
      : delayed === SEND_GRACE_REASONS.length
        ? k.graceAll
        : k.graceSome(String(delayed), String(SEND_GRACE_REASONS.length))

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
              { id: 'mod-enter', label: k.modeModEnter }
            ]}
            value={prefs.mode}
          />
        }
        description={k.description}
        title={k.title}
      />

      {/* Not offered in `enter`: the press has already committed by the time a
          hold could register, so the switch would be a lie there. */}
      {prefs.mode !== 'enter' && (
        <ListRow
          action={<Switch aria-label={k.holdToggleTitle} checked={prefs.sendOnHold} onCheckedChange={setHold} />}
          description={k.holdToggleDescription}
          title={k.holdToggleTitle}
        />
      )}

      {prefs.sendOnHold && prefs.mode !== 'enter' && (
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

      {/* One checkbox per situation, in a popover: four switches would swamp the
          section, and the set is a detail of one decision. */}
      <ListRow
        action={
          <Popover>
            <PopoverTrigger asChild>
              <button
                className="min-w-32 rounded-lg border border-(--ui-stroke-tertiary) px-3 py-1.5 text-left text-sm"
                type="button"
              >
                {summary}
              </button>
            </PopoverTrigger>
            <PopoverContent align="end" className="w-64 p-3">
              <p className="mb-2 text-xs text-(--ui-text-tertiary)">{k.gracePopoverHint}</p>
              <div className="flex flex-col gap-2">
                {SEND_GRACE_REASONS.map(reason => (
                  <label className="flex items-center gap-2 text-sm" key={reason}>
                    <Checkbox
                      checked={prefs.sendGraceFor.includes(reason)}
                      onCheckedChange={checked => toggleGrace(reason, checked === true)}
                    />
                    {reasonLabel[reason]}
                  </label>
                ))}
              </div>
            </PopoverContent>
          </Popover>
        }
        description={k.graceDescription}
        title={k.graceTitle}
      />

      {delayed > 0 && (
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
