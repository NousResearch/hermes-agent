import { composerConfigFromPrefs } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Switch } from '@/components/ui/switch'
import { saveHermesConfig } from '@/hermes'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import {
  $composerSendPrefs,
  clampDoubleEnterMs,
  clampHoldMs,
  clampIdleSendMs,
  clampSendGraceMs,
  clampTypingIdleMs,
  DOUBLE_ENTER_MAX_MS,
  DOUBLE_ENTER_MIN_MS,
  HOLD_MAX_MS,
  HOLD_MIN_MS,
  IDLE_SEND_MAX_MS,
  IDLE_SEND_MIN_MS,
  SEND_GRACE_MAX_MS,
  SEND_GRACE_MIN_MS,
  SEND_GRACE_REASONS,
  type SendGraceReason,
  TYPING_IDLE_MAX_MS,
  TYPING_IDLE_MIN_MS
} from '@/store/composer-prefs'
import { notify, notifyError } from '@/store/notifications'

import { composerSendHint } from '../chat/composer/send-hint'
import { useHermesConfigRecord } from '../hooks/use-config-record'

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
 *  Three kinds of setting, deliberately not one control:
 *
 *  1. the gate — whether a lone Enter is allowed to send at all. The primary
 *     control, and a switch rather than a remap: the feature exists to stop
 *     accidental sends, so the row states what it prevents;
 *  2. the newline — what a press does instead, once the gate is closed. A
 *     separate question, because a break and a send are different outcomes and
 *     "nothing at all" is a third answer;
 *  3. the gestures that ALSO commit a draft — independent, because they overlap
 *     freely and an enum could only ever arm one.
 *
 *  The newline row and the gestures are hidden while a bare Enter sends, since
 *  neither can matter then. Values are kept, so switching back restores what was
 *  chosen. */
export function ComposerSendSettings() {
  const { t } = useI18n()
  const k = t.keybinds.composerSend
  const prefs = useStore($composerSendPrefs)
  // The gateway config owns these values, so the hint names the record rather
  // than a file this app writes.
  const { writeScope } = useHermesConfigRecord()
  const fileHint = k.fileHint('desktop.composer in config.yaml')
  const gesturesLocked = prefs.enterSends

  // Say what the settings currently amount to. The gestures are invisible until
  // someone types, and the placeholder hint only exists while the composer is
  // empty — this is the one moment they are looking at the panel and can be told.
  const announce = (changed: string) => {
    notify({
      kind: 'info',
      message: composerSendHint($composerSendPrefs.get(), {
        chord: t.composer.placeholderSendChord,
        doubleTap: t.composer.placeholderSendDoubleTap,
        enterSends: t.composer.placeholderSendEnterSends,
        hold: t.composer.placeholderSendHold,
        idle: t.composer.placeholderSendIdle,
        newline: t.composer.placeholderSendNewline,
        pause: t.composer.placeholderSendPause
      }),
      title: changed
    })
  }

  /** Persist through the gateway config, which owns `desktop.composer.*`. The
   *  patch lands on the prefs already in hand, so switching one gesture on never
   *  writes a neighbour back to its default. The atom moves first and rolls back
   *  if the write fails, because the composer reads it on every keystroke. */
  const save = (patch: Partial<typeof prefs>, title?: string) => {
    const previous = $composerSendPrefs.get()
    const next = { ...previous, ...patch }

    $composerSendPrefs.set(next)

    void saveHermesConfig({ desktop: { composer: composerConfigFromPrefs(next) } }, writeScope)
      .then(result => {
        if (!result.ok) {
          throw new Error('composer send settings were not saved')
        }

        if (title) {
          announce(title)
        }
      })
      .catch(error => {
        $composerSendPrefs.set(previous)
        notifyError(error, t.settings.config.autosaveFailed)
      })
  }

  const toggleGrace = (reason: SendGraceReason, enabled: boolean) => {
    triggerHaptic('selection')

    const next = enabled
      ? SEND_GRACE_REASONS.filter(candidate => candidate === reason || prefs.sendGraceFor.includes(candidate))
      : prefs.sendGraceFor.filter(candidate => candidate !== reason)

    save({ sendGraceFor: next })
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

  /** One gesture: the switch, and its window once it is on. */
  const gesture = (
    key: 'doubleTap' | 'hold' | 'idle' | 'pause',
    title: string,
    description: string,
    pref: 'sendOnDoubleTap' | 'sendOnHold' | 'sendOnIdle' | 'sendOnPause',
    timing: { clamp: (value: unknown) => number; max: number; min: number; pref: 'doubleEnterMs' | 'holdMs' | 'idleSendMs' | 'typingIdleMs'; title: string; unit: string; description: string }
  ) => (
    <>
      <ListRow
        action={
          <Switch
            aria-label={title}
            checked={prefs[pref]}
            disabled={gesturesLocked}
            onCheckedChange={checked => {
              triggerHaptic('selection')
              save({ [pref]: checked }, title)
              announce(title)
            }}
          />
        }
        description={description}
        title={title}
      />
      {prefs[pref] && (
        <ListRow
          action={
            <MsField
              clamp={timing.clamp}
              label={timing.title}
              max={timing.max}
              min={timing.min}
              onChange={value => save({ [timing.pref]: value })}
              unit={timing.unit}
              value={prefs[timing.pref]}
            />
          }
          description={timing.description}
          hint={fileHint}
          title={timing.title}
        />
      )}
    </>
  )

  return (
    <>
      {/* The primary control is the GATE, not a remap: the feature exists to stop
          accidental sends, so the row states what it prevents rather than what
          the key does instead. Stored the natural way round (`enterSends`) and
          inverted here, once, on purpose. */}
      <ListRow
        action={
          <Switch
            aria-label={k.gateLabel}
            checked={!prefs.enterSends}
            onCheckedChange={checked => {
              triggerHaptic('selection')
              save({ enterSends: !checked })
              announce(k.gateLabel)
            }}
          />
        }
        description={k.gateDescription}
        title={k.gateLabel}
      />

      {/* What the press does instead, as its own question: a line break and a
          send are different outcomes, and "nothing at all" is a third answer. */}
      {!prefs.enterSends && (
        <ListRow
          action={
            <Switch
              aria-label={k.newlineLabel}
              checked={prefs.enterNewline}
              onCheckedChange={checked => {
                triggerHaptic('selection')
                save({ enterNewline: checked })
                announce(k.newlineLabel)
              }}
            />
          }
          description={k.newlineDescription}
          title={k.newlineLabel}
        />
      )}

      {gesturesLocked && <ListRow description={k.gesturesDisabled} title={k.gesturesTitle} />}

      {/* Hidden, not disabled, while Enter sends on the press: none of them
          can fire, and a row of dead switches reads as a broken panel. The
          values are kept, so switching back restores what was chosen. */}
      {!gesturesLocked && (
        <>
          {gesture('doubleTap', k.gestureDoubleTap, k.gestureDoubleTapDesc, 'sendOnDoubleTap', {
            clamp: clampDoubleEnterMs,
            description: k.doubleTapDescription,
            max: DOUBLE_ENTER_MAX_MS,
            min: DOUBLE_ENTER_MIN_MS,
            pref: 'doubleEnterMs',
            title: k.doubleTapTitle,
            unit: k.doubleTapUnit
          })}

          {gesture('pause', k.gesturePause, k.gesturePauseDesc, 'sendOnPause', {
            clamp: clampTypingIdleMs,
            description: k.typingIdleDescription,
            max: TYPING_IDLE_MAX_MS,
            min: TYPING_IDLE_MIN_MS,
            pref: 'typingIdleMs',
            title: k.typingIdleTitle,
            unit: k.typingIdleUnit
          })}

          {gesture('hold', k.gestureHold, k.gestureHoldDesc, 'sendOnHold', {
            clamp: clampHoldMs,
            description: k.holdMsDescription,
            max: HOLD_MAX_MS,
            min: HOLD_MIN_MS,
            pref: 'holdMs',
            title: k.holdMsTitle,
            unit: k.holdMsUnit
          })}

          {gesture('idle', k.gestureIdle, k.gestureIdleDesc, 'sendOnIdle', {
            clamp: clampIdleSendMs,
            description: k.idleMsDescription,
            max: IDLE_SEND_MAX_MS,
            min: IDLE_SEND_MIN_MS,
            pref: 'idleSendMs',
            title: k.idleMsTitle,
            unit: k.idleMsUnit
          })}
        </>
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
              onChange={sendGraceMs => save({ sendGraceMs })}
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
