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
  type ComposerSendMode,
  DOUBLE_ENTER_MAX_MS,
  DOUBLE_ENTER_MIN_MS,
  setComposerSendPrefs
} from '@/store/composer-send'
import { notify } from '@/store/notifications'

import { composerSendModeHint } from '../chat/composer/send-mode-hint'

import { ListRow } from './primitives'

/** Settings → Keyboards: how a draft gets committed.
 *
 *  This is the one composer shortcut that is a preference rather than a
 *  rebindable action — the three modes disagree about what a bare Enter means,
 *  so they can't all be expressed as one combo. The rows below the control
 *  (the shortcuts map) follow the choice. */
export function ComposerSendSettings() {
  const { t } = useI18n()
  const k = t.keybinds.composerSend
  const prefs = useStore($composerSendPrefs)
  const configPath = useStore($composerSendConfigPath)

  // Local text mirror so a half-typed number ("4", on the way to "45") isn't
  // clamped and committed on every keystroke.
  const [draftMs, setDraftMs] = useState(String(prefs.doubleEnterMs))

  useEffect(() => {
    setDraftMs(String(prefs.doubleEnterMs))
  }, [prefs.doubleEnterMs])

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

  const commitDoubleEnterMs = (value: number | string) => {
    const next = clampDoubleEnterMs(value)
    setDraftMs(String(next))
    void setComposerSendPrefs({ doubleEnterMs: next })
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
              { id: 'mod-enter', label: k.modeModEnter }
            ]}
            value={prefs.mode}
          />
        }
        description={k.description}
        title={k.title}
      />

      {prefs.mode === 'double-enter' && (
        <ListRow
          action={
            <div className="flex items-center gap-3">
              <input
                aria-label={k.doubleTapTitle}
                className="h-1.5 min-w-40 flex-1 cursor-pointer appearance-none rounded-full bg-(--ui-bg-tertiary)"
                max={DOUBLE_ENTER_MAX_MS}
                min={DOUBLE_ENTER_MIN_MS}
                onChange={event => commitDoubleEnterMs(event.currentTarget.value)}
                step={10}
                style={{ accentColor: 'var(--dt-primary)' }}
                type="range"
                value={clampDoubleEnterMs(draftMs)}
              />
              <Input
                aria-label={k.doubleTapTitle}
                className="w-20 text-right tabular-nums"
                max={DOUBLE_ENTER_MAX_MS}
                min={DOUBLE_ENTER_MIN_MS}
                onBlur={() => commitDoubleEnterMs(draftMs)}
                onChange={event => setDraftMs(event.currentTarget.value)}
                onKeyDown={event => {
                  if (event.key === 'Enter') {
                    commitDoubleEnterMs(draftMs)
                  }
                }}
                suffix={k.doubleTapUnit}
                type="number"
                value={draftMs}
              />
            </div>
          }
          description={k.doubleTapDescription}
          hint={configPath ? k.doubleTapFileHint(configPath) : undefined}
          title={k.doubleTapTitle}
        />
      )}
    </>
  )
}
