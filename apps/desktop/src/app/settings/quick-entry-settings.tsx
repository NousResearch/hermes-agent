import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'
import { isSubmitEnter } from '@/lib/ime'
import {
  $quickEntry,
  canUseQuickEntry,
  loadQuickEntrySettings,
  QUICK_ENTRY_DEFAULT_POSITION,
  QUICK_ENTRY_DEFAULT_SHORTCUT,
  saveQuickEntrySettings
} from '@/store/quick-entry'

import { ListRow, ToggleRow } from './primitives'
import { SETTING_IDS, settingElementId } from './settings-manifest'

/** An axis of the placement setting: `x` is the window's centre, `y` its top. */
type PositionAxis = keyof typeof QUICK_ENTRY_DEFAULT_POSITION

/**
 * Stored fraction → what the field shows: a percentage with at most one
 * decimal and never a trailing ".0", so the default reads `50`, not `50.0`.
 */
function percentLabel(fraction: number): string {
  return `${Math.round(fraction * 1000) / 10}`
}

/**
 * Committed percentage → stored fraction, or null to IGNORE the edit so an
 * unusable value never overwrites what is already saved. Blank resets that
 * axis to its shipped default; anything usable clamps into 0–100.
 */
function percentToFraction(raw: string, axis: PositionAxis): number | null {
  const text = raw.trim()

  if (!text) {
    return QUICK_ENTRY_DEFAULT_POSITION[axis]
  }

  const percent = Number(text)

  if (!Number.isFinite(percent)) {
    return null
  }

  return Math.min(1, Math.max(0, percent / 100))
}

/**
 * Quick Entry — the global-hotkey mini composer's settings rows.
 *
 * The MAIN process is authoritative (it owns the OS accelerator and applies the
 * saved placement to the window it summons), so this reads the live
 * registration state on mount and surfaces the failure the feature must never
 * swallow: a chord another app already owns comes back `registered: false` with
 * `error: 'taken'` and says so, right under the field.
 */
export function QuickEntrySettings() {
  const { t } = useI18n()
  const q = t.settings.quickEntry
  const state = useStore($quickEntry)
  // The shortcut field is a local draft: the accelerator is only committed on
  // blur/Enter, so a half-typed chord ("Alt+") never tears down the live
  // registration. The two position fields share that guard so a half-typed
  // percentage cannot yank the window mid-edit either.
  const [draft, setDraft] = useState<null | string>(null)
  const [xDraft, setXDraft] = useState<null | string>(null)
  const [yDraft, setYDraft] = useState<null | string>(null)

  useEffect(() => {
    void loadQuickEntrySettings()
  }, [])

  if (!canUseQuickEntry()) {
    return null
  }

  const commit = () => {
    const next = (draft ?? '').trim()
    setDraft(null)

    if (next && next !== state.shortcut) {
      void saveQuickEntrySettings({ shortcut: next })
    }
  }

  const commitPosition = (axis: PositionAxis) => {
    const raw = axis === 'x' ? xDraft : yDraft

    if (axis === 'x') {
      setXDraft(null)
    } else {
      setYDraft(null)
    }

    // Focused but never edited: blurring alone must not reset a saved position.
    if (raw === null) {
      return
    }

    const fraction = percentToFraction(raw, axis)

    if (fraction === null || fraction === state.position[axis]) {
      return
    }

    void saveQuickEntrySettings({ position: { ...state.position, [axis]: fraction } })
  }

  // Horizontal centre first, vertical top second — the same order the
  // description reads in, so the two fields never need their own captions.
  const positionField = (axis: PositionAxis, label: string) => (
    <Input
      aria-label={label}
      className="w-20"
      disabled={!state.enabled}
      inputMode="decimal"
      max={100}
      min={0}
      onBlur={() => commitPosition(axis)}
      onChange={event => (axis === 'x' ? setXDraft : setYDraft)(event.target.value)}
      onKeyDown={event => {
        if (isSubmitEnter(event)) {
          event.preventDefault()
          commitPosition(axis)
        }
      }}
      type="number"
      value={(axis === 'x' ? xDraft : yDraft) ?? percentLabel(state.position[axis])}
    />
  )

  const status =
    state.registered === null
      ? null
      : state.error === 'taken'
        ? q.takenBy
        : state.error === 'invalid'
          ? q.invalidShortcut
          : state.enabled && state.registered
            ? q.active
            : null

  return (
    <>
      <ToggleRow
        checked={state.enabled}
        description={q.enabledDesc}
        id={settingElementId(SETTING_IDS.advanced.quickEntry)}
        label={q.enabledTitle}
        onChange={enabled => void saveQuickEntrySettings({ enabled })}
      />
      <ListRow
        action={
          <Input
            aria-label={q.shortcutTitle}
            disabled={!state.enabled}
            onBlur={commit}
            onChange={event => setDraft(event.target.value)}
            onKeyDown={event => {
              if (isSubmitEnter(event)) {
                event.preventDefault()
                commit()
              }
            }}
            placeholder={QUICK_ENTRY_DEFAULT_SHORTCUT}
            value={draft ?? state.shortcut}
          />
        }
        below={
          status && (
            <div
              className={
                state.error
                  ? 'mt-1 text-[length:var(--conversation-caption-font-size)] text-amber-500/90'
                  : 'mt-1 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)'
              }
            >
              {status}
            </div>
          )
        }
        description={q.shortcutDesc}
        id={settingElementId(SETTING_IDS.advanced.quickEntryShortcut)}
        title={q.shortcutTitle}
      />
      <ListRow
        action={
          <>
            {positionField('x', q.positionHorizontal)}
            {positionField('y', q.positionVertical)}
          </>
        }
        description={q.positionDesc}
        hint={q.positionDefaultHint}
        id={settingElementId(SETTING_IDS.advanced.quickEntryPosition)}
        title={q.positionTitle}
      />
    </>
  )
}
