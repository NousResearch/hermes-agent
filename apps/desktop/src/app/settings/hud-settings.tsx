import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'
import { HUD_PET_CHOICES, type HudPetChoice } from '@/lib/hud-prefs'
import { $hudPrefs, canUseHud, loadHudPrefs, setHudPrefs, watchHudPrefs } from '@/store/hud'
import { $profiles, normalizeProfileKey, profileLabel, refreshProfiles } from '@/store/profile'

import { ListRow, ToggleRow } from './primitives'

/**
 * HUD follow / ask rows. Main is authoritative (it owns the OS chord and the
 * optional input hook), so this reads the live status on mount and shows the
 * two failures the feature must never swallow: a chord another app owns, and
 * a right-click hook that is not installed on this machine.
 */
export function HudSettings() {
  const { t } = useI18n()
  const s = t.settings.hud
  const prefs = useStore($hudPrefs)
  const profiles = useStore($profiles)
  // Local draft: the chord commits on blur/Enter so a half-typed "Alt+" never
  // tears down the live registration.
  const [draft, setDraft] = useState<null | string>(null)

  useEffect(() => {
    void loadHudPrefs()
    void refreshProfiles()

    return watchHudPrefs()
  }, [])

  if (!canUseHud() || !prefs) {
    return null
  }

  const commit = () => {
    const next = (draft ?? '').trim()
    setDraft(null)

    if (next && next !== prefs.askShortcut) {
      void setHudPrefs({ askShortcut: next })
    }
  }

  const status =
    prefs.askError === 'taken'
      ? s.takenBy
      : prefs.askError === 'invalid'
        ? s.invalidShortcut
        : prefs.askRegistered
          ? s.active
          : null

  return (
    <>
      <ToggleRow
        checked={prefs.follow}
        description={prefs.followSupported ? s.followDesc : s.followUnsupported}
        disabled={!prefs.followSupported}
        label={s.followTitle}
        onChange={on => void setHudPrefs({ follow: on })}
      />
      <ListRow
        action={
          <Input
            aria-label={s.askShortcutTitle}
            className="w-56 font-mono text-xs"
            onBlur={commit}
            onChange={event => setDraft(event.target.value)}
            onKeyDown={event => {
              if (event.key === 'Enter') {
                event.preventDefault()
                event.currentTarget.blur()
              }
            }}
            spellCheck={false}
            value={draft ?? prefs.askShortcut}
          />
        }
        description={s.askShortcutDesc}
        hint={status}
        title={s.askShortcutTitle}
      />
      <ToggleRow
        checked={prefs.pets}
        description={s.petsDesc}
        label={s.petsTitle}
        onChange={on => void setHudPrefs({ pets: on })}
      />
      {prefs.pets ? (
        <ListRow
          below={
            <div className="grid gap-1.5">
              {profiles.map((profile, index) => {
                const key = normalizeProfileKey(profile.name)
                const current = prefs.petByAgent[key] ?? (index === 0 ? 'hank' : index === 1 ? 'mina' : 'avatar')

                return (
                  <label className="flex items-center justify-between gap-3 text-xs" key={key}>
                    <span className="truncate">{profileLabel(profile)}</span>
                    <select
                      aria-label={`${s.petByAgentTitle}: ${profileLabel(profile)}`}
                      className="rounded border border-(--ui-stroke-secondary) bg-(--ui-bg-elevated) px-2 py-1 text-xs"
                      onChange={event =>
                        void setHudPrefs({
                          petByAgent: { ...prefs.petByAgent, [key]: event.target.value as HudPetChoice }
                        })
                      }
                      value={current}
                    >
                      {HUD_PET_CHOICES.map(choice => (
                        <option key={choice} value={choice}>
                          {s.petChoice[choice]}
                        </option>
                      ))}
                    </select>
                  </label>
                )
              })}
            </div>
          }
          description={s.petByAgentDesc}
          title={s.petByAgentTitle}
        />
      ) : null}
      <ToggleRow
        checked={prefs.askOnRightClick}
        description={prefs.askHookAvailable ? s.rightClickDesc : s.rightClickUnavailable(prefs.askHookReason ?? '')}
        disabled={!prefs.askHookAvailable}
        label={s.rightClickTitle}
        onChange={on => void setHudPrefs({ askOnRightClick: on })}
      />
    </>
  )
}
