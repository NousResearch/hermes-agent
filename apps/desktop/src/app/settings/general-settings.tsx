import { useStore } from '@nanostores/react'
import { type ReactNode, useEffect } from 'react'

import { useI18n } from '@/i18n'
import { Settings2 } from '@/lib/icons'
import { cn } from '@/lib/utils'
import {
  $trayAvailable,
  $trayLaunchAtLoginSupported,
  $trayPlatform,
  $trayPreferences,
  $trayPrefsStatus,
  loadTrayPreferences,
  setTrayPreference
} from '@/store/tray-preferences'

import { SectionHeading, SettingsContent, ToggleRow } from './primitives'

const CAPTION = 'text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)'

function Caption({ children, className }: { children: ReactNode; className?: string }) {
  return <p className={cn(CAPTION, className)}>{children}</p>
}

/**
 * Desktop general preferences (Windows tray + launch at login).
 * Main-process owned via tray-preferences.json + OS login items.
 */
export function GeneralSettings() {
  const { t } = useI18n()
  const copy = t.settings.general
  const prefs = useStore($trayPreferences)
  const status = useStore($trayPrefsStatus)
  const trayAvailable = useStore($trayAvailable)
  const launchSupported = useStore($trayLaunchAtLoginSupported)
  const platform = useStore($trayPlatform)
  const isWindows = platform === 'win32' || launchSupported

  useEffect(() => {
    void loadTrayPreferences()
  }, [])

  if (status === 'error') {
    return (
      <SettingsContent>
        <SectionHeading icon={Settings2} title={copy.title} />
        <Caption className="mb-2 leading-(--conversation-caption-line-height)">{copy.loadFailed}</Caption>
      </SettingsContent>
    )
  }

  if (status === 'ready' && platform && platform !== 'win32' && !launchSupported) {
    return (
      <SettingsContent>
        <SectionHeading icon={Settings2} title={copy.title} />
        <Caption className="mb-2 leading-(--conversation-caption-line-height)">{copy.windowsOnly}</Caption>
      </SettingsContent>
    )
  }

  const masterOn = prefs.enabled
  const subDisabled = !masterOn

  return (
    <SettingsContent>
      <SectionHeading icon={Settings2} title={copy.title} />
      <Caption className="mb-2 leading-(--conversation-caption-line-height)">{copy.intro}</Caption>

      {!trayAvailable && masterOn ? (
        <Caption className="mb-2 text-amber-600 dark:text-amber-400">{copy.trayUnavailable}</Caption>
      ) : null}

      <ToggleRow
        checked={prefs.enabled}
        description={copy.enableDesc}
        label={copy.enable}
        onChange={on => void setTrayPreference('enabled', on)}
      />

      <ToggleRow
        checked={prefs.closeToTray}
        description={copy.closeToTrayDesc}
        disabled={subDisabled}
        label={copy.closeToTray}
        onChange={on => void setTrayPreference('closeToTray', on)}
      />

      <ToggleRow
        checked={prefs.startInTray}
        description={copy.startInTrayDesc}
        disabled={subDisabled}
        label={copy.startInTray}
        onChange={on => void setTrayPreference('startInTray', on)}
      />

      <ToggleRow
        checked={prefs.popOutPetOnStartup}
        description={copy.popOutPetOnStartupDesc}
        disabled={subDisabled}
        label={copy.popOutPetOnStartup}
        onChange={on => void setTrayPreference('popOutPetOnStartup', on)}
      />

      <ToggleRow
        checked={prefs.launchAtLogin}
        description={copy.launchAtLoginDesc}
        disabled={!launchSupported}
        label={copy.launchAtLogin}
        onChange={on => void setTrayPreference('launchAtLogin', on)}
      />
    </SettingsContent>
  )
}

/** Back-compat alias while patches settle. */
export const TraySettings = GeneralSettings
export const TraySettingsSection = GeneralSettings
