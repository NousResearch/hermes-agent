import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { isLinuxPlatform } from '@/lib/platform'
import { $hudAlwaysOnTop, setHudAlwaysOnTop, watchHudAlwaysOnTop } from '@/store/hud'
import { notifyError } from '@/store/notifications'

import { ToggleRow } from './primitives'

interface TrayPreferences {
  launchAtLogin: boolean
  openNewConversationOnClick: boolean
}

/**
 * Quick access: what the tray icon does, whether the Mini Assistant floats,
 * and whether Hermes comes back on its own at login.
 *
 * Split from `MinimizeToTraySetting` on purpose. That row is the master
 * switch — it decides whether the tray icon exists at all — while everything
 * here is inert without it, so the two read as one group in the Window layout
 * page but keep their own authorities (main owns the icon and the login item;
 * the Mini Assistant's pin is the HUD's own persisted preference).
 */
export function TrayAccessSettings() {
  const { t } = useI18n()
  const c = t.settings.config
  const bridge = window.hermesDesktop?.trayPreferences
  const alwaysOnTopBridge = window.hermesDesktop?.hud?.alwaysOnTop
  const alwaysOnTop = useStore($hudAlwaysOnTop)
  const [preferences, setPreferences] = useState<TrayPreferences | null>(null)
  const [saving, setSaving] = useState<'launchAtLogin' | 'openNewConversationOnClick' | null>(null)
  const [loadFailed, setLoadFailed] = useState(false)
  const revision = useRef(0)

  const load = useCallback(async () => {
    if (!bridge) {
      return
    }

    const version = ++revision.current
    setLoadFailed(false)

    try {
      const next = await bridge.get()

      if (revision.current === version) {
        setPreferences(next)
      }
    } catch (error) {
      if (revision.current === version) {
        setLoadFailed(true)
        notifyError(error, c.failedLoad)
      }
    }
  }, [bridge, c.failedLoad])

  useEffect(() => {
    if (!bridge) {
      return
    }

    const unsubscribe = bridge.onChanged(next => {
      revision.current++
      setPreferences(next)
      setLoadFailed(false)
    })

    void load()

    return () => {
      revision.current++
      unsubscribe()
    }
  }, [bridge, load])

  // The pin state is main's, and it outlives any window — so follow it rather
  // than assuming the bar's current z-order is the preference.
  useEffect(() => watchHudAlwaysOnTop(), [])

  if (!bridge) {
    return null
  }

  const save = async (patch: Partial<TrayPreferences>, key: 'launchAtLogin' | 'openNewConversationOnClick') => {
    const previous = preferences
    const version = ++revision.current
    setSaving(key)
    setPreferences({ ...(preferences ?? { launchAtLogin: false, openNewConversationOnClick: true }), ...patch })

    try {
      const next = await bridge.set(patch)

      if (revision.current === version) {
        setPreferences(next)
      }
    } catch (error) {
      if (revision.current === version) {
        setPreferences(previous)
      }

      notifyError(error, c.autosaveFailed)
    } finally {
      if (revision.current === version) {
        setSaving(null)
      }
    }
  }

  const busy = (key: 'launchAtLogin' | 'openNewConversationOnClick') => saving === key

  return (
    <>
      <ToggleRow
        checked={preferences?.openNewConversationOnClick ?? false}
        description={c.trayNewConversationOnClickDesc}
        disabled={!preferences || busy('openNewConversationOnClick')}
        label={c.trayNewConversationOnClickTitle}
        onChange={on => void save({ openNewConversationOnClick: on }, 'openNewConversationOnClick')}
      />

      {/* The bar is inherently a floating surface, so this only exists when the
          shell can actually answer for it. On by default — that IS the shipped
          behavior — and turning it off gives the Mini Assistant an ordinary
          window's z-order. */}
      {alwaysOnTopBridge && (
        <ToggleRow
          checked={alwaysOnTop}
          description={c.miniAssistantAlwaysOnTopDesc}
          label={c.miniAssistantAlwaysOnTopTitle}
          onChange={on => setHudAlwaysOnTop(on)}
        />
      )}

      {/* Linux has no login-item API: the desktop's own autostart facilities
          own that job, so the row is absent rather than offering a dead lever. */}
      {!isLinuxPlatform() && (
        <ToggleRow
          checked={preferences?.launchAtLogin ?? false}
          description={c.launchAtLoginDesc}
          disabled={!preferences || busy('launchAtLogin')}
          label={c.launchAtLoginTitle}
          onChange={on => void save({ launchAtLogin: on }, 'launchAtLogin')}
        />
      )}

      {loadFailed && (
        <Button onClick={() => void load()} size="sm" variant="secondary">
          {t.settings.screenshot.retry}
        </Button>
      )}
    </>
  )
}
