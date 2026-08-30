import { useCallback, useState } from 'react'

import { type ProfileScope, saveHermesConfigRecord } from '@/hermes'
import { useI18n } from '@/i18n'
import { notify, notifyError } from '@/store/notifications'

import { hermesConfigCacheWriter, useHermesConfigRecord } from '../hooks/use-config-record'

import { ToggleRow } from './primitives'

interface BrowserRealProfilePanelProps {
  /** Capabilities profile-scope override — the toggle reads/writes THIS
   *  profile's config.yaml instead of the app-wide active one. */
  profile?: ProfileScope
}

function readUseRealProfile(record: Record<string, unknown> | undefined): boolean {
  const browser = record?.browser

  if (browser && typeof browser === 'object' && !Array.isArray(browser)) {
    return Boolean((browser as Record<string, unknown>).use_real_profile)
  }

  return false
}

/**
 * The `browser.use_real_profile` consent toggle, rendered at the top of the
 * Capabilities → Tools → Browser detail pane (above the backend/provider
 * matrix). This is the GUI home of the real-profile browsing switch: without
 * it the only desktop path was the generic Settings → Config editor, which
 * users reasonably never found ("no toggle in the browser section").
 *
 * Semantics mirror the config comment: turning it ON consents to snapshotting
 * the default browser's profile (cookies/logins) into a Hermes-owned copy;
 * turning it OFF deletes the snapshot store on next use. The toggle writes
 * config.yaml through the same deep-merging PUT /api/config every other
 * settings surface uses — applies to new sessions.
 */
export function BrowserRealProfilePanel({ profile }: BrowserRealProfilePanelProps) {
  const { t } = useI18n()
  const copy = t.settings.toolsets.browserRealProfile
  const { data: config } = useHermesConfigRecord(profile)
  const setConfig = hermesConfigCacheWriter(profile)
  const [busy, setBusy] = useState(false)

  const enabled = readUseRealProfile(config)

  const toggle = useCallback(
    async (on: boolean) => {
      if (!config) {
        return
      }

      const browser =
        config.browser && typeof config.browser === 'object' && !Array.isArray(config.browser)
          ? (config.browser as Record<string, unknown>)
          : {}

      const next = { ...config, browser: { ...browser, use_real_profile: on } }

      setBusy(true)
      setConfig(next)

      try {
        await saveHermesConfigRecord(next, profile)
        notify({
          kind: 'info',
          title: on ? copy.enabledTitle : copy.disabledTitle,
          message: on ? copy.enabledMessage : copy.disabledMessage
        })
      } catch (err) {
        setConfig(config)
        notifyError(err, copy.failedSave)
      } finally {
        setBusy(false)
      }
    },
    [config, copy, profile, setConfig]
  )

  return (
    <ToggleRow
      checked={enabled}
      description={copy.description}
      disabled={busy || !config}
      label={copy.label}
      onChange={on => void toggle(on)}
    />
  )
}
