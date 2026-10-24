import { useQuery } from '@tanstack/react-query'
import { useCallback, useState } from 'react'

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { getHermesConfigSchema, type ProfileScope, saveHermesConfigRecord } from '@/hermes'
import { useI18n } from '@/i18n'
import { notify, notifyError } from '@/store/notifications'
import { normalizeProfileKey } from '@/store/profile'

import { hermesConfigCacheWriter, useHermesConfigRecord } from '../hooks/use-config-record'

import { CONTROL_TEXT, EMPTY_SELECT_VALUE, REAL_PROFILE_BROWSER_LABELS } from './constants'
import { ListRow, ToggleRow } from './primitives'

interface BrowserRealProfilePanelProps {
  /** Capabilities profile-scope override — the toggle reads/writes THIS
   *  profile's config.yaml instead of the app-wide active one. */
  profile?: ProfileScope
}

/** Shared with the Browser pane's first-open consent prompt, so both surfaces
 *  agree on what "on" means for `browser.use_real_profile`. */
export function readUseRealProfile(record: Record<string, unknown> | undefined): boolean {
  const browser = record?.browser

  if (browser && typeof browser === 'object' && !Array.isArray(browser)) {
    return Boolean((browser as Record<string, unknown>).use_real_profile)
  }

  return false
}

const ALL_REAL_PROFILE_BROWSERS = Object.keys(REAL_PROFILE_BROWSER_LABELS)

function browserSection(config: Record<string, unknown>): Record<string, unknown> {
  return config.browser && typeof config.browser === 'object' && !Array.isArray(config.browser)
    ? (config.browser as Record<string, unknown>)
    : {}
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
  const { data: config, writeScope } = useHermesConfigRecord(profile)
  const setConfig = hermesConfigCacheWriter(profile)
  const [busy, setBusy] = useState(false)

  const enabled = readUseRealProfile(config)

  // Same schema query (and cache key) as Settings → Config: the backend lists
  // only the Chromium browsers installed on this machine plus the saved value.
  const schemaProfile = typeof profile === 'object' && profile !== null ? profile.profile : profile

  const { data: schemaResponse } = useQuery({
    queryKey:
      schemaProfile == null ? ['hermes-config-schema'] : ['hermes-config-schema', normalizeProfileKey(schemaProfile)],
    queryFn: () => getHermesConfigSchema(schemaProfile),
    staleTime: 5 * 60 * 1000
  })

  const toggle = useCallback(
    async (on: boolean) => {
      if (!config) {
        return
      }

      const next = { ...config, browser: { ...browserSection(config), use_real_profile: on } }

      setBusy(true)
      setConfig(next)

      try {
        // Sparse patch: PUT /api/config deep-merges, and echoing the cached
        // snapshot would overwrite keys other surfaces changed since it loaded.
        await saveHermesConfigRecord({ browser: { use_real_profile: on } }, writeScope ?? profile)

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
    [config, copy, profile, setConfig, writeScope]
  )

  const selectedBrowser = config ? String(browserSection(config).real_profile_browser ?? '') : ''
  const schemaOptions = schemaResponse?.fields?.['browser.real_profile_browser']?.options

  // Fall back to every supported browser until (or unless) the schema answers.
  const browserOptions = (schemaOptions ? schemaOptions.map(String) : ALL_REAL_PROFILE_BROWSERS).filter(Boolean)

  if (selectedBrowser && !browserOptions.includes(selectedBrowser)) {
    browserOptions.push(selectedBrowser)
  }

  const chooseBrowser = useCallback(
    async (key: string) => {
      if (!config) {
        return
      }

      setBusy(true)
      setConfig({ ...config, browser: { ...browserSection(config), real_profile_browser: key } })

      try {
        await saveHermesConfigRecord({ browser: { real_profile_browser: key } }, writeScope ?? profile)
        notify({ kind: 'info', title: copy.browserSavedTitle, message: copy.browserSavedMessage })
      } catch (err) {
        setConfig(config)
        notifyError(err, copy.failedSave)
      } finally {
        setBusy(false)
      }
    },
    [config, copy, profile, setConfig, writeScope]
  )

  return (
    <>
      <ToggleRow
        checked={enabled}
        description={copy.description}
        disabled={busy || !config}
        label={copy.label}
        onChange={on => void toggle(on)}
      />
      {enabled && (
        <ListRow
          action={
            <Select
              disabled={busy || !config}
              onValueChange={next => void chooseBrowser(next === EMPTY_SELECT_VALUE ? '' : next)}
              value={selectedBrowser || EMPTY_SELECT_VALUE}
            >
              <SelectTrigger aria-label={copy.browserLabel} className={CONTROL_TEXT}>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value={EMPTY_SELECT_VALUE}>{t.settings.config.systemDefault}</SelectItem>
                {browserOptions.map(key => (
                  <SelectItem key={key} value={key}>
                    {REAL_PROFILE_BROWSER_LABELS[key] ?? key}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          }
          description={copy.browserDescription}
          title={copy.browserLabel}
        />
      )}
    </>
  )
}
