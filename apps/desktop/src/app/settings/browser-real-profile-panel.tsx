import { useCallback, useState } from 'react'

import type { PreviewSigninsResult } from '@/global'
import { type ProfileScope, saveHermesConfigRecord } from '@/hermes'
import { useI18n } from '@/i18n'
import { claimPreviewSigninsSync, syncPreviewSignins } from '@/lib/preview-signins'
import { notify, notifyError } from '@/store/notifications'

import { hermesConfigCacheWriter, useHermesConfigRecord } from '../hooks/use-config-record'

import { RowFootnoteAction, ToggleRow } from './primitives'

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
 * settings surface uses — applies to new sessions. It also signs the in-app
 * Browser tab in with the profile's cookies (Electron main imports them for a
 * local backend; off removes them), and "Re-import sign-ins" refreshes them.
 */
export function BrowserRealProfilePanel({ profile }: BrowserRealProfilePanelProps) {
  const { t } = useI18n()
  const copy = t.settings.toolsets.browserRealProfile
  const { data: config, writeScope } = useHermesConfigRecord(profile)
  const setConfig = hermesConfigCacheWriter(profile)
  const [busy, setBusy] = useState(false)

  const enabled = readUseRealProfile(config)

  const reportSignins = useCallback(
    (result: null | PreviewSigninsResult, announce: boolean) => {
      if (!result) {
        return
      }

      if (result.status === 'error') {
        notify({ kind: 'warning', title: copy.signinsFailed, message: result.error || '' })
      } else if (result.status === 'imported' && (announce || result.imported > 0)) {
        notify({ kind: 'info', title: copy.signinsImportedTitle, message: copy.signinsImported(result.imported) })
      } else if (result.status === 'remote' && announce) {
        notify({ kind: 'info', title: copy.signinsImportedTitle, message: copy.signinsRemote })
      }
    },
    [copy]
  )

  const reimport = useCallback(async () => {
    setBusy(true)

    try {
      reportSignins(await syncPreviewSignins(true, writeScope), true)
    } catch (err) {
      notifyError(err, copy.signinsFailed)
    } finally {
      setBusy(false)
    }
  }, [copy, reportSignins, writeScope])

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
      claimPreviewSigninsSync(on, writeScope)
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
        // After the save: the backend re-checks the key it just wrote before exporting anything.
        reportSignins(await syncPreviewSignins(on, writeScope), false)
      } catch (err) {
        setConfig(config)
        notifyError(err, copy.failedSave)
      } finally {
        setBusy(false)
      }
    },
    [config, copy, profile, reportSignins, setConfig, writeScope]
  )

  return (
    <ToggleRow
      below={
        enabled ? (
          <RowFootnoteAction onClick={() => void reimport()}>{copy.reimportSignins}</RowFootnoteAction>
        ) : undefined
      }
      checked={enabled}
      description={copy.description}
      disabled={busy || !config}
      label={copy.label}
      onChange={on => void toggle(on)}
    />
  )
}
