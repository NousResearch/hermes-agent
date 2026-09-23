import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { useCallback, useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { getSharedMetricsConsent, getStatus, setSharedMetricsConsent, type SharedMetricsConsentState } from '@/hermes'
import { useI18n } from '@/i18n'
import { ExternalLink } from '@/lib/external-link'
import { BarChart3, Database, FolderOpen, Send } from '@/lib/icons'
import { queryClient } from '@/lib/query-client'
import { notify, notifyError } from '@/store/notifications'
import { $connection } from '@/store/session'
import { $settingsScopeProfile } from '@/store/settings-scope'

import { ListRow, SettingsContent, SettingsSection, ToggleRow } from './primitives'

const DOCS_URL =
  'https://github.com/NousResearch/hermes-agent/blob/main/website/docs/developer-guide/relay-shared-metrics.md'
const TELEMETRY_SUBDIR = 'telemetry/shared_metrics'

const CAPTION =
  'text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)'

export const sharedMetricsConsentKey = (profile?: null | string) =>
  ['telemetry', 'shared-metrics', profile ?? ''] as const

/** Effective consent for a profile, as the backend resolves it (profile keys → global answer → off). */
export function useSharedMetricsConsent(profile?: null | string) {
  return useQuery({
    queryKey: sharedMetricsConsentKey(profile),
    queryFn: () => getSharedMetricsConsent(profile ?? undefined),
    staleTime: 0
  })
}

export const invalidateSharedMetricsConsent = (profile?: null | string) =>
  queryClient.invalidateQueries({ queryKey: sharedMetricsConsentKey(profile) })

function joinPath(home: string, sub: string): string {
  const sep = home.includes('\\') && !home.includes('/') ? '\\' : '/'

  return `${home.replace(/[\\/]+$/, '')}${sep}${sub.split('/').join(sep)}`
}

export function TelemetrySettings() {
  const { t } = useI18n()
  const copy = t.settings.telemetry
  const profile = useStore($settingsScopeProfile)
  const connection = useStore($connection)
  const { data: state } = useSharedMetricsConsent(profile)
  const [busy, setBusy] = useState(false)
  const [hermesHome, setHermesHome] = useState<null | string>(null)

  // The two config keys are one product switch here: "share" = collect + send.
  // A collect-only profile (CLI users can set that) shows as off but keeps its
  // `enabled` until the user flips the switch, which then writes both.
  const sharing = Boolean(state?.enabled && state.send)
  const isLocal = connection?.mode !== 'remote'

  useEffect(() => {
    let cancelled = false
    void getStatus()
      .then(status => {
        if (!cancelled) {
          setHermesHome(status.hermes_home || null)
        }
      })
      .catch(() => undefined)

    return () => {
      cancelled = true
    }
  }, [profile])

  const toggle = useCallback(
    async (on: boolean) => {
      if (!state || busy) {
        return
      }

      const key = sharedMetricsConsentKey(profile)
      setBusy(true)
      // Optimistic: paint the switch from the intent; the route is the truth.
      queryClient.setQueryData<SharedMetricsConsentState>(key, {
        ...state,
        enabled: on,
        send: on,
        decided: true,
        source: 'profile'
      })

      try {
        const result = await setSharedMetricsConsent({ enabled: on, send: on }, profile)
        queryClient.setQueryData<SharedMetricsConsentState>(key, {
          enabled: result.enabled,
          send: result.send,
          decided: result.decided,
          source: result.source
        })
        notify({
          kind: 'info',
          title: on ? copy.enabledTitle : copy.disabledTitle,
          message: on ? copy.enabledMessage : copy.disabledMessage
        })
      } catch (error) {
        queryClient.setQueryData(key, state)
        notifyError(error, copy.failedSave)
      } finally {
        setBusy(false)
      }
    },
    [busy, copy, profile, state]
  )

  const telemetryDir = hermesHome ? joinPath(hermesHome, TELEMETRY_SUBDIR) : null

  const openFolder = async () => {
    if (!telemetryDir) {
      return
    }

    const result = await window.hermesDesktop?.openDir?.(telemetryDir)

    if (!result?.ok) {
      notifyError(result?.error ?? copy.openFolderFailed, copy.openFolderFailed)
    }
  }

  return (
    <SettingsContent>
      <SettingsSection icon={BarChart3} title={copy.title}>
        <p className={CAPTION}>{copy.intro}</p>
        <ul className={`${CAPTION} mt-2 list-disc space-y-1 pl-5`}>
          <li>{copy.pointCounters}</li>
          <li>{copy.pointNoContent}</li>
          <li>{copy.pointInstallId}</li>
          <li>{copy.pointConsentWindow}</li>
        </ul>
        <p className={`${CAPTION} mt-2`}>
          <ExternalLink href={DOCS_URL} showExternalIcon>
            {copy.docsLink}
          </ExternalLink>
        </p>
      </SettingsSection>

      <SettingsSection icon={Send} title={copy.sharingTitle}>
        <ToggleRow
          checked={sharing}
          description={copy.shareDesc}
          disabled={busy || !state}
          label={copy.shareLabel}
          onChange={on => void toggle(on)}
        />
        {state?.source === 'global' ? <p className={`${CAPTION} pb-2`}>{copy.inheritedNote}</p> : null}
        {state?.source === 'env' ? <p className={`${CAPTION} pb-2`}>{copy.deploymentNote}</p> : null}
        {state?.enabled && !state.send ? <p className={`${CAPTION} pb-2`}>{copy.collectOnlyNote}</p> : null}
        <p className={`${CAPTION} pb-2`}>{copy.cliHint}</p>
      </SettingsSection>

      <SettingsSection icon={Database} title={copy.filesTitle}>
        <ListRow
          action={
            isLocal ? (
              <Button
                disabled={!telemetryDir}
                onClick={() => void openFolder()}
                size="sm"
                type="button"
                variant="outline"
              >
                <FolderOpen className="size-3.5" />
                {copy.openFolder}
              </Button>
            ) : null
          }
          description={isLocal ? copy.filesDesc : copy.filesRemoteDesc}
          hint={telemetryDir ?? undefined}
          title={copy.filesLabel}
        />
      </SettingsSection>
    </SettingsContent>
  )
}
