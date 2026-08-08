import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'

import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { ErrorBanner } from '@/components/ui/error-state'
import { Input } from '@/components/ui/input'
import {
  authMcpServer,
  getActionStatus,
  getMcpCatalog,
  getMcpOAuthFlow,
  installMcpCatalogEntry,
  type McpCatalogEntry,
  setMcpServerEnabled
} from '@/hermes'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Wrench } from '@/lib/icons'
import { completeMcpDesktopOAuth } from '@/lib/mcp-dashboard-oauth'
import { $gateway } from '@/store/gateway'
import { notify, notifyError } from '@/store/notifications'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import { $activeSessionId } from '@/store/session'

import { prettyName } from './helpers'
import { ListRow, SettingsGroup, SettingsSection } from './primitives'

const CATALOG_INSTALL_POLL_MS = 1500

type IntegrationAction = 'activate' | 'install' | 'ready' | 'sign-in'

interface IntegrationsCatalogProps {
  /** Optional hub search term; matching is deliberately client-side because the curated catalog is small. */
  query?: string
  /** Keeps installed entries in the Hub's shared Installed section instead of repeating them below. */
  hideInstalled?: boolean
  /** Shows only integrations that have already been added to this profile. */
  onlyInstalled?: boolean
  /** Do not leave an empty section behind when another content type has results. */
  hideWhenEmpty?: boolean
  /** Lets the Browse Hub use its own section copy without duplicating install behavior. */
  title?: string
  /** Allows an embedded catalog to omit the Settings-specific ready count. */
  meta?: null | string
  /** `null` intentionally suppresses the otherwise helpful default description. */
  description?: null | string
  /** Optional empty-state copy for embedded surfaces with a narrower scope. */
  emptyDescription?: string
}

function Tag({ children }: { children: string }) {
  return (
    <span className="rounded-[var(--radius-sm)] bg-(--ui-bg-tertiary) px-1.5 py-0.5 text-[0.625rem] font-medium text-(--ui-text-secondary)">
      {children}
    </span>
  )
}

function requiredEnvironmentIsMissing(entry: McpCatalogEntry, values: Record<string, string>): boolean {
  return entry.required_env.some(env => env.required && !values[env.name]?.trim())
}

function actionFor(entry: McpCatalogEntry): IntegrationAction {
  if (!entry.installed) {
    return 'install'
  }

  if (!entry.enabled) {
    return 'activate'
  }

  if (entry.auth_type === 'oauth' && entry.authenticated === false) {
    return 'sign-in'
  }

  return 'ready'
}

export function matchesIntegrationCatalogQuery(entry: McpCatalogEntry, query: string): boolean {
  const term = query.trim().toLocaleLowerCase()

  if (!term) {
    return true
  }

  return [entry.name, prettyName(entry.name), entry.description, entry.source, entry.transport, entry.auth_type].some(
    value => value.toLocaleLowerCase().includes(term)
  )
}

/**
 * One install surface for curated external tools. Server configuration and OAuth
 * tokens stay in the profile; a successful install reloads the active gateway
 * immediately, while the normal gateway startup path revives it on later app
 * launches. There is deliberately no separate connect/manage screen.
 */
export function IntegrationsCatalog({
  description,
  emptyDescription,
  hideInstalled = false,
  hideWhenEmpty = false,
  meta,
  onlyInstalled = false,
  query = '',
  title
}: IntegrationsCatalogProps) {
  const { t } = useI18n()
  const p = t.settings.plugins
  const activeProfile = useStore($activeGatewayProfile)
  const activeSessionId = useStore($activeSessionId)
  const gateway = useStore($gateway)
  const [installing, setInstalling] = useState<null | string>(null)
  const [envDrafts, setEnvDrafts] = useState<Record<string, Record<string, string>>>({})
  const [envOpenFor, setEnvOpenFor] = useState<null | string>(null)

  const catalogQuery = useQuery({
    queryKey: ['integrations-catalog', normalizeProfileKey(activeProfile)],
    queryFn: getMcpCatalog
  })

  const entries = [...(catalogQuery.data?.entries ?? [])]
    .filter(entry => matchesIntegrationCatalogQuery(entry, query))
    .filter(entry => (onlyInstalled ? entry.installed : !hideInstalled || !entry.installed))
    .sort((a, b) => Number(b.installed) - Number(a.installed) || prettyName(a.name).localeCompare(prettyName(b.name)))

  const readyCount = entries.filter(entry => entry.installed && entry.enabled && entry.authenticated !== false).length
  const sectionTitle = title ?? p.integrationsTitle
  const sectionDescription = description === undefined ? p.integrationsDescription : description
  const sectionMeta = meta === undefined ? p.integrationsCount(readyCount) : meta

  if (hideWhenEmpty && !catalogQuery.isLoading && !catalogQuery.isError && entries.length === 0) {
    return null
  }

  const reloadInstalledTools = async (): Promise<boolean> => {
    if (!gateway) {
      return false
    }

    // This is an explicit user install/activation, so the normal reload consent
    // is already satisfied. Reload starts every persisted integration, not just
    // the one that was clicked, and refreshes the current agent's tool snapshot.
    try {
      await gateway.request('reload.mcp', { confirm: true, session_id: activeSessionId ?? undefined })

      return true
    } catch {
      // The durable config write already succeeded. A disconnected gateway can
      // simply pick it up through normal startup on the next app launch.
      return false
    }
  }

  const finishOAuth = async (entry: McpCatalogEntry) => {
    const openExternal = window.hermesDesktop?.openExternal

    if (!openExternal) {
      throw new Error(p.integrationsBrowserUnavailable)
    }

    await completeMcpDesktopOAuth({
      serverName: entry.name,
      start: authMcpServer,
      status: getMcpOAuthFlow,
      openExternal: url => openExternal(url)
    })
  }

  const waitForBackgroundInstall = async (entry: McpCatalogEntry, action: string) => {
    for (;;) {
      const status = await getActionStatus(action, 1)

      if (!status.running) {
        if (status.exit_code !== 0) {
          throw new Error(p.integrationsInstallFailed(entry.name))
        }

        return
      }

      await new Promise<void>(resolve => window.setTimeout(resolve, CATALOG_INSTALL_POLL_MS))
    }
  }

  const runAction = async (entry: McpCatalogEntry) => {
    const action = actionFor(entry)
    const draft = envDrafts[entry.name] ?? {}

    if (action === 'install' && entry.required_env.length > 0 && envOpenFor !== entry.name) {
      setEnvOpenFor(entry.name)

      return
    }

    if (action === 'install' && requiredEnvironmentIsMissing(entry, draft)) {
      notify({
        kind: 'error',
        title: p.integrationsCredentialsTitle(entry.name),
        message: p.integrationsCredentialsRequired
      })

      return
    }

    setInstalling(entry.name)

    try {
      if (action === 'install') {
        const result = await installMcpCatalogEntry(entry.name, draft)

        if (result.background && result.action) {
          await waitForBackgroundInstall(entry, result.action)
        }

        if (entry.auth_type === 'oauth') {
          await finishOAuth(entry)
        }
      } else if (action === 'sign-in') {
        await finishOAuth(entry)
      } else if (action === 'activate') {
        await setMcpServerEnabled(entry.name, true)

        if (entry.auth_type === 'oauth' && entry.authenticated === false) {
          await finishOAuth(entry)
        }
      }

      const startedNow = await reloadInstalledTools()
      triggerHaptic('success')
      setEnvOpenFor(null)
      notify({
        kind: 'success',
        title: p.integrationReady(entry.name),
        message: startedNow ? p.integrationReadyMessage : p.integrationReadyNextLaunch
      })
    } catch (error) {
      notifyError(error, p.integrationsInstallFailed(entry.name))
    } finally {
      setInstalling(null)
      void catalogQuery.refetch()
    }
  }

  return (
    <SettingsSection icon={Wrench} meta={sectionMeta ?? undefined} title={sectionTitle}>
      {sectionDescription && (
        <p className="mb-3 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
          {sectionDescription}
        </p>
      )}

      {catalogQuery.isLoading ? (
        <SettingsGroup>
          <PageLoader className="min-h-28" label={p.integrationsLoading} />
        </SettingsGroup>
      ) : catalogQuery.isError ? (
        <ErrorBanner>
          <span className="flex flex-wrap items-center gap-2">
            {p.integrationsLoadFailed}
            <Button onClick={() => void catalogQuery.refetch()} size="xs" variant="textStrong">
              {p.integrationsRetry}
            </Button>
          </span>
        </ErrorBanner>
      ) : entries.length === 0 ? (
        <SettingsGroup>
          <ListRow description={emptyDescription ?? p.integrationsEmpty} title={sectionTitle} />
        </SettingsGroup>
      ) : (
        <SettingsGroup>
          {entries.map(entry => {
            const action = actionFor(entry)
            const busy = installing === entry.name
            const draft = envDrafts[entry.name] ?? {}
            const credentialsOpen = envOpenFor === entry.name && entry.required_env.length > 0

            const status =
              action === 'ready'
                ? p.integrationsReady
                : action === 'sign-in'
                  ? p.integrationsNeedsSignIn
                  : action === 'activate'
                    ? p.integrationsDisabled
                    : null

            const actionLabel = busy
              ? p.integrationsInstalling
              : action === 'ready'
                ? p.integrationsReady
                : action === 'sign-in'
                  ? p.integrationsSignIn
                  : action === 'activate'
                    ? p.integrationsActivate
                    : p.integrationsInstall

            return (
              <ListRow
                action={
                  <Button
                    aria-label={`${actionLabel} ${prettyName(entry.name)}`}
                    disabled={action === 'ready' || installing !== null}
                    onClick={() => void runAction(entry)}
                    size="xs"
                    variant={action === 'ready' ? 'secondary' : 'default'}
                  >
                    {actionLabel}
                  </Button>
                }
                below={
                  credentialsOpen && (
                    <div className="mt-3 grid max-w-md gap-2 rounded-[var(--radius-md)] border border-(--ui-stroke-secondary) bg-(--ui-bg-tertiary)/45 p-3">
                      <p className="text-[0.6875rem] font-medium text-foreground">
                        {p.integrationsCredentialsTitle(entry.name)}
                      </p>
                      <p className="text-[0.6875rem] leading-[1.35] text-(--ui-text-tertiary)">
                        {p.integrationsCredentialsDescription}
                      </p>
                      {entry.required_env.map(env => (
                        <label className="grid gap-1" key={env.name}>
                          <span className="text-[0.6875rem] text-(--ui-text-secondary)">
                            {env.prompt || env.name}
                            {env.required ? ' *' : ''}
                          </span>
                          <Input
                            autoComplete="off"
                            className="h-8 text-xs"
                            disabled={installing !== null}
                            onChange={event =>
                              setEnvDrafts(current => ({
                                ...current,
                                [entry.name]: { ...current[entry.name], [env.name]: event.currentTarget.value }
                              }))
                            }
                            type="password"
                            value={draft[env.name] ?? ''}
                          />
                        </label>
                      ))}
                    </div>
                  )
                }
                description={entry.description}
                key={entry.name}
                title={
                  <span className="flex flex-wrap items-center gap-1.5">
                    <span>{prettyName(entry.name)}</span>
                    <Tag>{entry.transport}</Tag>
                    {entry.auth_type === 'oauth' && <Tag>{p.integrationsOAuth}</Tag>}
                    {entry.auth_type === 'api_key' && <Tag>{p.integrationsApiKey}</Tag>}
                    {entry.needs_install && !entry.installed && <Tag>{p.integrationsNeedsBuild}</Tag>}
                    {status && <span className="text-[0.6875rem] font-medium text-(--ui-text-tertiary)">{status}</span>}
                  </span>
                }
              />
            )
          })}
        </SettingsGroup>
      )}
    </SettingsSection>
  )
}
