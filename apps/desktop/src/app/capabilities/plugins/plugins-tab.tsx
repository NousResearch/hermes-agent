import { useStore } from '@nanostores/react'
import { memo, type ReactNode, useCallback, useEffect, useMemo, useState } from 'react'

import { setEnvVar } from '@/api/config'
import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Switch } from '@/components/ui/switch'
import { Tip } from '@/components/ui/tooltip'
import { $pluginRecords, type PluginRecord, setPluginEnabled } from '@/contrib/plugins-store'
import { discoverRuntimePlugins, uninstallDiskPlugin } from '@/contrib/runtime-loader'
import type { ProfileScope } from '@/hermes'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { FolderOpen, Loader2, Monitor, Package, RefreshCw, Trash2 } from '@/lib/icons'
import { cn } from '@/lib/utils'
import {
  $agentPluginBusy,
  $agentPlugins,
  $agentPluginsError,
  $agentPluginsStatus,
  type AgentPluginRow,
  type AgentPluginServerState,
  type AgentPluginUpdateOutcome,
  type GatewayRequest,
  isDesktopRelevantPlugin,
  loadAgentPlugins,
  removeAgentPlugin,
  saveAgentPluginSettings,
  toggleAgentPlugin,
  updateAgentPlugin
} from '@/store/agent-plugins'
import { confirm } from '@/store/confirm'
import { notify, notifyError } from '@/store/notifications'
import { openCatalogPluginInstall } from '@/store/plugin-catalog-install'
import { openPluginInstallRequest } from '@/store/plugin-install-request'
import { $connection } from '@/store/session'

import { Pill } from '../../settings/primitives'
import { useDeepLinkHighlight } from '../../settings/use-deep-link-highlight'
import { CatalogAlert } from '../catalog/catalog-alert'
import { CatalogBrowser } from '../catalog/catalog-browser'
import { type CatalogEntry, parseCatalog } from '../catalog/catalog-data'

import { mergePluginPackages, type PackageKind, type PluginPackage } from './plugin-packages'
import { PluginSettingsForm } from './plugin-settings-form'

/** Deep-link anchor for a package row (`/capabilities?tab=plugins&plugin=<key>`).
 *  Accepts the agent key, the agent name, or the desktop record id. */
export const pluginElementId = (target: string) => `plugin-${target}`

/** Derive the bare profile name a `plugins.manage` call should target. */
function profileParam(scope: ProfileScope): null | string {
  if (!scope) {
    return null
  }

  return typeof scope === 'string' ? scope : (scope.profile ?? null)
}

function reveal(file: string) {
  void window.hermesDesktop?.revealPath?.(file)?.catch(() => undefined)
}

async function revealPluginsDir() {
  try {
    // Electron owns the app-level plugin root — deriving it from the backend's
    // hermes_home breaks against a remote backend (#66899).
    const dir = await window.hermesDesktop?.desktopPluginsRoot?.()

    if (!dir) {
      notifyError('Desktop plugins are unavailable', 'Could not resolve the plugins folder')

      return
    }

    const result = await window.hermesDesktop?.openDir?.(dir)

    if (result && !result.ok) {
      notifyError(result.error ?? 'unknown error', 'Could not open the plugins folder')
    }
  } catch (err) {
    notifyError(err, 'Could not resolve the plugins folder')
  }
}

/** Copy any changed unified desktop halves into the app root FIRST, then
 *  rescan the root — a concurrent scan would read the pre-copy state. */
async function rescanAll(requestGateway: GatewayRequest, scope: null | string) {
  await window.hermesDesktop?.reconcileDesktopPlugins?.().catch(() => undefined)
  await discoverRuntimePlugins()
  await loadAgentPlugins(requestGateway, scope)
}

/** Open the dual-target install modal pre-filled to install ONLY the agent
 *  half of a unified package into the scoped profile (the desktop half is
 *  already here). Provenance comes from the package marker Electron stamped
 *  when it copied the half out (catalog sidecar or git remote). */
function installAgentHalfHere(record: PluginRecord, profile: null | string) {
  const origin = record.packageOrigin

  if (!origin?.repo) {
    return
  }

  openPluginInstallRequest({
    catalogName: origin.catalogName,
    legacyHint: 'agent',
    profile,
    repo: origin.repo,
    sha: origin.sha
  })
}

const SERVER_TONE = {
  connected: 'success',
  app_not_running: 'warn',
  endpoint_unavailable: 'warn',
  no_interactive_session: 'warn',
  unknown: 'warn',
  version_too_old: 'destructive',
  missing_app: 'destructive'
} as const satisfies Record<AgentPluginServerState, 'destructive' | 'success' | 'warn'>

function KindBadge({ kind }: { kind: PackageKind }) {
  const { t } = useI18n()
  const p = t.skills.plugins

  return (
    <span className="inline-flex items-center gap-1 rounded border border-(--ui-stroke-tertiary) px-1.5 py-px text-[0.65rem] text-(--ui-text-tertiary)">
      {kind !== 'desktop' && <Package aria-hidden className="size-3" />}
      {kind !== 'agent' && <Monitor aria-hidden className="size-3" />}
      {kind === 'both' ? p.kindBoth : kind === 'agent' ? p.kindAgent : p.kindDesktop}
    </span>
  )
}

/** Provenance pill: where the package came from. */
function ProvenancePill({ pkg }: { pkg: PluginPackage }) {
  const { t } = useI18n()
  const p = t.skills.plugins

  if (pkg.agent?.catalog_name) {
    return (
      <Tip label={p.catalogProvenance(pkg.agent.installed_sha?.slice(0, 8) ?? '')}>
        <span>
          <Pill>{pkg.agent.catalog_tier === 'official' ? p.tierOfficial : p.tierCommunity}</Pill>
        </span>
      </Tip>
    )
  }

  if (pkg.agent?.pinned_sha) {
    return (
      <Tip label={p.pinnedProvenance(pkg.agent.pinned_sha.slice(0, 8))}>
        <span>
          <Pill>
            <span className="font-mono">{p.pinnedBadge(pkg.agent.pinned_sha.slice(0, 8))}</span>
          </Pill>
        </span>
      </Tip>
    )
  }

  if (pkg.agent) {
    return <Pill>{pkg.agent.source}</Pill>
  }

  if (pkg.desktop) {
    return <Pill>{t.settings.plugins.kinds[pkg.desktop.kind]}</Pill>
  }

  return null
}

/** Controls for one installed plugin half in the detail pane. */
function HalfCell({ label, labelContent, children }: { label: string; labelContent?: ReactNode; children: ReactNode }) {
  return (
    <div aria-label={label} className="flex w-full items-center gap-3" role="cell">
      <span className="min-w-0 flex-1 text-xs text-(--ui-text-tertiary)">{labelContent ?? label}</span>
      {children}
    </div>
  )
}

function Dash() {
  return (
    <span aria-hidden className="w-9 text-center text-(--ui-text-quaternary)">
      —
    </span>
  )
}

function PackageRow({
  pkg,
  scope,
  profile,
  scopeLabel,
  scopeSelector,
  busy,
  request,
  onAgentToggle,
  onAgentUpdate,
  onAgentRemove,
  onDesktopRemove
}: {
  pkg: PluginPackage
  scope: null | string
  profile: ProfileScope
  scopeLabel: string
  scopeSelector?: ReactNode
  busy: boolean
  request: GatewayRequest
  onAgentToggle: (row: AgentPluginRow, enable: boolean) => void
  onAgentUpdate: (row: AgentPluginRow) => void
  onAgentRemove: (row: AgentPluginRow) => void
  onDesktopRemove: (record: PluginRecord) => void
}) {
  const { t } = useI18n()
  const p = t.skills.plugins
  const d = t.settings.plugins
  const desktop = pkg.desktop
  const agent = pkg.agent
  // Manifest `config_schema` → an inline settings form under the row (#46600, #87934).
  const settingsFields = agent?.settings_schema ?? []
  const hasSettings = Boolean(agent?.key) && settingsFields.length > 0
  const [settingsOpen, setSettingsOpen] = useState(false)
  const desktopOn = desktop ? desktop.status !== 'disabled' : false
  const agentOn = agent?.status === 'enabled'
  const agentToggleable = Boolean(agent?.key)
  // Only what lives under the profile's plugins dir ("user", or "git" when it
  // was cloned there) can be uninstalled here: bundled plugins are refused by
  // the backend and entrypoint (pip-installed) ones go with their package.
  const agentRemovable = agent?.source === 'user' || agent?.source === 'git'
  const unavailableServers = agent?.servers?.filter(server => server.state !== 'connected') ?? []
  // A STANDALONE desktop plugin (a folder in <HERMES_HOME>/desktop-plugins with
  // no agent package behind it) is deleted by Electron. A unified package's
  // desktop half is not offered here: uninstalling the agent half prunes it.
  const desktopRemovable = desktop?.kind === 'disk' && !desktop.packageName && !agent
  // Electron's desktop-half reconcile only walks THIS machine's homes, so a
  // package installed on a remote backend can never materialize here (#114079).
  const remoteBackend = useStore($connection)?.mode === 'remote'

  return (
    <>
      <div
        className="flex flex-col gap-5"
        data-testid={`plugin-row-${pkg.key}`}
        id={pluginElementId(agent?.key ?? agent?.name ?? desktop?.id ?? pkg.key)}
        role="row"
      >
        <div className="flex w-full min-w-0 flex-1 items-start gap-2" role="cell">
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2 text-[length:var(--conversation-text-font-size)] font-medium text-foreground">
              <span>{pkg.name}</span>
              {agent?.version && <span className="text-(--ui-text-quaternary)">v{agent.version}</span>}
              <KindBadge kind={pkg.kind} />
              <ProvenancePill pkg={pkg} />
              {agent?.portable && <Pill>{p.portableBadge}</Pill>}
              {agent?.servers?.map(server => (
                <Pill data-testid={`server-pill-${server.name}`} key={server.name} tone={SERVER_TONE[server.state]}>
                  {server.name}: {p.serverStates[server.state]}
                </Pill>
              ))}
              {desktop?.status === 'error' && <Pill tone="primary">{d.failed}</Pill>}
            </div>
            {(desktop?.status === 'error' ? desktop.error : pkg.description) && (
              <div
                className={cn(
                  'mt-0.5 text-[length:var(--conversation-caption-font-size)] break-words',
                  desktop?.status === 'error' ? 'text-(--ui-danger,#f87171)' : 'text-(--ui-text-tertiary)'
                )}
              >
                {desktop?.status === 'error' ? desktop.error : pkg.description}
              </div>
            )}
            {unavailableServers.map(server =>
              server.sentence ? (
                <div
                  className="mt-0.5 text-[length:var(--conversation-caption-font-size)] break-words text-(--ui-text-secondary)"
                  key={server.name}
                >
                  {server.sentence}
                </div>
              ) : null
            )}
          </div>
          {/* Fixed slot so the switch column stays straight whether or not
            this row has a folder to reveal (bundled plugins have none). */}
          <span className="flex size-7 shrink-0 items-center justify-center">
            {desktop?.file && (
              <Tip label={d.reveal}>
                <Button onClick={() => reveal(desktop.file!)} size="icon" variant="ghost">
                  <Codicon name="folder-opened" size="0.85rem" />
                </Button>
              </Tip>
            )}
          </span>
          {/* Fixed slot for the settings gear: only plugins whose manifest declares
            a config_schema get one. */}
          <span className="flex size-7 shrink-0 items-center justify-center">
            {hasSettings && (
              <Tip label={p.settingsToggle(pkg.name)}>
                <Button
                  aria-expanded={settingsOpen}
                  aria-label={p.settingsToggle(pkg.name)}
                  className={cn(settingsOpen && 'text-foreground')}
                  onClick={() => setSettingsOpen(open => !open)}
                  size="icon"
                  variant="ghost"
                >
                  <Codicon name="settings-gear" size="0.85rem" />
                </Button>
              </Tip>
            )}
          </span>
          {/* Same fixed-slot treatment for Uninstall: present on every row so the
            halves line up, populated when the agent half is a user install or
            the row is a standalone desktop plugin. */}
          <span className="flex size-7 shrink-0 items-center justify-center">
            {agent && agentRemovable ? (
              <Tip label={p.uninstallTip(pkg.name, scopeLabel)}>
                <Button
                  aria-label={`${p.uninstall}: ${pkg.name}`}
                  className="text-(--ui-text-tertiary) hover:text-(--ui-danger,#f87171)"
                  disabled={busy}
                  onClick={() => onAgentRemove(agent)}
                  size="icon"
                  variant="ghost"
                >
                  <Trash2 className="size-3.5" />
                </Button>
              </Tip>
            ) : desktop && desktopRemovable ? (
              <Tip label={p.uninstallDesktopTip(pkg.name)}>
                <Button
                  aria-label={`${p.uninstall}: ${pkg.name}`}
                  className="text-(--ui-text-tertiary) hover:text-(--ui-danger,#f87171)"
                  onClick={() => onDesktopRemove(desktop)}
                  size="icon"
                  variant="ghost"
                >
                  <Trash2 className="size-3.5" />
                </Button>
              </Tip>
            ) : null}
          </span>
        </div>

        {/* The two halves. Desktop is app-level and reads the same whichever
          profile is selected; Agent follows the selector. A half the package
          lacks shows a dash; a half it has but which is missing on this side
          shows the install affordance. */}
        <HalfCell label={p.halfDesktop}>
          {desktop ? (
            <Switch
              aria-label={`${p.halfDesktop}: ${pkg.name}`}
              checked={desktopOn}
              onCheckedChange={on => {
                triggerHaptic('selection')
                void setPluginEnabled(desktop.id, on)
              }}
            />
          ) : pkg.desktopMissing ? (
            <Tip label={remoteBackend ? p.desktopHalfRemoteTip : p.desktopHalfPendingTip}>
              <span className="text-[0.65rem] text-(--ui-text-tertiary)">
                {remoteBackend ? p.desktopHalfRemote : p.desktopHalfPending}
              </span>
            </Tip>
          ) : (
            <Dash />
          )}
        </HalfCell>

        <HalfCell label={p.halfAgentIn(scopeLabel)} labelContent={scopeSelector}>
          {agent ? (
            <>
              {agent.update_available && (
                <Button
                  className="h-5 px-1.5 text-[0.65rem]"
                  disabled={busy}
                  onClick={() => onAgentUpdate(agent)}
                  size="xs"
                  variant="outline"
                >
                  {p.updateToPin(agent.catalog_version ?? agent.catalog_sha?.slice(0, 8) ?? '')}
                </Button>
              )}
              {busy && <Loader2 className="size-3.5 animate-spin text-(--ui-text-tertiary)" />}
              {agentToggleable ? (
                <Switch
                  aria-label={`${p.halfAgent}: ${pkg.name}`}
                  checked={agentOn}
                  disabled={busy}
                  onCheckedChange={on => onAgentToggle(agent, on)}
                />
              ) : (
                <Tip label={p.legacyBackend}>
                  <span>
                    <Switch aria-label={`${p.halfAgent}: ${pkg.name}`} checked={agentOn} disabled />
                  </span>
                </Tip>
              )}
            </>
          ) : pkg.agentMissingInProfile && desktop ? (
            <Tip label={desktop.packageOrigin?.repo ? p.installAgentHereTip(scopeLabel) : p.installAgentHereNoOrigin}>
              <span>
                <Button
                  className="h-5 px-1.5 text-[0.65rem]"
                  disabled={!desktop.packageOrigin?.repo}
                  onClick={() => installAgentHalfHere(desktop, scope)}
                  size="xs"
                  variant="outline"
                >
                  {p.installAgentHere}
                </Button>
              </span>
            </Tip>
          ) : (
            <Dash />
          )}
        </HalfCell>
      </div>
      {hasSettings && settingsOpen && agent?.key && (
        <div className="mt-5">
          <PluginSettingsForm
            disabled={busy}
            fields={settingsFields}
            idPrefix={`plugin-settings-${agent.key}`}
            onSave={async changes => {
              const ok = await saveAgentPluginSettings(request, {
                key: agent.key!,
                values: changes.values,
                secrets: changes.secrets,
                writeSecret: (env, value) => setEnvVar(env, value, profile),
                failMessage: p.settingsForm.saveFailed(pkg.name),
                profile: scope
              })

              if (ok) {
                notify({ kind: 'success', message: p.settingsForm.saved(pkg.name) })
              }

              return ok
            }}
          />
        </div>
      )}
    </>
  )
}

export function PluginActions({ profile }: { profile: ProfileScope }) {
  const { t } = useI18n()
  const d = t.settings.plugins
  const { requestGateway } = useGatewayRequest()
  const scope = profileParam(profile)

  return (
    <>
      <Button
        className="underline"
        onClick={() => openPluginInstallRequest({ profile: scope, repo: '' })}
        size="xs"
        variant="text"
      >
        {d.installModal.installFromGit}
      </Button>
      <Tip label={d.openFolder}>
        <Button aria-label={d.openFolder} onClick={() => void revealPluginsDir()} size="icon-xs" variant="ghost">
          <FolderOpen />
        </Button>
      </Tip>
      <Tip label={d.rescan}>
        <Button
          aria-label={d.rescan}
          onClick={() => {
            triggerHaptic('selection')
            void rescanAll(requestGateway, scope)
          }}
          size="icon-xs"
          variant="ghost"
        >
          <RefreshCw />
        </Button>
      </Tip>
    </>
  )
}

/** Installed packages retain both halves' controls in the native detail pane.
 *  Browse shares the Skills catalog; navigation and search belong to the shell. */
export const PluginsTab = memo(function PluginsTab({
  profile,
  scopeSelector,
  scopeLabel,
  query,
  onQueryChange
}: {
  profile: ProfileScope
  query?: string
  onQueryChange?: (value: string) => void
  /** The profile selector governs only the Agent half, not the app-level Desktop half. */
  scopeSelector?: ReactNode
  /** Display name of the selected profile for the Agent half label. */
  scopeLabel?: string
}) {
  const { t } = useI18n()
  const p = t.skills.plugins
  const { requestGateway } = useGatewayRequest()

  const desktopRecords = useStore($pluginRecords)
  const agentRows = useStore($agentPlugins)
  const status = useStore($agentPluginsStatus)
  const error = useStore($agentPluginsError)
  const busyKey = useStore($agentPluginBusy)

  const scope = profileParam(profile)
  const label = scopeLabel ?? scope ?? t.skills.plugins.defaultProfile

  useEffect(() => {
    void loadAgentPlugins(requestGateway, scope)
  }, [requestGateway, scope])

  const packages = useMemo(
    () => mergePluginPackages(Object.values(desktopRecords), agentRows.filter(isDesktopRelevantPlugin)),
    [agentRows, desktopRecords]
  )

  const [selectedEntryId, setSelectedEntryId] = useState<string | null>(null)

  const packageForTarget = useCallback(
    (target: string) =>
      packages.find(pkg => [pkg.key, pkg.agent?.key, pkg.agent?.name, pkg.desktop?.id].includes(target)),
    [packages]
  )

  useDeepLinkHighlight({
    param: 'plugin',
    ready: target => Boolean(packageForTarget(target)),
    elementId: target => {
      const pkg = packageForTarget(target)

      return pluginElementId(pkg?.agent?.key ?? pkg?.agent?.name ?? pkg?.desktop?.id ?? target)
    },
    onResolve: useCallback(
      (target: string) => {
        const pkg = packageForTarget(target)

        if (pkg) {
          onQueryChange?.('')
          setSelectedEntryId(`installed:${pkg.key}`)
        }
      },
      [onQueryChange, packageForTarget]
    )
  })

  const agentBusy = (row: AgentPluginRow) => busyKey === (row.key ?? row.name) || busyKey === row.name

  const installedEntries = useMemo(
    () =>
      parseCatalog(
        'plugins',
        packages.map(pkg => ({
          name: pkg.name,
          identifier: pkg.key,
          description: pkg.description,
          category: pkg.kind === 'desktop' ? 'desktop' : 'general',
          tier: pkg.agent?.catalog_tier ?? pkg.agent?.source ?? pkg.desktop?.kind ?? '',
          repo: pkg.desktop?.packageOrigin?.repo ?? '',
          sha: pkg.agent?.installed_sha ?? pkg.desktop?.packageOrigin?.sha ?? '',
          version: pkg.agent?.version ?? ''
        }))
      ).map(entry => ({ ...entry, id: `installed:${entry.identifier}` })),
    [packages]
  )

  const packageById = useMemo(() => new Map(packages.map(pkg => [`installed:${pkg.key}`, pkg])), [packages])

  const installedByCatalogName = useMemo(() => {
    const byName = new Map<string, CatalogEntry>()

    for (const entry of installedEntries) {
      const pkg = packageById.get(entry.id)
      const name = pkg?.agent?.catalog_name ?? pkg?.desktop?.packageOrigin?.catalogName

      if (name) {
        byName.set(name, entry)
      }
    }

    return byName
  }, [installedEntries, packageById])

  const matchInstalled = useCallback(
    (entry: CatalogEntry) => installedByCatalogName.get(entry.name),
    [installedByCatalogName]
  )
  const isInstalled = (entry: CatalogEntry) => packageById.has(entry.id)

  const handleAgentRemove = useCallback(
    (row: AgentPluginRow) => {
      void confirm({
        confirmLabel: p.uninstall,
        description: p.uninstallConfirmBody(row.name, label),
        destructive: true,
        title: p.uninstallConfirmTitle(row.name)
      }).then(async ok => {
        if (!ok) {
          return
        }

        if (await removeAgentPlugin(requestGateway, row.name, p.uninstallFailed(row.name), scope)) {
          notify({ kind: 'success', message: p.uninstalled(row.name) })
          // Prunes the app-level desktop half whose source package just went away.
          void rescanAll(requestGateway, scope)
        }
      })
    },
    [label, p, requestGateway, scope]
  )

  const handleDesktopRemove = useCallback(
    (record: PluginRecord) => {
      void confirm({
        confirmLabel: p.uninstall,
        description: p.uninstallDesktopConfirmBody(record.name),
        destructive: true,
        title: p.uninstallConfirmTitle(record.name)
      }).then(async ok => {
        if (!ok) {
          return
        }

        const result = await uninstallDiskPlugin(record.id)

        if (result.ok) {
          notify({ kind: 'success', message: p.uninstalledDesktop(record.name) })
        } else {
          notifyError(result.error, p.uninstallFailed(record.name))
        }
      })
    },
    [p]
  )

  // The card switch turns the whole package on or off, like a skill's; per-half
  // switches and uninstall (trash + confirm) live in the detail's PackageRow.
  const packageSwitch = (pkg: PluginPackage) => {
    const { agent, desktop } = pkg

    const setEnabled = (enable: boolean) => {
      if (agent?.key) {
        void toggleAgentPlugin(requestGateway, agent.key, enable, p.toggleFailed(agent.name), scope)
      }

      if (desktop) {
        void setPluginEnabled(desktop.id, enable)
      }
    }

    return (
      <Switch
        aria-label={pkg.name}
        checked={agent ? agent.status === 'enabled' : desktop?.status !== 'disabled'}
        disabled={agent ? !agent.key || agentBusy(agent) : false}
        onCheckedChange={setEnabled}
        size="xs"
      />
    )
  }

  const notice =
    status === 'error' ? (
      <CatalogAlert
        onRetry={() => void loadAgentPlugins(requestGateway, scope)}
        retryLabel={t.skills.refresh}
        title={p.loadFailed}
      >
        {error}
      </CatalogAlert>
    ) : null

  return (
    <CatalogBrowser
      headerActions={<PluginActions profile={profile} />}
      installedEntries={installedEntries}
      installedPending={status !== 'ready'}
      isInstalled={isInstalled}
      kind="plugins"
      matchInstalled={matchInstalled}
      notice={notice}
      onInstall={entry => openCatalogPluginInstall(entry, scope)}
      onQueryChange={onQueryChange}
      query={query}
      renderInstalledAction={entry => {
        const pkg = packageById.get(entry.id)

        return pkg ? packageSwitch(pkg) : null
      }}
      renderInstalledDetail={entry => {
        const pkg = packageById.get(entry.id)

        if (!pkg) {
          return null
        }

        return (
          <PackageRow
            busy={pkg.agent ? agentBusy(pkg.agent) : false}
            key={pkg.key}
            onAgentRemove={handleAgentRemove}
            onAgentToggle={(row, enable) => {
              if (!row.key) {
                return
              }

              void toggleAgentPlugin(requestGateway, row.key, enable, p.toggleFailed(row.name), scope)
            }}
            onAgentUpdate={row => {
              const finish = (outcome: AgentPluginUpdateOutcome) => {
                if (outcome.kind === 'applied') {
                  notify({ kind: 'success', message: p.updated(row.name) })
                  void rescanAll(requestGateway, scope)
                }
              }

              void updateAgentPlugin(requestGateway, row.name, p.updateFailed(row.name), scope).then(async outcome => {
                if (outcome.kind !== 'consent') {
                  finish(outcome)

                  return
                }

                // The new pin widens the plugin (tools, hooks, deps, capabilities, a Desktop
                // half); the backend changed nothing until the user confirms the delta.
                const ok = await confirm({
                  confirmLabel: p.updateConsentConfirm,
                  description: [p.updateConsentBody(row.name, outcome.sha), ...outcome.deltaLines].join('\n'),
                  title: p.updateConsentTitle(row.name)
                })

                if (ok) {
                  finish(await updateAgentPlugin(requestGateway, row.name, p.updateFailed(row.name), scope, true))
                }
              })
            }}
            onDesktopRemove={handleDesktopRemove}
            pkg={pkg}
            profile={profile}
            request={requestGateway}
            scope={scope}
            scopeLabel={label}
            scopeSelector={scopeSelector}
          />
        )
      }}
      selectedEntryId={selectedEntryId}
    />
  )
})
