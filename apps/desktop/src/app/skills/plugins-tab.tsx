import { useStore } from '@nanostores/react'
import { memo, type ReactNode, useEffect, useMemo } from 'react'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Switch } from '@/components/ui/switch'
import { Tip } from '@/components/ui/tooltip'
import { $pluginRecords, type PluginRecord, setPluginEnabled } from '@/contrib/plugins-store'
import { discoverRuntimePlugins } from '@/contrib/runtime-loader'
import type { ProfileScope } from '@/hermes'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { FolderOpen, Loader2, Monitor, Package, RefreshCw } from '@/lib/icons'
import { cn } from '@/lib/utils'
import {
  $agentPluginBusy,
  $agentPlugins,
  type AgentPluginRow,
  type GatewayRequest,
  isDesktopRelevantPlugin,
  loadAgentPlugins,
  toggleAgentPlugin,
  updateAgentPlugin
} from '@/store/agent-plugins'
import { notify, notifyError } from '@/store/notifications'
import { openPluginInstallRequest } from '@/store/plugin-install-request'
import { $connection } from '@/store/session'

import { Pill } from '../settings/primitives'
import { useDeepLinkHighlight } from '../settings/use-deep-link-highlight'

import type { CapabilityView } from './capability-tabs'
import { CatalogBrowser } from './catalog-browser'
import { type CatalogEntry, parseCatalog } from './catalog-data'
import { mergePluginPackages, type PackageKind, type PluginPackage } from './plugin-packages'

/** Deep-link anchor for a package row (`/skills?tab=plugins&plugin=<key>`).
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
function HalfCell({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div aria-label={label} className="flex w-full items-center gap-3" role="cell">
      <span className="min-w-0 flex-1 text-xs text-(--ui-text-tertiary)">{label}</span>
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
  scopeLabel,
  busy,
  onAgentToggle,
  onAgentUpdate
}: {
  pkg: PluginPackage
  scope: null | string
  scopeLabel: string
  busy: boolean
  onAgentToggle: (row: AgentPluginRow, enable: boolean) => void
  onAgentUpdate: (row: AgentPluginRow) => void
}) {
  const { t } = useI18n()
  const p = t.skills.plugins
  const d = t.settings.plugins
  const desktop = pkg.desktop
  const agent = pkg.agent
  const desktopOn = desktop ? desktop.status !== 'disabled' : false
  const agentOn = agent?.status === 'enabled'
  const agentToggleable = Boolean(agent?.key)
  // Electron's desktop-half reconcile only walks THIS machine's homes, so a
  // package installed on a remote backend can never materialize here (#114079).
  const remoteBackend = useStore($connection)?.mode === 'remote'

  return (
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

      <HalfCell label={p.halfAgentIn(scopeLabel)}>
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
          onClick={() => void rescanAll(requestGateway, scope)}
          size="icon-xs"
          variant="ghost"
        >
          <RefreshCw />
        </Button>
      </Tip>
    </>
  )
}

/** THE plugins surface: one row per package. Each row shows its Desktop half
 *  (this app — the same for every profile, gateway, or machine) and its Agent
 *  half (the selected profile's backend). Browse uses the shared native
 *  catalog; Install from Git remains available for unlisted packages. */
export const PluginsTab = memo(function PluginsTab({
  profile,
  scopeLabel,
  view = 'installed',
  query = '',
  onQueryChange
}: {
  query?: string
  onQueryChange?: (value: string) => void
  view?: CapabilityView
  profile: ProfileScope
  /** Display name of the selected profile for the Agent column label. */
  scopeLabel?: string
}) {
  const { t } = useI18n()
  const p = t.skills.plugins
  const { requestGateway } = useGatewayRequest()

  const desktopRecords = useStore($pluginRecords)
  const agentRows = useStore($agentPlugins)
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

  useDeepLinkHighlight({ param: 'plugin', ready: () => true, elementId: pluginElementId })

  const agentBusy = (row: AgentPluginRow) => busyKey === (row.key ?? row.name) || busyKey === row.name

  const installedEntries = useMemo(
    () =>
      parseCatalog(
        'plugins',
        packages.map(pkg => ({
          name: pkg.name,
          identifier: pkg.agent?.catalog_name ?? pkg.desktop?.packageOrigin?.catalogName ?? pkg.key,
          description: pkg.description,
          category: pkg.kind === 'desktop' ? 'desktop' : 'general',
          tier: pkg.agent?.catalog_tier ?? pkg.agent?.source ?? pkg.desktop?.kind ?? '',
          repo: pkg.desktop?.packageOrigin?.repo ?? '',
          sha: pkg.agent?.installed_sha ?? pkg.desktop?.packageOrigin?.sha ?? '',
          version: pkg.agent?.version ?? ''
        }))
      ).map((entry, index) => ({ ...entry, id: `installed:${packages[index].key}` })),
    [packages]
  )

  const packageById = useMemo(() => new Map(packages.map(pkg => [`installed:${pkg.key}`, pkg])), [packages])

  const isInstalled = (entry: CatalogEntry) =>
    packageById.has(entry.id) ||
    agentRows.some(row => (row.catalog_name === entry.name || row.name === entry.name) && !row.update_available)

  const install = (entry: CatalogEntry) =>
    openPluginInstallRequest({
      catalogName: entry.name,
      profile: scope,
      repo: entry.subdir ? `${entry.repo}#${entry.subdir}` : entry.repo,
      sha: entry.sha
    })

  return (
    <div className="flex h-full min-h-0 flex-col">
      <CatalogBrowser
        installedEntries={installedEntries}
        isInstalled={isInstalled}
        kind="plugins"
        onInstall={install}
        onQueryChange={onQueryChange}
        query={query}
        renderInstalledDetail={entry => {
          const pkg = packageById.get(entry.id)

          if (!pkg) {
            return null
          }

          return (
            <PackageRow
              busy={pkg.agent ? agentBusy(pkg.agent) : false}
              key={pkg.key}
              onAgentToggle={(row, enable) => {
                if (row.key) {
                  void toggleAgentPlugin(requestGateway, row.key, enable, p.toggleFailed(row.name), scope)
                }
              }}
              onAgentUpdate={row => {
                void updateAgentPlugin(requestGateway, row.name, p.updateFailed(row.name), scope).then(applied => {
                  if (applied) {
                    notify({ kind: 'success', message: p.updated(row.name) })
                    void rescanAll(requestGateway, scope)
                  }
                })
              }}
              pkg={pkg}
              scope={scope}
              scopeLabel={label}
            />
          )
        }}
        view={view}
      />
    </div>
  )
})
