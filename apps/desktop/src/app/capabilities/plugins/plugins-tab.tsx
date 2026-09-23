import { useStore } from '@nanostores/react'
import {
  memo,
  type ReactNode,
  type PointerEvent as ReactPointerEvent,
  useEffect,
  useMemo,
  useRef,
  useState
} from 'react'

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
import { CATALOG_ORIGIN, CATALOG_PICKER_URL } from '@/lib/plugin-catalog'
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
import { $paneHeightOverride, setPaneHeightOverride } from '@/store/panes'
import { openCatalogPluginInstall } from '@/store/plugin-catalog-install'
import { openPluginInstallRequest } from '@/store/plugin-install-request'
import { $connection } from '@/store/session'

import { PanelEmpty } from '../../overlays/panel'
import { Pill } from '../../settings/primitives'
import { useDeepLinkHighlight } from '../../settings/use-deep-link-highlight'

import { mergePluginPackages, type PackageKind, type PluginPackage } from './plugin-packages'
import { PluginSettingsForm } from './plugin-settings-form'

// The REAL Plugin Catalog page (docs site) embedded as a one-click picker —
// the same pattern as the Skills tab's EmbeddedHubPicker. `?embed=picker`
// hides the docs chrome and adds "+ Add to this Agent" per card, which posts
//   { type: 'hermes-plugin-pick', name, repo, sha, subdir, tier, installCmd }
// to the parent window. We validate the origin and open the shared
// dual-target install modal (agent half → catalog-pinned install into the
// scoped profile; desktop half → this app), so unified packages install both
// halves in one flow. URLs live in `@/lib/plugin-catalog` so the
// `hermes://plugin/install?catalog=` deep link resolves against the same feed.

// Catalog viewport: persisted through the shared pane store, dragged from the
// section's TOP edge ("pull the catalog up"), clamped so neither the catalog
// nor the plugin list above can vanish. Same contract as EmbeddedHubPicker.
const CATALOG_PANE_ID = 'capabilities-plugin-catalog'
const CATALOG_DEFAULT_PX = 380
const CATALOG_MIN_PX = 120
const CATALOG_MAX_VH = 0.75
const CATALOG_COLLAPSED_PX = 4
const CATALOG_LIST_RESERVED_PX = 176

interface PluginPickMessage {
  installCmd?: string
  name?: string
  repo?: string
  sha?: string
  subdir?: string
  tier?: string
  type?: string
}

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
  mcp_not_connected: 'warn',
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

/** Column widths shared by the header and every row so the two control
 *  columns line up down the page like a table. */
const HALF_COL = 'flex w-36 shrink-0 items-center gap-1.5'

/** One control cell; `label` is the accessible name for screen readers only
 *  (the visible column label lives once, in the header). */
function HalfCell({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div aria-label={label} className={HALF_COL} role="cell">
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
        className={cn(
          'flex items-center gap-3 border-b border-(--ui-stroke-tertiary) px-3 py-2.5',
          !settingsOpen && 'last:border-b-0'
        )}
        data-testid={`plugin-row-${pkg.key}`}
        id={pluginElementId(agent?.key ?? agent?.name ?? desktop?.id ?? pkg.key)}
        role="row"
      >
        <div className="flex min-w-0 flex-1 items-start gap-2" role="cell">
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
      {hasSettings && settingsOpen && agent?.key && (
        <div className="border-b border-(--ui-stroke-tertiary) bg-(--ui-bg-secondary,transparent) px-3 py-3 last:border-b-0">
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

/** THE plugins surface: one row per package. Each row shows its Desktop half
 *  (this app — the same for every profile, gateway, or machine) and its Agent
 *  half (the selected profile's backend). Discovery sits underneath: the live
 *  catalog picker plus Install from Git for anything not in the catalog. */
export const PluginsTab = memo(function PluginsTab({
  profile,
  scopeSelector,
  scopeLabel
}: {
  profile: ProfileScope
  /** The Capabilities profile selector; rendered in the Agent column header so
   *  it visibly governs only that column. */
  scopeSelector?: ReactNode
  /** Display name of the selected profile for the Agent column label. */
  scopeLabel?: string
}) {
  const { t } = useI18n()
  const p = t.skills.plugins
  const d = t.settings.plugins
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

  useDeepLinkHighlight({ param: 'plugin', ready: () => true, elementId: pluginElementId })

  // Catalog picker viewport (persisted height, collapse toggle, top-edge sash).
  const heightOverride = useStore($paneHeightOverride(CATALOG_PANE_ID))
  const height = heightOverride ?? CATALOG_DEFAULT_PX
  const open = height > CATALOG_COLLAPSED_PX
  const [pickerMounted, setPickerMounted] = useState(open)
  const [dragging, setDragging] = useState(false)
  const sectionRef = useRef<HTMLElement>(null)

  if (open && !pickerMounted) {
    setPickerMounted(true)
  }

  const startDrag = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) {
      return
    }

    event.preventDefault()
    const startY = event.clientY
    const startHeight = height
    const column = sectionRef.current?.parentElement
    const columnMax = column ? column.clientHeight - CATALOG_LIST_RESERVED_PX : Number.POSITIVE_INFINITY
    const max = Math.max(CATALOG_MIN_PX, Math.round(Math.min(window.innerHeight * CATALOG_MAX_VH, columnMax)))
    setDragging(true)

    const onMove = (move: globalThis.PointerEvent) => {
      setPaneHeightOverride(
        CATALOG_PANE_ID,
        Math.round(Math.min(max, Math.max(CATALOG_MIN_PX, startHeight + (startY - move.clientY))))
      )
    }

    const onUp = () => {
      window.removeEventListener('pointermove', onMove)
      setDragging(false)
    }

    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp, { once: true })
  }

  useEffect(() => {
    if (!open) {
      return undefined
    }

    const onMessage = (event: MessageEvent) => {
      if (event.origin !== CATALOG_ORIGIN) {
        return
      }

      const data = event.data as null | PluginPickMessage

      if (!data || data.type !== 'hermes-plugin-pick' || !data.name || !data.repo) {
        return
      }

      // Already-installed short-circuit + the dialog itself live in the shared
      // helper so a catalog deep link behaves identically to this pick.
      openCatalogPluginInstall(
        {
          name: String(data.name),
          repo: String(data.repo),
          sha: data.sha ? String(data.sha) : undefined,
          subdir: data.subdir ? String(data.subdir) : undefined
        },
        scope
      )
    }

    window.addEventListener('message', onMessage)

    return () => window.removeEventListener('message', onMessage)
  }, [open, p, scope])

  const agentBusy = (row: AgentPluginRow) => busyKey === (row.key ?? row.name) || busyKey === row.name

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="min-h-32 flex-1 overflow-y-auto">
        {/* Header: what the two columns mean, and the controls that act on
            the whole page (install, folder, rescan). */}
        <div className="flex flex-wrap items-start justify-between gap-3 px-3 pt-3 pb-2">
          <p className="min-w-0 flex-1 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
            {p.pageBlurb}
          </p>
          <div className="flex shrink-0 items-center gap-1">
            <Button
              onClick={() => openPluginInstallRequest({ profile: scope, repo: '' })}
              size="sm"
              type="button"
              variant="secondary"
            >
              {d.installModal.installFromGit}
            </Button>
            <Tip label={d.openFolder}>
              <Button
                aria-label={d.openFolder}
                onClick={() => void revealPluginsDir()}
                size="icon"
                type="button"
                variant="ghost"
              >
                <FolderOpen className="size-3.5" />
              </Button>
            </Tip>
            <Tip label={d.rescan}>
              <Button
                aria-label={d.rescan}
                onClick={() => {
                  triggerHaptic('selection')
                  void rescanAll(requestGateway, scope)
                }}
                size="icon"
                type="button"
                variant="ghost"
              >
                <RefreshCw className="size-3.5" />
              </Button>
            </Tip>
          </div>
        </div>

        {status === 'error' ? (
          <PanelEmpty
            action={
              <Button onClick={() => void loadAgentPlugins(requestGateway, scope)} size="sm">
                {t.skills.refresh}
              </Button>
            }
            description={error ?? undefined}
            icon="error"
            title={p.loadFailed}
          />
        ) : packages.length === 0 && status === 'ready' ? (
          <p className="px-3 py-3 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
            {p.emptyAll} {p.emptyHint}
          </p>
        ) : (
          <div className="flex flex-col" role="table">
            {/* Column header: the visible labels for the two control columns,
                aligned with the cells below. The profile selector sits INSIDE
                the Agent header so it visibly governs only that column. */}
            <div
              className="flex items-center gap-3 border-y border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) px-3 py-1.5 text-[0.68rem] text-(--ui-text-tertiary)"
              role="row"
            >
              <div className="min-w-0 flex-1" role="columnheader" />
              <div className={HALF_COL} role="columnheader">
                <Monitor aria-hidden className="size-3.5 shrink-0" />
                <Tip label={p.halfDesktopHint}>
                  <span className="font-medium">{p.halfDesktop}</span>
                </Tip>
              </div>
              <div className={HALF_COL} role="columnheader">
                <Package aria-hidden className="size-3.5 shrink-0" />
                {scopeSelector ?? <span className="truncate font-medium">{p.halfAgentIn(label)}</span>}
              </div>
            </div>
            {packages.map(pkg => (
              <PackageRow
                busy={pkg.agent ? agentBusy(pkg.agent) : false}
                key={pkg.key}
                onAgentRemove={row => {
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
                }}
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

                  void updateAgentPlugin(requestGateway, row.name, p.updateFailed(row.name), scope).then(
                    async outcome => {
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
                    }
                  )
                }}
                onDesktopRemove={record => {
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
                }}
                pkg={pkg}
                profile={profile}
                request={requestGateway}
                scope={scope}
                scopeLabel={label}
              />
            ))}
          </div>
        )}
      </div>

      <section
        className="relative flex min-h-9 flex-col overflow-hidden border-t border-(--ui-stroke-secondary)"
        ref={sectionRef}
      >
        <div
          className="group/catsash absolute inset-x-0 top-0 z-10 h-1 -translate-y-1/2 cursor-row-resize"
          data-testid="plugin-catalog-sash"
          onDoubleClick={() => setPaneHeightOverride(CATALOG_PANE_ID, undefined)}
          onPointerDown={startDrag}
        >
          <div
            className={cn(
              'absolute inset-x-0 top-1/2 h-px -translate-y-1/2 transition-colors',
              dragging ? 'bg-(--ui-stroke-secondary)' : 'group-hover/catsash:bg-(--ui-stroke-secondary)'
            )}
          />
        </div>
        <div className="flex shrink-0 items-center justify-between px-3 py-1.5">
          <span className="text-[0.62rem] font-medium tracking-wide uppercase text-(--ui-text-quaternary)">
            {p.catalogTitle}
          </span>
          <Button onClick={() => setPaneHeightOverride(CATALOG_PANE_ID, open ? 0 : undefined)} size="xs" variant="text">
            {open ? p.catalogHide : p.catalogBrowse}
          </Button>
        </div>
        {pickerMounted && (
          <div className={cn('flex min-h-0 flex-col gap-1 px-3 pb-2', !open && 'hidden')}>
            <div
              style={{
                border: '1px solid var(--ui-stroke-secondary)',
                borderRadius: 8,
                flex: `0 1 ${height}px`,
                maxWidth: '100%',
                minHeight: 0,
                minWidth: 320,
                overflow: 'hidden',
                position: 'relative',
                width: '100%'
              }}
            >
              <iframe
                sandbox="allow-scripts allow-same-origin"
                src={CATALOG_PICKER_URL}
                style={{
                  background: 'transparent',
                  border: 'none',
                  height: '133.34%',
                  pointerEvents: dragging ? 'none' : 'auto',
                  transform: 'scale(0.75)',
                  transformOrigin: 'top left',
                  width: '133.34%'
                }}
                title={p.catalogTitle}
              />
            </div>
            <p className="shrink-0 px-1 text-[0.65rem] leading-4 text-(--ui-text-quaternary)">{p.catalogHint}</p>
          </div>
        )}
      </section>
    </div>
  )
})
