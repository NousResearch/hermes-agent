import { useStore } from '@nanostores/react'
import { memo, useEffect, useMemo, useState } from 'react'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { Button } from '@/components/ui/button'
import { Switch } from '@/components/ui/switch'
import { Tip } from '@/components/ui/tooltip'
import type { ProfileScope } from '@/hermes'
import { useI18n } from '@/i18n'
import { Loader2, Package } from '@/lib/icons'
import { cn } from '@/lib/utils'
import {
  $agentPluginBusy,
  $agentPlugins,
  $agentPluginsError,
  $agentPluginsStatus,
  type AgentPluginRow,
  isDesktopRelevantPlugin,
  loadAgentPlugins,
  toggleAgentPlugin,
  updateAgentPlugin
} from '@/store/agent-plugins'
import { notify } from '@/store/notifications'
import { $paneHeightOverride, setPaneHeightOverride } from '@/store/panes'
import { openPluginInstallRequest } from '@/store/plugin-install-request'

import { PanelEmpty } from '../overlays/panel'

// The REAL Plugin Catalog page (docs site) embedded as a one-click picker —
// the same pattern as the Skills tab's EmbeddedHubPicker. `?embed=picker`
// hides the docs chrome and adds "+ Add to this Agent" per card, which posts
//   { type: 'hermes-plugin-pick', name, repo, sha, subdir, tier, installCmd }
// to the parent window. We validate the origin and open the shared
// dual-target install modal (agent half → catalog-pinned install into the
// scoped profile; desktop half → local app), so bundled agent+desktop
// packages install both halves in one flow.
const CATALOG_ORIGIN = 'https://hermes-agent.nousresearch.com'
const CATALOG_PICKER_URL = `${CATALOG_ORIGIN}/docs/plugins?embed=picker`

const CATALOG_PANE_ID = 'capabilities-plugin-catalog'
const CATALOG_DEFAULT_PX = 380
const CATALOG_COLLAPSED_PX = 4

interface PluginPickMessage {
  installCmd?: string
  name?: string
  repo?: string
  sha?: string
  subdir?: string
  tier?: string
  type?: string
}

/** Derive the bare profile name a `plugins.manage` call should target. */
function profileParam(scope: ProfileScope): null | string {
  if (!scope) {
    return null
  }

  return typeof scope === 'string' ? scope : (scope.profile ?? null)
}

function PluginRow({
  row,
  busy,
  onToggle,
  onUpdate
}: {
  row: AgentPluginRow
  busy: boolean
  onToggle: (enable: boolean) => void
  onUpdate?: () => void
}) {
  const { t } = useI18n()
  const address = row.key ?? ''
  const canToggle = Boolean(address)
  const enabled = row.status === 'enabled'

  return (
    <div className="flex items-start gap-3 border-b border-(--ui-stroke-tertiary) px-3 py-2 last:border-b-0">
      <Package aria-hidden className="mt-0.5 size-4 shrink-0 text-(--ui-text-tertiary)" />
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-2 text-[length:var(--conversation-text-font-size)] font-medium text-foreground">
          {row.name}
          {row.version && <span className="text-(--ui-text-quaternary)">v{row.version}</span>}
          {row.portable && (
            <span className="rounded border border-(--ui-stroke-tertiary) px-1 text-[0.65rem] text-(--ui-text-tertiary)">
              {t.skills.plugins.portableBadge}
            </span>
          )}
          {row.catalog_name && (
            <Tip label={t.skills.plugins.catalogProvenance(row.installed_sha?.slice(0, 8) ?? '')}>
              <span className="rounded border border-(--ui-stroke-tertiary) px-1 text-[0.65rem] text-(--ui-text-tertiary)">
                {row.catalog_tier === 'official' ? t.skills.plugins.tierOfficial : t.skills.plugins.tierCommunity}
              </span>
            </Tip>
          )}
          {row.update_available && onUpdate && (
            <Button
              className="h-5 px-1.5 text-[0.65rem]"
              disabled={busy}
              onClick={onUpdate}
              size="xs"
              variant="outline"
            >
              {t.skills.plugins.updateToPin(row.catalog_sha?.slice(0, 8) ?? '')}
            </Button>
          )}
        </div>
        {row.description && (
          <div className="mt-0.5 text-[length:var(--conversation-caption-font-size)] break-words text-(--ui-text-tertiary)">
            {row.description}
          </div>
        )}
      </div>
      <div className="flex shrink-0 items-center gap-2">
        {busy && <Loader2 className="size-3.5 animate-spin text-(--ui-text-tertiary)" />}
        {canToggle ? (
          <Switch aria-label={row.name} checked={enabled} disabled={busy} onCheckedChange={onToggle} />
        ) : (
          <Tip label={t.skills.plugins.legacyBackend}>
            <span>
              <Switch aria-label={row.name} checked={enabled} disabled />
            </span>
          </Tip>
        )}
      </div>
    </div>
  )
}

/** Agent plugins for the Capabilities page: the scoped profile's installed
 *  plugins on top (toggleable), the live catalog picker underneath — same
 *  management-plus-discovery shape as the Skills tab. */
export const PluginsTab = memo(function PluginsTab({ profile }: { profile: ProfileScope }) {
  const { t } = useI18n()
  const p = t.skills.plugins
  const { requestGateway } = useGatewayRequest()

  const rows = useStore($agentPlugins)
  const status = useStore($agentPluginsStatus)
  const error = useStore($agentPluginsError)
  const busyKey = useStore($agentPluginBusy)

  const scope = profileParam(profile)

  useEffect(() => {
    void loadAgentPlugins(requestGateway, scope)
  }, [requestGateway, scope])

  const visible = useMemo(() => rows.filter(isDesktopRelevantPlugin), [rows])

  // Catalog picker viewport (persisted height, collapse toggle) — same pane
  // store contract as EmbeddedHubPicker.
  const heightOverride = useStore($paneHeightOverride(CATALOG_PANE_ID))
  const height = heightOverride ?? CATALOG_DEFAULT_PX
  const open = height > CATALOG_COLLAPSED_PX
  const [pickerMounted, setPickerMounted] = useState(open)

  if (open && !pickerMounted) {
    setPickerMounted(true)
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

      // Already installed at (or past) this pin in the scoped profile →
      // tell the user instead of re-running the install ceremony. Rows with
      // update_available keep their explicit Update chip in the list above.
      const existing = $agentPlugins.get().find(row => row.catalog_name === data.name || row.name === data.name)

      if (existing && !existing.update_available) {
        notify({ kind: 'success', message: t.skills.plugins.alreadyInstalled(String(data.name)) })

        return
      }

      // Open the shared dual-target install modal: it probes the repo for
      // agent/desktop halves, installs the agent half at the catalog pin
      // into the scoped profile, and offers the desktop half locally.
      openPluginInstallRequest({
        catalogName: String(data.name),
        profile: scope,
        repo: data.subdir ? `${String(data.repo)}#${String(data.subdir)}` : String(data.repo),
        sha: data.sha ? String(data.sha) : undefined
      })
    }

    window.addEventListener('message', onMessage)

    return () => window.removeEventListener('message', onMessage)
  }, [open, scope, t])

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="min-h-32 flex-1 overflow-y-auto">
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
        ) : visible.length === 0 && status === 'ready' ? (
          <PanelEmpty description={p.emptyHint} icon="package" title={p.empty} />
        ) : (
          <div className="flex flex-col">
            {visible.map(row => (
              <PluginRow
                busy={busyKey === (row.key ?? row.name) || busyKey === row.name}
                key={row.key ?? row.name}
                onToggle={enable => {
                  if (!row.key) {
                    return
                  }

                  void toggleAgentPlugin(requestGateway, row.key, enable, p.toggleFailed(row.name), scope)
                }}
                onUpdate={
                  row.update_available
                    ? () => {
                        void updateAgentPlugin(requestGateway, row.name, p.updateFailed(row.name), scope).then(
                          applied => {
                            if (applied) {
                              notify({ kind: 'success', message: p.updated(row.name) })
                            }
                          }
                        )
                      }
                    : undefined
                }
                row={row}
              />
            ))}
          </div>
        )}
      </div>

      <section className={cn('relative flex min-h-9 flex-col overflow-hidden border-t border-(--ui-stroke-secondary)')}>
        <div className="flex shrink-0 items-center justify-between px-3 py-1.5">
          <span className="text-[0.7rem] font-medium text-(--ui-text-tertiary)">{p.catalogTitle}</span>
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
