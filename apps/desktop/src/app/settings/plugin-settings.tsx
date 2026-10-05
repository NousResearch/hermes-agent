import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useMemo } from 'react'
import { useLocation, useNavigate } from 'react-router'

import { setEnvVar } from '@/api/config'
import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { Button } from '@/components/ui/button'
import { Codicon, codiconIcon } from '@/components/ui/codicon'
import { $pluginRecords } from '@/contrib/plugins-store'
import { ContribBoundary, ContribRender } from '@/contrib/react/boundary'
import { useContributions } from '@/contrib/react/use-contributions'
import {
  pluginSettingsEntries,
  type PluginSettingsEntry,
  type PluginSettingsTarget,
  resolvePluginSettingsTarget,
  SETTINGS_PLUGINS_AREA
} from '@/contrib/settings-pages'
import { useI18n } from '@/i18n'
import type { IconComponent } from '@/lib/icons'
import {
  $agentPluginBusy,
  $agentPlugins,
  $agentPluginsStatus,
  isDesktopRelevantPlugin,
  loadAgentPlugins,
  saveAgentPluginSettings
} from '@/store/agent-plugins'
import { notify } from '@/store/notifications'
import { $settingsRequestProfile } from '@/store/settings-scope'

import { desktopPackageName } from '../capabilities/plugins/plugin-packages'
import type { OverlayNavLink } from '../overlays/overlay-split-layout'
import { CAPABILITIES_ROUTE } from '../routes'

import { PluginSettingsForm } from './plugin-settings-form'
import { EmptyState, ListRowSkeleton, SectionHeading, SettingsContent } from './primitives'
import { SettingsProfileScope } from './profile-scope'

/** Opens Settings ▸ Plugins at an entry (by key) and optional sub-page;
 *  `null` = the overview. */
export type OpenPluginSettings = (plugin: null | string, page?: string) => void

const MANAGE_PLUGINS_HREF = `#${CAPABILITIES_ROUTE}?tab=plugins`

// codiconIcon mints a component per call; cache so rail rows keep a stable
// element type across renders (no icon remount on every navigation).
const iconCache = new Map<string, IconComponent>()

function iconFor(name: string): IconComponent {
  let icon = iconCache.get(name)

  if (!icon) {
    icon = codiconIcon(name)
    iconCache.set(name, icon)
  }

  return icon
}

export const PLUGINS_NAV_ICON = iconFor('extensions')

const entryIcon = (entry: PluginSettingsEntry) => iconFor(entry.icon ?? 'plug')

/** Query params owned by one settings page; switching pages drops them. */
export const PAGE_SCOPED_PARAMS = [
  'page',
  'field',
  'setting',
  'key',
  'aux',
  'session',
  'kind',
  'label',
  'origin',
  'plugin',
  'ppage'
] as const

/** Settings ▸ Plugins routing: `?tab=plugins&plugin=<entry key | plugin id |
 *  agent:<key>>&ppage=<sub-page>`. `active` = the Plugins view is showing. */
export function usePluginSettingsRoute(active: boolean) {
  const navigate = useNavigate()
  const { hash, pathname, search } = useLocation()
  const entries = usePluginSettingsEntries()
  const status = useStore($agentPluginsStatus)
  const params = new URLSearchParams(search)
  const requested = params.get('plugin')
  const target = active ? resolvePluginSettingsTarget(entries, requested, params.get('ppage')) : null
  // A deep link to an agent plugin's page can't resolve until the list loads;
  // only call it missing once the load has settled.
  const settled = status !== 'idle' && status !== 'loading'

  const open = useCallback<OpenPluginSettings>(
    (plugin, page) => {
      const next = new URLSearchParams(search)

      for (const key of PAGE_SCOPED_PARAMS) {
        next.delete(key)
      }

      next.set('tab', 'plugins')

      if (plugin) {
        next.set('plugin', plugin)
      }

      if (plugin && page) {
        next.set('ppage', page)
      }

      navigate({ hash, pathname, search: `?${next}` }, { replace: true })
    },
    [hash, navigate, pathname, search]
  )

  return { entries, missing: Boolean(requested) && !target && settled, open, target }
}

/** Every plugin settings page, live: Desktop plugins' registered pages plus an
 *  automatic page for each agent plugin that declares a `config_schema`
 *  (scoped to the Settings profile selector). */
export function usePluginSettingsEntries(): PluginSettingsEntry[] {
  const { t } = useI18n()
  const { requestGateway } = useGatewayRequest()
  const contributions = useContributions(SETTINGS_PLUGINS_AREA)
  const rows = useStore($agentPlugins)
  const records = useStore($pluginRecords)
  const scope = useStore($settingsRequestProfile)

  // Cheap backend disk scan; the same loader Capabilities ▸ Plugins uses.
  useEffect(() => {
    void loadAgentPlugins(requestGateway, scope ?? null)
  }, [requestGateway, scope])

  const configTitle = t.settings.pluginPages.agentSettings

  return useMemo(
    () =>
      pluginSettingsEntries({
        configTitle,
        contributions,
        packageOf: pluginId => {
          const record = records[pluginId]

          return record ? desktopPackageName(record) : null
        },
        rows: rows.filter(isDesktopRelevantPlugin)
      }),
    [configTitle, contributions, records, rows]
  )
}

/** The Settings rail's children under "Plugins": one row per entry, its
 *  sub-pages folded beneath it while it is selected. */
export function pluginSettingsNavChildren(
  entries: readonly PluginSettingsEntry[],
  target: null | PluginSettingsTarget,
  open: OpenPluginSettings
): OverlayNavLink[] {
  return entries.map(entry => {
    const active = target?.entry.key === entry.key

    return {
      active,
      children: entry.children.map(child => ({
        active: active && target?.child?.id === child.id,
        icon: iconFor(child.agentKey ? 'settings-gear' : 'list-flat'),
        id: `plugins:${entry.key}:${child.id}`,
        label: child.title,
        onSelect: () => open(entry.key, child.id)
      })),
      icon: entryIcon(entry),
      id: `plugins:${entry.key}`,
      label: entry.title,
      onSelect: () => open(entry.key)
    }
  })
}

// Lead copy under a page heading, the same caption rhythm native pages use.
const BLURB_CLASS =
  'mb-2 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)'

function PluginSettingsOverview({
  entries,
  missing,
  onOpen
}: {
  entries: readonly PluginSettingsEntry[]
  missing: boolean
  onOpen: OpenPluginSettings
}) {
  const { t } = useI18n()
  const copy = t.settings.pluginPages

  return (
    <SettingsContent>
      <SectionHeading
        aside={
          // Page-level action on the heading row, like Passwords & Logins' Add.
          <Button asChild className="gap-1.5" size="sm" variant="outline">
            <a href={MANAGE_PLUGINS_HREF}>
              {copy.manage}
              <Codicon name="arrow-right" size="0.75rem" />
            </a>
          </Button>
        }
        icon={PLUGINS_NAV_ICON}
        page
        title={t.settings.nav.plugins}
      />
      <p className={BLURB_CLASS}>{copy.blurb}</p>
      {missing && (
        <p className="mb-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-secondary)" role="status">
          {copy.missing}
        </p>
      )}
      {entries.length === 0 ? (
        <EmptyState title={copy.empty} />
      ) : (
        <div className="mt-2 grid overflow-hidden rounded-lg border border-(--ui-stroke-tertiary)">
          {entries.map(entry => {
            const Icon = entryIcon(entry)
            // The landing page counts as one; sub-pages add to it.
            const pages = entry.children.length + 1

            return (
              <button
                className="flex min-h-11 items-center gap-3 border-b border-(--ui-stroke-tertiary) px-3 text-left transition-colors last:border-b-0 hover:bg-(--chrome-action-hover)"
                data-testid={`plugin-settings-row-${entry.key}`}
                key={entry.key}
                onClick={() => onOpen(entry.key)}
                type="button"
              >
                <Icon className="size-4 shrink-0 text-(--ui-text-tertiary)" />
                <span className="min-w-0 flex-1 truncate text-[length:var(--conversation-text-font-size)]">
                  {entry.title}
                </span>
                {pages > 1 && (
                  <span className="shrink-0 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                    {copy.pageCount(pages)}
                  </span>
                )}
                <Codicon className="text-(--ui-text-tertiary)" name="chevron-right" size="0.8rem" />
              </button>
            )
          })}
        </div>
      )}
    </SettingsContent>
  )
}

/** Automatic page for an agent plugin's manifest `config_schema`: the schema
 *  form, saved through `plugins.manage settings` (+ `.env` for secrets) for
 *  the profile the Settings scope selector targets. */
function PluginSchemaSettingsPage({ agentKey }: { agentKey: string }) {
  const { t } = useI18n()
  const p = t.skills.plugins
  const { requestGateway } = useGatewayRequest()
  const row = useStore($agentPlugins).find(candidate => candidate.key === agentKey)
  const busy = useStore($agentPluginBusy) === agentKey
  const scope = useStore($settingsRequestProfile)

  return (
    <SettingsContent>
      <SettingsProfileScope className="mb-5" />
      {row?.settings_schema?.length ? (
        <PluginSettingsForm
          disabled={busy}
          fields={row.settings_schema}
          idPrefix={`plugin-settings-${agentKey}`}
          intro={row.description ? <p className={BLURB_CLASS}>{row.description}</p> : undefined}
          onSave={async changes => {
            const ok = await saveAgentPluginSettings(requestGateway, {
              failMessage: p.settingsForm.saveFailed(row.name),
              key: agentKey,
              profile: scope ?? null,
              secrets: changes.secrets,
              values: changes.values,
              writeSecret: (env, value) => setEnvVar(env, value, scope)
            })

            if (ok) {
              notify({ kind: 'success', message: p.settingsForm.saved(row.name) })
            }

            return ok
          }}
          title={row.name}
        />
      ) : (
        <div className="grid gap-1">
          <ListRowSkeleton />
          <ListRowSkeleton />
          <ListRowSkeleton />
        </div>
      )}
    </SettingsContent>
  )
}

/** Right-hand side of Settings ▸ Plugins: the selected page, or the overview
 *  (every plugin with settings) when nothing — or something stale — is
 *  selected. */
export function PluginSettingsPane({
  entries,
  missing = false,
  onOpen,
  target
}: {
  entries: readonly PluginSettingsEntry[]
  /** A `?plugin=` was requested but matches no page (disabled/uninstalled). */
  missing?: boolean
  onOpen: OpenPluginSettings
  target: null | PluginSettingsTarget
}) {
  if (!target) {
    return <PluginSettingsOverview entries={entries} missing={missing} onOpen={onOpen} />
  }

  const node = target.child ?? target.entry

  if (node.agentKey) {
    return <PluginSchemaSettingsPage agentKey={node.agentKey} key={node.agentKey} />
  }

  const render = node.render

  if (!render) {
    return <PluginSettingsOverview entries={entries} missing onOpen={onOpen} />
  }

  // Keyed per page so a sub-page switch remounts the plugin's tree (fresh
  // hook state, fresh error boundary) instead of reconciling across pages.
  return (
    <SettingsContent key={`${target.entry.key}:${target.child?.id ?? ''}`}>
      <ContribBoundary id={target.entry.key}>
        <ContribRender render={render} />
      </ContribBoundary>
    </SettingsContent>
  )
}
