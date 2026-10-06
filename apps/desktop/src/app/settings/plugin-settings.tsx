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
  PLUGIN_SETTINGS_PROFILE_PARAM,
  pluginSettingsEntries,
  type PluginSettingsEntry,
  pluginSettingsNavId,
  type PluginSettingsNode,
  type PluginSettingsRoute,
  pluginSettingsRouteFrom,
  pluginSettingsRouteOf,
  type PluginSettingsTarget,
  resolvePluginSettingsTarget,
  SETTINGS_PLUGINS_AREA
} from '@/contrib/settings-pages'
import { useI18n } from '@/i18n'
import type { IconComponent } from '@/lib/icons'
import {
  $agentPluginBusy,
  $agentPlugins,
  $agentPluginsProfile,
  $agentPluginsStatus,
  isDesktopRelevantPlugin,
  loadAgentPlugins,
  saveAgentPluginSettings
} from '@/store/agent-plugins'
import { notify, notifyError } from '@/store/notifications'
import { $settingsRequestProfile, setSettingsScope } from '@/store/settings-scope'

import { desktopPackageName } from '../capabilities/plugins/plugin-packages'
import type { OverlayNavLink } from '../overlays/overlay-split-layout'
import { CAPABILITIES_ROUTE } from '../routes'

import { PluginSettingsForm } from './plugin-settings-form'
import { EmptyState, ListRowSkeleton, SectionHeading, SettingsContent } from './primitives'
import { SettingsProfileScope } from './profile-scope'

/** Opens Settings ▸ Plugins at a route; `null` = the overview. */
export type OpenPluginSettings = (route: null | PluginSettingsRoute) => void

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
  'ppage',
  'agent',
  PLUGIN_SETTINGS_PROFILE_PARAM
] as const

/** The profile the Settings plugin pages edit, as `plugins.manage` and the
 *  agent-plugin store key it (`null` = the backend's launch profile). */
const ownerKey = (scope: string | undefined): null | string => scope ?? null

/** The agent plugin rows loaded for `owner`, or none while the shared list
 *  still holds another profile's rows (a switch in flight, or Capabilities
 *  loaded a different profile). */
function useOwnedAgentPlugins(owner: null | string) {
  const rows = useStore($agentPlugins)
  const loadedFor = useStore($agentPluginsProfile)
  const status = useStore($agentPluginsStatus)
  const owned = loadedFor === owner

  return { owned, rows: owned ? rows : [], settled: status === 'error' || (status === 'ready' && owned), status }
}

/** Settings ▸ Plugins routing: `?tab=plugins&plugin=<entry key | plugin id>
 *  &ppage=<sub-page>` for desktop pages, `?tab=plugins&agent=<key>` for an
 *  agent plugin's schema form. A `&profile=` hand-off (the Capabilities gear)
 *  is adopted into the Settings scope, then dropped from the URL. `active` =
 *  the Plugins view is showing. */
export function usePluginSettingsRoute(active: boolean) {
  const navigate = useNavigate()
  const { hash, pathname, search } = useLocation()
  const entries = usePluginSettingsEntries()
  const { settled } = useOwnedAgentPlugins(ownerKey(useStore($settingsRequestProfile)))
  const params = new URLSearchParams(search)
  const route = pluginSettingsRouteFrom(params)
  const handoff = active ? params.get(PLUGIN_SETTINGS_PROFILE_PARAM) : null
  // Until the hand-off is adopted, the current scope is not the one the link
  // asked for: resolve nothing, so no page renders against the wrong profile.
  const target = active && !handoff ? resolvePluginSettingsTarget(entries, route) : null

  useEffect(() => {
    if (!handoff) {
      return
    }

    setSettingsScope(handoff)

    const next = new URLSearchParams(search)

    next.delete(PLUGIN_SETTINGS_PROFILE_PARAM)
    navigate({ hash, pathname, search: `?${next}` }, { replace: true })
  }, [handoff, hash, navigate, pathname, search])

  const open = useCallback<OpenPluginSettings>(
    next => {
      const query = new URLSearchParams(search)

      for (const key of PAGE_SCOPED_PARAMS) {
        query.delete(key)
      }

      query.set('tab', 'plugins')

      if (next && 'agent' in next) {
        query.set('agent', next.agent)
      } else if (next) {
        query.set('plugin', next.plugin)

        if (next.page) {
          query.set('ppage', next.page)
        }
      }

      navigate({ hash, pathname, search: `?${query}` }, { replace: true })
    },
    [hash, navigate, pathname, search]
  )

  // A deep link to an agent plugin's page can't resolve until the scope's
  // list loads; only call it missing once that load has settled.
  const waiting = Boolean(route) && !target && (Boolean(handoff) || !settled)

  return { entries, missing: Boolean(route) && !target && !waiting, open, pending: active && waiting, target }
}

/** Every plugin settings page, live: Desktop plugins' registered pages plus an
 *  automatic page for each agent plugin that declares a `config_schema`
 *  (scoped to the Settings profile selector). */
export function usePluginSettingsEntries(): PluginSettingsEntry[] {
  const { t } = useI18n()
  const { requestGateway } = useGatewayRequest()
  const contributions = useContributions(SETTINGS_PLUGINS_AREA)
  const records = useStore($pluginRecords)
  const owner = ownerKey(useStore($settingsRequestProfile))
  const { owned, rows, status } = useOwnedAgentPlugins(owner)

  // Cheap backend disk scan; the same loader Capabilities ▸ Plugins uses.
  useEffect(() => {
    void loadAgentPlugins(requestGateway, owner)
  }, [requestGateway, owner])

  // The list is shared: if another surface replaced it with a different
  // profile's rows, fetch this scope's again rather than show none.
  useEffect(() => {
    if (status === 'ready' && !owned) {
      void loadAgentPlugins(requestGateway, owner)
    }
  }, [owned, owner, requestGateway, status])

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

const isActiveNode = (target: null | PluginSettingsTarget, entry: PluginSettingsEntry, child: PluginSettingsNode) =>
  target?.entry.uid === entry.uid &&
  (child.agentKey
    ? target.child?.agentKey === child.agentKey
    : !target.child?.agentKey && target.child?.id === child.id)

/** The Settings rail's children under "Plugins": one row per entry, its
 *  sub-pages folded beneath it while it is selected. */
export function pluginSettingsNavChildren(
  entries: readonly PluginSettingsEntry[],
  target: null | PluginSettingsTarget,
  open: OpenPluginSettings
): OverlayNavLink[] {
  return entries.map(entry => {
    const active = target?.entry.uid === entry.uid

    return {
      active,
      children: entry.children.map(child => ({
        active: isActiveNode(target, entry, child),
        icon: iconFor(child.agentKey ? 'settings-gear' : 'list-flat'),
        id: pluginSettingsNavId(entry, child),
        label: child.title,
        onSelect: () => open(pluginSettingsRouteOf(entry, child))
      })),
      icon: entryIcon(entry),
      id: pluginSettingsNavId(entry),
      label: entry.title,
      onSelect: () => open(pluginSettingsRouteOf(entry))
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
      {/* The agent half of this list is per profile: keep the selector here
          too, or a profile without the plugin strands the user (no way back). */}
      <SettingsProfileScope className="mb-5" />
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
        <p
          className="mb-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-secondary)"
          role="status"
        >
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
                data-testid={`plugin-settings-row-${entry.uid}`}
                key={entry.uid}
                onClick={() => onOpen(pluginSettingsRouteOf(entry))}
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

function PluginSettingsLoading() {
  return (
    <SettingsContent>
      <SettingsProfileScope className="mb-5" />
      <div className="grid gap-1">
        <ListRowSkeleton />
        <ListRowSkeleton />
        <ListRowSkeleton />
      </div>
    </SettingsContent>
  )
}

/** Automatic page for an agent plugin's manifest `config_schema`: the schema
 *  form, saved through `plugins.manage settings` (+ `.env` for secrets).
 *
 *  Bound to ONE profile for its whole life: the pane keys it by
 *  (profile, plugin) so a scope switch remounts it (the draft goes with the
 *  old profile), it renders only rows loaded for that profile, and its save
 *  writes that profile or nothing. */
function PluginSchemaSettingsPage({ agentKey, owner }: { agentKey: string; owner: null | string }) {
  const { t } = useI18n()
  const p = t.skills.plugins
  const { requestGateway } = useGatewayRequest()
  const { rows } = useOwnedAgentPlugins(owner)
  const row = rows.find(candidate => candidate.key === agentKey)
  const busy = useStore($agentPluginBusy) === agentKey

  if (!row?.settings_schema?.length) {
    return <PluginSettingsLoading />
  }

  return (
    <SettingsContent>
      <SettingsProfileScope className="mb-5" />
      <PluginSettingsForm
        disabled={busy}
        fields={row.settings_schema}
        idPrefix={`plugin-settings-${agentKey}`}
        intro={row.description ? <p className={BLURB_CLASS}>{row.description}</p> : undefined}
        onSave={async changes => {
          // Belt and braces: the key remount already discards a draft when
          // the scope moves, but a submit must never write a profile other
          // than the one these values were loaded from.
          if (ownerKey($settingsRequestProfile.get()) !== owner || $agentPluginsProfile.get() !== owner) {
            notifyError(null, p.settingsForm.saveFailed(row.name))

            return false
          }

          const ok = await saveAgentPluginSettings(requestGateway, {
            failMessage: p.settingsForm.saveFailed(row.name),
            key: agentKey,
            profile: owner,
            secrets: changes.secrets,
            values: changes.values,
            writeSecret: (env, value) => setEnvVar(env, value, owner ?? undefined)
          })

          if (ok) {
            notify({ kind: 'success', message: p.settingsForm.saved(row.name) })
          }

          return ok
        }}
        title={row.name}
      />
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
  pending = false,
  target
}: {
  entries: readonly PluginSettingsEntry[]
  /** A page was requested but matches none (disabled/uninstalled). */
  missing?: boolean
  onOpen: OpenPluginSettings
  /** A page was requested and the scope's plugin list is still loading. */
  pending?: boolean
  target: null | PluginSettingsTarget
}) {
  const owner = ownerKey(useStore($settingsRequestProfile))

  if (!target) {
    return pending ? (
      <PluginSettingsLoading />
    ) : (
      <PluginSettingsOverview entries={entries} missing={missing} onOpen={onOpen} />
    )
  }

  const node = target.child ?? target.entry

  if (node.agentKey) {
    // Keyed by (profile, plugin): a scope switch remounts the form, so a
    // draft never outlives the profile it was typed against.
    return (
      <PluginSchemaSettingsPage agentKey={node.agentKey} key={JSON.stringify([owner, node.agentKey])} owner={owner} />
    )
  }

  const render = node.render

  if (!render) {
    return <PluginSettingsOverview entries={entries} missing onOpen={onOpen} />
  }

  // Keyed per page so a sub-page switch remounts the plugin's tree (fresh
  // hook state, fresh error boundary) instead of reconciling across pages.
  return (
    <SettingsContent key={JSON.stringify([target.entry.uid, target.child?.id ?? null])}>
      <ContribBoundary id={target.entry.key}>
        <ContribRender render={render} />
      </ContribBoundary>
    </SettingsContent>
  )
}
