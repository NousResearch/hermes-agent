'use client'

import type { CatalogPluginPresence, PluginsManageResult } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { Puzzle } from 'lucide-react'
import { useEffect } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { useConnectionOwner } from '@/components/assistant-ui/use-connection-owner'
import { Button } from '@/components/ui/button'
import { Chip } from '@/components/ui/chip'
import { useI18n } from '@/i18n'
import { type CatalogLookupReply, lookupPluginCatalogEntry } from '@/lib/plugin-catalog'
import { cn } from '@/lib/utils'
import {
  $agentPlugins,
  $agentPluginsByOwner,
  $agentPluginsError,
  $agentPluginsErrorByOwner,
  $agentPluginsLoadingByOwner,
  $agentPluginsProfile,
  $agentPluginsStatus,
  type AgentPluginRow,
  type AgentPluginsStatus,
  loadAgentPlugins,
  pluginInventoryKey
} from '@/store/agent-plugins'
import type { ConnectionOwner } from '@/store/connection-request'
import { requestGatewayForAgent } from '@/store/gateway'
import { notifyPluginCatalogLookupError, openCatalogPluginInstall } from '@/store/plugin-catalog-install'

import { CARD_ACTION_CLASS, CARD_CHIP_CLASS, CARD_LABEL_CLASS } from './card-type'

async function readPresence(owner: ConnectionOwner, names: string[]): Promise<CatalogPluginPresence[]> {
  const result = await requestGatewayForAgent<PluginsManageResult>(
    owner.connectionId,
    owner.profile,
    'plugins.manage',
    {
      action: 'presence',
      names
    }
  )

  return result.presence ?? []
}

async function openInstall(name: string, title: string, owner: ConnectionOwner): Promise<void> {
  const lookup = await lookupPluginCatalogEntry(name, {
    request: (method, params) =>
      requestGatewayForAgent<CatalogLookupReply>(owner.connectionId, owner.profile, method, params)
  })

  if (lookup.ok) {
    // The card stays in the chat and flips the row to Installed when the plugin list reloads.
    openCatalogPluginInstall(lookup.entry, owner, { stayAfterInstall: true, title })
  } else {
    notifyPluginCatalogLookupError(name, lookup.error)
  }
}

function loadOwnerInventory(owner: ConnectionOwner): Promise<void> {
  return loadAgentPlugins(
    (method, params, timeoutMs) => requestGatewayForAgent(owner.connectionId, owner.profile, method, params, timeoutMs),
    owner.profile,
    owner.connectionId
  )
}

function cardInventory(
  owner: ConnectionOwner | null,
  state: {
    snapshots: Record<string, AgentPluginRow[]>
    errors: Record<string, string | null>
    loading: Record<string, boolean>
    shared: AgentPluginRow[]
    sharedError: string | null
    sharedStatus: AgentPluginsStatus
    loadedProfile: string | null | undefined
  }
) {
  const key = owner?.connectionId ? pluginInventoryKey(owner.connectionId, owner.profile) : null

  const rows = key
    ? state.snapshots[key]
    : owner && state.loadedProfile === owner.profile && state.sharedStatus === 'ready'
      ? state.shared
      : undefined

  const error = key ? state.errors[key] : state.sharedStatus === 'error' ? state.sharedError : null
  const pending = key ? rows === undefined || Boolean(state.loading[key]) : Boolean(owner) && rows === undefined
  const installed = new Set((rows ?? []).flatMap(row => (row.catalog_name ? [row.catalog_name, row.name] : [row.name])))

  return { installed, error, pending }
}

const PLUGIN_ICON = (
  <span className="grid size-7 shrink-0 place-items-center rounded-full bg-background text-muted-foreground">
    <Puzzle className="size-4" />
  </span>
)

function PluginChips({
  rows,
  installed,
  owner,
  installedLabel,
  installLabel
}: {
  rows: CatalogPluginPresence[]
  installed: Set<string>
  owner: ConnectionOwner | null
  installedLabel: string
  installLabel: string
}) {
  return (
    <div className="grid grid-cols-2 gap-2 p-1">
      {rows.map(row => {
        const on = installed.has(row.name)

        return (
          <Chip
            className={cn('w-full', CARD_CHIP_CLASS)}
            icon={PLUGIN_ICON}
            key={row.name}
            label={row.title || row.name}
            on={on}
            onToggle={() => {
              if (!on && owner) {
                void openInstall(row.name, row.title || row.name, owner)
              }
            }}
            sub={on ? installedLabel : row.sentence || installLabel}
          />
        )
      })}
    </div>
  )
}

/** The plugins this OS runs, from the card payload; Install opens the reviewed install dialog in place,
 *  Continue answers with the ones installed; it reads Skip until one is. */
export function PluginsCard({
  names,
  onContinue
}: {
  names: string[]
  /** The installed catalog names, and the titles the card showed for them. */
  onContinue: (installed: string[], titles: string[]) => void
}) {
  const { t } = useI18n()
  const copy = t.firstRun
  const storedId = useStore(useSessionView().$storedId)
  const owner = useConnectionOwner(storedId, true)
  const inventories = useStore($agentPluginsByOwner)
  const ownerErrors = useStore($agentPluginsErrorByOwner)
  const ownerLoading = useStore($agentPluginsLoadingByOwner)
  const shared = useStore($agentPlugins)
  const sharedError = useStore($agentPluginsError)
  const sharedStatus = useStore($agentPluginsStatus)
  const loadedProfile = useStore($agentPluginsProfile)
  // The shared plugin list may be empty or another profile's: load the chat's own.
  useEffect(() => {
    if (owner) {
      void loadOwnerInventory(owner)
    }
  }, [owner])

  const presence = useQuery({
    enabled: Boolean(owner) && names.length > 0,
    queryFn: () => readPresence(owner!, names),
    queryKey: ['onboarding', 'plugins.presence', owner?.connectionId, owner?.profile, ...names],
    staleTime: Infinity
  })

  const {
    installed,
    error: inventoryError,
    pending: inventoryPending
  } = cardInventory(owner, {
    snapshots: inventories,
    errors: ownerErrors,
    loading: ownerLoading,
    shared,
    sharedError,
    sharedStatus,
    loadedProfile
  })

  const rows = presence.data ?? []

  return (
    <div className="grid gap-3">
      {presence.isPending && names.length > 0 ? (
        <p className={CARD_LABEL_CLASS} role="status">
          {copy.loading}
        </p>
      ) : presence.isError ? (
        <div role="alert">
          <p className={CARD_LABEL_CLASS}>
            {t.skills.plugins.loadFailed}: {String(presence.error)}
          </p>
          <Button onClick={() => void presence.refetch()} size="sm" variant="outline">
            {t.catalog.retry}
          </Button>
        </div>
      ) : inventoryError ? (
        <div role="alert">
          <p className={CARD_LABEL_CLASS}>
            {t.skills.plugins.loadFailed}: {inventoryError}
          </p>
          {owner && (
            <Button onClick={() => void loadOwnerInventory(owner)} size="sm" variant="outline">
              {t.catalog.retry}
            </Button>
          )}
        </div>
      ) : inventoryPending ? (
        <p className={CARD_LABEL_CLASS} role="status">
          {copy.loading}
        </p>
      ) : rows.length === 0 ? (
        <p className={CARD_LABEL_CLASS}>{copy.noPlugins}</p>
      ) : (
        <PluginChips
          installed={installed}
          installedLabel={copy.installed}
          installLabel={copy.install}
          owner={owner}
          rows={rows}
        />
      )}
      <div>
        <Button
          className={CARD_ACTION_CLASS}
          onClick={() => {
            const done = names.filter(name => installed.has(name))
            const title = (name: string) => rows.find(row => row.name === name)?.title || name

            onContinue(done, done.map(title))
          }}
          size="sm"
        >
          {names.some(name => installed.has(name)) ? t.common.continue : t.common.skip}
        </Button>
      </div>
    </div>
  )
}
