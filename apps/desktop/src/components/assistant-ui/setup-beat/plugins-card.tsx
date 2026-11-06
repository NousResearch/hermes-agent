'use client'

import type { CatalogPluginPresence, PluginsManageResult } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { Puzzle } from 'lucide-react'
import { useEffect } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { useConnectionOwner } from '@/components/assistant-ui/use-connection-owner'
import { Button } from '@/components/ui/button'
import { Chip } from '@/components/ui/chip'
import { useI18n } from '@/i18n'
import { lookupPluginCatalogEntry } from '@/lib/plugin-catalog'
import { cn } from '@/lib/utils'
import { $agentPlugins, $agentPluginsProfile, loadAgentPlugins } from '@/store/agent-plugins'
import type { ConnectionOwner } from '@/store/connection-request'
import { requestGatewayForAgent } from '@/store/gateway'
import { openCatalogPluginInstall } from '@/store/plugin-catalog-install'

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

async function openInstall(name: string, title: string, profile: string): Promise<void> {
  const lookup = await lookupPluginCatalogEntry(name)

  if (lookup.ok) {
    // The card stays in the chat and flips the row to Installed when the plugin list reloads.
    openCatalogPluginInstall(lookup.entry, profile, { stayAfterInstall: true, title })
  }
}

const PLUGIN_ICON = (
  <span className="grid size-7 shrink-0 place-items-center rounded-full bg-background text-muted-foreground">
    <Puzzle className="size-4" />
  </span>
)

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
  const plugins = useStore($agentPlugins)
  const pluginsProfile = useStore($agentPluginsProfile)
  const { requestGateway } = useGatewayRequest()

  // The shared plugin list may be empty or another profile's: load the chat's own.
  useEffect(() => {
    if (owner) {
      void loadAgentPlugins(requestGateway, owner.profile)
    }
  }, [owner, requestGateway])

  const presence = useQuery({
    enabled: Boolean(owner) && names.length > 0,
    queryFn: () => readPresence(owner!, names),
    queryKey: ['onboarding', 'plugins.presence', owner?.connectionId, owner?.profile, ...names],
    staleTime: Infinity
  })

  const ownRows = owner && pluginsProfile === owner.profile ? plugins : []
  const installed = new Set(ownRows.flatMap(row => (row.catalog_name ? [row.catalog_name, row.name] : [row.name])))

  const rows = presence.data ?? []

  return (
    <div className="grid gap-3">
      {presence.isPending && names.length > 0 ? (
        <p className={CARD_LABEL_CLASS} role="status">
          {copy.loading}
        </p>
      ) : rows.length === 0 ? (
        <p className={CARD_LABEL_CLASS}>{copy.noPlugins}</p>
      ) : (
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
                    void openInstall(row.name, row.title || row.name, owner.profile)
                  }
                }}
                sub={on ? copy.installed : row.sentence || copy.install}
              />
            )
          })}
        </div>
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
