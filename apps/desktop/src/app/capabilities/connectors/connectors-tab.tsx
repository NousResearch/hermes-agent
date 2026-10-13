import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { useLocation, useNavigate } from 'react-router'

import { Button } from '@/components/ui/button'
import type { HermesGateway, ProfileScope } from '@/hermes'
import { useI18n } from '@/i18n'
import { $freeTierStatus } from '@/store/free-tier'
import { notifyError } from '@/store/notifications'

import { installBundledEntry } from '../mcp/install-catalog-entry'
import { useMcpServers } from '../mcp/use-mcp-servers'

import { AddServerDialog } from './add-dialog'
import { ConnectorsDirectory } from './connectors-directory'
import { $abandonedConnects } from './data/account-operations'
import { claimRouteLink } from './data/deep-link'
import { useHostedActions } from './data/hosted-actions'
import { joinBundledEntries, joinLocalServers } from './data/join'
import { invalidateConnectors } from './data/keys'
import { useRenameAccount } from './data/mutations'
import { seedLocalServers, startConnectorPersistence, storeLocalServers } from './data/persist'
import { prefetchConnectorTools, usePrefetchConnectedTools } from './data/prefetch'
import { useHostedConnectors, usePluginServers } from './data/queries'
import {
  cardKey,
  deriveCards,
  EMPTY_CONNECTORS_FILTER,
  forgetAbandoned,
  hostedCardKey,
  localServerName
} from './derive'
import { DisconnectConfirm, type Disconnecting } from './disconnect-confirm'
import { HostedConnectorDialog, type PendingRename } from './hosted-dialog'
import { HostedNotice } from './hosted-notice'
import { LocalConnectorDialog } from './local-dialog'
import { RemoveServerConfirm } from './local-slots'
import { openToolsList, resetOpenedTools } from './tools-summary'
import type { AccountRow, ConnectorCardModel, ConnectorsFilter } from './types'

const toolsListKey = (card: ConnectorCardModel) => (card.residency === 'local' ? localServerName(card) : card.slug)

export interface ConnectorsTabProps {
  gateway: HermesGateway | null
  profile: ProfileScope
}

export function ConnectorsTab({ gateway, profile }: ConnectorsTabProps) {
  const { t } = useI18n()
  const copy = t.connectorsPage

  const hosted = useHostedConnectors(profile)
  const mcp = useMcpServers({ gateway, profile })
  const abandoned = useStore($abandonedConnects)
  const freeTier = useStore($freeTierStatus)

  const [filter, setFilter] = useState<ConnectorsFilter>(EMPTY_CONNECTORS_FILTER)
  const [openKey, setOpenKey] = useState<null | string>(null)

  const hostedActions = useHostedActions(profile, hosted.accounts, {
    onStarted: card => setOpenKey(cardKey(card))
  })

  const { connectAccount, connector, remover, retry, startConnect, switcher, write } = hostedActions
  const renamer = useRenameAccount(profile)

  const [addOpen, setAddOpen] = useState(false)
  const [removeServer, setRemoveServer] = useState<null | ConnectorCardModel>(null)
  const [disconnecting, setDisconnecting] = useState<Disconnecting | null>(null)
  const [pendingRename, setPendingRename] = useState<null | PendingRename>(null)
  const renameOpened = useCallback(() => setPendingRename(null), [])
  const [installing, setInstalling] = useState<null | string>(null)

  const local = useMemo(
    () =>
      joinLocalServers({
        catalog: mcp.catalog,
        servers: mcp.servers,
        status: mcp.statuses,
        toolCounts: mcp.toolCounts,
        usage: mcp.usageByServer
      }),
    [mcp.catalog, mcp.servers, mcp.statuses, mcp.toolCounts, mcp.usageByServer]
  )

  const bundled = useMemo(() => joinBundledEntries(mcp.availableCatalog), [mcp.availableCatalog])

  const lastKnownServers = useMemo(() => seedLocalServers(profile), [profile])
  const pluginServers = usePluginServers(profile)

  const servers = useMemo(
    () => [...(mcp.configLoading ? lastKnownServers : local), ...pluginServers],
    [lastKnownServers, local, mcp.configLoading, pluginServers]
  )

  useEffect(() => {
    if (!mcp.configLoading) {
      storeLocalServers(profile, local)
    }
  }, [local, mcp.configLoading, profile])

  const cards = useMemo(
    () =>
      deriveCards({
        bundled,
        hosted: forgetAbandoned(hosted.rows, new Set(abandoned)),
        local: servers,
        titles: hosted.titles
      }),
    [abandoned, bundled, hosted.rows, hosted.titles, servers]
  )

  useEffect(startConnectorPersistence, [])

  useEffect(() => {
    resetOpenedTools()
    $abandonedConnects.set([])
  }, [profile])

  usePrefetchConnectedTools(profile, cards)

  const openCard = useMemo(() => cards.find(card => cardKey(card) === openKey) ?? null, [cards, openKey])

  useOpenFromRoute(cards, profile, setOpenKey, setPendingRename)

  const bundledEntry = (card: ConnectorCardModel) =>
    mcp.availableCatalog.find(candidate => candidate.name === card.ways.local?.entryName) ?? null

  const startInstall = async (card: ConnectorCardModel, env: Record<string, string>) => {
    const entry = bundledEntry(card)

    if (!entry) {
      return
    }

    setInstalling(cardKey(card))

    try {
      await installBundledEntry(entry, env, profile)
      await mcp.onCatalogInstalled()
      setOpenKey(cardKey(card))
    } catch (error) {
      notifyError(error, t.settings.mcp.catalogInstallFailed(card.name))
    } finally {
      setInstalling(null)
    }
  }

  const runVerb = (card: ConnectorCardModel) => {
    if (hostedActions.runHostedVerb(card)) {
      return
    }

    switch (card.verb) {
      case 'authenticate':
        void mcp.authenticate(localServerName(card))

        return

      case 'install':
        if (card.ways.local?.entryName && card.ways.local.needsEnv !== true) {
          void startInstall(card, {})

          return
        }

        break

      default:
        break
    }

    setOpenKey(cardKey(card))
  }

  const busySlug = hostedActions.busySlug
  const busyKey = installing ?? (busySlug === null ? null : hostedCardKey(busySlug))

  return (
    <div className="flex h-full min-h-0 flex-col gap-3 px-4 pb-2">
      <ConnectorsDirectory
        addYourOwn={
          <Button disabled={mcp.profilePending} onClick={() => setAddOpen(true)} size="xs" variant="outline">
            {copy.add.action}
          </Button>
        }
        busyKey={busyKey}
        cards={cards}
        filter={filter}
        hostedFailed={hosted.phase === 'failed'}
        loading={(hosted.phase === 'loading' || mcp.configLoading || mcp.catalogLoading) && cards.length === 0}
        notices={<HostedNotice hasGuest={freeTier?.has_guest === true} phase={hosted.phase} />}
        onFilterChange={setFilter}
        onOpen={card => setOpenKey(cardKey(card))}
        onPrefetch={card => prefetchConnectorTools(profile, card)}
        onRetryHosted={hosted.refetch}
        onServerToggle={(card, next) => {
          if (card.plugin === undefined) {
            void mcp.setServerEnabled(localServerName(card), next)
          }
        }}
        onVerb={runVerb}
        selectedKey={openKey}
      />

      {openCard && openCard.ways.hosted === null ? (
        <LocalConnectorDialog
          card={openCard}
          controller={mcp}
          installFields={bundledEntry(openCard)?.required_env}
          installing={installing === cardKey(openCard)}
          onClose={() => setOpenKey(null)}
          onConnect={() => void startConnect(openCard, false)}
          onInstall={env => void startInstall(openCard, env)}
          onReconnect={() => void startConnect(openCard, true)}
          onRemoveServer={() => setRemoveServer(openCard)}
          profile={profile}
        />
      ) : null}

      {openCard?.ways.hosted ? (
        <HostedConnectorDialog
          accountActions={{
            add: async alias => {
              const outcome = await connectAccount(openCard, false, { alias })

              return outcome.ok ? { ok: true } : outcome
            },
            pendingRename,
            reconnect: (account: AccountRow) => void startConnect(openCard, true, account),
            reconnecting: connector.pending === openCard.slug,
            retry: operation => retry(openCard, operation),
            remove: (account: AccountRow) => setDisconnecting({ account, card: openCard }),
            rename: renamer.rename,
            renameOpened
          }}
          card={openCard}
          controller={mcp}
          hosted={hosted}
          installFields={bundledEntry(openCard)?.required_env}
          installing={installing === cardKey(openCard)}
          onClose={() => {
            setOpenKey(null)
            setPendingRename(null)
          }}
          onConnect={() => void startConnect(openCard, false)}
          onDisconnect={() => setDisconnecting({ card: openCard })}
          onGiveUp={opId => void write(connector.giveUp(opId))}
          onInstall={env => void startInstall(openCard, env)}
          onReconnect={() => void startConnect(openCard, true)}
          onRemoveServer={() => setRemoveServer(openCard)}
          onToggleForMe={next => void write(switcher.setEnabled(openCard.slug, next))}
          onVerb={() => runVerb(openCard)}
          profile={profile}
          togglePending={switcher.pending === openCard.slug}
        />
      ) : null}

      <AddServerDialog controller={mcp} onOpenChange={setAddOpen} open={addOpen} profile={profile} />

      <RemoveServerConfirm
        card={removeServer}
        controller={mcp}
        onClose={() => setRemoveServer(null)}
        onRemoved={() => setOpenKey(null)}
      />

      <DisconnectConfirm
        accounts={hosted.accounts}
        disconnect={remover.disconnect}
        onClose={() => setDisconnecting(null)}
        onDisconnected={() => setOpenKey(null)}
        target={disconnecting}
      />
    </div>
  )
}

function useOpenFromRoute(
  cards: readonly ConnectorCardModel[],
  scope: ProfileScope,
  open: (key: string) => void,
  rename: (target: PendingRename) => void
): void {
  const { hash, key: locationKey, pathname, search } = useLocation()
  const navigate = useNavigate()

  useEffect(() => {
    const params = new URLSearchParams(search)
    const server = params.get('server')
    const slug = params.get('connector')

    if (!server && !slug) {
      return
    }

    const target = server
      ? cards.find(card => card.residency === 'local' && card.slug === server)
      : cards.find(card => card.slug === slug)

    // A tab for another profile leaves the link to the page once it takes the link's scope, and of two
    // mounted pages (one docked) only one acts on it.
    if (!target || !claimRouteLink(locationKey, params, scope)) {
      return
    }

    if (params.get('tool')) {
      openToolsList(toolsListKey(target))
    }

    const connectionId = params.get('rename')

    if (connectionId) {
      // The cached list can predate an account connected since; the editor decides on a fresh read.
      invalidateConnectors(scope, 'accounts')
      rename({ connectionId, slug: target.slug })
    }

    open(cardKey(target))
    params.delete('server')
    params.delete('connector')
    params.delete('tool')
    params.delete('rename')
    params.delete('profile')
    params.delete('connection')

    const query = params.toString()
    navigate({ hash, pathname, search: query ? `?${query}` : '' }, { replace: true })
  }, [cards, hash, locationKey, navigate, open, pathname, rename, scope, search])
}
