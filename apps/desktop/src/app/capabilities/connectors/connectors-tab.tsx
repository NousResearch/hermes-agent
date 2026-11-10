import './connectors.css'

import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useState } from 'react'
import { useLocation, useNavigate } from 'react-router'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import type { HermesGateway, ProfileScope } from '@/hermes'
import { useI18n } from '@/i18n'
import { FREE_LOCK_MESSAGE, freeConnectorAllowed } from '../free-tier'
import { $brandSession, capabilitiesUnlocked } from '@/store/brand-session'
import { isCurrentBrandMcp } from '@/app/brand/sync-brand-mcp'
import { $freeTierStatus } from '@/store/free-tier'
import { openFreeTierSignIn } from '@/store/free-tier-sign-in'
import { notify, notifyError, readableError } from '@/store/notifications'

import { installBundledEntry } from '../mcp/install-catalog-entry'
import { useMcpServers } from '../mcp/use-mcp-servers'

import { AddServerDialog } from './add-dialog'
import { DiscordPanel } from './discord-panel'
import { ConnectorsDirectory } from './connectors-directory'
import { $abandonedConnects, $accountOperations, abandonConnect, accountOperationFor } from './data/account-operations'
import { joinBundledEntries, joinLocalServers, pickAccount } from './data/join'
import { useConnectConnector, useConnectorSwitch, useDisconnectAccount } from './data/mutations'
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
import { HostedConnectorDialog } from './hosted-dialog'
import { LocalConnectorDialog } from './local-dialog'
import { RemoveServerConfirm } from './local-slots'
import { openToolsList, resetOpenedTools } from './tools-summary'
import type { ConnectorCardModel, ConnectorsFilter, HostedPhase } from './types'

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
  const operations = useStore($accountOperations)
  const abandoned = useStore($abandonedConnects)
  const freeTier = useStore($freeTierStatus)

  const connector = useConnectConnector(profile)
  const switcher = useConnectorSwitch(profile)
  const remover = useDisconnectAccount(profile)

  const [filter, setFilter] = useState<ConnectorsFilter>(EMPTY_CONNECTORS_FILTER)
  const [openKey, setOpenKey] = useState<null | string>(null)
  const [addOpen, setAddOpen] = useState(false)
  const [removeServer, setRemoveServer] = useState<null | ConnectorCardModel>(null)
  const [disconnecting, setDisconnecting] = useState<null | ConnectorCardModel>(null)
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

  const brand = useStore($brandSession)
  const servers = useMemo(() => {
    const appId = brand.signedIn ? brand.activeAppId : ''

    return [...(mcp.configLoading ? lastKnownServers : local), ...pluginServers].filter(server =>
      isCurrentBrandMcp(server.name, appId)
    )
  }, [brand.activeAppId, brand.signedIn, lastKnownServers, local, mcp.configLoading, pluginServers])

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

  useOpenFromRoute(cards, setOpenKey)

  const write = async (pending: Promise<{ error?: unknown; ok: boolean }>) => {
    const outcome = await pending

    if (!outcome.ok) {
      notifyError(outcome.error, copy.page.writeFailed)
    }
  }

  const startConnect = async (card: ConnectorCardModel, reconnect: boolean) => {
    const outcome = await connector.connect(card.slug, { reconnect })

    if (!outcome.ok) {
      notifyError(outcome.error, t.connectors.connectErrorFor(card.name))

      return
    }

    const url = outcome.operation.targets.find(target => target.connectUrl)?.connectUrl

    if (url) {
      void window.hermesDesktop?.openExternal?.(url)
    }

    setOpenKey(cardKey(card))
  }

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
    const open = accountOperationFor(operations, card.slug)

    switch (card.verb) {
      case 'authenticate':
        void mcp.authenticate(localServerName(card))

        return

      case 'connect':
        void startConnect(card, false)

        return

      case 'install':
        if (card.ways.local?.entryName && card.ways.local.needsEnv !== true) {
          void startInstall(card, {})

          return
        }

        break

      case 'reconnect':
        void startConnect(card, true)

        return

      case 'tryAgain':
        if (card.residency === 'local' && card.ways.local?.installed === true) {
          void mcp.runProbe(localServerName(card))

          return
        }

        void startConnect(card, true)

        return
      case 'stopWaiting': {
        if (open) {
          void write(connector.giveUp(open.opId))

          return
        }

        abandonConnect([card.slug])

        const pending = pickAccount(hosted.accounts, card.slug)

        if (pending) {
          void write(remover.disconnect(pending.connection_id))
        }

        return
      }

      case 'turnBackOn':
        void write(switcher.setEnabled(card.slug, true))

        return

      default:
        break
    }

    setOpenKey(cardKey(card))
  }

  const busySlug = switcher.pending ?? connector.pending
  const busyKey = installing ?? (busySlug === null ? null : hostedCardKey(busySlug))

  const addConnectorButton = (
    <Button
      className="h-auto gap-2 rounded-full border-0 bg-gradient-to-r from-[#4361ee] to-[#c04de2] px-[18px] py-[11px] text-[14px] font-bold text-white shadow-none hover:brightness-105 hover:text-white"
      data-connectors-cta
      disabled={mcp.profilePending}
      onClick={() => setAddOpen(true)}
      size="sm"
      type="button"
      variant="ghost"
    >
      <Codicon name="add" size="1rem" />
      {copy.page.addConnector}
    </Button>
  )

  const addYourOwnButton = (
    <Button
      className="h-auto rounded-full border border-[#e2e8f0] bg-white px-3.5 py-2 text-[13px] font-semibold text-[#0f172a] shadow-none hover:bg-[#f8fafc]"
      disabled={mcp.profilePending}
      onClick={() => setAddOpen(true)}
      size="xs"
      type="button"
      variant="outline"
    >
      {copy.add.action}
    </Button>
  )

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden" data-connectors-page>
      <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain">
        <header className="flex items-start justify-between gap-4 px-7 pb-4 pt-5">
          <div className="min-w-0 space-y-1.5">
            <h1 className="text-[40px] font-extrabold leading-none tracking-tight text-[#0f172a] dark:text-foreground">
              {copy.title}
            </h1>
            <p className="max-w-2xl text-[15px] text-[#475569] dark:text-(--ui-text-tertiary)">
              {copy.page.subtitle}
            </p>
          </div>
          {addConnectorButton}
        </header>

        <div className="flex flex-col gap-4 px-7 pb-5">
          <DiscordPanel profile={profile} />
          <ConnectorsDirectory
          addConnector={addConnectorButton}
          addYourOwn={addYourOwnButton}
          busyKey={busyKey}
          cards={cards}
          filter={filter}
          hostedFailed={hosted.phase === 'failed'}
          isLocked={card => !freeConnectorAllowed(card.slug, capabilitiesUnlocked(brand))}
          lockedLabel={FREE_LOCK_MESSAGE}
          loading={(hosted.phase === 'loading' || mcp.configLoading || mcp.catalogLoading) && cards.length === 0}
          notices={<HostedNotice hasGuest={freeTier?.has_guest === true} phase={hosted.phase} />}
          onFilterChange={setFilter}
          onOpen={card => {
            if (!freeConnectorAllowed(card.slug, capabilitiesUnlocked(brand))) {
              notify({ kind: 'info', title: 'Locked', message: FREE_LOCK_MESSAGE })
              return
            }

            setOpenKey(cardKey(card))
          }}
          onPrefetch={card => {
            if (card.ways.hosted) {
              prefetchConnectorTools(profile, card.slug)
            }
          }}
          onRetryHosted={hosted.refetch}
          onServerToggle={(card, next) => {
            const unlocked = capabilitiesUnlocked($brandSession.get())

            if (next && !freeConnectorAllowed(card.slug, unlocked)) {
              notify({ kind: 'info', title: 'Locked', message: FREE_LOCK_MESSAGE })
              return
            }

            if (card.plugin === undefined) {
              void mcp.setServerEnabled(localServerName(card), next)
            }
          }}
          onVerb={runVerb}
          selectedKey={openKey}
        />
        </div>
      </div>

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
          card={openCard}
          controller={mcp}
          hosted={hosted}
          installFields={bundledEntry(openCard)?.required_env}
          installing={installing === cardKey(openCard)}
          onClose={() => setOpenKey(null)}
          onConnect={() => void startConnect(openCard, false)}
          onDisconnect={() => setDisconnecting(openCard)}
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

      <ConfirmDialog
        confirmLabel={copy.dialog.disconnect}
        description={copy.dialog.disconnectBody}
        destructive
        onClose={() => setDisconnecting(null)}
        onConfirm={async () => {
          const card = disconnecting
          const account = card ? pickAccount(hosted.accounts, card.slug) : null

          if (!account) {
            throw new Error(copy.page.disconnectNoAccount)
          }

          const outcome = await remover.disconnect(account.connection_id)

          if (!outcome.ok) {
            throw new Error(
              outcome.error.reason === 'ACCOUNTS_UNAVAILABLE'
                ? copy.page.disconnectRefused
                : readableError(outcome.error, copy.page.writeFailed).message
            )
          }

          setOpenKey(null)
        }}
        open={disconnecting !== null}
        title={copy.dialog.disconnectTitle(disconnecting?.name ?? '')}
      />
    </div>
  )
}

function HostedNotice({ hasGuest, phase }: { hasGuest: boolean; phase: HostedPhase }) {
  const { t } = useI18n()
  const copy = t.connectorsPage.page

  if (phase === 'signedOut') {
    return (
      <div className="flex shrink-0 items-center gap-2 px-3.5 py-2.5 text-[13px] font-medium" data-connectors-notice>
        <span>{copy.signInLine}</span>
        <Button onClick={() => openFreeTierSignIn()} size="xs" variant="text">
          {copy.signIn}
        </Button>
      </div>
    )
  }

  if (phase === 'unavailable') {
    return (
      <div className="shrink-0 px-3.5 py-2.5 text-[13px] font-medium" data-connectors-notice>
        {copy.managedUnavailable}
      </div>
    )
  }

  return hasGuest ? (
    <div className="shrink-0 px-3.5 py-2.5 text-[13px] font-medium" data-connectors-notice>
      {copy.freeTierNote}
    </div>
  ) : null
}

function useOpenFromRoute(cards: readonly ConnectorCardModel[], open: (key: string) => void): void {
  const { hash, pathname, search } = useLocation()
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

    if (!target) {
      return
    }

    if (params.get('tool')) {
      openToolsList(toolsListKey(target))
    }

    open(cardKey(target))
    params.delete('server')
    params.delete('connector')
    params.delete('tool')

    const query = params.toString()
    navigate({ hash, pathname, search: query ? `?${query}` : '' }, { replace: true })
  }, [cards, hash, navigate, open, pathname, search])
}
