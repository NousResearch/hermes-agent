// The Connectors tab: one page for every app Hermes can use, wherever it runs.
//
// This file is the only place the three halves meet. `data/` knows the wire,
// `derive.ts` knows the shapes, `capabilities/mcp/` knows the servers on this
// Mac; the container feeds one into the other and fills the directory's and the
// dialog's slots. It holds no state that either of those layers could hold —
// only what is open, what is being confirmed, and what the filter row says.

import { compactNumber } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useState } from 'react'
import { useLocation, useNavigate } from 'react-router'

import { Button } from '@/components/ui/button'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import type { HermesGateway, ProfileScope } from '@/hermes'
import { useI18n } from '@/i18n'
import { openExternalLink } from '@/lib/external-link'
import { openFreeTierSignIn } from '@/store/free-tier-sign-in'
import { notifyError, readableError } from '@/store/notifications'

import { McpJsonEditor } from '../mcp/mcp-editor'
import { statusOf } from '../mcp/mcp-status'
import { useMcpServers } from '../mcp/use-mcp-servers'

import { ConnectElement } from './connect-element'
import { ConnectorDialog } from './connector-dialog'
import { ConnectorsDirectory } from './connectors-directory'
import { $accountOperations, type AccountOperation, accountOperationFor } from './data/account-operations'
import { joinLocalServers, pickAccount } from './data/join'
import { useConnectConnector, useConnectorSwitch, useDisconnectAccount } from './data/mutations'
import { useConnectorsAdminUrl } from './data/portal'
import { useConnectorTools, useHostedConnectors } from './data/queries'
import { deriveCards } from './derive'
import { ConnectorDialogMenu } from './dialog-menu'
import { LocalAdvanced, McpDocumentDialog, RemoveServerConfirm } from './local-slots'
import { HostedToolsPanel, LocalToolsPanel, orgDisabledCount } from './tools-panel'
import type { ConnectorCardModel, ConnectorsFilter, LocalServerStatus } from './types'

const EMPTY_FILTER: ConnectorsFilter = { category: null, pill: 'all', query: '', residency: null }

const cardKey = (card: ConnectorCardModel) => `${card.residency}:${card.slug}`

/** Whether the dialog's left column belongs to the connect element. A settled
 *  operation keeps it whenever it did NOT connect: that per-target summary is
 *  the only place an authorization that failed or ran out of time is ever said.
 *  One that connected hands the column back to the account it just made. */
function showsConnect(operation: AccountOperation | null): operation is AccountOperation {
  return operation !== null && (!operation.settled || !operation.targets.every(target => target.state === 'connected'))
}

export interface ConnectorsTabProps {
  /** Only backs the live `reload.mcp` RPC; withheld for a cross-backend scope. */
  gateway: HermesGateway | null
  profile: ProfileScope
  /** The scoped profile's display name, for the dialog's `Configuring:` line. */
  scopeLabel?: string
}

export function ConnectorsTab({ gateway, profile, scopeLabel }: ConnectorsTabProps) {
  const { t } = useI18n()
  const copy = t.connectorsPage

  const hosted = useHostedConnectors(profile)
  const mcp = useMcpServers({ gateway, profile })
  const operations = useStore($accountOperations)

  const connector = useConnectConnector(profile)
  const switcher = useConnectorSwitch(profile)
  const remover = useDisconnectAccount(profile)
  const adminUrl = useConnectorsAdminUrl()

  const [filter, setFilter] = useState<ConnectorsFilter>(EMPTY_FILTER)
  const [openKey, setOpenKey] = useState<null | string>(null)
  const [addOpen, setAddOpen] = useState(false)
  const [removeServer, setRemoveServer] = useState<null | string>(null)
  const [disconnecting, setDisconnecting] = useState<null | ConnectorCardModel>(null)

  // The probe table the MCP tab already keeps, in the vocabulary the card
  // derivation speaks. The two enums are the same six words.
  const status = useMemo(() => {
    const table: Record<string, LocalServerStatus> = {}

    for (const [name, server] of Object.entries(mcp.servers)) {
      table[name] = statusOf(server, mcp.probes[name])
    }

    return table
  }, [mcp.probes, mcp.servers])

  const local = useMemo(
    () =>
      joinLocalServers({
        catalog: mcp.catalog,
        categories: hosted.categories,
        servers: mcp.servers,
        status,
        toolCounts: mcp.toolCounts,
        usage: mcp.usageByServer
      }),
    [hosted.categories, mcp.catalog, mcp.servers, mcp.toolCounts, mcp.usageByServer, status]
  )

  const cards = useMemo(
    () => deriveCards({ hosted: hosted.rows, local, titles: hosted.titles }),
    [hosted.rows, hosted.titles, local]
  )

  const openCard = useMemo(() => cards.find(card => cardKey(card) === openKey) ?? null, [cards, openKey])

  useOpenFromRoute(cards, setOpenKey)

  // The same query the panel reads, from the same cache — the count the left
  // column prints and the rows the right column renders must be one answer.
  const openTools = useConnectorTools(profile, openCard?.residency === 'hosted' ? openCard.slug : null)

  const row = openCard ? hosted.rows.find(entry => entry.slug === openCard.slug) : undefined
  const operation = openCard ? accountOperationFor(operations, openCard.slug) : null

  /** Report a refused write once, in the page's own words. The hooks answer
   *  with an outcome rather than throwing, because the sentence is i18n's. */
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

    // The first target's link is minted with the reply and nowhere else: open it
    // now, and the frames that follow paint the row.
    const url = outcome.operation.targets.find(target => target.connectUrl)?.connectUrl

    if (url) {
      void window.hermesDesktop?.openExternal?.(url)
    }

    setOpenKey(cardKey(card))
  }

  const runVerb = (card: ConnectorCardModel) => {
    const open = accountOperationFor(operations, card.slug)

    switch (card.verb) {
      case 'authenticate':
        void mcp.authenticate(card.slug)

        return

      case 'connect':
        void startConnect(card, false)

        return

      // Both re-mint the authorization; only the word on the card differs.
      case 'reconnect':

      case 'tryAgain':
        void startConnect(card, true)

        return

      case 'stopWaiting':
        // The word comes from the account row, which outlives this window; the
        // operation does not. After a relaunch, in a second window, or when the
        // connect was started elsewhere there is nothing here to stop, so the
        // press falls through to the dialog rather than doing nothing at all.
        if (open) {
          void connector.giveUp(open.opId)

          return
        }

        break

      case 'turnBackOn':
        void write(switcher.setEnabled(card.slug, true))

        return

      default:
        break
    }

    // `openLogs`, a `stopWaiting` this window cannot act on, and anything a
    // later backend adds: the dialog holds the logs and everything else, so
    // opening it is always a true answer.
    setOpenKey(cardKey(card))
  }

  const startAdd = () => {
    mcp.addServer()
    setAddOpen(true)
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-hidden px-4 pb-2">
      {hosted.phase === 'signedOut' ? (
        <div className="flex items-center gap-2 rounded-md border border-(--ui-stroke-tertiary) px-3 py-2">
          <span className="flex-1 text-[0.72rem] text-(--ui-text-secondary)">{copy.page.signedOutNote}</span>
          <Button onClick={() => openFreeTierSignIn()} size="xs">
            {copy.page.signIn}
          </Button>
        </div>
      ) : null}

      <ConnectorsDirectory
        addYourOwn={
          <Button disabled={mcp.profilePending} onClick={startAdd} size="xs" variant="outline">
            {copy.addYourOwn}
          </Button>
        }
        // Both writes a card's verb can start, in one answer: the switch's
        // `Turn back on` and the connect the other verbs open.
        busySlug={switcher.pending ?? connector.pending}
        cards={cards}
        filter={filter}
        hostedFailed={hosted.phase === 'failed'}
        loading={hosted.phase === 'loading' && cards.length === 0}
        onAddServer={startAdd}
        onFilterChange={setFilter}
        onOpen={card => setOpenKey(cardKey(card))}
        onRetryHosted={hosted.refetch}
        onServerToggle={(card, next) => void mcp.setServerEnabled(card.slug, next)}
        onVerb={runVerb}
        selectedSlug={openCard?.slug ?? null}
      />

      {openCard ? (
        <ConnectorDialog
          accountLabel={row?.accountLabel}
          advanced={
            openCard.residency === 'local' ? (
              <LocalAdvanced controller={mcp} name={openCard.slug} onRemove={() => setRemoveServer(openCard.slug)} />
            ) : undefined
          }
          card={openCard}
          connectedOn={formatDate(row?.connectedAt)}
          connectElement={
            showsConnect(operation) ? (
              <ConnectElement
                busy={connector.pending === openCard.slug}
                onReissue={() => void startConnect(openCard, true)}
                onStopWaiting={() => void connector.giveUp(operation.opId)}
                operation={operation}
              />
            ) : undefined
          }
          cost={openCard.residency === 'local' ? localCost(mcp, openCard.slug) : undefined}
          menu={
            <ConnectorDialogMenu
              onReconnect={openCard.residency === 'hosted' ? () => void startConnect(openCard, true) : undefined}
              onRefreshTools={
                openCard.residency === 'hosted' ? openTools.refresh : () => void mcp.runProbe(openCard.slug)
              }
            />
          }
          onDisconnect={openCard.residency === 'hosted' ? () => setDisconnecting(openCard) : undefined}
          onOpenAdmin={() => openExternalLink(adminUrl)}
          onOpenChange={next => setOpenKey(next ? openKey : null)}
          onServerToggle={
            openCard.residency === 'local' ? next => void mcp.setServerEnabled(openCard.slug, next) : undefined
          }
          onToggleForMe={
            openCard.residency === 'hosted' ? next => void write(switcher.setEnabled(openCard.slug, next)) : undefined
          }
          open
          orgDisabledCount={
            openCard.residency === 'hosted' ? orgDisabledCount(hosted.policy, openCard.slug, openTools.tools) : 0
          }
          profileName={scopeLabel}
          togglePending={switcher.pending === openCard.slug}
          tools={
            openCard.residency === 'local' ? (
              <LocalToolsPanel card={openCard} controller={mcp} onRemove={() => setRemoveServer(openCard.slug)} />
            ) : (
              <HostedToolsPanel
                card={openCard}
                onDisconnect={() => setDisconnecting(openCard)}
                onRetry={hosted.refetch}
                onSignIn={() => openFreeTierSignIn()}
                policy={hosted.policy}
                scope={profile}
              />
            )
          }
        />
      ) : null}

      <McpDocumentDialog onOpenChange={setAddOpen} open={addOpen} title={copy.group.localAddServer}>
        <McpJsonEditor controller={mcp} />
      </McpDocumentDialog>

      <RemoveServerConfirm
        controller={mcp}
        name={removeServer}
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

          if (account) {
            const outcome = await remover.disconnect(account.connection_id)

            // `ConfirmDialog` owns the inline error: throwing keeps it open and
            // says so, where swallowing the failure would run its done beat and
            // close over a connection that is still there. `readableError` is
            // how the app turns a wire failure into that one sentence.
            if (!outcome.ok) {
              throw new Error(readableError(outcome.error, copy.page.writeFailed).message)
            }
          }

          // `ConfirmDialog` owns its own close beat; the page only has to let
          // go of the app whose account just went away.
          setOpenKey(null)
        }}
        open={disconnecting !== null}
        title={copy.dialog.disconnectTitle(disconnecting?.name ?? '')}
      />
    </div>
  )
}

/** `?connector=<slug>` and `?server=<name>` open that app's dialog. The param is
 *  dropped only once the card it names exists — the config and the hosted list
 *  arrive after the first paint, and deleting it earlier would lose the link. */
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

    open(cardKey(target))
    params.delete('server')
    params.delete('connector')

    const query = params.toString()
    navigate({ hash, pathname, search: query ? `?${query}` : '' }, { replace: true })
  }, [cards, hash, navigate, open, pathname, search])
}

/** The two usage numbers a local server's dialog prints. Each half is omitted
 *  when it is unknown rather than printed as a zero nobody measured. */
function localCost(mcp: ReturnType<typeof useMcpServers>, name: string) {
  const entry = mcp.servers[name]

  if (!entry) {
    return undefined
  }

  const cost = mcp.costFor(name, entry)

  return {
    tokensPerCall: cost.tokens === null ? undefined : compactNumber(cost.tokens),
    usesPerMonth: cost.uses === null ? undefined : compactNumber(cost.uses)
  }
}

const formatDate = (iso: string | undefined): string | undefined => {
  if (!iso) {
    return undefined
  }

  const at = new Date(iso)

  return Number.isNaN(at.getTime()) ? undefined : at.toLocaleDateString()
}
