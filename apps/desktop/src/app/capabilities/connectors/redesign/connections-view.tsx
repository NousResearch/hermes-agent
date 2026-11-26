import './connections.css'

import { useStore } from '@nanostores/react'
import { type ReactNode, useEffect, useMemo, useState } from 'react'

import type { ProfileScope } from '@/hermes'
import { $interfaceMode } from '@/store/interface-mode'

import type { McpServersController } from '../../mcp/use-mcp-servers'
import type { WriteOutcome } from '../data/mutations'
import type { HostedConnectorsView } from '../data/queries'
import { cardKey } from '../derive'
import type { PendingRename } from '../hosted-dialog'
import type { InstallField } from '../local-server-control'
import type { AccountRow, ConnectorCardModel } from '../types'

import { ConnectionsCategoryDetail } from './category-detail'
import { ConnectionSheet } from './connection-sheet'
import { ConnectionToolsPage } from './connection-tools-page'
import { ConnectionsDirectoryRedesign } from './connections-directory'
import { type ConnectionsFilterState, EMPTY_CONNECTIONS_FILTER } from './filter-logic'
import { cardCtaKind, type ConnectionCategoryId, type ConnectionInterfaceMode } from './taxonomy'

export interface ConnectionsViewProps {
  addDisabled?: boolean
  agentControls: boolean
  busyKey?: null | string
  cards: readonly ConnectorCardModel[]
  controller: McpServersController
  hosted: HostedConnectorsView
  hostedFailed?: boolean
  /** The catalog fields a local install needs, resolved for the app the sheet is showing. */
  installFieldsFor?: (card: ConnectorCardModel) => readonly InstallField[] | undefined
  installingKey?: null | string
  loading?: boolean
  notices?: ReactNode
  onAddServer?: () => void
  onCloseCard: () => void
  onConnect: (card: ConnectorCardModel, alias?: string) => Promise<WriteOutcome | void>
  onDisconnectApp: (card: ConnectorCardModel) => Promise<WriteOutcome>
  onGiveUp: (opId: string) => void
  onInstall: (card: ConnectorCardModel, env: Record<string, string>) => void
  onOpenCard: (card: ConnectorCardModel) => void
  onPrefetch?: (card: ConnectorCardModel) => void
  onReconnect: (card: ConnectorCardModel, account?: AccountRow) => void
  onRemoveAccount: (card: ConnectorCardModel, account: AccountRow) => Promise<WriteOutcome>
  onRemoveLocalServer: (card: ConnectorCardModel) => void
  onRenameAccount: (connectionId: string, alias: string) => Promise<WriteOutcome>
  onRenameOpened?: () => void
  onRetryHosted?: () => void
  /** Stops a connect the drawer is waiting on when no account operation is tracked. */
  onStopWaiting?: (card: ConnectorCardModel) => void
  /** A category See-All page or an app's Tools page owns the page header while open. */
  onSubviewChange?: (inSubview: boolean) => void
  onToggleForMe?: (card: ConnectorCardModel, next: boolean) => void
  onVerb: (card: ConnectorCardModel) => void
  /** The app drawer this page opens; the parent keys it, so route links and sign-ins can open it. */
  openCard: ConnectorCardModel | null
  pendingRename?: PendingRename | null
  profile: ProfileScope
  togglePendingKey?: null | string
}

/**
 * The portal shell's Connections page: the directory, a category's See-All page,
 * and one app's Tools page share one filter and one drill-down. The page takes
 * its Simple/Advanced ordering from the window's interface mode, and defers
 * every connect, install and account write to the Connectors tab that owns them.
 */
export function ConnectionsView({
  addDisabled = false,
  agentControls,
  busyKey = null,
  cards,
  controller,
  hosted,
  hostedFailed = false,
  installFieldsFor,
  installingKey = null,
  loading = false,
  notices,
  onAddServer,
  onCloseCard,
  onConnect,
  onDisconnectApp,
  onGiveUp,
  onInstall,
  onOpenCard,
  onPrefetch,
  onReconnect,
  onRemoveAccount,
  onRemoveLocalServer,
  onRenameAccount,
  onRenameOpened,
  onRetryHosted,
  onStopWaiting,
  onSubviewChange,
  onToggleForMe,
  onVerb,
  openCard,
  pendingRename = null,
  profile,
  togglePendingKey = null
}: ConnectionsViewProps) {
  const interfaceMode = useStore($interfaceMode)
  const mode: ConnectionInterfaceMode = interfaceMode === 'advanced' ? 'advanced' : 'simple'
  const [filter, setFilter] = useState<ConnectionsFilterState>(EMPTY_CONNECTIONS_FILTER)
  const [activeCategory, setActiveCategory] = useState<ConnectionCategoryId | null>(null)
  const [toolsCardKey, setToolsCardKey] = useState<null | string>(null)
  // Set only by the Tools page's Disconnect action, so the drawer opens its own confirmation.
  const [disconnectKey, setDisconnectKey] = useState<null | string>(null)
  // The Tools page a Disconnect request came from: canceling it goes straight back there.
  const [disconnectOriginToolsKey, setDisconnectOriginToolsKey] = useState<null | string>(null)

  const toolsCard = useMemo(
    () => (toolsCardKey === null ? null : (cards.find(card => cardKey(card) === toolsCardKey) ?? null)),
    [cards, toolsCardKey]
  )
  const inSubview = activeCategory !== null || toolsCard !== null
  const wantsDisconnect = openCard !== null && disconnectKey !== null && disconnectKey === cardKey(openCard)

  useEffect(() => {
    onSubviewChange?.(inSubview)
  }, [inSubview, onSubviewChange])

  useEffect(() => () => onSubviewChange?.(false), [onSubviewChange])

  useEffect(() => {
    setActiveCategory(null)
    setToolsCardKey(null)
    setDisconnectKey(null)
    setDisconnectOriginToolsKey(null)
  }, [profile])

  useEffect(() => {
    if (openCard === null && disconnectKey !== null) {
      setDisconnectKey(null)
      setDisconnectOriginToolsKey(null)
    }
  }, [disconnectKey, openCard])

  const openSheet = (card: ConnectorCardModel) => {
    setDisconnectKey(null)
    setDisconnectOriginToolsKey(null)
    onOpenCard(card)
  }

  const closeSheet = () => {
    setDisconnectKey(null)
    setDisconnectOriginToolsKey(null)
    onCloseCard()
  }

  // Cancel or Escape on the Disconnect modal: a Tools page request returns to that same page.
  const dismissDisconnect = () => {
    setDisconnectKey(null)
    const origin = disconnectOriginToolsKey
    setDisconnectOriginToolsKey(null)

    if (origin !== null) {
      setToolsCardKey(origin)
      onCloseCard()
    }
  }

  const openDisconnect = (card: ConnectorCardModel) => {
    setDisconnectOriginToolsKey(toolsCardKey)
    setToolsCardKey(null)
    setDisconnectKey(cardKey(card))
    onOpenCard(card)
  }

  const openToolsPage = (card: ConnectorCardModel) => {
    setToolsCardKey(cardKey(card))
    closeSheet()
  }

  // A local server is removed, not disconnected through the hosted account door.
  const disconnectFromTools = (card: ConnectorCardModel) => {
    if (card.ways.hosted === null) {
      onRemoveLocalServer(card)

      return
    }

    openDisconnect(card)
  }

  const handleCta = (card: ConnectorCardModel) => {
    const local = card.ways.local
    const needsFields = local !== null && card.ways.hosted === null && local.needsEnv === true

    if (cardCtaKind(card) === 'manage' || needsFields) {
      openSheet(card)

      return
    }

    // Connect/install starts the operation and opens the drawer itself.
    onVerb(card)
  }

  return (
    <>
      {toolsCard !== null ? (
        <ConnectionToolsPage
          agentControls={agentControls}
          card={toolsCard}
          controller={controller}
          hosted={hosted}
          key={cardKey(toolsCard)}
          onBack={() => setToolsCardKey(null)}
          onDisconnect={() => disconnectFromTools(toolsCard)}
          onReconnect={() => onReconnect(toolsCard)}
          onRemoveLocalServer={() => onRemoveLocalServer(toolsCard)}
          profile={profile}
        />
      ) : activeCategory !== null ? (
        <ConnectionsCategoryDetail
          busyKey={busyKey}
          cards={cards}
          category={activeCategory}
          filter={filter}
          mode={mode}
          onBack={() => setActiveCategory(null)}
          onCtaClick={handleCta}
          onFilterChange={setFilter}
          onOpenCard={openSheet}
          onPrefetch={onPrefetch}
        />
      ) : (
        <ConnectionsDirectoryRedesign
          addDisabled={addDisabled}
          busyKey={busyKey}
          cards={cards}
          filter={filter}
          hideHeader
          hostedFailed={hostedFailed}
          loading={loading}
          mode={mode}
          notices={notices}
          onAddServer={onAddServer}
          onCtaClick={handleCta}
          onFilterChange={setFilter}
          onOpenCard={openSheet}
          onPrefetch={onPrefetch}
          onRetryHosted={onRetryHosted}
          onSelectCategory={setActiveCategory}
        />
      )}

      {toolsCard === null && openCard !== null ? (
        <ConnectionSheet
          agentControls={agentControls}
          card={openCard}
          controller={controller}
          hosted={hosted}
          initialDisconnect={wantsDisconnect}
          installFields={installFieldsFor?.(openCard)}
          installing={installingKey === cardKey(openCard)}
          key={cardKey(openCard)}
          onClose={closeSheet}
          onConnect={alias => onConnect(openCard, alias)}
          onDisconnectApp={() => onDisconnectApp(openCard)}
          onDisconnectDismiss={dismissDisconnect}
          onGiveUp={onGiveUp}
          onInstall={env => onInstall(openCard, env)}
          onOpenToolsPage={openToolsPage}
          onReconnect={account => onReconnect(openCard, account)}
          onRemoveAccount={account => onRemoveAccount(openCard, account)}
          onRemoveLocalServer={() => onRemoveLocalServer(openCard)}
          onRenameAccount={onRenameAccount}
          onRenameOpened={onRenameOpened}
          onStopWaiting={onStopWaiting}
          onToggleForMe={onToggleForMe === undefined ? undefined : next => onToggleForMe(openCard, next)}
          pendingRename={pendingRename}
          profile={profile}
          togglePending={togglePendingKey === openCard.slug}
        />
      ) : null}
    </>
  )
}
