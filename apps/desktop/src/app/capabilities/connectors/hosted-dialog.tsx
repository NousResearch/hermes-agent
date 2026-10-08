import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import type { ProfileScope } from '@/hermes'
import { openFreeTierSignIn } from '@/store/free-tier-sign-in'

import type { McpServersController } from '../mcp/use-mcp-servers'

import { type AccountEdit, AccountsSection, type AccountsSectionProps } from './accounts-section'
import { ConnectElement } from './connect-element'
import { ConnectorDialog } from './connector-dialog'
import {
  $accountOperations,
  type AccountOperation,
  accountOperationFor,
  clearAccountOperation
} from './data/account-operations'
import type { WriteOutcome } from './data/mutations'
import { openConnectorsAdmin } from './data/portal'
import { type HostedConnectorsView, useConnectorTools } from './data/queries'
import { localServerName } from './derive'
import { ConnectorDialogMenu } from './dialog-menu'
import { localCost } from './local-dialog'
import type { InstallField } from './local-server-control'
import { LocalAdvanced } from './local-slots'
import { HostedToolsPanel, LocalToolsPanel, orgDisabledCount } from './tools-panel'
import type { AccountRow, ConnectorCardModel } from './types'
import { type WayChoice, wayInUse } from './ways-section'

export interface HostedConnectorDialogProps {
  accountActions: HostedAccountActions
  card: ConnectorCardModel
  controller: McpServersController
  hosted: HostedConnectorsView
  installFields?: readonly InstallField[]
  installing?: boolean
  onClose: () => void
  onConnect: () => void
  onDisconnect: () => void
  onGiveUp: (opId: string) => void
  onInstall: (env: Record<string, string>) => void
  onReconnect: () => void
  onRemoveServer: () => void
  onToggleForMe: (next: boolean) => void
  onVerb: () => void
  profile: ProfileScope
  togglePending: boolean
}

export interface PendingRename {
  alias: string
  slug: string
}

export interface HostedAccountActions {
  add: (alias: string) => Promise<WriteOutcome>
  /** An account a deep link asked to rename; that app's dialog opens its editor once the account is listed. */
  pendingRename: null | PendingRename
  reconnect: (account: AccountRow) => void
  reconnecting: boolean
  remove: (account: AccountRow) => void
  rename: (connectionId: string, alias: string) => Promise<WriteOutcome>
  renameOpened: () => void
}

export function HostedConnectorDialog({
  accountActions,
  card,
  controller,
  hosted,
  installFields,
  installing,
  onClose,
  onConnect,
  onDisconnect,
  onGiveUp,
  onInstall,
  onReconnect,
  onRemoveServer,
  onToggleForMe,
  onVerb,
  profile,
  togglePending
}: HostedConnectorDialogProps) {
  const tools = useConnectorTools(profile, card.slug, hosted.listSlugs.has(card.slug))
  const operation = accountOperationFor(useStore($accountOperations), card.slug, profile)
  const inUse = wayInUse(card.ways)
  const [way, setWay] = useState<WayChoice>(inUse ?? 'hosted')

  useEffect(() => {
    if (inUse) {
      setWay(inUse)
    }
  }, [inUse])

  const accounts = card.ways.hosted?.accounts ?? []
  const accountList = useAccountList(card, accountActions, operation !== null && !operation.settled)

  const local = card.ways.local
  const serverName = localServerName(card)
  const installed = local?.installed === true && card.plugin === undefined

  const hostedPanel = (
    <HostedToolsPanel
      card={card}
      disabledTools={card.ways.hosted?.disabledTools}
      onDisconnect={onDisconnect}
      onRetryRules={hosted.retryRules}
      onSignIn={() => openFreeTierSignIn()}
      policy={hosted.policy}
      readOnly={hosted.rulesFailed}
      rulesSignedOut={hosted.rulesSignedOut}
      scope={profile}
      tools={tools}
    />
  )

  const element = stillOpen(operation) ? (
    <ConnectElement onStopWaiting={() => onGiveUp(operation.opId)} operation={operation} />
  ) : undefined

  return (
    <ConnectorDialog
      accounts={<AccountsSection {...accountList.props} />}
      advanced={
        installed ? <LocalAdvanced controller={controller} name={serverName} onRemove={onRemoveServer} /> : undefined
      }
      card={card}
      connectElement={element}
      cost={installed ? localCost(controller, serverName) : undefined}
      installFields={installFields}
      installing={installing}
      menu={
        <ConnectorDialogMenu
          onDisconnect={card.ways.hosted?.connected === true && accounts.length === 0 ? onDisconnect : undefined}
          onReconnect={menuReconnect(accounts, onReconnect, accountActions.reconnect)}
          onRefreshTools={tools.refresh}
        />
      }
      onAuthenticate={() => void controller.authenticate(localServerName(card))}
      onConnect={onConnect}
      onDisconnect={onDisconnect}
      onEscape={accountList.cancelEdit}
      onInstall={onInstall}
      onOpenAdmin={() => void openConnectorsAdmin()}
      onOpenChange={next => {
        if (!next) {
          if (operation?.settled) {
            clearAccountOperation(operation.opId)
          }

          onClose()
        }
      }}
      onReconnect={onReconnect}
      onServerToggle={next => void controller.setServerEnabled(serverName, next)}
      onToggleForMe={onToggleForMe}
      onVerb={onVerb}
      onWayChange={local ? setWay : undefined}
      open
      orgDisabledCount={orgDisabledCount(hosted.policy, card.slug, tools.tools)}
      rulesReadOnly={hosted.rulesFailed}
      togglePending={togglePending}
      tools={
        way === 'local' && local?.installed === true ? (
          <LocalToolsPanel card={card} controller={controller} onRemove={onRemoveServer} />
        ) : (
          hostedPanel
        )
      }
      way={way}
    />
  )
}

function stillOpen(operation: AccountOperation | null): operation is AccountOperation {
  return operation !== null && (!operation.settled || !operation.targets.every(target => target.state === 'connected'))
}

/** The menu's Reconnect speaks for the app only while it has one account; with several, each row has its own. */
function menuReconnect(
  accounts: readonly AccountRow[],
  reconnectApp: () => void,
  reconnectAccount: (account: AccountRow) => void
): (() => void) | undefined {
  if (accounts.length > 1) {
    return undefined
  }

  const only = accounts[0]

  return only ? () => reconnectAccount(only) : reconnectApp
}

/** The account list's props and its one piece of local state: which name is being edited. A deep link's pending
 *  rename opens that account's editor once this app lists it. */
function useAccountList(card: ConnectorCardModel, actions: HostedAccountActions, connecting: boolean) {
  const [edit, setEdit] = useState<AccountEdit | null>(null)
  const accounts = card.ways.hosted?.accounts ?? []
  const { pendingRename, renameOpened } = actions

  const renameTarget =
    pendingRename?.slug === card.slug ? accounts.find(row => row.alias === pendingRename.alias) : undefined

  useEffect(() => {
    if (renameTarget) {
      setEdit({ connectionId: renameTarget.connection_id, kind: 'rename' })
      renameOpened()
    }
  }, [renameOpened, renameTarget])

  const props: AccountsSectionProps = {
    accounts,
    connecting,
    edit,
    onAdd: actions.add,
    onEditChange: setEdit,
    onReconnect: actions.reconnect,
    onRemove: actions.remove,
    onRename: actions.rename,
    reconnecting: actions.reconnecting,
    retired: card.ways.hosted?.retiredAccounts ?? []
  }

  // Escape cancels an open name edit and reports it, so the dialog stays open.
  const cancelEdit = (): boolean => {
    if (edit === null) {
      return false
    }

    setEdit(null)

    return true
  }

  return { cancelEdit, props }
}
