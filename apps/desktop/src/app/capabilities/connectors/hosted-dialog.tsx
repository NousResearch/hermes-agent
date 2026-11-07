import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import type { ProfileScope } from '@/hermes'
import { translateNow } from '@/i18n'
import { openFreeTierSignIn } from '@/store/free-tier-sign-in'
import { notify } from '@/store/notifications'

import { type AccountEdit, AccountsSection, type AccountsSectionProps } from './accounts-section'
import { ConnectElement } from './connect-element'
import { ConnectorDialog } from './connector-dialog'
import {
  $accountOperations,
  type AccountOperation,
  accountOperationFor,
  clearAccountOperation,
  retryable,
  retryableAccount
} from './data/account-operations'
import type { WriteOutcome } from './data/mutations'
import { openConnectorsAdmin } from './data/portal'
import { type HostedConnectorsView, useConnectorTools } from './data/queries'
import { ConnectorDialogMenu } from './dialog-menu'
import { HostedToolsPanel, orgDisabledCount } from './tools-panel'
import type { AccountRow, ConnectorCardModel } from './types'

export interface HostedConnectorDialogProps {
  accountActions: HostedAccountActions
  card: ConnectorCardModel
  hosted: HostedConnectorsView
  onClose: () => void
  onDisconnect: () => void
  onGiveUp: (opId: string) => void
  onReconnect: () => void
  onToggleForMe: (next: boolean) => void
  onVerb: () => void
  profile: ProfileScope
  togglePending: boolean
}

export interface PendingRename {
  /** The account goes by id: a stale link must not open another account that took its old name. */
  connectionId: string
  slug: string
}

/** The account a pending rename link opens in this app's dialog: the row with that id, 'gone' once the list
 *  has answered without it, or null while there is nothing to do yet. Never a row matched by name. */
export function renameLinkTarget(
  pending: null | PendingRename,
  slug: string,
  accounts: readonly AccountRow[],
  loaded: boolean
): 'gone' | AccountRow | null {
  if (pending?.slug !== slug) {
    return null
  }

  return accounts.find(row => row.connection_id === pending.connectionId) ?? (loaded ? 'gone' : null)
}

export interface HostedAccountActions {
  add: (alias: string) => Promise<WriteOutcome>
  /** An account a deep link asked to rename; that app's dialog opens its editor once the account is listed. */
  pendingRename: null | PendingRename
  reconnect: (account: AccountRow) => void
  reconnecting: boolean
  /** Sends a failed operation's own request again, so the backend mints a fresh link on that operation. */
  retry: (operation: AccountOperation) => void
  remove: (account: AccountRow) => void
  rename: (connectionId: string, alias: string) => Promise<WriteOutcome>
  renameOpened: () => void
}

export function HostedConnectorDialog({
  accountActions,
  card,
  hosted,
  onClose,
  onDisconnect,
  onGiveUp,
  onReconnect,
  onToggleForMe,
  onVerb,
  profile,
  togglePending
}: HostedConnectorDialogProps) {
  const tools = useConnectorTools(profile, card.slug, hosted.listSlugs.has(card.slug))
  const operation = accountOperationFor(useStore($accountOperations), card.slug, profile)
  const accounts = card.ways.hosted?.accounts ?? []

  const accountList = useAccountList(
    card,
    hosted.accountsLoaded,
    accountActions,
    operation !== null && !operation.settled,
    retryableAccount(operation)
  )

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
    <ConnectElement
      onRetry={retryable(operation) ? () => accountActions.retry(operation) : undefined}
      onStopWaiting={() => onGiveUp(operation.opId)}
      operation={operation}
      retrying={accountActions.reconnecting}
    />
  ) : undefined

  return (
    <ConnectorDialog
      accounts={<AccountsSection {...accountList.props} />}
      card={card}
      connectElement={element}
      menu={
        <ConnectorDialogMenu
          onDisconnect={card.ways.hosted?.connected === true && accounts.length === 0 ? onDisconnect : undefined}
          onReconnect={menuReconnect(accounts, onReconnect, accountActions.reconnect)}
          onRefreshTools={tools.refresh}
        />
      }
      onDisconnect={onDisconnect}
      onEscape={accountList.cancelEdit}
      onOpenAdmin={() => void openConnectorsAdmin()}
      onOpenChange={next => {
        if (!next) {
          if (operation?.settled) {
            clearAccountOperation(operation.opId)
          }

          onClose()
        }
      }}
      onToggleForMe={onToggleForMe}
      onVerb={onVerb}
      open
      orgDisabledCount={orgDisabledCount(hosted.policy, card.slug, tools.tools)}
      rulesReadOnly={hosted.rulesFailed}
      togglePending={togglePending}
      tools={hostedPanel}
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
function useAccountList(
  card: ConnectorCardModel,
  accountsLoaded: boolean,
  actions: HostedAccountActions,
  connecting: boolean,
  retryable: null | string
) {
  const [edit, setEdit] = useState<AccountEdit | null>(null)
  const accounts = card.ways.hosted?.accounts ?? []
  const { pendingRename, renameOpened } = actions

  const linked = renameLinkTarget(pendingRename, card.slug, accounts, accountsLoaded)
  const renameTarget = linked === 'gone' ? undefined : (linked ?? undefined)
  const renameGone = linked === 'gone'

  useEffect(() => {
    if (renameTarget) {
      setEdit({ connectionId: renameTarget.connection_id, kind: 'rename' })
      renameOpened()
    } else if (renameGone) {
      // Removed or replaced by a reconnect since the link was made: open nothing and say so.
      notify({ kind: 'info', message: translateNow('connectorsPage.accounts.renameGone') })
      renameOpened()
    }
  }, [renameGone, renameOpened, renameTarget])

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
    retired: card.ways.hosted?.retiredAccounts ?? [],
    retryable
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
