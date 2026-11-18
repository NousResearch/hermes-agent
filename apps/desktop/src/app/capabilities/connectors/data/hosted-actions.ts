import { useStore } from '@nanostores/react'

import type { ProfileScope } from '@/hermes'
import { useI18n } from '@/i18n'
import { notifyError } from '@/store/notifications'

import type { AccountRow, ConnectorCardModel } from '../types'

import { $accountOperations, abandonConnect, type AccountOperation, accountOperationFor } from './account-operations'
import { accountAddress, pickAccount } from './join'
import {
  type AccountDisconnect,
  type ConnectorConnect,
  type ConnectorSwitch,
  type ConnectOutcome,
  useConnectConnector,
  useConnectorSwitch,
  useDisconnectAccount,
  type WriteOutcome
} from './mutations'
import type { ConnectAddress } from './rpc'

export interface HostedActions {
  busySlug: null | string
  connectAccount: (card: ConnectorCardModel, reconnect: boolean, address?: ConnectAddress) => Promise<ConnectOutcome>
  connector: ConnectorConnect
  remover: AccountDisconnect
  /** Runs a failed sign-in's own request again, so it mints a fresh link for the same account. */
  retry: (card: ConnectorCardModel, operation: AccountOperation) => void
  /** Runs a hosted card's verb; false when the verb is not a hosted one (a local server's). */
  runHostedVerb: (card: ConnectorCardModel) => boolean
  startConnect: (card: ConnectorCardModel, reconnect: boolean, account?: AccountRow) => Promise<void>
  switcher: ConnectorSwitch
  write: (pending: Promise<WriteOutcome>) => Promise<void>
}

export interface HostedActionOptions {
  /** Runs after a sign-in starts. */
  onStarted?: (card: ConnectorCardModel) => void
  /** Where the browser's return shows the user; the Connectors page when absent. */
  returnRoute?: string
}

/** The hosted connect, reconnect, give-up and turn-back-on paths, shared by the Connectors page and the chat's
 *  directory card. */
export function useHostedActions(
  scope: ProfileScope,
  accounts: readonly AccountRow[],
  { onStarted, returnRoute }: HostedActionOptions = {}
): HostedActions {
  const { t } = useI18n()
  const operations = useStore($accountOperations)
  const connector = useConnectConnector(scope)
  const switcher = useConnectorSwitch(scope)
  const remover = useDisconnectAccount(scope)

  const write = async (pending: Promise<WriteOutcome>) => {
    const outcome = await pending

    if (!outcome.ok) {
      notifyError(outcome.error, t.connectorsPage.page.writeFailed)
    }
  }

  const connectAccount = async (card: ConnectorCardModel, reconnect: boolean, address: ConnectAddress = {}) => {
    const outcome = await connector.connect(card.slug, { ...address, reconnect, returnRoute })

    if (outcome.ok) {
      const url = outcome.operation.targets.find(target => target.connectUrl)?.connectUrl

      if (url) {
        void window.hermesDesktop?.openExternal?.(url)
      }
    }

    return outcome
  }

  // A row passes its own account. An app-level reconnect or Try again (no account) names the account the
  // summary speaks for, or nothing; either goes by id, so it still matches the sign-in that is already open.
  const startConnect = async (card: ConnectorCardModel, reconnect: boolean, account?: AccountRow) => {
    const target = account ?? (reconnect ? pickAccount(accounts, card.slug) : undefined)
    const address = target ? accountAddress(target) : {}
    const outcome = await connectAccount(card, reconnect, address)

    if (!outcome.ok) {
      notifyError(outcome.error, t.connectors.connectErrorFor(card.name))

      return
    }

    onStarted?.(card)
  }

  const retry = (card: ConnectorCardModel, operation: AccountOperation) => {
    const { reconnect, ...address } = operation.request

    void connectAccount(card, reconnect, address).then(outcome => {
      if (!outcome.ok) {
        notifyError(outcome.error, t.connectors.connectErrorFor(card.name))
      }
    })
  }

  const stopWaiting = (card: ConnectorCardModel) => {
    const open = accountOperationFor(operations, card.slug, scope)

    if (open) {
      void write(connector.giveUp(open.opId))

      return
    }

    abandonConnect([card.slug])

    const pending = pickAccount(accounts, card.slug)

    if (pending) {
      void write(remover.disconnect(pending.connection_id))
    }
  }

  const runHostedVerb = (card: ConnectorCardModel): boolean => {
    switch (card.verb) {
      case 'connect':
        void startConnect(card, false)

        return true

      case 'reconnect':

      case 'tryAgain':
        void startConnect(card, true)

        return true

      case 'stopWaiting':
        stopWaiting(card)

        return true

      case 'turnBackOn':
        void write(switcher.setEnabled(card.slug, true))

        return true

      default:
        return false
    }
  }

  return {
    busySlug: switcher.pending ?? connector.pending,
    connectAccount,
    connector,
    remover,
    retry,
    runHostedVerb,
    startConnect,
    switcher,
    write
  }
}
