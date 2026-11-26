import './connections.css'

import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'

import type { ProfileScope } from '@/hermes'
import { useI18n } from '@/i18n'
import { notify, readableError } from '@/store/notifications'

import type { McpServersController } from '../../mcp/use-mcp-servers'
import { $accountOperations, accountOperationFor } from '../data/account-operations'
import { accountName } from '../data/join'
import { invalidatePluginProbe } from '../data/keys'
import type { WriteOutcome } from '../data/mutations'
import { type HostedConnectorsView, useConnectorTools } from '../data/queries'
import { localServerName } from '../derive'
import { type PendingRename, renameLinkTarget } from '../hosted-dialog'
import type { InstallField } from '../local-server-control'
import type { AccountRow, ConnectorCardModel } from '../types'

import {
  AccountSection,
  AvailabilitySection,
  DisconnectModal,
  HeaderMenu,
  SheetBanner,
  type SheetBannerState,
  SheetHeader,
  ToolPermissionsCard,
  useLocalTools,
  useSheetFocusTrap
} from './connection-sheet-parts'
import { useConnectionsCopy } from './connections-copy'

const NO_ACCOUNTS: readonly AccountRow[] = []

export interface ConnectionSheetProps {
  agentControls: boolean
  card: ConnectorCardModel
  controller: McpServersController
  hosted: HostedConnectorsView
  /** Opens the drawer on its disconnect confirmation (the Tools page's own Disconnect action). */
  initialDisconnect?: boolean
  installFields?: readonly InstallField[]
  installing: boolean
  onClose: () => void
  onConnect: (alias?: string) => Promise<WriteOutcome | void>
  onDisconnectApp: () => Promise<WriteOutcome>
  /** The tools page opened the drawer's confirmation; clearing it keeps the parent key in sync. */
  onDisconnectDismiss?: () => void
  onGiveUp: (opId: string) => void
  onInstall: (env: Record<string, string>) => void
  onOpenToolsPage: (card: ConnectorCardModel) => void
  onReconnect: (account?: AccountRow) => void
  onRemoveAccount: (account: AccountRow) => Promise<WriteOutcome>
  onRemoveLocalServer: () => void
  onRenameAccount: (connectionId: string, alias: string) => Promise<WriteOutcome>
  onRenameOpened?: () => void
  onStopWaiting?: (card: ConnectorCardModel) => void
  onToggleForMe?: (next: boolean) => void
  pendingRename?: PendingRename | null
  profile: ProfileScope
  togglePending: boolean
}

export function ConnectionSheet({
  agentControls,
  card,
  controller,
  hosted,
  initialDisconnect = false,
  installFields,
  installing,
  onClose,
  onConnect,
  onDisconnectApp,
  onDisconnectDismiss,
  onGiveUp,
  onInstall,
  onOpenToolsPage,
  onReconnect,
  onRemoveAccount,
  onRemoveLocalServer,
  onRenameAccount,
  onRenameOpened,
  onStopWaiting,
  onToggleForMe,
  pendingRename = null,
  profile,
  togglePending
}: ConnectionSheetProps) {
  const copy = useConnectionsCopy()
  const { t } = useI18n()
  const sheet = useRef<HTMLDivElement | null>(null)
  const operations = useStore($accountOperations)
  const hostedWay = card.ways.hosted
  const local = card.ways.local
  const accounts = hostedWay?.accounts ?? NO_ACCOUNTS
  const operation = accountOperationFor(operations, card.slug, profile)
  const hostedConnected = hostedWay?.connected === true
  const localInstalled = local?.installed === true
  const localMode = !hostedConnected && localInstalled
  const tools = useConnectorTools(
    profile,
    hostedWay === null || localMode ? null : card.slug,
    hosted.listSlugs.has(card.slug)
  )
  const localTools = useLocalTools(controller, card, localMode)
  const connected = hostedConnected || localInstalled
  const waiting = !hostedConnected && (hostedWay?.state === 'connecting' || (operation !== null && !operation.settled))

  const [banner, setBanner] = useState<SheetBannerState | null>(null)
  const [busy, setBusy] = useState(false)
  const [disconnectOpen, setDisconnectOpen] = useState(initialDisconnect)
  const [renameAccountId, setRenameAccountId] = useState<null | string>(null)
  const [handledRename, setHandledRename] = useState<null | string>(null)

  // A Tools page Disconnect request can arrive while the sheet is already mounted: open on the
  // flip to true, with no `key` churn and no remount.
  useEffect(() => {
    if (initialDisconnect) {
      setDisconnectOpen(true)
    }
  }, [initialDisconnect])

  const closeDisconnect = useCallback(() => {
    setDisconnectOpen(false)
    onDisconnectDismiss?.()
    sheet.current?.focus()
  }, [onDisconnectDismiss])

  const escaped = useCallback(() => {
    if (disconnectOpen) {
      closeDisconnect()

      return
    }

    onClose()
  }, [closeDisconnect, disconnectOpen, onClose])

  useSheetFocusTrap(sheet, escaped)

  const renameTarget = useMemo(
    () => renameLinkTarget(pendingRename, card.slug, accounts, hosted.accountsLoaded),
    [accounts, card.slug, hosted.accountsLoaded, pendingRename]
  )

  useEffect(() => {
    if (pendingRename?.slug !== card.slug || handledRename === pendingRename.connectionId) {
      return
    }

    if (renameTarget === 'gone') {
      setHandledRename(pendingRename.connectionId)
      notify({ kind: 'info', message: t.connectorsPage.accounts.renameGone })
      onRenameOpened?.()

      return
    }

    if (renameTarget) {
      setHandledRename(pendingRename.connectionId)
      setRenameAccountId(renameTarget.connection_id)
      onRenameOpened?.()
    }
  }, [card.slug, handledRename, onRenameOpened, pendingRename, renameTarget, t.connectorsPage.accounts.renameGone])

  const connect = async (alias?: string) => {
    if (busy) {
      return
    }

    setBanner(null)
    setBusy(true)

    try {
      const outcome = await onConnect(alias)

      if (outcome && !outcome.ok) {
        setBanner({
          kind: 'error',
          retry: () => void connect(alias),
          text: readableError(outcome.error, copy.errors.connectBody).message
        })
      }
    } finally {
      setBusy(false)
    }
  }

  const removeAccount = async (account: AccountRow) => {
    if (busy) {
      return
    }

    const name = accountName(account)
    setBanner(null)
    setBusy(true)

    try {
      const outcome = await onRemoveAccount(account)

      if (!outcome.ok) {
        setBanner({
          kind: 'error',
          retry: () => void removeAccount(account),
          text: copy.sheet.removeFailed(name)
        })

        return
      }

      setBanner({ kind: 'success', text: copy.sheet.removedSuccess(name) })
    } finally {
      setBusy(false)
    }
  }

  const disconnectApp = async () => {
    if (busy) {
      return
    }

    setBanner(null)
    setDisconnectOpen(false)
    setBusy(true)

    try {
      const outcome = await onDisconnectApp()

      if (!outcome.ok) {
        setBanner({
          kind: 'error',
          retry: () => void disconnectApp(),
          text: copy.sheet.disconnectFailed(card.name)
        })

        return
      }
    } finally {
      setBusy(false)
    }

    onClose()
  }

  const refreshTools = () => {
    if (hostedConnected) {
      tools.refresh()

      return
    }

    const name = localServerName(card)

    if (card.plugin) {
      invalidatePluginProbe(profile, name)
    }

    void controller.runProbe(name)
  }

  const reconnect = () => onReconnect(accounts.length === 1 ? accounts[0] : undefined)
  const toolsEditable = localMode
    ? agentControls && card.plugin === undefined
    : hostedConnected && agentControls && !hosted.rulesFailed

  const menu = connected ? (
    <HeaderMenu
      disabled={busy}
      kind={hostedConnected ? 'hosted' : 'local'}
      onDisconnect={() => setDisconnectOpen(true)}
      onReconnect={reconnect}
      onRefreshTools={refreshTools}
      onRemoveLocal={card.plugin === undefined ? onRemoveLocalServer : undefined}
    />
  ) : null

  return createPortal(
    <>
      <div className="conn-sheet-backdrop fixed inset-0" onClick={onClose} />
      <div
        aria-label={card.name}
        aria-modal="true"
        className="conn-sheet fixed inset-y-0 right-0 flex w-full max-w-[40rem] flex-col outline-none"
        data-slot="connection-sheet"
        ref={sheet}
        role="dialog"
        tabIndex={-1}
      >
        <div aria-hidden={disconnectOpen || undefined} className="flex min-h-0 flex-1 flex-col" inert={disconnectOpen}>
          <SheetHeader card={card} connected={connected} menu={menu} onClose={onClose} />

          <div className="conn-sheet-body min-h-0 flex-1 overflow-y-auto px-6 pb-6">
            {banner ? <SheetBanner banner={banner} busy={busy} onDismiss={() => setBanner(null)} /> : null}

            <div className="grid gap-3">
              <AvailabilitySection
                agentControls={agentControls}
                hostedWay={hostedWay}
                local={local}
                onServerEnabledChange={next => void controller.setServerEnabled(localServerName(card), next)}
                onToggleForMe={onToggleForMe}
                plugin={card.plugin}
                togglePending={togglePending}
              />

              <AccountSection
                accounts={accounts}
                appName={card.name}
                busy={busy}
                card={card}
                hostedConnected={hostedConnected}
                installFields={installFields}
                installing={installing}
                key={card.slug}
                local={local}
                localInstalled={localInstalled}
                onAdd={connect}
                onAuthenticate={() => void controller.authenticate(localServerName(card))}
                onGiveUp={onGiveUp}
                onInstall={onInstall}
                onRemove={account => void removeAccount(account)}
                onRename={onRenameAccount}
                onRenamed={alias => setBanner({ kind: 'success', text: copy.toast.renamed(alias) })}
                onStopWaiting={onStopWaiting}
                operation={operation}
                renameAccountId={renameAccountId}
                waiting={waiting}
              />

              <ToolPermissionsCard
                card={card}
                connected={connected}
                disabledTools={hostedWay?.disabledTools}
                editable={toolsEditable}
                key={card.slug}
                localTools={localMode ? localTools : undefined}
                onError={message => setBanner({ kind: 'error', text: message })}
                onOpen={() => onOpenToolsPage(card)}
                policy={hosted.policy}
                profile={profile}
                tools={tools}
              />
            </div>
          </div>
        </div>

        {disconnectOpen ? (
          <DisconnectModal
            appName={card.name}
            busy={busy}
            onCancel={closeDisconnect}
            onConfirm={() => void disconnectApp()}
          />
        ) : null}
      </div>
    </>,
    document.body
  )
}
