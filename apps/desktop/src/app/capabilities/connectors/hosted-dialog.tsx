import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import type { ProfileScope } from '@/hermes'
import { openFreeTierSignIn } from '@/store/free-tier-sign-in'

import type { McpServersController } from '../mcp/use-mcp-servers'

import { ConnectElement } from './connect-element'
import { ConnectorDialog } from './connector-dialog'
import {
  $accountOperations,
  type AccountOperation,
  accountOperationFor,
  clearAccountOperation
} from './data/account-operations'
import { openConnectorsAdmin } from './data/portal'
import { type HostedConnectorsView, useConnectorTools } from './data/queries'
import { localServerName } from './derive'
import { ConnectorDialogMenu } from './dialog-menu'
import { localCost } from './local-dialog'
import type { InstallField } from './local-server-control'
import { LocalAdvanced } from './local-slots'
import { HostedToolsPanel, LocalToolsPanel, orgDisabledCount } from './tools-panel'
import type { ConnectorCardModel } from './types'
import { type WayChoice, wayInUse } from './ways-section'

export interface HostedConnectorDialogProps {
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

export function HostedConnectorDialog({
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
  const operation = accountOperationFor(useStore($accountOperations), card.slug)
  const inUse = wayInUse(card.ways)
  const [way, setWay] = useState<WayChoice>(inUse ?? 'hosted')

  useEffect(() => {
    if (inUse) {
      setWay(inUse)
    }
  }, [inUse])

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
          onDisconnect={card.ways.hosted?.connected === true ? onDisconnect : undefined}
          onReconnect={onReconnect}
          onRefreshTools={tools.refresh}
        />
      }
      onAuthenticate={() => void controller.authenticate(localServerName(card))}
      onConnect={onConnect}
      onDisconnect={onDisconnect}
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
