import { compactNumber } from '@hermes/shared'
import { useLocation, useNavigate } from 'react-router'

import type { ProfileScope } from '@/hermes'

import type { McpServersController } from '../mcp/use-mcp-servers'

import { ConnectorDialog } from './connector-dialog'
import { invalidatePluginProbe } from './data/keys'
import { localServerName } from './derive'
import { ConnectorDialogMenu } from './dialog-menu'
import { LocalAdvanced } from './local-slots'
import { PluginToolsPanel } from './plugin-tools-panel'
import { LocalToolsPanel } from './tools-panel'
import type { ConnectorCardModel } from './types'

export interface LocalConnectorDialogProps {
  card: ConnectorCardModel
  controller: McpServersController
  onClose: () => void
  onRemoveServer: () => void
  profile: ProfileScope
}

export function LocalConnectorDialog({
  card,
  controller,
  onClose,
  onRemoveServer,
  profile
}: LocalConnectorDialogProps) {
  const name = localServerName(card)
  const openPlugins = useOpenPluginsTab()
  const plugin = card.plugin

  const refreshTools = () => {
    if (plugin === undefined) {
      void controller.runProbe(name)
    } else {
      invalidatePluginProbe(profile, name)
    }
  }

  return (
    <ConnectorDialog
      advanced={
        plugin === undefined ? (
          <LocalAdvanced controller={controller} name={name} onRemove={onRemoveServer} />
        ) : undefined
      }
      card={card}
      cost={localCost(controller, name)}
      menu={<ConnectorDialogMenu onRefreshTools={refreshTools} />}
      onAuthenticate={() => void controller.authenticate(name)}
      onOpenChange={next => {
        if (!next) {
          onClose()
        }
      }}
      onServerToggle={next => void controller.setServerEnabled(name, next)}
      open
      tools={
        plugin !== undefined ? (
          <PluginToolsPanel card={card} onOpenPlugins={openPlugins} plugin={plugin} scope={profile} />
        ) : (
          <LocalToolsPanel card={card} controller={controller} onRemove={onRemoveServer} />
        )
      }
    />
  )
}

function useOpenPluginsTab(): () => void {
  const navigate = useNavigate()
  const { hash, pathname, search } = useLocation()

  return () => {
    const params = new URLSearchParams(search)
    params.set('tab', 'plugins')
    navigate({ hash, pathname, search: `?${params.toString()}` })
  }
}

export function localCost(controller: McpServersController, name: string) {
  const entry = controller.servers[name]

  if (!entry) {
    return undefined
  }

  const cost = controller.costFor(name, entry)

  return {
    tokensPerCall: cost.tokens === null ? undefined : compactNumber(cost.tokens),
    usesPerMonth: cost.uses === null ? undefined : compactNumber(cost.uses)
  }
}
