import { useMemo } from 'react'

import type { ProfileScope } from '@/hermes'
import { isToolEnabled } from '@/lib/mcp-tool-filter'

import { okProbe } from '../mcp/mcp-status'
import type { McpServersController } from '../mcp/use-mcp-servers'

import {
  type ConnectorPolicyView,
  connectorToolRows,
  memberDisabledTools,
  memberRevision,
  orgLockedTools
} from './data/join'
import { useConnectorToolsSave } from './data/mutations'
import type { ConnectorToolsView } from './data/queries'
import { localServerName } from './derive'
import { conflictDifference, toolDisplayName, toolRows } from './derive-tools'
import { ToolsList } from './tools-list'
import type { ConnectorCardModel, ToolInput } from './types'
import { type SaveResult, useToolsEditor } from './use-tools-editor'

export interface HostedToolsPanelProps {
  card: ConnectorCardModel
  disabledTools?: readonly string[]
  onDisconnect: () => void
  onRetryRules?: () => void
  onSignIn: () => void
  policy: ConnectorPolicyView
  readOnly?: boolean
  rulesSignedOut?: boolean
  scope: ProfileScope
  tools: ConnectorToolsView
}

export function HostedToolsPanel({
  card,
  disabledTools = [],
  onDisconnect,
  onRetryRules,
  onSignIn,
  policy,
  readOnly = false,
  rulesSignedOut = false,
  scope,
  tools
}: HostedToolsPanelProps) {
  const saver = useConnectorToolsSave(scope, card.slug, memberRevision(policy))

  const rows = useMemo(
    () =>
      readOnly ? toolRows(tools.tools, new Set(disabledTools)) : connectorToolRows(policy, card.slug, tools.tools),
    [card.slug, disabledTools, policy, readOnly, tools.tools]
  )

  const savedDisabled = useMemo(
    () => (readOnly ? [...disabledTools] : [...memberDisabledTools(policy, card.slug)]),
    [card.slug, disabledTools, policy, readOnly]
  )

  const editor = useToolsEditor({
    editorKey: card.slug,
    onSave: saver.onSave,
    savedDisabled,
    status: tools.status,
    tools: rows
  })

  return (
    <ToolsList
      appOff={card.state === 'off'}
      conflict={saver.theirs ? conflictDifference(saver.theirs, editor.local) : undefined}
      connectorName={card.name}
      editor={editor}
      listKey={card.slug}
      onReload={saver.reload}
      onRemove={onDisconnect}
      onRetry={tools.retry}
      onRetryRules={onRetryRules}
      onSignIn={onSignIn}
      preview={card.ways.hosted?.connected !== true}
      readOnly={readOnly}
      rulesSignedOut={rulesSignedOut}
      signedOut={tools.signedOut}
      tools={rows}
    />
  )
}

export interface LocalToolsPanelProps {
  card: ConnectorCardModel
  controller: McpServersController
  onRemove: () => void
}

export function LocalToolsPanel({ card, controller, onRemove }: LocalToolsPanelProps) {
  const name = localServerName(card)
  const probe = controller.probes[name]
  const entry = controller.servers[name]

  const status =
    card.ways.local?.serverEnabled === false
      ? 'off'
      : card.ways.local?.reason?.key === 'serverNeedsAuth'
        ? 'needsAuth'
        : !probe || probe === 'probing'
          ? 'loading'
          : probe.ok
            ? null
            : 'unavailable'

  const discovered = useMemo(() => okProbe(probe)?.tools.map(tool => tool.name) ?? [], [probe])

  const inputs = useMemo<ToolInput[]>(
    () =>
      okProbe(probe)?.tools.map(tool => ({
        categories: [],
        deprecated: false,
        description: tool.description ?? '',
        facet: 'unclassified',
        hints: [],
        name: toolDisplayName(tool.name),
        slug: tool.name
      })) ?? [],
    [probe]
  )

  const savedDisabled = useMemo(
    () => discovered.filter(tool => entry !== undefined && !isToolEnabled(entry, tool)),
    [discovered, entry]
  )

  const rows = useMemo(() => toolRows(inputs, new Set(savedDisabled)), [inputs, savedDisabled])

  const onSave = async (disabled: string[]): Promise<SaveResult> =>
    (await controller.setServerTools(name, disabled, discovered)) ? 'saved' : 'failed'

  const editor = useToolsEditor({ editorKey: name, onSave, savedDisabled, status, tools: rows })

  return (
    <ToolsList
      connectorName={card.name}
      editor={editor}
      listKey={name}
      onReload={() => void controller.runProbe(name)}
      onRemove={onRemove}
      onRetry={() => void controller.runProbe(name)}
      preview={card.state !== 'connected'}
      tools={rows}
    />
  )
}

export function orgDisabledCount(policy: ConnectorPolicyView, slug: string, tools: readonly ToolInput[]): number {
  return orgLockedTools(policy, slug, tools).size
}
