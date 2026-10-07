import { useMemo } from 'react'

import { isToolEnabled } from '@/lib/mcp-tool-filter'

import { okProbe } from '../mcp/mcp-status'
import type { McpServersController } from '../mcp/use-mcp-servers'

import { localServerName } from './derive'
import { toolDisplayName, toolRows } from './derive-tools'
import { ToolsList } from './tools-list'
import type { ConnectorCardModel, ToolInput } from './types'
import { type SaveResult, useToolsEditor } from './use-tools-editor'

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
    card.ways?.serverEnabled === false
      ? 'off'
      : card.ways?.reason?.key === 'serverNeedsAuth'
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
