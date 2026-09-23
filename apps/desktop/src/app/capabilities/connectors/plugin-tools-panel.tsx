import { useQuery } from '@tanstack/react-query'
import { useMemo } from 'react'

import { PanelEmpty } from '@/app/overlays/panel'
import { Button } from '@/components/ui/button'
import { type ProfileScope, testMcpServer } from '@/hermes'
import { useI18n } from '@/i18n'
import { PROBE_TTL_MS } from '@/lib/mcp-probe-cache'

import { CONNECTOR_GC_TIME, pluginProbeQueryKey } from './data/keys'
import { localServerName } from './derive'
import { toolDisplayName, toolRows } from './derive-tools'
import { ToolsList } from './tools-list'
import type { ConnectorCardModel, ToolInput } from './types'
import { useToolsEditor } from './use-tools-editor'

const NOTHING_DISABLED: readonly string[] = []

export interface PluginToolsPanelProps {
  card: ConnectorCardModel
  onOpenPlugins: () => void
  plugin: string
  scope: ProfileScope
}

export function PluginToolsPanel({ card, onOpenPlugins, plugin, scope }: PluginToolsPanelProps) {
  const { t } = useI18n()
  const copy = t.connectorsPage
  const name = localServerName(card)

  const probe = useQuery({
    gcTime: CONNECTOR_GC_TIME,
    queryFn: () => testMcpServer(name, scope),
    queryKey: pluginProbeQueryKey(scope, name),
    retry: false,
    staleTime: PROBE_TTL_MS
  })

  const inputs = useMemo<ToolInput[]>(
    () =>
      probe.data?.tools.map(tool => ({
        categories: [],
        deprecated: false,
        description: tool.description ?? '',
        facet: 'unclassified',
        hints: [],
        name: toolDisplayName(tool.name),
        slug: tool.name
      })) ?? [],
    [probe.data]
  )

  const rows = useMemo(() => toolRows(inputs, new Set()), [inputs])

  const editor = useToolsEditor({
    editorKey: name,
    onSave: async () => 'failed',
    savedDisabled: NOTHING_DISABLED,
    status: probe.isPending ? 'loading' : null,
    tools: rows
  })

  const failure = probe.error
    ? probe.error instanceof Error
      ? probe.error.message
      : String(probe.error)
    : probe.data && !probe.data.ok
      ? (probe.data.error ?? '')
      : null

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex shrink-0 items-center gap-2 border-b border-(--ui-stroke-tertiary) px-3.5 py-1.5">
        <span className="min-w-0 flex-1 text-[0.7rem] text-(--ui-text-tertiary)">
          {copy.dialog.providedByPlugin(plugin)}
        </span>
        <Button onClick={onOpenPlugins} size="inline" variant="textStrong">
          {copy.dialog.openPlugins}
        </Button>
      </div>

      {failure === null ? (
        <ToolsList
          connectorName={card.name}
          editor={editor}
          listKey={name}
          onReload={() => void probe.refetch()}
          onRetry={() => void probe.refetch()}
          preview
          readOnly
          summaryTitle={card.state === 'connected' ? copy.tools.summaryTitle(card.name) : undefined}
          tools={rows}
        />
      ) : (
        <PanelEmpty
          action={
            <Button onClick={() => void probe.refetch()} size="inline" variant="textStrong">
              {copy.tools.retry}
            </Button>
          }
          description={failure}
          icon="warning"
          title={copy.tools.unavailableLine}
        />
      )}
    </div>
  )
}
