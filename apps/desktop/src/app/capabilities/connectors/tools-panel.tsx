// The dialog's right column, bound to data.
//
// Two panels, because the two backings answer to different owners. A hosted
// app's rule is the member policy layer, written compare-and-set against the
// revision the person saw. A server on this Mac has no policy at all: its rule
// is `tools.include` / `tools.exclude` in mcp.json, and the editor writes the
// whole off-list in one config save.
//
// Both render the SAME `ToolsList` through the SAME `useToolsEditor`, so the
// dirty footer, the quick actions and the discard all behave identically.

import { useMemo } from 'react'

import type { ProfileScope } from '@/hermes'
import { isToolEnabled } from '@/lib/mcp-tool-filter'

import type { McpServersController } from '../mcp/use-mcp-servers'

import {
  type ConnectorPolicyView,
  connectorToolRows,
  memberDisabledTools,
  memberRevision,
  orgLockedTools
} from './data/join'
import { useConnectorToolsSave } from './data/mutations'
import { useConnectorTools } from './data/queries'
import { conflictDifference, toolRows } from './derive'
import { ToolsList } from './tools-list'
import type { ConnectorCardModel, ToolInput } from './types'
import { type SaveResult, useToolsEditor } from './use-tools-editor'

export interface HostedToolsPanelProps {
  card: ConnectorCardModel
  /** The 'gone' state's way out: the connector is no longer on the account. */
  onDisconnect: () => void
  /** Re-read everything the page holds for this account. */
  onRetry: () => void
  onSignIn: () => void
  policy: ConnectorPolicyView
  scope: ProfileScope
}

export function HostedToolsPanel({ card, onDisconnect, onRetry, onSignIn, policy, scope }: HostedToolsPanelProps) {
  const tools = useConnectorTools(scope, card.slug)
  const saver = useConnectorToolsSave(scope, card.slug, memberRevision(policy))

  const rows = useMemo(() => connectorToolRows(policy, card.slug, tools.tools), [card.slug, policy, tools.tools])

  const savedDisabled = useMemo(() => [...memberDisabledTools(policy, card.slug)], [card.slug, policy])

  const editor = useToolsEditor({
    editorKey: card.slug,
    onSave: saver.onSave,
    savedDisabled,
    status: tools.status,
    tools: rows
  })

  return (
    <ToolsList
      conflict={saver.theirs ? conflictDifference(saver.theirs, editor.local) : undefined}
      connectorName={card.name}
      counts={editor.counts}
      currentAction={editor.currentAction}
      dirty={editor.dirty}
      freshness={tools.freshness ?? undefined}
      isOn={editor.isOn}
      onApplyQuickAction={editor.applyQuickAction}
      onDiscard={editor.discard}
      onKeepMine={() => void editor.keepMine()}
      onRefresh={tools.refresh}
      onReload={saver.reload}
      onRemove={onDisconnect}
      onRetry={onRetry}
      onSave={() => void editor.save()}
      onSignIn={onSignIn}
      onToggle={editor.toggle}
      phase={editor.phase}
      tools={rows}
    />
  )
}

export interface LocalToolsPanelProps {
  card: ConnectorCardModel
  controller: McpServersController
  /** Remove the server — the 'gone' state's way out. */
  onRemove: () => void
}

export function LocalToolsPanel({ card, controller, onRemove }: LocalToolsPanelProps) {
  const probe = controller.probes[card.slug]
  const entry = controller.servers[card.slug]

  // A probe that has not answered yet, or answered with a failure, is exactly
  // the tool list's own loading / unavailable state — no second vocabulary.
  const status = !probe || probe === 'probing' ? 'loading' : probe.ok ? null : 'unavailable'

  const discovered = useMemo(
    () => (probe && probe !== 'probing' && probe.ok ? probe.tools.map(tool => tool.name) : []),
    [probe]
  )

  // A discovered MCP tool carries a name and a description and nothing else: no
  // facet, no hints, no categories. The vocabulary map already renders an
  // unknown facet as "Unknown effect", so the row reads honestly and the quick
  // actions leave every one of them alone.
  const inputs = useMemo<ToolInput[]>(
    () =>
      probe && probe !== 'probing' && probe.ok
        ? probe.tools.map(tool => ({
            categories: [],
            deprecated: false,
            description: tool.description ?? '',
            facet: 'unclassified',
            hints: [],
            name: tool.name,
            slug: tool.name
          }))
        : [],
    [probe]
  )

  const savedDisabled = useMemo(
    () => discovered.filter(name => entry !== undefined && !isToolEnabled(entry, name)),
    [discovered, entry]
  )

  const rows = useMemo(() => toolRows(inputs, new Set(savedDisabled)), [inputs, savedDisabled])

  const onSave = async (disabled: string[]): Promise<SaveResult> =>
    (await controller.setServerTools(card.slug, disabled, discovered)) ? 'saved' : 'failed'

  const editor = useToolsEditor({ editorKey: card.slug, onSave, savedDisabled, status, tools: rows })

  return (
    <ToolsList
      connectorName={card.name}
      counts={editor.counts}
      currentAction={editor.currentAction}
      dirty={editor.dirty}
      isOn={editor.isOn}
      onApplyQuickAction={editor.applyQuickAction}
      onDiscard={editor.discard}
      onKeepMine={() => undefined}
      onRefresh={() => void controller.runProbe(card.slug)}
      onReload={() => void controller.runProbe(card.slug)}
      onRemove={onRemove}
      onRetry={() => void controller.runProbe(card.slug)}
      onSave={() => void editor.save()}
      onSignIn={() => void controller.authenticate(card.slug)}
      onToggle={editor.toggle}
      phase={editor.phase}
      tools={rows}
    />
  )
}

/** How many tools the organisation took away from one hosted app. */
export function orgDisabledCount(policy: ConnectorPolicyView, slug: string, tools: readonly ToolInput[]): number {
  return orgLockedTools(policy, slug, tools).size
}
