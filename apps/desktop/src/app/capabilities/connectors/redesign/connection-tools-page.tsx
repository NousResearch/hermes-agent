import './connections.css'

import { useCallback, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { DisclosureCaret } from '@/components/ui/disclosure-caret'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { EmptyState } from '@/components/ui/empty-state'
import { Switch } from '@/components/ui/switch'
import { Tip } from '@/components/ui/tooltip'
import type { ProfileScope } from '@/hermes'
import { useI18n } from '@/i18n'

import type { McpServersController } from '../../mcp/use-mcp-servers'
import { connectorToolRows, memberDisabledTools, memberRevision } from '../data/join'
import { type ConnectorToolsSaver, useConnectorToolsSave } from '../data/mutations'
import { type HostedConnectorsView, useConnectorTools } from '../data/queries'
import { localServerName } from '../derive'
import { categoryLabel, conflictDifference, toolDisplayName, UNCATEGORISED } from '../derive-tools'
import type { ConnectorCardModel, ToolRowModel, ToolsEditorPhase, ToolsEditorStatus } from '../types'
import { type ToolsEditor, useToolsEditor } from '../use-tools-editor'

import { useLocalTools } from './connection-sheet-parts'
import { ConnectionToolsToolbar } from './connection-tools-filter'
import { type ConnectionsCopy, useConnectionsCopy } from './connections-copy'
import {
  type ConnectionToolAreaGroup,
  type ConnectionToolsFilterState,
  EMPTY_CONNECTION_TOOLS_FILTER,
  filterConnectionTools,
  groupToolsByArea
} from './filter-logic'

/**
 * Figma `1560:28037` (`{App}` tools page), `1565:30694` (tool rows + active
 * chips) and `1565:31077` (Tool Filter). The page owns one
 * `ConnectionToolsFilterState` and one `useToolsEditor` draft, so a tool row and
 * a group's bulk action move the same dirty state and save together.
 */
export interface ConnectionToolsPageProps {
  agentControls: boolean
  card: ConnectorCardModel
  controller: McpServersController
  hosted: HostedConnectorsView
  onBack: () => void
  onDisconnect: () => void
  onReconnect: () => void
  onRemoveLocalServer?: () => void
  profile: ProfileScope
}

function useConnectionToolsSource(
  card: ConnectorCardModel,
  controller: McpServersController,
  hosted: HostedConnectorsView,
  profile: ProfileScope,
  agentControls: boolean
) {
  const revision = memberRevision(hosted.policy)
  const hostedConnected = card.ways.hosted?.connected === true
  const localInstalled = card.ways.local?.installed === true
  const localMode = !hostedConnected && localInstalled
  const name = localServerName(card)
  const tools = useConnectorTools(profile, localMode ? null : card.slug, hosted.listSlugs.has(card.slug))
  const localTools = useLocalTools(controller, card, localMode)
  const saver = useConnectorToolsSave(profile, card.slug, revision)

  const rows = useMemo(
    () => (localMode ? localTools.rows : connectorToolRows(hosted.policy, card.slug, tools.tools)),
    [card.slug, hosted.policy, localMode, localTools.rows, tools.tools]
  )
  const savedDisabled = useMemo(
    () => (localMode ? localTools.savedDisabled : memberDisabledTools(hosted.policy, card.slug)),
    [card.slug, hosted.policy, localMode, localTools.savedDisabled]
  )
  const editor = useToolsEditor({
    editorKey: localMode ? name : card.slug,
    onSave: localMode ? localTools.onSave : saver.onSave,
    savedDisabled,
    status: localMode ? localTools.status : tools.status,
    tools: rows
  })

  const connected = hostedConnected || localInstalled
  const canEdit = localMode
    ? card.plugin === undefined && agentControls
    : connected && agentControls && !hosted.rulesFailed && revision !== undefined

  return { canEdit, connected, editor, localMode, name, rows, saver, tools }
}

export function ConnectionToolsPage({
  agentControls,
  card,
  controller,
  hosted,
  onBack,
  onDisconnect,
  onReconnect,
  onRemoveLocalServer,
  profile
}: ConnectionToolsPageProps) {
  const copy = useConnectionsCopy()
  const { canEdit, connected, editor, localMode, name, rows, saver, tools } = useConnectionToolsSource(
    card,
    controller,
    hosted,
    profile,
    agentControls
  )
  const plugin = card.plugin

  const [filter, setFilter] = useState<ConnectionToolsFilterState>(EMPTY_CONNECTION_TOOLS_FILTER)
  const patchFilter = useCallback((patch: Partial<ConnectionToolsFilterState>) => {
    setFilter(previous => ({ ...previous, ...patch }))
  }, [])

  const phase = editor.phase
  const ready = isReadyPhase(phase)
  const onCount = rows.filter(tool => editor.isOn(tool.slug)).length
  const subtitle =
    canEdit && ready && rows.length > 0
      ? copy.tools.pageSummary(onCount, rows.length, card.name)
      : copy.tools.subtitle(card.name)

  const refreshTools = () => {
    if (localMode) {
      void controller.runProbe(name)

      return
    }

    tools.refresh()
  }

  const reconnect = localMode ? () => void controller.authenticate(name) : onReconnect

  return (
    <section className="conn-tools-page" data-slot="connection-tools-page">
      <button className="conn-back-link" onClick={onBack} type="button">
        <Codicon aria-hidden className="conn-back-glyph" name="arrow-left" size="0.875rem" /> {card.name}
      </button>

      <header className="conn-tools-header">
        <div className="conn-tools-title-row">
          <h1 className="conn-tools-title">{copy.tools.pageTitle(card.name)}</h1>
          {connected ? (
            <span className="conn-status-pill" data-conn-state="connected">
              {copy.card.connected}
            </span>
          ) : null}
          {connected ? (
            <ToolsMenu
              copy={copy}
              local={localMode}
              onDisconnect={onDisconnect}
              onReconnect={reconnect}
              onRefresh={refreshTools}
              onRemove={onRemoveLocalServer}
              plugin={plugin}
            />
          ) : null}
        </div>
        <p className="conn-tools-subtitle">{subtitle}</p>
      </header>

      {isReadyPhase(phase) ? (
        rows.length === 0 ? (
          <EmptyState
            className="conn-tools-empty"
            description={copy.tools.emptyBody}
            title={copy.tools.emptyTitle}
          />
        ) : (
          <>
            <ConnectionToolsToolbar
              appName={card.name}
              filter={filter}
              onFilterChange={patchFilter}
              tools={rows}
            />
            <ConnectionToolsBody
              canEdit={canEdit}
              copy={copy}
              editor={editor}
              filter={filter}
              onClearFilter={() => setFilter(EMPTY_CONNECTION_TOOLS_FILTER)}
              rows={rows}
            />
            <ToolsSaveBar copy={copy} editor={editor} saver={saver} />
          </>
        )
      ) : (
        <ToolsStatus
          card={card}
          copy={copy}
          onDisconnect={localMode ? (onRemoveLocalServer ?? onDisconnect) : onDisconnect}
          onReconnect={reconnect}
          phase={phase}
          retry={localMode ? () => void controller.runProbe(name) : tools.retry}
        />
      )}
    </section>
  )
}

function isReadyPhase(phase: ToolsEditorPhase): phase is 'conflict' | 'ready' | 'saving' {
  return phase === 'ready' || phase === 'saving' || phase === 'conflict'
}

function ToolsMenu({
  copy,
  local,
  onDisconnect,
  onReconnect,
  onRefresh,
  onRemove,
  plugin
}: {
  copy: ConnectionsCopy
  local: boolean
  onDisconnect: () => void
  onReconnect: () => void
  onRefresh: () => void
  onRemove?: () => void
  plugin?: string
}) {
  const canRemove = local ? plugin === undefined && onRemove !== undefined : true

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button aria-label={copy.card.menu} className="conn-tools-menu ml-auto" size="icon-sm" variant="ghost">
          <Codicon aria-hidden name="ellipsis" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        {local ? null : <DropdownMenuItem onSelect={onReconnect}>{copy.card.reconnect}</DropdownMenuItem>}
        <DropdownMenuItem onSelect={() => onRefresh()}>{copy.card.refreshTools}</DropdownMenuItem>
        {canRemove ? (
          <>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              onSelect={() => (local ? onRemove?.() : onDisconnect())}
              variant="destructive"
            >
              {local ? copy.card.remove : `${copy.card.disconnect}…`}
            </DropdownMenuItem>
          </>
        ) : null}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

interface ToolsStatusProps {
  card: ConnectorCardModel
  copy: ConnectionsCopy
  onDisconnect: () => void
  onReconnect: () => void
  phase: ToolsEditorStatus
  retry: () => void
}

function ToolsStatus({ card, copy, onDisconnect, onReconnect, phase, retry }: ToolsStatusProps) {
  const { t } = useI18n()
  const strings = t.connectorsPage.tools

  if (phase === 'loading') {
    return <ToolsSkeleton copy={copy} />
  }

  if (phase === 'unavailable') {
    return (
      <EmptyState
        action={
          <Button className="conn-cta-btn is-manage" onClick={retry} size="sm" variant="outline">
            {copy.errors.tryAgain}
          </Button>
        }
        className="conn-tools-error"
        description={copy.tools.errorBody}
        title={copy.tools.errorTitle}
      />
    )
  }

  if (phase === 'needsAuth' || phase === 'signedOut') {
    const needsAuth = phase === 'needsAuth'

    return (
      <EmptyState
        action={
          <Button className="conn-primary" onClick={onReconnect} size="sm" variant="default">
            {copy.card.reconnect}
          </Button>
        }
        className="conn-tools-error"
        description={needsAuth ? strings.needsAuthBody : strings.signedOutBody}
        title={needsAuth ? strings.needsAuthTitle(card.name) : strings.signedOutTitle}
      />
    )
  }

  if (phase === 'gone') {
    return (
      <EmptyState
        action={
          <Button className="conn-cta-btn is-manage" onClick={onDisconnect} size="sm" variant="outline">
            {copy.card.remove}
          </Button>
        }
        className="conn-tools-error"
        description={strings.goneBody}
        title={strings.goneTitle(card.name)}
      />
    )
  }

  return (
    <EmptyState
      className="conn-tools-empty"
      description={strings.offBody}
      title={strings.offTitle(card.name)}
    />
  )
}

function ToolsSkeleton({ copy }: { copy: ConnectionsCopy }) {
  return (
    <div aria-busy className="conn-tools-skeleton" role="status">
      <span className="sr-only">{copy.tools.loading}</span>
      {[0, 1, 2, 3, 4, 5].map(row => (
        <span className="conn-tools-skeleton-row" data-slot="skeleton" key={row} />
      ))}
    </div>
  )
}

interface ConnectionToolsBodyProps {
  canEdit: boolean
  copy: ConnectionsCopy
  editor: ToolsEditor
  filter: ConnectionToolsFilterState
  onClearFilter: () => void
  rows: readonly ToolRowModel[]
}

function ConnectionToolsBody({
  canEdit,
  copy,
  editor,
  filter,
  onClearFilter,
  rows
}: ConnectionToolsBodyProps) {
  const filtered = useMemo(() => filterConnectionTools(rows, filter), [filter, rows])
  const groups = useMemo(() => groupToolsByArea(filtered), [filtered])
  const [expanded, setExpanded] = useState<ReadonlySet<string>>(() => new Set())

  const toggleExpanded = useCallback((slug: string) => {
    setExpanded(previous => {
      const next = new Set(previous)

      if (next.has(slug)) {
        next.delete(slug)
      } else {
        next.add(slug)
      }

      return next
    })
  }, [])

  if (filtered.length === 0) {
    return (
      <EmptyState
        action={
          <Button className="conn-cta-btn is-manage" onClick={onClearFilter} size="sm" variant="outline">
            {copy.toolFilters.clear}
          </Button>
        }
        className="conn-tools-empty"
        description={copy.tools.noResultsBody}
        title={copy.tools.noResultsTitle}
      />
    )
  }

  return (
    <div className="conn-tools-groups" data-slot="connection-tools-groups">
      {groups.map(group => (
        <AreaGroup
          canEdit={canEdit}
          copy={copy}
          editor={editor}
          expanded={expanded}
          group={group}
          key={group.area}
          onToggleExpanded={toggleExpanded}
        />
      ))}
    </div>
  )
}

function AreaGroup({
  canEdit,
  copy,
  editor,
  expanded,
  group,
  onToggleExpanded
}: {
  canEdit: boolean
  copy: ConnectionsCopy
  editor: ToolsEditor
  expanded: ReadonlySet<string>
  group: ConnectionToolAreaGroup
  onToggleExpanded: (slug: string) => void
}) {
  const area = group.area === UNCATEGORISED ? copy.toolFilters.uncategorised : categoryLabel(group.area)
  const mutable = group.tools.filter(tool => tool.lockedBy === null)
  const onCount = mutable.filter(tool => editor.isOn(tool.slug)).length
  const allOn = mutable.length > 0 && onCount === mutable.length
  const action = allOn ? copy.bulk.turnAllOff : onCount === 0 ? copy.bulk.turnAllOn : copy.bulk.someOn

  // "Some on" turns every mutable tool on, as its label promises.
  const turnAll = () => {
    const target = !allOn

    for (const tool of mutable) {
      if (editor.isOn(tool.slug) !== target) {
        editor.toggle(tool.slug)
      }
    }
  }

  return (
    <section className="conn-tools-group" data-slot="connection-tools-group">
      <header className="conn-tools-group-header">
        <h2 className="conn-tools-group-title">{`${area} (${group.tools.length})`}</h2>
        {canEdit && mutable.length > 0 ? (
          <button
            aria-label={`${allOn ? copy.bulk.turnAllOff : copy.bulk.turnAllOn}: ${area}`}
            className="conn-tools-group-action"
            onClick={turnAll}
            type="button"
          >
            {action}
          </button>
        ) : null}
      </header>

      <div className="conn-tools-group-card">
        {group.tools.map(tool => (
          <ToolRowItem
            canEdit={canEdit}
            copy={copy}
            editor={editor}
            expanded={expanded.has(tool.slug)}
            key={tool.slug}
            onToggleExpanded={() => onToggleExpanded(tool.slug)}
            tool={tool}
          />
        ))}
      </div>
    </section>
  )
}

const FACET_BADGES = {
  destructive: { className: 'conn-badge conn-badge-destructive', symbol: '▲' },
  read: { className: 'conn-badge conn-badge-read', symbol: '○' },
  write: { className: 'conn-badge conn-badge-write', symbol: '◐' }
} as const

type KnownFacet = keyof typeof FACET_BADGES

function isKnownFacet(facet: string): facet is KnownFacet {
  return facet === 'read' || facet === 'write' || facet === 'destructive'
}

function ToolRowItem({
  canEdit,
  copy,
  editor,
  expanded,
  onToggleExpanded,
  tool
}: {
  canEdit: boolean
  copy: ConnectionsCopy
  editor: ToolsEditor
  expanded: boolean
  onToggleExpanded: () => void
  tool: ToolRowModel
}) {
  const { t } = useI18n()
  const strings = t.connectorsPage.tools
  const name = toolDisplayName(tool.name || tool.slug)
  const on = editor.isOn(tool.slug)
  const badge =
    on && isKnownFacet(tool.facet)
      ? { ...FACET_BADGES[tool.facet], label: copy.badges[tool.facet] }
      : null

  return (
    <div className="conn-tools-row" data-off={on ? undefined : ''} data-tool={tool.slug}>
      <div className="conn-tools-row-main">
        <button
          aria-expanded={expanded}
          aria-label={expanded ? strings.hideDetails(name) : strings.showDetails(name)}
          className="conn-tool-chevron"
          onClick={onToggleExpanded}
          type="button"
        >
          <DisclosureCaret open={expanded} />
        </button>

        <button
          aria-expanded={expanded}
          className="conn-tool-name-button"
          onClick={onToggleExpanded}
          type="button"
        >
          <span className="conn-tool-name">{name}</span>
        </button>

        <span className="conn-tools-row-badge">
          {badge ? (
            <span className={badge.className}>
              <span aria-hidden>{badge.symbol}</span> {badge.label}
            </span>
          ) : on ? null : (
            <span className="conn-badge conn-badge-off">{copy.badges.off}</span>
          )}
        </span>

        <ToolSwitch canEdit={canEdit} copy={copy} editor={editor} tool={tool} />
      </div>

      {expanded ? (
        <div className="conn-tool-details">
          {tool.description.trim() === '' ? null : (
            <p className="conn-tool-description">{tool.description}</p>
          )}
          <code className="conn-tool-slug">
            {copy.tools.toolNamePrefix} {tool.slug}
          </code>
        </div>
      ) : null}
    </div>
  )
}

function ToolSwitch({
  canEdit,
  copy,
  editor,
  tool
}: {
  canEdit: boolean
  copy: ConnectionsCopy
  editor: ToolsEditor
  tool: ToolRowModel
}) {
  const { t } = useI18n()
  const strings = t.connectorsPage.tools
  const on = editor.isOn(tool.slug)
  const locked = tool.lockedBy === 'org'
  const control = (
    <Switch
      aria-label={on ? strings.turnToolOff(tool.name) : strings.turnToolOn(tool.name)}
      checked={on}
      disabled={!canEdit || locked}
      onCheckedChange={() => editor.toggle(tool.slug)}
    />
  )

  if (!locked) {
    return <span className="conn-tools-switch">{control}</span>
  }

  return (
    <Tip label={copy.tools.lockedByOrg}>
      <span className="conn-tools-switch">{control}</span>
    </Tip>
  )
}

function ToolsSaveBar({
  copy,
  editor,
  saver
}: {
  copy: ConnectionsCopy
  editor: ToolsEditor
  saver: ConnectorToolsSaver
}) {
  const { t } = useI18n()
  const strings = t.connectorsPage.tools
  const difference = conflictDifference(saver.theirs ?? [], editor.local)
  const changed = editor.counts.off + editor.counts.backOn
  const saving = editor.phase === 'saving'
  const reloadTheirs = () => {
    editor.discard()
    saver.reload()
  }

  return (
    <>
      {editor.phase === 'conflict' ? (
        <div className="conn-conflict-bar" role="alert">
          <div className="conn-conflict-copy">
            <p className="conn-conflict-title">{strings.conflictTitle}</p>
            <p className="conn-conflict-body">
              {strings.conflictBody(difference.theyOff, difference.theyOn)}
            </p>
          </div>
          <div className="conn-save-actions">
            <Button className="conn-cta-btn is-manage" onClick={reloadTheirs} size="sm" variant="outline">
              {strings.conflictReload}
            </Button>
            <Button
              className="conn-primary"
              loading={saving}
              onClick={() => void editor.keepMine()}
              size="sm"
              variant="default"
            >
              {strings.conflictSave}
            </Button>
          </div>
        </div>
      ) : null}

      {editor.dirty && editor.phase !== 'conflict' ? (
        <div aria-label={copy.tools.unsavedChanges(changed)} className="conn-save-bar" role="region">
          <span className="conn-save-count">{copy.tools.unsavedChanges(changed)}</span>
          <div className="conn-save-actions">
            <Button
              className="conn-cta-btn is-manage"
              disabled={saving}
              onClick={editor.discard}
              size="sm"
              variant="ghost"
            >
              {copy.bulk.discard}
            </Button>
            <Button
              className="conn-primary"
              loading={saving}
              onClick={() => void editor.save()}
              size="sm"
              variant="default"
            >
              {copy.bulk.save}
            </Button>
          </div>
        </div>
      ) : null}
    </>
  )
}
