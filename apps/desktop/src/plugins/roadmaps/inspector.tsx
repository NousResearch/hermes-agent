/**
 * Roadmaps plugin — Inspector panel.
 *
 * Details of the selected node (description, todos, parent) plus versioned
 * manual steering: claim / progress / complete / block / unblock and todo
 * start / finish / cancel / reopen. Every write is a versioned mutation
 * (expected_version = the active version of the loaded snapshot) via the REST
 * door; the caller reloads an authoritative snapshot after each success.
 * Backend messages are never rendered — stable English guidance by code only.
 */

import { Button, cn, Codicon, CopyButton, EmptyState, Input, Separator, StatusDot } from '@hermes/plugin-sdk'
import { useCallback, useEffect, useMemo, useState } from 'react'

import { advanceNode, blockNode, claimNode, completeNode, unblockNode, updateProgress, updateTodo } from './api'
import {
  formatDate,
  isValidIdentifier,
  type MutationError,
  mutationErrorCopy,
  nodeBlockers,
  nodeBlocks,
  nodeDependants,
  nodeDepsInfo,
  nodeLabel,
  plural,
  rpcError,
  validateProgress
} from './data'
import type { RoadmapTodo, RoadmapVersion, Scope, SnapshotResponse } from './types'
import { NodeStateTag, SectionTitle } from './ui'

type NodeMutation = (profile: string, projectId: string, roadmapId: string, nodeId: string, body: Record<string, unknown>) => Promise<unknown>

const NODE_OPS: Record<string, NodeMutation> = {
  claim_node: claimNode,
  advance: advanceNode,
  update_progress: updateProgress,
  complete_node: completeNode,
  block_node: blockNode,
  unblock_node: unblockNode
}

function MutationButton({
  label,
  codicon,
  onClick,
  busy,
  disabled,
  tone
}: {
  label: string
  codicon: string
  onClick: () => void
  busy: boolean
  disabled?: boolean
  tone?: 'danger'
}) {
  return (
    <Button
      className="gap-1"
      disabled={disabled || busy}
      onClick={onClick}
      size="xs"
      type="button"
      variant={tone === 'danger' ? 'destructive' : 'secondary'}
    >
      <Codicon name={codicon} size="0.75rem" />
      {label}
    </Button>
  )
}

export function TodoRow({
  todo,
  onMutate,
  busyTodoId
}: {
  todo: RoadmapTodo
  onMutate: (todoId: string, state: string) => void
  busyTodoId: string | null
}) {
  const done = todo.state === 'done' || todo.state === 'cancelled'

  return (
    <div className="flex items-center gap-2 px-0.5 py-0.5 text-xs">
      <StatusDot tone={done ? 'muted' : 'good'} />
      <span className={cn('min-w-0 flex-1 truncate', done && 'line-through opacity-60')}>{todo.title}</span>
      <div className="flex shrink-0 items-center gap-1">
        {todo.state === 'open' ? (
          <MutationButton busy={busyTodoId === todo.todo_id} codicon="play" label="Start" onClick={() => onMutate(todo.todo_id, 'in_progress')} />
        ) : null}
        {todo.state === 'in_progress' ? (
          <MutationButton busy={busyTodoId === todo.todo_id} codicon="pass-filled" label="Finish" onClick={() => onMutate(todo.todo_id, 'done')} />
        ) : null}
        {!done ? (
          <MutationButton busy={busyTodoId === todo.todo_id} codicon="close" label="Cancel" onClick={() => onMutate(todo.todo_id, 'cancelled')} tone="danger" />
        ) : null}
        {todo.state === 'cancelled' ? (
          <MutationButton busy={busyTodoId === todo.todo_id} codicon="debug-restart" label="Reopen" onClick={() => onMutate(todo.todo_id, 'open')} />
        ) : null}
      </div>
    </div>
  )
}

/** Relation chips: "Blockers / Dependants / Blocks" compact labeled rows. */
function RelationChips({
  title,
  codicon,
  items,
  onSelect,
  destructive
}: {
  title: string
  codicon: string
  items: Array<{ id: string; label: string; hint?: string }>
  onSelect: (id: string) => void
  destructive?: boolean
}) {
  if (items.length === 0) {return null}

  return (
    <div className="flex items-start gap-2 text-[0.625rem]">
      <span className="mt-px inline-flex w-16 shrink-0 items-center gap-1 font-medium uppercase tracking-wide text-(--ui-text-tertiary)">
        <Codicon name={codicon} size="0.65rem" />
        {title}
      </span>
      <div className="flex min-w-0 flex-wrap gap-1">
        {items.map((it) => (
          <button
            className={cn(
              'min-w-0 max-w-48 truncate rounded-[3px] px-1 py-px transition-colors',
              destructive ? 'text-destructive hover:bg-(--chrome-action-hover)' : 'text-(--ui-text-secondary) hover:bg-(--chrome-action-hover) hover:text-foreground'
            )}
            key={it.id}
            onClick={() => onSelect(it.id)}
            title={it.hint}
            type="button"
          >
            {it.label}
          </button>
        ))}
      </div>
    </div>
  )
}

export function Inspector({
  snapshot,
  version,
  nodeId,
  scope,
  onMutated,
  compact,
  actor,
  setActor,
  onSelect
}: {
  snapshot: SnapshotResponse | null | undefined
  version: RoadmapVersion | null | undefined
  nodeId: string
  scope: Scope | null
  onMutated: () => void
  compact: boolean
  actor: string
  setActor: (v: string) => void
  onSelect: (id: string) => void
}) {
  const [progressInput, setProgressInput] = useState('')
  const [reason, setReason] = useState('')
  const [busyOp, setBusyOp] = useState<string | null>(null)
  const [busyTodoId, setBusyTodoId] = useState<string | null>(null)
  const [error, setError] = useState<MutationError | null>(null)

  const node = (version?.nodes ?? []).find((n) => n.node_id === nodeId) ?? null
  const todos = (version?.todos ?? []).filter((t) => t.node_id === nodeId)
  const expectedVersion = snapshot?.roadmap?.active_version

  const deps = useMemo(() => (node ? nodeDepsInfo(node, version) : null), [node, version])
  const dependants = useMemo(() => (node ? nodeDependants(node, version) : []), [node, version])
  const blockers = useMemo(() => (node ? nodeBlockers(node, version) : []), [node, version])
  const blocks = useMemo(() => (node ? nodeBlocks(node, version) : []), [node, version])

  // Reset transient form/error state when the inspected node changes.
  useEffect(() => {
    setProgressInput('')
    setReason('')
    setError(null)
  }, [nodeId])

  const guardActor = useCallback((): boolean => {
    const sent = actor.trim() || 'user'

    if (isValidIdentifier(sent)) {return true}
    setError({
      code: null,
      hint: 'Actor must be a valid identifier: non-empty, at most 128 characters, no control characters.'
    })

    return false
  }, [actor])

  const mutate = useCallback(
    async (op: string, extra?: Record<string, unknown>) => {
      if (!node || !scope || !guardActor()) {return}

      if (op === 'update_progress') {
        const p = Number(extra?.progress)

        if (!validateProgress(p)) {
          setError({ code: null, hint: 'Progress must be an integer between 0 and 100.' })

          return
        }
      }

      setBusyOp(op)
      setError(null)

      try {
        await NODE_OPS[op](scope.profile, scope.projectId, scope.roadmapId, node.node_id, {
          actor: actor.trim() || 'user',
          expected_version: expectedVersion,
          ...(extra ?? {})
        })
        onMutated()
      } catch (err) {
        setError({ code: rpcError(err).code })
      } finally {
        setBusyOp(null)
      }
    },
    [actor, expectedVersion, guardActor, node, onMutated, scope]
  )

  const mutateTodo = useCallback(
    async (todoId: string, state: string) => {
      if (!scope || !guardActor()) {return}
      setBusyTodoId(todoId)
      setError(null)

      try {
        await updateTodo(scope.profile, scope.projectId, scope.roadmapId, todoId, {
          actor: actor.trim() || 'user',
          state,
          expected_version: expectedVersion
        })
        onMutated()
      } catch (err) {
        setError({ code: rpcError(err).code })
      } finally {
        setBusyTodoId(null)
      }
    },
    [actor, expectedVersion, guardActor, onMutated, scope]
  )

  if (!node) {
    return <EmptyState description="Pick a node in the Thread, Map, or Milestones view." title="No node selected" />
  }

  const ec = mutationErrorCopy(error)

  return (
    <div className="flex flex-col gap-2">
      <SectionTitle>Inspector</SectionTitle>

      <div className="flex items-start justify-between gap-2 px-0.5">
        <div className="min-w-0">
          <div className="flex items-center gap-1 text-[0.625rem] uppercase tracking-wide text-(--ui-text-tertiary)">
            <span className="truncate">{`${node.kind} · ${node.node_id}`}</span>
            <CopyButton
              appearance="icon"
              buttonSize="icon-xs"
              buttonVariant="ghost"
              label="Copy node ID"
              text={node.node_id}
              title="Copy node ID"
            />
          </div>
          <div className="truncate text-[0.8125rem] font-medium">{nodeLabel(node)}</div>
        </div>
        <NodeStateTag state={node.state} />
      </div>

      {node.description ? (
        <p className="whitespace-pre-wrap break-words px-0.5 text-xs leading-relaxed text-(--ui-text-tertiary)">{node.description}</p>
      ) : null}

      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 px-0.5 text-[0.625rem] text-(--ui-text-tertiary)">
        <span>{`Progress: ${node.progress ?? 0} %`}</span>
        {node.owner_agent ? <span>{`Owner: ${node.owner_agent}`}</span> : <span>Owner: —</span>}
        {node.parent_node_id ? <span>{`Parent: ${node.parent_node_id}`}</span> : null}
        {node.created_at ? <span className="tabular-nums">{formatDate(node.created_at)}</span> : null}
      </div>

      {/* Dependencies — the depends_on drill-down (satisfied or not). */}
      <div className="flex flex-col gap-0.5 px-0.5">
        <SectionTitle
          right={
            deps ? (
              <span
                className={cn(
                  'tabular-nums',
                  deps.satisfied === deps.total ? 'text-(--ui-text-tertiary)' : 'text-amber-500/90 dark:text-amber-300/90'
                )}
              >
                {deps.total === 0 ? 'none' : `${deps.satisfied}/${deps.total} satisfied`}
              </span>
            ) : null
          }
        >
          Dependencies
        </SectionTitle>
        {deps && deps.total > 0 ? (
          <div className="flex flex-col divide-y divide-(--ui-stroke-tertiary)">
            {deps.deps.map((d) => (
              <div className="flex items-center gap-1.5 py-0.5 text-[0.625rem]" key={d.targetId}>
                <Codicon
                  className={d.satisfied ? 'shrink-0 text-(--ui-accent)' : 'shrink-0 text-amber-500/90 dark:text-amber-300/90'}
                  name={d.satisfied ? 'check' : 'hourglass'}
                  size="0.65rem"
                />
                {d.target ? (
                  <button className="min-w-0 truncate hover:underline" onClick={() => onSelect(d.target!.node_id)} type="button">
                    {nodeLabel(d.target)}
                  </button>
                ) : (
                  <span className="min-w-0 truncate font-mono">{d.targetId}</span>
                )}
                <span className="ml-auto shrink-0 text-(--ui-text-quaternary)">{d.target ? d.target.state : 'missing'}</span>
              </div>
            ))}
          </div>
        ) : (
          <div className="px-0.5 text-[0.625rem] text-(--ui-text-quaternary)">No depends_on relations on this node.</div>
        )}
      </div>

      {/* Graph relations — blockers in, dependants, and what this node blocks. */}
      <div className="flex flex-col gap-1 px-0.5">
        <RelationChips
          codicon="debug-disconnect"
          destructive
          items={blockers.map((b) => ({ id: b.from.node_id, label: nodeLabel(b.from), hint: b.reason ?? undefined }))}
          onSelect={onSelect}
          title="Blockers"
        />
        <RelationChips codicon="arrow-down" items={dependants.map((n) => ({ id: n.node_id, label: nodeLabel(n) }))} onSelect={onSelect} title="Dependants" />
        <RelationChips codicon="arrow-up" items={blocks.map((n) => ({ id: n.node_id, label: nodeLabel(n) }))} onSelect={onSelect} title="Blocks" />
      </div>

      {todos.length > 0 ? (
        <div className="flex flex-col gap-0.5 px-0.5">
          <SectionTitle right={<span className="tabular-nums text-(--ui-text-quaternary)">{plural(todos.length, 'todo')}</span>}>
            Todos
          </SectionTitle>
          <div className="flex flex-col divide-y divide-(--ui-stroke-tertiary)">
            {todos.map((t) => (
              <TodoRow busyTodoId={busyTodoId} key={t.todo_id} onMutate={(todoId, state) => void mutateTodo(todoId, state)} todo={t} />
            ))}
          </div>
        </div>
      ) : null}

      <Separator className="my-0.5" />

      {/* Actor + expected_version context row — always visible before acting. */}
      <div className="flex flex-wrap items-center gap-2 px-0.5">
        <label className="flex items-center gap-1.5 text-[0.625rem] text-(--ui-text-tertiary)">
          Actor
          <Input
            aria-label="Actor"
            className="h-6 w-28 px-1.5 text-xs"
            onChange={(ev) => setActor(ev.target.value)}
            spellCheck={false}
            value={actor}
          />
        </label>
        <span className="font-mono text-[0.6rem] text-(--ui-text-quaternary)">{`expected_version = ${String(expectedVersion)}`}</span>
      </div>

      {error && ec ? (
        <div className="flex items-start gap-2 rounded-[3px] bg-destructive/10 px-2 py-1.5 text-xs text-destructive">
          <Codicon className="mt-px shrink-0" name="error" size="0.85rem" />
          <div className="min-w-0 flex-1">
            <div className="font-medium">{`${ec.title}${ec.code != null ? ` (code ${ec.code})` : ''}`}</div>
            <div className="mt-0.5 opacity-90">{ec.hint}</div>
          </div>
          {error.code === 5064 || error.code === 5065 ? (
            <Button onClick={onMutated} size="xs" type="button" variant="secondary">
              Reload snapshot
            </Button>
          ) : null}
        </div>
      ) : null}

      {/* Node actions — availability mirrors the node's lifecycle state. */}
      <div className="flex flex-wrap items-center gap-1.5 px-0.5">
        <MutationButton
          busy={busyOp === 'claim_node'}
          codicon="person-add"
          disabled={node.state !== 'ready'}
          label="Claim"
          onClick={() => void mutate('claim_node')}
        />
        <div className="flex items-center gap-1.5">
          <Input
            aria-label="Progress (0-100)"
            className="h-6 w-16 px-1.5 text-xs tabular-nums"
            inputMode="numeric"
            onChange={(ev) => setProgressInput(ev.target.value.replace(/[^0-9]/g, ''))}
            placeholder="0-100"
            value={progressInput}
          />
          <MutationButton
            busy={busyOp === 'update_progress'}
            codicon="arrow-up"
            disabled={node.state !== 'in_progress' || progressInput === ''}
            label="Progress"
            onClick={() => void mutate('update_progress', { progress: Number(progressInput) })}
          />
        </div>
        <MutationButton
          busy={busyOp === 'complete_node'}
          codicon="pass-filled"
          disabled={node.state !== 'in_progress'}
          label="Complete"
          onClick={() => void mutate('complete_node')}
        />
        {node.state === 'blocked' ? (
          <MutationButton busy={busyOp === 'unblock_node'} codicon="debug-restart" label="Unblock" onClick={() => void mutate('unblock_node')} />
        ) : (
          <div className="flex items-center gap-1.5">
            <Input
              aria-label="Block reason"
              className="h-6 w-40 px-1.5 text-xs"
              onChange={(ev) => setReason(ev.target.value)}
              placeholder={compact ? 'reason…' : 'block reason (required)'}
              value={reason}
            />
            <MutationButton
              busy={busyOp === 'block_node'}
              codicon="debug-disconnect"
              disabled={(node.state === 'ready' || node.state === 'in_progress') && !reason.trim()}
              label="Block"
              onClick={() => void mutate('block_node', { reason: reason.trim() })}
              tone="danger"
            />
          </div>
        )}
      </div>
    </div>
  )
}
