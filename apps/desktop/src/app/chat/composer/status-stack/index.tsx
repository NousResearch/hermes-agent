import { useStore } from '@nanostores/react'
import { type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router'

import { blurComposerInput } from '@/app/chat/composer/focus'
import { AGENTS_ROUTE } from '@/app/routes'
import { BillingBanner } from '@/components/billing-banner'
import { composerDockCard } from '@/components/chat/composer-dock'
import { StatusSection } from '@/components/chat/status-section'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import { Tip, TipKeybindLabel } from '@/components/ui/tooltip'
import { type Translations, useI18n } from '@/i18n'
import { parseTodoSnapshot } from '@/lib/todos'
import { useSessionSlice, useStoreSelector } from '@/lib/use-session-slice'
import { cn } from '@/lib/utils'
import { $billingBlock } from '@/store/billing-block'
import {
  $statusItemsBySession,
  type ComposerStatusItem,
  dismissBackgroundProcess,
  groupStatusItems,
  refreshBackgroundProcesses,
  resetBackgroundPollingGuard,
  type StatusGroup,
  stopBackgroundProcess
} from '@/store/composer-status'
import { refreshSessionGoal } from '@/store/goals'
import { $previewStatusBySession, dismissPreviewArtifact } from '@/store/preview-status'
import { $threadScrolledUp } from '@/store/thread-scroll'
import {
  createTodoMutationController,
  humanTodoTarget,
  todoGatewayErrorCode,
  type TodoGatewayRequest,
  type TodoHumanStatus,
  TodoMutationFailure
} from '@/store/todo-mutation'
import {
  $sessionTodoSnapshots,
  applyOptimisticTodoStatus,
  currentSessionTodoSnapshot,
  setSessionTodoSnapshot
} from '@/store/todos'
import { openSessionInNewWindow } from '@/store/windows'

import { PreviewStatusRow } from './preview-row'
import { StatusItemRow } from './status-row'

// Slow safety-net poll for silent exits (processes without notify_on_complete
// emit no event when they die). Only armed while a running row is on screen.
const BACKGROUND_POLL_MS = 5_000

// A localhost/loopback preview is only meaningful while its dev server is up, so
// we tie it to a live background process rather than persisting dismissals or
// letting dead URLs pile up. File previews (a real on-disk artifact) stand alone.
const isLocalhostPreview = (target: string): boolean => /\b(?:localhost|127\.0\.0\.1|0\.0\.0\.0)\b/i.test(target)

// Real codicons per group (no sparkles): a checklist for todos, the agent glyph
// for subagents, a background process glyph for background tasks.
const GROUP_ICON: Record<StatusGroup['type'], string> = {
  goal: 'target',
  todo: 'checklist',
  subagent: 'agent',
  background: 'server-process'
}

const groupLabel = (group: StatusGroup, s: Translations['statusStack']) => {
  if (group.type === 'goal') {
    const status = group.items[0]?.goalStatus

    return status === 'paused'
      ? s.goalPaused
      : status === 'waiting'
        ? s.goalWaiting
        : status === 'done'
          ? s.goalDone
          : s.goalActive
  }

  if (group.type === 'todo') {
    const counted = group.items.filter(item => item.todoStatus !== 'cancelled')

    return s.todos(counted.filter(item => item.todoStatus === 'completed').length, counted.length)
  }

  return group.type === 'subagent' ? s.subagents(group.items.length) : s.background(group.items.length)
}

const hasRunningTodo = (group: StatusGroup) =>
  group.type === 'todo' && group.items.some(item => item.todoStatus === 'in_progress' && item.state === 'running')

interface TodoConfirmTarget {
  authority: { generation: number; revision: number }
  item: ComposerStatusItem
  status: TodoHumanStatus
}

type TodoSyncState = 'failed' | 'ready' | 'session' | 'syncing' | 'unsupported'

interface ComposerStatusStackProps {
  /** The queue, built by the composer (it owns the queue's callbacks). Rendered
   *  as the last group so it stays fused to the composer like before. */
  queue: ReactNode
  requestGateway?: TodoGatewayRequest
  sessionId: null | string
}

/**
 * The status "sink" above the composer: one card (the queue's chrome) holding
 * every session-scoped status — subagents, background tasks, queue — grouped by
 * type and separated by light dividers. Collapses to nothing when empty.
 */
export function ComposerStatusStack({ queue, requestGateway, sessionId }: ComposerStatusStackProps) {
  const { t } = useI18n()
  const navigate = useNavigate()
  // Subscribe to THIS session's slice only. Both maps churn on other
  // sessions' activity (subagent ticks, background polls, preview updates in
  // any tile); a whole-map `useStore` re-rendered every mounted stack — one
  // per open tile — on all of it. The per-key arrays are referentially stable
  // across unrelated writes, so the slice hook bails out unless OUR session's
  // items actually changed.
  const items = useSessionSlice($statusItemsBySession, sessionId)
  const previews = useSessionSlice($previewStatusBySession, sessionId)
  const scrolledUp = useStore($threadScrolledUp)
  const billing = useStore($billingBlock)

  const todoAuthority = useStoreSelector($sessionTodoSnapshots, snapshots =>
    sessionId ? (snapshots[sessionId] ?? null) : null
  )

  const [todoConfirm, setTodoConfirm] = useState<TodoConfirmTarget | null>(null)
  const [pendingTodoItemId, setPendingTodoItemId] = useState<string | null>(null)
  const [todoCollapsed, setTodoCollapsed] = useState(false)
  const [todoSyncState, setTodoSyncState] = useState<TodoSyncState>(requestGateway ? 'syncing' : 'unsupported')
  const todoActionOriginRef = useRef<HTMLButtonElement | null>(null)
  const todoDisclosureRef = useRef<HTMLButtonElement | null>(null)
  const todoSyncControllerRef = useRef<AbortController | null>(null)

  const todoController = useMemo(
    () =>
      requestGateway
        ? createTodoMutationController({
            applySnapshot: setSessionTodoSnapshot,
            getSnapshot: currentSessionTodoSnapshot,
            optimisticStatus: applyOptimisticTodoStatus,
            request: requestGateway
          })
        : null,
    [requestGateway]
  )

  const groups = useMemo(() => groupStatusItems(items), [items])

  const refreshTodoSnapshot = useCallback(async () => {
    if (!sessionId || !requestGateway) {
      setTodoSyncState('unsupported')

      return
    }

    todoSyncControllerRef.current?.abort()
    const controller = new AbortController()
    todoSyncControllerRef.current = controller
    setTodoSyncState('syncing')

    try {
      const raw = await requestGateway<unknown>('todo.snapshot', { session_id: sessionId })
      const snapshot = parseTodoSnapshot(raw)

      if (controller.signal.aborted) {
        return
      }

      if (!snapshot || snapshot.session_id !== sessionId) {
        setTodoSyncState('failed')

        return
      }

      setSessionTodoSnapshot(snapshot)
      setTodoSyncState('ready')
    } catch (error) {
      if (controller.signal.aborted) {
        return
      }

      const code = todoGatewayErrorCode(error)
      setTodoSyncState(code === -32601 ? 'unsupported' : code === 4001 ? 'session' : 'failed')
    }
  }, [requestGateway, sessionId])

  // Seed from the registry on session open; event-driven refreshes (terminal /
  // process tool completions) live in use-message-stream.
  useEffect(() => {
    if (sessionId) {
      // Opening/rebinding a session is a fresh runtime binding: clear any
      // gone-latch left by a previous runtime under this id so the poll below
      // is allowed to run again (see resetBackgroundPollingGuard).
      resetBackgroundPollingGuard(sessionId)
      void refreshBackgroundProcesses(sessionId)
      void refreshSessionGoal(sessionId)
    }
  }, [sessionId])

  useEffect(() => {
    setTodoConfirm(null)
    setPendingTodoItemId(null)
    setTodoCollapsed(false)

    if (!sessionId || !requestGateway) {
      setTodoSyncState('unsupported')

      return
    }

    void refreshTodoSnapshot()

    return () => {
      todoSyncControllerRef.current?.abort()
    }
  }, [refreshTodoSnapshot, requestGateway, sessionId])

  const hasRunningBackground = groups.some(g => g.type === 'background' && g.items.some(i => i.state === 'running'))

  // Drop localhost previews once no dev server is left running — that's what made
  // dead `localhost:5174` chips stick around. On-disk file previews are kept.
  const visiblePreviews = previews.filter(item => hasRunningBackground || !isLocalhostPreview(item.target))

  useEffect(() => {
    if (!sessionId || !hasRunningBackground) {
      return
    }

    const timer = setInterval(() => void refreshBackgroundProcesses(sessionId), BACKGROUND_POLL_MS)

    return () => clearInterval(timer)
  }, [hasRunningBackground, sessionId])

  const openAgents = () => navigate(AGENTS_ROUTE)

  const openSubagent = (item: ComposerStatusItem) =>
    item.sessionId ? void openSessionInNewWindow(item.sessionId, { watch: true }) : openAgents()

  const requestTodoAction = useCallback((item: ComposerStatusItem, origin: HTMLButtonElement) => {
    if (!item.todoStatus || !item.todoItemId || !todoAuthority || todoSyncState !== 'ready') {
      return
    }

    todoActionOriginRef.current = origin
    setTodoConfirm({
      authority: { generation: todoAuthority.generation, revision: todoAuthority.revision },
      item,
      status: humanTodoTarget(item.todoStatus)
    })
  }, [todoAuthority, todoSyncState])

  const closeTodoConfirm = useCallback(() => {
    setTodoConfirm(null)
    setPendingTodoItemId(null)
    queueMicrotask(() => {
      const origin = todoActionOriginRef.current
      const focusTarget = origin?.isConnected ? origin : todoDisclosureRef.current
      focusTarget?.focus()
    })
  }, [])

  const confirmTodoAction = useCallback(async () => {
    if (!todoConfirm?.item.todoItemId || !sessionId || !todoController) {
      return
    }

    const itemId = todoConfirm.item.todoItemId
    setPendingTodoItemId(itemId)

    try {
      const result = await todoController.run({
        content: todoConfirm.item.title,
        expectedGeneration: todoConfirm.authority.generation,
        expectedRevision: todoConfirm.authority.revision,
        itemId,
        sessionId,
        status: todoConfirm.status
      })

      const allTerminal =
        result.todos.length > 0 &&
        result.todos.every(todo => todo.status === 'completed' || todo.status === 'cancelled')

      setTodoCollapsed(allTerminal)
    } catch (error) {
      const kind = error instanceof TodoMutationFailure ? error.kind : 'unavailable'

      const message =
        kind === 'stale'
          ? t.statusStack.taskChanged
          : kind === 'missing'
            ? t.statusStack.taskMissing
            : kind === 'session'
              ? t.statusStack.taskSessionMissing
              : t.statusStack.taskUpdateFailed

      throw new Error(message, { cause: error })
    } finally {
      setPendingTodoItemId(null)
    }
  }, [sessionId, t.statusStack, todoConfirm, todoController])

  const todoMutationUnavailableLabel =
    todoSyncState === 'failed'
      ? t.statusStack.taskSyncFailed
      : todoSyncState === 'session'
        ? t.statusStack.taskSessionMissing
        : todoSyncState === 'unsupported'
          ? t.statusStack.taskUpdatesUnavailable
          : t.statusStack.syncingTask

  const todoRetry =
    requestGateway && (todoSyncState === 'failed' || todoSyncState === 'session') ? (
      <Button
        className="text-muted-foreground/75 hover:text-foreground/90"
        onClick={() => void refreshTodoSnapshot()}
        size="micro"
        type="button"
        variant="text"
      >
        {t.statusStack.retryTaskSync}
      </Button>
    ) : undefined

  // Preview links live as child rows of the background group — a localhost dev
  // server and its preview are the same thing — so they no longer float as an
  // odd, differently-indented standalone block under the stack.
  const previewRows =
    visiblePreviews.length > 0 && sessionId
      ? visiblePreviews.map(item => (
          <PreviewStatusRow item={item} key={item.id} onDismiss={id => dismissPreviewArtifact(sessionId, id)} />
        ))
      : []

  const hasBackgroundGroup = groups.some(g => g.type === 'background')

  const previewBlock = <div className="px-1 py-0.5">{previewRows}</div>

  const sections: { key: string; node: ReactNode }[] = []

  // Billing wall sits at the very top of the stack — it's the most important
  // thing above the composer when the account is out of credits. Rendered here
  // (not as a composer-disable) so slash commands stay usable.
  if (billing && sessionId && billing.sessionId === sessionId) {
    sections.push({ key: 'billing', node: <BillingBanner sessionId={sessionId} /> })
  }

  for (const group of groups) {
    sections.push({
      key: group.type,
      node: (
        <StatusSection
          accessory={
            group.type === 'todo' ? (
              todoRetry
            ) : group.type === 'subagent' ? (
              <Tip label={<TipKeybindLabel actionId="nav.agents" text={t.statusStack.agents} />}>
                <Button
                  className="text-muted-foreground/75 hover:text-foreground/90"
                  onClick={openAgents}
                  size="micro"
                  type="button"
                  variant="text"
                >
                  {t.statusStack.agents}
                </Button>
              </Tip>
            ) : undefined
          }
          collapsed={group.type === 'todo' ? todoCollapsed : undefined}
          collapsedIndicator={
            hasRunningTodo(group) ? (
              <GlyphSpinner
                ariaLabel={t.statusStack.running}
                className="text-[0.8rem] leading-none text-muted-foreground/80"
                spinner="braille"
              />
            ) : undefined
          }
          defaultCollapsed={group.type !== 'todo' && group.type !== 'goal'}
          icon={<Codicon className="text-muted-foreground/70" name={GROUP_ICON[group.type]} size="0.8rem" />}
          label={groupLabel(group, t.statusStack)}
          onCollapsedChange={group.type === 'todo' ? setTodoCollapsed : undefined}
          triggerRef={group.type === 'todo' ? todoDisclosureRef : undefined}
        >
          {group.items.map(item => (
            <StatusItemRow
              item={item}
              key={item.id}
              onDismiss={sessionId ? id => dismissBackgroundProcess(sessionId, id) : undefined}
              onOpen={() => openSubagent(item)}
              onStop={sessionId ? id => void stopBackgroundProcess(sessionId, id) : undefined}
              onTodoAction={requestTodoAction}
              todoMutationEnabled={Boolean(requestGateway && todoAuthority && todoSyncState === 'ready')}
              todoMutationPending={Boolean(item.todoItemId && pendingTodoItemId === item.todoItemId)}
              todoMutationUnavailableLabel={todoMutationUnavailableLabel}
            />
          ))}
        </StatusSection>
      )
    })

    // Preview links belong to the background group (a localhost dev server and
    // its preview are the same thing), but they must stay VISIBLE even when that
    // group is collapsed — the whole point is a one-tap open. Render them as an
    // always-visible block right after the background section, not as collapsible
    // children that get swallowed the moment a background task appears.
    if (group.type === 'background' && previewRows.length > 0) {
      sections.push({ key: 'preview', node: previewBlock })
    }
  }

  // No background group to host them (e.g. a standalone on-disk file preview):
  // still render them as their own always-visible block.
  if (previewRows.length > 0 && !hasBackgroundGroup) {
    sections.push({ key: 'preview', node: previewBlock })
  }

  if (queue) {
    sections.push({ key: 'queue', node: queue })
  }

  // Micro actions are the TOP-MOST thing in the whole overlay lane — above the
  // status card, above the billing wall, above everything. They're the only
  // rows up here you press instead of read, so nothing may ever stack on top
  // of them. Rendered outside the card (below) so the pills float.
  const visible = sections.length > 0
  const hasTodoRows = groups.some(group => group.type === 'todo' && group.items.length > 0)

  // No height to publish: the stack is an in-flow child of the composer dock,
  // so the dock's own measurement (--composer-measured-height) already covers
  // it and the thread clears both with one number.

  if (!visible && !todoConfirm) {
    return null
  }

  const completingTodo = todoConfirm?.status === 'completed'

  return (
    <>
      {visible && (
        <div
          // In flow in the dock column, directly above the composer. The dock is
          // bottom-anchored, so this grows upward over the thread without needing
          // to be positioned — and it shares the dock's left edge for free.
          className="flex max-h-[40vh] min-h-0 flex-col overflow-y-auto"
          data-slot="composer-status-stack"
          onPointerDownCapture={() => blurComposerInput()}
        >
          {/* The card paints the shared --composer-fill (rest / scrolled / focused
              all match the composer surface by construction); on scroll we only
              ghost the CONTENT — element opacity on the card would kill the blur.
              Rounded top, square bottom; the bottom border is TRANSPARENT — the
              composer surface's visible top border (which sits at a higher z) is the
              single shared seam, so the two read as one fused capsule. */}
          <div
            className={cn(
              composerDockCard('top'),
              // Inset (mx-2) so the stack reads slightly narrower than the composer
              // surface below it — the original look.
              'mx-2 overflow-hidden rounded-b-none border-b border-b-transparent pt-0.5',
              'transition-opacity duration-200 ease-out',
              scrolledUp && !hasTodoRows ? 'opacity-30 group-hover/composer:opacity-100' : 'opacity-100'
            )}
          >
            {sections.map(section => (
              <div key={section.key}>{section.node}</div>
            ))}
          </div>
        </div>
      )}
      <ConfirmDialog
        busyLabel={completingTodo ? t.statusStack.markingDone : t.statusStack.reopening}
        confirmLabel={completingTodo ? t.statusStack.markDone : t.statusStack.reopen}
        description={
          todoConfirm ? (
            <span className="space-y-2">
              <span className="block break-words text-foreground">{todoConfirm.item.title}</span>
              <span className="block">
                {completingTodo ? t.statusStack.markDoneDescription : t.statusStack.reopenDescription}
              </span>
            </span>
          ) : undefined
        }
        doneLabel={completingTodo ? t.statusStack.markedDone : t.statusStack.reopened}
        onClose={closeTodoConfirm}
        onConfirm={confirmTodoAction}
        open={Boolean(todoConfirm)}
        title={completingTodo ? t.statusStack.markDoneTitle : t.statusStack.reopenTitle}
      />
    </>
  )
}
