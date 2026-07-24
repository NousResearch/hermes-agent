import { useCallback, useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { getKanbanBoard, type KanbanBoardResponse, type KanbanBoardSummary, listKanbanBoards } from '@/hermes'
import { useI18n } from '@/i18n'
import { Clipboard, RefreshCw } from '@/lib/icons'
import { notifyError } from '@/store/notifications'

import { PanelEmpty } from '../overlays/panel'

import { Pill, SectionHeading, SettingsContent } from './primitives'

type BoardsStatus = 'error' | 'loading' | 'ready' | 'unavailable'

interface KanbanSettingsProps {
  gatewayId?: string
}

function EmptyState({ description, title }: { description: string; title: string }) {
  return (
    <div className="flex min-h-64">
      <PanelEmpty description={description} title={title} />
    </div>
  )
}

function LoadingState({ label }: { label: string }) {
  return (
    <div className="flex min-h-64">
      <PanelEmpty icon="loading~spin" title={label} />
    </div>
  )
}

function taskTitle(task: unknown, fallback: string): string {
  if (!task || typeof task !== 'object') {
    return fallback
  }

  const value = (task as { id?: unknown; title?: unknown }).title ?? (task as { id?: unknown }).id

  return typeof value === 'string' && value.trim() ? value : fallback
}

function taskMeta(task: unknown): string {
  if (!task || typeof task !== 'object') {
    return ''
  }

  const value = task as { assignee?: unknown; priority?: unknown; status?: unknown }

  return [
    value.status,
    value.assignee ? `@${String(value.assignee)}` : null,
    value.priority !== undefined ? `p${String(value.priority)}` : null
  ]
    .filter(Boolean)
    .join(' · ')
}

function isUnavailableError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error)

  return /(?:^|error:\s*)404(?:\s|:|$)/i.test(message) || /endpoint is likely missing/i.test(message)
}

export function KanbanSettings({ gatewayId }: KanbanSettingsProps = {}) {
  const { t } = useI18n()
  const copy = t.settings.kanban
  const [boards, setBoards] = useState<KanbanBoardSummary[]>([])
  const [boardsStatus, setBoardsStatus] = useState<BoardsStatus>('loading')
  const [selectedBoard, setSelectedBoard] = useState('')
  const [board, setBoard] = useState<KanbanBoardResponse | null>(null)
  const [boardLoadFailed, setBoardLoadFailed] = useState(false)
  const [loadingBoard, setLoadingBoard] = useState(false)
  const boardsRequest = useRef(0)
  const boardRequest = useRef(0)

  const loadBoards = useCallback(async () => {
    const request = ++boardsRequest.current
    boardRequest.current += 1
    setBoardsStatus('loading')
    setBoards([])
    setSelectedBoard('')
    setBoard(null)
    setBoardLoadFailed(false)
    setLoadingBoard(false)

    try {
      const result = await listKanbanBoards({ gatewayId })

      if (request !== boardsRequest.current) {
        return
      }

      setBoards(result.boards)
      setSelectedBoard(
        result.current || result.boards.find(value => value.is_current)?.slug || result.boards[0]?.slug || ''
      )
      setBoardsStatus('ready')
    } catch (error) {
      if (request !== boardsRequest.current) {
        return
      }

      if (isUnavailableError(error)) {
        setBoardsStatus('unavailable')
      } else {
        setBoardsStatus('error')
        notifyError(error, copy.boardsLoadFailed)
      }
    }
  }, [copy.boardsLoadFailed, gatewayId])

  useEffect(() => {
    void loadBoards()

    return () => {
      boardsRequest.current += 1
      boardRequest.current += 1
    }
  }, [loadBoards])

  const loadBoard = useCallback(
    async (slug: string) => {
      const request = ++boardRequest.current
      setBoard(null)
      setBoardLoadFailed(false)

      if (!slug) {
        setLoadingBoard(false)

        return
      }

      setLoadingBoard(true)

      try {
        const result = await getKanbanBoard({ board: slug, gatewayId })

        if (request === boardRequest.current) {
          setBoard(result)
        }
      } catch (error) {
        if (request === boardRequest.current) {
          setBoard(null)
          setBoardLoadFailed(true)
          notifyError(error, copy.boardLoadFailed)
        }
      } finally {
        if (request === boardRequest.current) {
          setLoadingBoard(false)
        }
      }
    },
    [copy.boardLoadFailed, gatewayId]
  )

  useEffect(() => {
    if (boardsStatus === 'ready') {
      void loadBoard(selectedBoard)
    }

    return () => {
      boardRequest.current += 1
    }
  }, [boardsStatus, loadBoard, selectedBoard])

  if (boardsStatus === 'loading') {
    return <LoadingState label={copy.loadingBoards} />
  }

  if (boardsStatus === 'unavailable' || boardsStatus === 'error') {
    const unavailable = boardsStatus === 'unavailable'

    return (
      <SettingsContent>
        <div className="flex min-h-80">
          <PanelEmpty
            action={
              <Button onClick={() => void loadBoards()} size="sm">
                {copy.retry}
              </Button>
            }
            description={unavailable ? copy.unavailableDescription : copy.loadFailedDescription}
            icon={unavailable ? 'plug' : 'error'}
            title={unavailable ? copy.unavailableTitle : copy.loadFailedTitle}
          />
        </div>
      </SettingsContent>
    )
  }

  return (
    <SettingsContent>
      <div className="mb-4 flex items-center justify-between gap-3">
        <SectionHeading icon={Clipboard} meta={copy.boardCount(boards.length)} title={copy.title} />
        <Button
          disabled={!selectedBoard || loadingBoard}
          onClick={() => void loadBoard(selectedBoard)}
          size="sm"
          variant="outline"
        >
          <RefreshCw className={loadingBoard ? 'animate-spin' : undefined} />
          {copy.refresh}
        </Button>
      </div>

      <div className="grid min-h-0 gap-4 lg:grid-cols-[17rem_minmax(0,1fr)]">
        <div className="grid content-start gap-1 rounded-xl bg-background/60 p-2">
          {boards.length === 0 ? (
            <EmptyState description={copy.emptyDescription} title={copy.emptyTitle} />
          ) : (
            boards.map(value => (
              <button
                className={`rounded-md px-2 py-2 text-left transition-colors hover:bg-(--chrome-action-hover) ${selectedBoard === value.slug ? 'bg-accent/45 text-foreground' : 'text-muted-foreground'}`}
                key={value.slug}
                onClick={() => {
                  setBoard(null)
                  setBoardLoadFailed(false)
                  setSelectedBoard(value.slug)
                }}
                type="button"
              >
                <div className="truncate text-sm font-medium">{value.name || value.slug}</div>
                <div className="mt-1 flex items-center gap-1.5">
                  <Pill>{value.slug}</Pill>
                  {value.is_current ? <Pill tone="primary">{copy.current}</Pill> : null}
                  {typeof value.total === 'number' ? <Pill>{copy.taskCount(value.total)}</Pill> : null}
                </div>
              </button>
            ))
          )}
        </div>

        <div className="min-h-0 overflow-x-auto rounded-xl bg-background/60 p-3">
          {loadingBoard ? (
            <LoadingState label={copy.loadingBoard} />
          ) : boardLoadFailed ? (
            <EmptyState description={copy.boardUnavailableDescription} title={copy.boardUnavailableTitle} />
          ) : !board ? (
            <EmptyState description={copy.noSelectionDescription} title={copy.noSelectionTitle} />
          ) : (
            <div
              className="grid min-w-[52rem] gap-3"
              style={{ gridTemplateColumns: `repeat(${Math.max(1, board.columns.length)}, minmax(12rem, 1fr))` }}
            >
              {board.columns.map(column => (
                <div className="rounded-lg bg-muted/20 p-2" key={column.name}>
                  <div className="mb-2 flex items-center justify-between gap-2">
                    <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                      {column.name}
                    </span>
                    <Pill>{column.tasks.length}</Pill>
                  </div>
                  <div className="grid gap-2">
                    {column.tasks.slice(0, 20).map((task, index) => {
                      const meta = taskMeta(task)

                      return (
                        <div className="rounded-md bg-background/70 p-2 text-xs" key={`${column.name}-${index}`}>
                          <div className="font-medium text-foreground">{taskTitle(task, copy.untitledTask)}</div>
                          {meta ? <div className="mt-1 text-[0.68rem] text-muted-foreground">{meta}</div> : null}
                        </div>
                      )
                    })}
                    {column.tasks.length > 20 ? (
                      <div className="text-[0.68rem] text-muted-foreground">
                        {copy.moreTasks(column.tasks.length - 20)}
                      </div>
                    ) : null}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </SettingsContent>
  )
}
