/** Human + agent shared Kanban projection. The server remains authoritative. */
import { Button, Codicon, Input, Loader, Textarea, useQuery, useQueryClient } from '@hermes/plugin-sdk'
import { type DragEvent, useEffect, useState } from 'react'

import {
  createHybridBoard,
  createHybridCard,
  createHybridColumn,
  deleteHybridBoard,
  deleteHybridCard,
  deleteHybridColumn,
  fetchHybridBoard,
  fetchHybridBoards,
  fetchHybridCard,
  moveHybridCard,
  moveHybridColumn,
  updateHybridCard
} from './api'
import type { HybridActivityItem, HybridBoard, HybridCard, HybridColumn } from './types'

const HYBRID_BOARDS_KEY = ['kanban', 'hybrid', 'boards'] as const
const hybridBoardKey = (id: string) => ['kanban', 'hybrid', 'board', id] as const
const hybridCardKey = (id: string) => ['kanban', 'hybrid', 'card', id] as const

function Add({ label, onAdd }: { label: string; onAdd: (name: string) => Promise<unknown> }) {
  const [value, setValue] = useState('')
  const [busy, setBusy] = useState(false)
  const submit = async () => {
    if (!value.trim()) return
    setBusy(true)
    try {
      await onAdd(value.trim())
      setValue('')
    } finally {
      setBusy(false)
    }
  }
  return (
    <div className="flex gap-1">
      <Input aria-label={label} onChange={e => setValue(e.target.value)} value={value} />
      <Button disabled={!value.trim() || busy} onClick={() => void submit()} size="xs">
        {busy ? '…' : label}
      </Button>
    </div>
  )
}
function Card({ card, onOpen }: { card: HybridCard; onOpen: (card: HybridCard) => void }) {
  const dragStart = (event: DragEvent<HTMLButtonElement>) => {
    event.dataTransfer.setData('text/hermes-hybrid-card', card.id)
  }
  return (
    <button
      className="w-full rounded-md border border-(--ui-stroke-secondary) bg-(--ui-bg-primary) p-2 text-left text-xs shadow-sm transition hover:border-(--ui-stroke-primary)"
      draggable
      onClick={() => onOpen(card)}
      onDragStart={dragStart}
      type="button"
    >
      <div className="font-medium">{card.title}</div>
      {card.description && <div className="mt-1 line-clamp-2 text-(--ui-text-tertiary)">{card.description}</div>}
    </button>
  )
}

function Column({
  board,
  column,
  refresh,
  onOpen
}: {
  board: HybridBoard
  column: HybridColumn
  refresh: () => void
  onOpen: (card: HybridCard) => void
}) {
  const [deleting, setDeleting] = useState(false)

  const dragStartCol = (event: DragEvent<HTMLElement>) => {
    event.dataTransfer.setData('text/hermes-hybrid-column', column.id)
  }

  const drop = async (event: DragEvent<HTMLElement>) => {
    event.preventDefault()
    const colId = event.dataTransfer.getData('text/hermes-hybrid-column')
    if (colId && colId !== column.id) {
      const sourceCol = board.columns.find(item => item.id === colId)
      if (sourceCol) {
        try {
          await moveHybridColumn(colId, column.id, undefined, sourceCol.revision)
          refresh()
        } catch {
          refresh()
        }
        return
      }
    }

    const card = event.dataTransfer.getData('text/hermes-hybrid-card')
    if (!card) return
    const source = board.columns.flatMap(item => item.cards).find(item => item.id === card)
    if (!source) return
    try {
      await moveHybridCard(card, column.id, source.revision)
      refresh()
    } catch {
      refresh()
    }
  }

  const handleDelete = async () => {
    if (column.cards.length > 0) {
      if (!window.confirm(`Delete column "${column.name}" and all its ${column.cards.length} card(s)?`)) {
        return
      }
    }
    setDeleting(true)
    try {
      await deleteHybridColumn(column.id)
      refresh()
    } finally {
      setDeleting(false)
    }
  }

  return (
    <section
      className="flex w-68 shrink-0 flex-col gap-2 rounded-lg bg-(--ui-bg-quaternary) p-2"
      onDragOver={event => event.preventDefault()}
      onDrop={event => void drop(event)}
    >
      <header
        className="flex cursor-grab items-center justify-between px-1 active:cursor-grabbing"
        draggable
        onDragStart={dragStartCol}
      >
        <div className="flex items-center gap-1.5">
          <Codicon className="text-(--ui-text-quaternary)" name="grabber" size="0.85rem" />
          <h2 className="text-xs font-semibold">{column.name}</h2>
          <span className="rounded-full bg-(--ui-bg-secondary) px-1.5 py-0.2 text-[0.65rem] text-(--ui-text-tertiary)">
            {column.cards.length}
          </span>
        </div>
        <Button
          aria-label="Delete column"
          className="text-(--ui-text-tertiary) hover:text-destructive"
          disabled={deleting}
          onClick={() => void handleDelete()}
          size="xs"
          variant="ghost"
        >
          <Codicon name="trash" size="0.75rem" />
        </Button>
      </header>
      <div className="flex min-h-10 flex-col gap-1.5">
        {column.cards.map(card => (
          <Card card={card} key={card.id} onOpen={onOpen} />
        ))}
      </div>
      <Add label="Add card" onAdd={name => createHybridCard(board.id, column.id, name).then(refresh)} />
    </section>
  )
}

function CardActivityDrawer({
  board,
  cardId,
  onClose,
  onRefresh
}: {
  board: HybridBoard
  cardId: string
  onClose: () => void
  onRefresh: () => void
}) {
  const { data, isLoading } = useQuery({
    queryKey: hybridCardKey(cardId),
    queryFn: () => fetchHybridCard(cardId),
    refetchInterval: 4_000
  })
  const card = data?.card

  const [title, setTitle] = useState('')
  const [description, setDescription] = useState('')
  const [columnId, setColumnId] = useState('')
  const [saving, setSaving] = useState(false)
  const [deleting, setDeleting] = useState(false)

  useEffect(() => {
    if (card) {
      setTitle(card.title)
      setDescription(card.description || '')
      setColumnId(card.column_id)
    }
  }, [card])

  if (isLoading || !card) {
    return (
      <aside className="border-t border-(--ui-stroke-secondary) bg-(--ui-bg-primary) p-4">
        <div className="flex items-center justify-center p-8">
          <Loader type="lemniscate-bloom" />
        </div>
      </aside>
    )
  }

  const handleSave = async () => {
    if (!title.trim()) return
    setSaving(true)
    try {
      let currentRevision = card.revision
      if (columnId !== card.column_id) {
        const moved = await moveHybridCard(card.id, columnId, currentRevision)
        currentRevision = moved.card.revision
      }
      await updateHybridCard(card.id, {
        title: title.trim(),
        description,
        revision: currentRevision
      })
      onRefresh()
      onClose()
    } catch {
      onRefresh()
    } finally {
      setSaving(false)
    }
  }

  const handleDelete = async () => {
    if (!window.confirm(`Delete card "${card.title}"?`)) return
    setDeleting(true)
    try {
      await deleteHybridCard(card.id)
      onRefresh()
      onClose()
    } finally {
      setDeleting(false)
    }
  }

  const activities = card.activity ?? []

  return (
    <aside className="border-t border-(--ui-stroke-secondary) bg-(--ui-bg-primary) p-4">
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Codicon name="note" size="0.9rem" />
          <span className="text-xs font-semibold">Card Details</span>
          <span className="font-mono text-[0.7rem] text-(--ui-text-tertiary)">{card.id}</span>
        </div>
        <div className="flex items-center gap-1">
          <Button disabled={saving || !title.trim()} onClick={() => void handleSave()} size="xs">
            {saving ? 'Saving…' : 'Save'}
          </Button>
          <Button
            className="text-(--ui-text-tertiary) hover:text-destructive"
            disabled={deleting}
            onClick={() => void handleDelete()}
            size="xs"
            variant="ghost"
          >
            <Codicon name="trash" size="0.8rem" />
          </Button>
          <Button onClick={onClose} size="xs" variant="ghost">
            <Codicon name="close" size="0.8rem" />
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <div className="flex flex-col gap-2">
          <div>
            <label className="mb-1 block text-[0.7rem] font-medium text-(--ui-text-secondary)">Title</label>
            <Input aria-label="Card title" onChange={e => setTitle(e.target.value)} value={title} />
          </div>

          <div>
            <label className="mb-1 block text-[0.7rem] font-medium text-(--ui-text-secondary)">Column</label>
            <select
              aria-label="Column selection"
              className="w-full rounded border border-(--ui-stroke-secondary) bg-(--ui-bg-quaternary) px-2 py-1 text-xs"
              onChange={e => setColumnId(e.target.value)}
              value={columnId}
            >
              {board.columns.map(c => (
                <option key={c.id} value={c.id}>
                  {c.name}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="mb-1 block text-[0.7rem] font-medium text-(--ui-text-secondary)">Description</label>
            <Textarea
              aria-label="Card description"
              className="min-h-20"
              onChange={e => setDescription(e.target.value)}
              value={description}
            />
          </div>
        </div>

        <div className="flex flex-col border-l border-(--ui-stroke-secondary) pl-4">
          <div className="mb-2 flex items-center gap-1.5 text-xs font-semibold text-(--ui-text-secondary)">
            <Codicon name="history" size="0.85rem" />
            <span>Activity & Provenance</span>
          </div>

          <div className="max-h-48 flex-1 overflow-y-auto pr-1">
            {activities.length === 0 ? (
              <p className="text-[0.7rem] text-(--ui-text-tertiary)">No activity recorded for this card.</p>
            ) : (
              <ul className="flex flex-col gap-2">
                {activities.map((item: HybridActivityItem) => {
                  const isAgent = item.actor_type === 'agent'
                  const isHuman = item.actor_type === 'human'
                  const actorLabel = isAgent ? `Agent (${item.actor_id || 'Hermes'})` : isHuman ? 'Human' : 'System'
                  const dateStr = new Date(item.created_at * 1000).toLocaleTimeString([], {
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit'
                  })

                  return (
                    <li
                      className="flex flex-col gap-0.5 rounded bg-(--ui-bg-quaternary) p-1.5 text-[0.7rem]"
                      key={item.id}
                    >
                      <div className="flex items-center justify-between">
                        <span
                          className={`rounded px-1.5 py-0.2 font-medium ${
                            isAgent
                              ? 'bg-purple-500/10 text-purple-400'
                              : isHuman
                              ? 'bg-sky-500/10 text-sky-400'
                              : 'bg-zinc-500/10 text-zinc-400'
                          }`}
                        >
                          {actorLabel}
                        </span>
                        <span className="font-mono text-[0.65rem] text-(--ui-text-tertiary)">{dateStr}</span>
                      </div>
                      <div className="text-(--ui-text-secondary)">
                        <span className="font-medium">{item.kind}</span>
                        {item.payload && Object.keys(item.payload).length > 0 && (
                          <span className="ml-1 text-(--ui-text-tertiary)">
                            ({JSON.stringify(item.payload).replace(/[{"}]/g, '')})
                          </span>
                        )}
                      </div>
                    </li>
                  )
                })}
              </ul>
            )}
          </div>
        </div>
      </div>
    </aside>
  )
}

export function HybridBoardPage() {
  const qc = useQueryClient()
  const [selectedId, setSelectedId] = useState('')
  const [openCardId, setOpenCardId] = useState<string | null>(null)
  const { data: boardsData } = useQuery({
    queryKey: HYBRID_BOARDS_KEY,
    queryFn: fetchHybridBoards,
    refetchInterval: 8_000
  })
  const boards = boardsData?.boards ?? []
  const boardId = selectedId || boards[0]?.id || ''
  const { data, isLoading } = useQuery({
    queryKey: hybridBoardKey(boardId),
    queryFn: () => fetchHybridBoard(boardId),
    enabled: Boolean(boardId),
    refetchInterval: 8_000
  })
  const board = data?.board

  const refresh = () => {
    void qc.invalidateQueries({ queryKey: HYBRID_BOARDS_KEY })
    void qc.invalidateQueries({ queryKey: hybridBoardKey(boardId) })
    if (openCardId) {
      void qc.invalidateQueries({ queryKey: hybridCardKey(openCardId) })
    }
  }

  const handleDeleteBoard = async () => {
    if (!board) return
    if (!window.confirm(`Delete board "${board.name}" and all its contents?`)) return
    try {
      await deleteHybridBoard(board.id)
      setSelectedId('')
      refresh()
    } catch {
      refresh()
    }
  }

  return (
    <main className="flex h-full min-h-0 flex-col">
      <header className="flex items-center gap-2 border-b border-(--ui-stroke-secondary) px-4 py-2">
        <Codicon name="project" size="1rem" />
        <h1 className="text-sm font-semibold">Hybrid Kanban</h1>
        <select
          aria-label="Hybrid board"
          className="ml-2 rounded bg-(--ui-bg-quaternary) px-2 py-1 text-xs"
          onChange={event => setSelectedId(event.target.value)}
          value={boardId}
        >
          {boards.map(item => (
            <option key={item.id} value={item.id}>
              {item.name}
            </option>
          ))}
        </select>
        {board && (
          <div className="flex items-center gap-1">
            <Button
              aria-label="Refresh board"
              className="text-(--ui-text-tertiary)"
              onClick={refresh}
              size="xs"
              variant="ghost"
            >
              <Codicon name="refresh" size="0.8rem" />
            </Button>
            <Button
              aria-label="Delete board"
              className="text-(--ui-text-tertiary) hover:text-destructive"
              onClick={() => void handleDeleteBoard()}
              size="xs"
              variant="ghost"
            >
              <Codicon name="trash" size="0.8rem" />
            </Button>
          </div>
        )}
        <div className="ml-auto w-72">
          <Add
            label="New board"
            onAdd={name =>
              createHybridBoard(name).then(result => {
                setSelectedId(result.board.id)
                refresh()
              })
            }
          />
        </div>
      </header>

      {isLoading ? (
        <div className="grid flex-1 place-items-center">
          <Loader type="lemniscate-bloom" />
        </div>
      ) : !board ? (
        <div className="grid flex-1 place-items-center text-sm text-(--ui-text-tertiary)">
          Create a shared board to start.
        </div>
      ) : (
        <>
          <div className="flex flex-1 gap-3 overflow-x-auto p-4">
            {board.columns.map(column => (
              <Column
                board={board}
                column={column}
                key={column.id}
                onOpen={c => setOpenCardId(c.id)}
                refresh={refresh}
              />
            ))}
            <section className="w-68 shrink-0 rounded-lg border border-dashed border-(--ui-stroke-secondary) p-2">
              <Add label="Add column" onAdd={name => createHybridColumn(board.id, name).then(refresh)} />
            </section>
          </div>
          {openCardId && (
            <CardActivityDrawer
              board={board}
              cardId={openCardId}
              onClose={() => setOpenCardId(null)}
              onRefresh={refresh}
            />
          )}
        </>
      )}
    </main>
  )
}
