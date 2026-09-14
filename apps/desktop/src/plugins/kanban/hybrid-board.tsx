/** Human + agent shared Kanban projection. The server remains authoritative. */
import { Button, Codicon, Input, Loader, Textarea, useQuery, useQueryClient } from '@hermes/plugin-sdk'
import { type DragEvent, useMemo, useState } from 'react'

import {
  createHybridBoard,
  createHybridCard,
  createHybridColumn,
  fetchHybridBoard,
  fetchHybridBoards,
  moveHybridCard,
  updateHybridCard
} from './api'
import type { HybridBoard, HybridCard, HybridColumn } from './types'

const HYBRID_BOARDS_KEY = ['kanban', 'hybrid', 'boards'] as const
const hybridBoardKey = (id: string) => ['kanban', 'hybrid', 'board', id] as const

function Add({ label, onAdd }: { label: string; onAdd: (name: string) => Promise<unknown> }) {
  const [value, setValue] = useState('')
  const [busy, setBusy] = useState(false)
  const submit = async () => {
    if (!value.trim()) return
    setBusy(true)
    try { await onAdd(value.trim()); setValue('') } finally { setBusy(false) }
  }
  return <div className="flex gap-1"><Input aria-label={label} onChange={setValue} value={value} /><Button disabled={!value.trim() || busy} onClick={() => void submit()} size="xs">{busy ? '…' : label}</Button></div>
}

function Card({ card, onOpen }: { card: HybridCard; onOpen: (card: HybridCard) => void }) {
  const dragStart = (event: DragEvent<HTMLButtonElement>) => event.dataTransfer.setData('text/hermes-hybrid-card', card.id)
  return <button className="w-full rounded-md border border-(--ui-stroke-secondary) bg-(--ui-bg-primary) p-2 text-left text-xs shadow-sm hover:border-(--ui-stroke-primary)" draggable onClick={() => onOpen(card)} onDragStart={dragStart} type="button"><div className="font-medium">{card.title}</div>{card.description && <div className="mt-1 line-clamp-2 text-(--ui-text-tertiary)">{card.description}</div>}</button>
}

function Column({ board, column, refresh, onOpen }: { board: HybridBoard; column: HybridColumn; refresh: () => void; onOpen: (card: HybridCard) => void }) {
  const drop = async (event: DragEvent<HTMLElement>) => {
    event.preventDefault()
    const card = event.dataTransfer.getData('text/hermes-hybrid-card')
    if (!card) return
    const source = board.columns.flatMap(item => item.cards).find(item => item.id === card)
    if (!source) return
    try { await moveHybridCard(card, column.id, source.revision); refresh() } catch { refresh() }
  }
  return <section className="flex w-68 shrink-0 flex-col gap-2 rounded-lg bg-(--ui-bg-quaternary) p-2" onDragOver={event => event.preventDefault()} onDrop={event => void drop(event)}><h2 className="px-1 text-xs font-semibold">{column.name}</h2><div className="flex min-h-10 flex-col gap-1.5">{column.cards.map(card => <Card card={card} key={card.id} onOpen={onOpen} />)}</div><Add label="Add card" onAdd={name => createHybridCard(board.id, column.id, name).then(refresh)} /></section>
}

export function HybridBoardPage() {
  const qc = useQueryClient()
  const [selectedId, setSelectedId] = useState('')
  const [openCard, setOpenCard] = useState<HybridCard | null>(null)
  const { data: boardsData } = useQuery({ queryKey: HYBRID_BOARDS_KEY, queryFn: fetchHybridBoards, refetchInterval: 8_000 })
  const boards = boardsData?.boards ?? []
  const boardId = selectedId || boards[0]?.id || ''
  const { data, isLoading } = useQuery({ queryKey: hybridBoardKey(boardId), queryFn: () => fetchHybridBoard(boardId), enabled: Boolean(boardId), refetchInterval: 8_000 })
  const board = data?.board
  const refresh = () => { void qc.invalidateQueries({ queryKey: HYBRID_BOARDS_KEY }); void qc.invalidateQueries({ queryKey: hybridBoardKey(boardId) }) }
  const selected = useMemo(() => board?.columns.flatMap(column => column.cards).find(card => card.id === openCard?.id) ?? openCard, [board, openCard])
  const save = async () => { if (!selected) return; await updateHybridCard(selected.id, { title: selected.title, description: selected.description, revision: selected.revision }); setOpenCard(null); refresh() }

  return <main className="flex h-full min-h-0 flex-col"><header className="flex items-center gap-2 border-b border-(--ui-stroke-secondary) px-4 py-2"><Codicon name="project" size="1rem" /><h1 className="text-sm font-semibold">Hybrid Kanban</h1><select aria-label="Hybrid board" className="ml-2 rounded bg-(--ui-bg-quaternary) px-2 py-1 text-xs" onChange={event => setSelectedId(event.target.value)} value={boardId}>{boards.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</select><div className="ml-auto w-72"><Add label="New board" onAdd={name => createHybridBoard(name).then(result => { setSelectedId(result.board.id); refresh() })} /></div></header>{isLoading ? <div className="grid flex-1 place-items-center"><Loader type="lemniscate-bloom" /></div> : !board ? <div className="grid flex-1 place-items-center text-sm text-(--ui-text-tertiary)">Create a shared board to start.</div> : <><div className="flex flex-1 gap-3 overflow-x-auto p-4">{board.columns.map(column => <Column board={board} column={column} key={column.id} onOpen={setOpenCard} refresh={refresh} />)}<section className="w-68 shrink-0 rounded-lg border border-dashed border-(--ui-stroke-secondary) p-2"><Add label="Add column" onAdd={name => createHybridColumn(board.id, name).then(refresh)} /></section></div>{selected && <aside className="border-t border-(--ui-stroke-secondary) bg-(--ui-bg-primary) p-4"><div className="mb-2 flex items-center gap-2"><Input aria-label="Card title" onChange={title => setOpenCard({ ...selected, title })} value={selected.title} /><Button onClick={() => void save()} size="xs">Save</Button><Button onClick={() => setOpenCard(null)} size="xs" variant="ghost">Close</Button></div><Textarea aria-label="Card description" className="min-h-24" onChange={description => setOpenCard({ ...selected, description })} value={selected.description} /><p className="mt-2 text-[0.7rem] text-(--ui-text-tertiary)">Changes are persisted through the canonical Hybrid Kanban domain. Activity is available from the card API.</p></aside>}</>}</main>
}
