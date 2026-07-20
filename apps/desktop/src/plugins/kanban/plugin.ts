// @ts-nocheck — this plugin uses the jsx() API for plain-JS compatibility
// and is verified working as a runtime plugin. If converting to JSX syntax,
// replace all `h(` calls with proper JSX elements.

import {
  host,
  useQuery,
  useMutation,
  useQueryClient,
  ROUTES_AREA,
  SIDEBAR_NAV_AREA,
  PALETTE_AREA,
  KEYBINDS_AREA,
  Button,
  Badge,
  Input,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Checkbox,
  Separator,
  SearchField,
  Textarea,
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  ConfirmDialog,
  GlyphSpinner,
  cn,
  icons,
  relativeTime,
  coarseElapsed,
  fmtDateTime,
  profileColor,
  profileColorSoft,
  ErrorState,
  EmptyState,
} from '@hermes/plugin-sdk'

/**
 * Hermes Kanban — Desktop Plugin
 *
 * Full kanban board for the Hermes desktop app, backed by the existing
 * kanban dashboard plugin's REST API at /api/plugins/kanban/.
 *
 * Features: board view with horizontal scrollable columns, drag-and-drop
 * cards, task drawer with detail/comments/events, create/edit tasks,
 * board switcher, filters (tenant/assignee/search/archived), bulk ops,
 * live WebSocket updates, delete via drag-to-trash.
 *
 * Single ESM file. Uses @hermes/plugin-sdk for UI and ctx.rest / ctx.socket.
 */

import { jsx } from 'react/jsx-runtime'
import { useState, useEffect, useCallback, useMemo, useRef } from 'react'
import {
  host, useQuery, useMutation, useQueryClient,
  ROUTES_AREA, SIDEBAR_NAV_AREA, PALETTE_AREA, KEYBINDS_AREA,
  Button, Badge, Input, Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
  Checkbox, Separator, SearchField, Textarea,
  Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle,
  ConfirmDialog, GlyphSpinner,
  cn, icons, relativeTime, coarseElapsed, fmtDateTime,
  profileColor, profileColorSoft,
  ErrorState, EmptyState,
} from '@hermes/plugin-sdk'

// ---------------------------------------------------------------------------
// Module state — captured once on registration
// ---------------------------------------------------------------------------
let _ctx = null

function rest(path, opts) {
  return _ctx.rest(path, opts || {})
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const COLUMNS = ['triage', 'todo', 'ready', 'running', 'blocked', 'done']

const COLUMN_LABELS = {
  triage: 'Triage',
  todo: 'Todo',
  ready: 'Ready',
  running: 'In Progress',
  blocked: 'Blocked',
  done: 'Done',
}

const COLUMN_COLORS = {
  triage: 'var(--ui-kanban-triage, #8b5cf6)',
  todo: 'var(--ui-kanban-todo, #3b82f6)',
  ready: 'var(--ui-kanban-ready, #22c55e)',
  running: 'var(--ui-kanban-running, #f59e0b)',
  blocked: 'var(--ui-kanban-blocked, #ef4444)',
  done: 'var(--ui-kanban-done, #6b7280)',
}

// Map status to an available icon name from the SDK's tabler set
function colIcon(status) {
  switch (status) {
    case 'triage': return icons.Layers3
    case 'todo': return icons.CircleIcon
    case 'ready': return icons.Play
    case 'running': return icons.Loader2
    case 'blocked': return icons.AlertCircle
    case 'done': return icons.CheckCircle2
    default: return icons.CircleIcon
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function colLabel(status) {
  return COLUMN_LABELS[status] || status
}

function colColor(status) {
  return COLUMN_COLORS[status] || 'var(--ui-text-secondary)'
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

function taskAge(task) {
  if (!task) return null
  if (task.completed_at) return coarseElapsed(task.completed_at * 1000)
  if (task.started_at) return coarseElapsed(task.started_at * 1000)
  if (task.created_at) return coarseElapsed(task.created_at * 1000)
  return null
}

function asgnColor(name) {
  return name ? profileColor(name) : 'var(--ui-text-quaternary)'
}

// Safe markdown renderer (subset)
function renderMd(src) {
  if (!src) return ''
  const blocks = []
  let w = String(src).replace(/```([\s\S]*?)```/g, (_, c) => {
    blocks.push(c)
    return `\x00CODE${blocks.length - 1}\x00`
  })
  const esc = escapeHtml(w)
  const lines = esc.split(/\r?\n/)
  const out = []
  let inList = false
  for (const r of lines) {
    const bullet = /^\s*[-*]\s+(.*)$/.exec(r)
    const heading = /^(#{1,4})\s+(.*)$/.exec(r)
    if (bullet) {
      if (!inList) { out.push('<ul>'); inList = true }
      out.push(`<li>${bullet[1]}</li>`)
      continue
    }
    if (inList) { out.push('</ul>'); inList = false }
    if (heading) {
      out.push(`<h${heading[1].length}>${heading[2]}</h${heading[1].length}>`)
    } else if (r.trim() === '') {
      out.push('')
    } else {
      out.push(`<p>${r}</p>`)
    }
  }
  if (inList) out.push('</ul>')
  let h = out.join('\n')
  h = h.replace(/\x00CODE(\d+)\x00/g, (_, i) =>
    `<pre><code>${escapeHtml(blocks[Number(i)])}</code></pre>`
  )
  h = h.replace(/`([^`\n]+)`/g, '<code>$1</code>')
  h = h.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
  h = h.replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<em>$2</em>')
  h = h.replace(
    /\[([^\]\n]+)\]\((https?:\/\/[^\s)]+|mailto:[^\s)]+)\)/g,
    (_, t, url) => `<a href="${url}" target="_blank" rel="noopener noreferrer">${t}</a>`
  )
  return h
}

// ---------------------------------------------------------------------------
// Injected CSS — self-contained, uses the app's theme variables
// ---------------------------------------------------------------------------

const PLUGIN_CSS = `
/* --- Kanban Plugin Styles --- */
.kanban-page {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
  background: var(--ui-bg, var(--background));
}
.kanban-loading {
  align-items: center;
  justify-content: center;
}
.kanban-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-bottom: 1px solid var(--ui-stroke-secondary, var(--border));
  flex-shrink: 0;
  flex-wrap: wrap;
}
.kanban-header-left {
  display: flex;
  align-items: center;
  gap: 4px;
}
.kanban-board-trigger {
  width: 160px;
}
.kanban-filters {
  display: flex;
  align-items: center;
  gap: 6px;
  flex: 1;
  flex-wrap: wrap;
}
.kanban-search {
  width: 180px;
}
.kanban-filter-select {
  width: 130px;
}
.kanban-archived-toggle {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  color: var(--ui-text-secondary, var(--muted-foreground));
  cursor: pointer;
}
.kanban-header-right {
  display: flex;
  align-items: center;
  gap: 4px;
}
.kanban-bulk-bar {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 12px;
  background: var(--ui-accent, var(--accent));
  color: var(--ui-accent-foreground, var(--accent-foreground));
  font-size: 13px;
  flex-shrink: 0;
}
.kanban-bulk-label {
  font-weight: 600;
  margin-right: 8px;
}
.kanban-bulk-actions {
  display: flex;
  gap: 4px;
}
.kanban-columns {
  display: flex;
  gap: 8px;
  padding: 8px 12px;
  flex: 1;
  overflow-x: auto;
  overflow-y: hidden;
  align-items: stretch;
}
.kanban-columns--dragging {
  cursor: grabbing;
}
.kanban-column {
  display: flex;
  flex-direction: column;
  min-width: 260px;
  max-width: 340px;
  flex: 1;
  border-radius: 8px;
  background: var(--ui-bg-muted, color-mix(in srgb, var(--ui-bg, var(--background)) 97%, currentColor));
  border: 1px solid var(--ui-stroke-secondary, var(--border));
}
.kanban-column-header {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 10px;
  font-size: 13px;
  font-weight: 600;
  border-bottom: 1px solid var(--ui-stroke-secondary, var(--border));
  flex-shrink: 0;
}
.kanban-column-title {
  display: flex;
  align-items: center;
  gap: 4px;
  flex: 1;
}
.kanban-column-count {
  font-size: 11px;
  margin-left: auto;
}
.kanban-column-cards {
  flex: 1;
  overflow-y: auto;
  padding: 4px 6px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-height: 60px;
}
.kanban-column-empty {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 60px;
  font-size: 12px;
  color: var(--ui-text-quaternary, color-mix(in srgb, currentColor 30%, transparent));
  border: 1px dashed var(--ui-stroke-secondary, var(--border));
  border-radius: 6px;
}
.kanban-card {
  padding: 8px 10px;
  border-radius: 6px;
  background: var(--ui-bg, var(--background));
  border: 1px solid var(--ui-stroke-secondary, var(--border));
  cursor: grab;
  transition: box-shadow 0.15s, border-color 0.15s;
  user-select: none;
}
.kanban-card:hover {
  border-color: var(--ui-accent, var(--ring));
  box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
.kanban-card--selected {
  border-color: var(--ui-accent, var(--ring));
  box-shadow: 0 0 0 1px var(--ui-accent, var(--ring));
}
.kanban-card--blocked {
  border-left: 3px solid var(--ui-kanban-blocked, #ef4444);
}
.kanban-card-top {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 4px;
}
.kanban-card-select {
  display: flex;
  align-items: center;
}
.kanban-card-priority {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  flex-shrink: 0;
}
.kanban-card-tenant {
  font-size: 10px;
  margin-left: auto;
}
.kanban-card-title {
  font-size: 13px;
  line-height: 1.4;
  color: var(--ui-text, var(--foreground));
  word-break: break-word;
  margin-bottom: 4px;
}
.kanban-card-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 11px;
  color: var(--ui-text-secondary, var(--muted-foreground));
}
.kanban-card-assignee {
  font-weight: 500;
}
.kanban-card-unassigned {
  font-style: italic;
  opacity: 0.5;
}
.kanban-card-age {
  opacity: 0.7;
}
.kanban-trash {
  display: none;
  align-items: center;
  justify-content: center;
  gap: 6px;
  min-width: 80px;
  border: 2px dashed var(--ui-stroke-secondary, var(--border));
  border-radius: 8px;
  color: var(--ui-text-quaternary, var(--muted-foreground));
  font-size: 12px;
  transition: all 0.2s;
}
.kanban-trash--visible {
  display: flex;
}
.kanban-trash:hover {
  border-color: var(--ui-destructive, #ef4444);
  color: var(--ui-destructive, #ef4444);
  background: color-mix(in srgb, var(--ui-destructive, #ef4444) 8%, transparent);
}

/* --- Drawer --- */
.kanban-drawer-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.3);
  z-index: 50;
  display: flex;
  justify-content: flex-end;
}
.kanban-drawer {
  width: 480px;
  max-width: 90vw;
  height: 100%;
  background: var(--ui-bg, var(--background));
  border-left: 1px solid var(--ui-stroke-secondary, var(--border));
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.kanban-drawer-header {
  padding: 12px;
  border-bottom: 1px solid var(--ui-stroke-secondary, var(--border));
  flex-shrink: 0;
}
.kanban-drawer-title-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}
.kanban-drawer-status-control {
  flex: 1;
}
.kanban-drawer-status-trigger {
  width: 140px;
}
.kanban-drawer-title-input {
  font-size: 16px;
  font-weight: 600;
}
.kanban-drawer-loading {
  display: flex;
  align-items: center;
  justify-content: center;
  flex: 1;
}
.kanban-drawer-content {
  flex: 1;
  overflow-y: auto;
  padding: 12px;
}
.kanban-drawer-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 16px;
  font-size: 12px;
  margin-bottom: 8px;
}
.kanban-drawer-meta-item {
  display: flex;
  gap: 4px;
}
.kanban-drawer-meta-label {
  font-weight: 500;
  color: var(--ui-text-secondary, var(--muted-foreground));
}
.kanban-drawer-section {
  margin: 8px 0;
}
.kanban-drawer-section-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 6px;
}
.kanban-drawer-section-title {
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--ui-text-secondary, var(--muted-foreground));
}
.kanban-drawer-body {
  font-size: 13px;
  line-height: 1.5;
}
.kanban-drawer-body code {
  background: color-mix(in srgb, currentColor 8%, transparent);
  border-radius: 3px;
  padding: 1px 4px;
  font-size: 12px;
}
.kanban-drawer-body pre {
  background: var(--ui-bg-muted, var(--muted));
  border-radius: 4px;
  padding: 8px;
  overflow-x: auto;
  font-size: 12px;
}
.kanban-drawer-body-edit {
  font-size: 13px;
  min-height: 80px;
}
.kanban-drawer-attachments {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.kanban-drawer-attachment {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  padding: 4px 6px;
  border-radius: 4px;
  background: var(--ui-bg-muted, var(--muted));
}
.kanban-drawer-attachment-size {
  margin-left: auto;
  opacity: 0.6;
}
.kanban-drawer-events {
  display: flex;
  flex-direction: column;
  gap: 2px;
  max-height: 200px;
  overflow-y: auto;
}
.kanban-drawer-event {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 11px;
  padding: 3px 6px;
  border-radius: 3px;
  background: var(--ui-bg-muted, var(--muted));
}
.kanban-drawer-event-kind {
  font-family: monospace;
  font-weight: 500;
}
.kanban-drawer-event-time {
  opacity: 0.6;
}
.kanban-drawer-comments {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-bottom: 8px;
}
.kanban-drawer-no-comments {
  font-size: 12px;
  font-style: italic;
  opacity: 0.5;
  padding: 8px 0;
}
.kanban-drawer-comment {
  padding: 6px 8px;
  border-radius: 6px;
  background: var(--ui-bg-muted, var(--muted));
}
.kanban-drawer-comment-author {
  font-size: 11px;
  font-weight: 600;
  margin-bottom: 2px;
  display: flex;
  align-items: center;
  gap: 6px;
}
.kanban-drawer-comment-time {
  font-weight: 400;
  opacity: 0.6;
}
.kanban-drawer-comment-body {
  font-size: 13px;
  line-height: 1.4;
}
.kanban-drawer-comment-input-row {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.kanban-drawer-comment-input {
  font-size: 13px;
}

/* --- Create task dialog --- */
.kanban-create-form {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 4px 0;
}
.kanban-create-form label {
  font-size: 12px;
  font-weight: 500;
}
.kanban-create-row {
  display: flex;
  gap: 12px;
}
.kanban-create-row > div {
  flex: 1;
}
.kanban-create-error {
  font-size: 12px;
  color: var(--ui-destructive, #ef4444);
  padding: 4px 0;
}
.kanban-switch-label {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  cursor: pointer;
}
`

// ---------------------------------------------------------------------------
// Style injector
// ---------------------------------------------------------------------------

let _stylesInjected = false

function injectStyles() {
  if (_stylesInjected) return
  _stylesInjected = true
  const el = document.createElement('style')
  el.setAttribute('data-hermes-plugin', 'kanban')
  el.textContent = PLUGIN_CSS
  document.head.appendChild(el)
}

// ---------------------------------------------------------------------------
// H — jsx shorthand
// ---------------------------------------------------------------------------
const h = jsx

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

function useKanbanBoard(board, tenantFilter, includeArchived) {
  return useQuery({
    queryKey: ['kanban', 'board', board, tenantFilter, includeArchived],
    queryFn: async () => {
      const p = new URLSearchParams()
      if (tenantFilter) p.set('tenant', tenantFilter)
      if (includeArchived) p.set('include_archived', 'true')
      const qs = p.toString()
      return rest(`/board${qs ? '?' + qs : ''}`)
    },
    refetchInterval: 30_000,
    staleTime: 5_000,
  })
}

function useKanbanBoardList() {
  return useQuery({
    queryKey: ['kanban', 'boards'],
    queryFn: () => rest('/boards'),
    staleTime: 30_000,
  })
}

function useKanbanConfig() {
  return useQuery({
    queryKey: ['kanban', 'config'],
    queryFn: () => rest('/config'),
    staleTime: 60_000,
  })
}

function useKanbanTask(taskId) {
  return useQuery({
    queryKey: ['kanban', 'task', taskId],
    queryFn: async () => {
      const resp = await rest(`/tasks/${taskId}`)
      // API returns { task: {...}, comments: [...], events: [...], ... }
      // Flatten so the drawer can access title, status etc. directly
      return { ...resp.task, comments: resp.comments || [], events: resp.events || [], attachments: resp.attachments || [], runs: resp.runs || [] }
    },
    enabled: !!taskId,
    staleTime: 5_000,
  })
}

function useKanbanAttachments(taskId) {
  return useQuery({
    queryKey: ['kanban', 'attachments', taskId],
    queryFn: () => rest(`/tasks/${taskId}/attachments`),
    enabled: !!taskId,
    staleTime: 10_000,
  })
}

function useKanbanAssignees() {
  return useQuery({
    queryKey: ['kanban', 'assignees'],
    queryFn: () => rest('/assignees'),
    staleTime: 30_000,
  })
}

// ---------------------------------------------------------------------------
// Mutations
// ---------------------------------------------------------------------------

function usePatchTask() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ taskId, patch }) => rest(`/tasks/${taskId}`, { method: 'PATCH', body: patch }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['kanban', 'board'] }) },
  })
}

function useCreateTaskMut() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (task) => rest('/tasks', { method: 'POST', body: task }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['kanban', 'board'] }) },
  })
}

function useDeleteTaskMut() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id) => rest(`/tasks/${id}`, { method: 'DELETE' }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['kanban', 'board'] }) },
  })
}

function useAddCommentMut() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ taskId, body }) => rest(`/tasks/${taskId}/comments`, { method: 'POST', body: { body } }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['kanban', 'task'] }) },
  })
}

function useCreateBoardMut() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (data) => rest('/boards', { method: 'POST', body: data }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['kanban', 'boards'] }) },
  })
}

function useSwitchBoardMut() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (slug) => rest(`/boards/${slug}/switch`, { method: 'POST' }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['kanban', 'boards'] }) },
  })
}

function useBulkPatchMut() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ ids, patch }) => rest('/tasks/bulk', { method: 'POST', body: { task_ids: Array.from(ids), patch } }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['kanban', 'board'] })
    },
  })
}

// ---------------------------------------------------------------------------
// KanbanPage — root component
// ---------------------------------------------------------------------------

function KanbanPage() {
  // Persisted board selection
  const [board, setBoard] = useState(() => {
    try { return window.localStorage.getItem('hermes.kanban.board') || null }
    catch { return null }
  })
  const [tenantFilter, setTenantFilter] = useState('')
  const [assigneeFilter, setAssigneeFilter] = useState('')
  const [includeArchived, setIncludeArchived] = useState(false)
  const [search, setSearch] = useState('')
  const [selectedTaskId, setSelectedTaskId] = useState(null)
  const [selectedIds, setSelectedIds] = useState(() => new Set())
  const [showCreate, setShowCreate] = useState(false)
  const [showNewBoard, setShowNewBoard] = useState(false)
  const [draggingId, setDraggingId] = useState(null)
  const [deleteConfirm, setDeleteConfirm] = useState(null)
  const [configApplied, setConfigApplied] = useState(false)
  const [eventTick, setEventTick] = useState({})
  const wsRef = useRef(null)
  const reloadTimer = useRef(null)

  // Inject CSS once
  useEffect(() => { injectStyles() }, [])

  // Data queries
  const { data: boardData, isLoading, error, refetch } = useKanbanBoard(board, tenantFilter, includeArchived)
  const { data: boardListResp } = useKanbanBoardList()
  const { data: config } = useKanbanConfig()
  const { data: assigneesResp } = useKanbanAssignees()

  const boardList = (boardListResp && boardListResp.boards) || []
  const assignees = (assigneesResp && assigneesResp.assignees) || []
  const tenants = (boardData && boardData.tenants) || []

  // Apply config defaults
  useEffect(() => {
    if (config && !configApplied) {
      if (config.default_tenant) setTenantFilter(config.default_tenant)
      setConfigApplied(true)
    }
  }, [config, configApplied])

  // Persist board selection
  useEffect(() => {
    try {
      if (board) window.localStorage.setItem('hermes.kanban.board', board)
      else window.localStorage.removeItem('hermes.kanban.board')
    } catch {}
  }, [board])

  // Resolve default board
  useEffect(() => {
    if (!board && boardList.length > 0) {
      const current = boardListResp && boardListResp.current
      if (current && current !== 'default') setBoard(current)
    }
  }, [board, boardList, boardListResp])

  // Fallback if stored board disappeared
  useEffect(() => {
    if (board && board !== 'default' && boardList.length > 0 &&
        !boardList.find(b => b.slug === board)) {
      setBoard('default')
    }
  }, [board, boardList])

  // WebSocket live updates via ctx.socket
  useEffect(() => {
    if (!boardData) return
    let closed = false
    let disposer = null
    try {
      const qs = new URLSearchParams({ since: String(boardData.latest_event_id || 0) })
      if (board) qs.set('board', board)
      wsRef.current = _ctx.socket(`/events?${qs}`, (data) => {
        if (data && data.task_id) {
          setEventTick(prev => ({ ...prev, [data.task_id]: (prev[data.task_id] || 0) + 1 }))
        }
      })
    } catch {}
    return () => { closed = true; if (wsRef.current) { wsRef.current(); wsRef.current = null } }
  }, [board, boardData])

  // Re-fetch task detail on live events
  const qc = useQueryClient()
  useEffect(() => {
    if (selectedTaskId && eventTick[selectedTaskId]) {
      qc.invalidateQueries({ queryKey: ['kanban', 'task', selectedTaskId] })
    }
  }, [eventTick, selectedTaskId, qc])

  // Mutations
  const patchTask = usePatchTask()
  const deleteTask = useDeleteTaskMut()
  const bulkPatch = useBulkPatchMut()

  // Drag-drop handlers
  const handleDrop = useCallback((taskId, newStatus) => {
    if (!taskId || !newStatus) return
    patchTask.mutate({ taskId, patch: { status: newStatus } })
  }, [patchTask])

  const handleDelete = useCallback((taskId) => {
    setDeleteConfirm(taskId)
  }, [])

  // Build column data
  const columns = useMemo(() => {
    if (!boardData || !boardData.columns) return {}
    const col = {}
    for (const c of COLUMNS) col[c] = []
    for (const colEntry of boardData.columns) {
      const status = colEntry.name
      if (col[status] && Array.isArray(colEntry.tasks)) {
        for (const t of colEntry.tasks) {
          col[status].push(t)
        }
      }
    }
    if (assigneeFilter || search) {
      const ls = search.toLowerCase()
      for (const c of COLUMNS) {
        col[c] = col[c].filter(t => {
          if (assigneeFilter && t.assignee !== assigneeFilter) return false
          if (search && !t.title.toLowerCase().includes(ls) &&
              !(t.body || '').toLowerCase().includes(ls)) return false
          return true
        })
      }
    }
    return col
  }, [boardData, assigneeFilter, search])

  const taskCounts = useMemo(() => {
    const c = {}
    for (const col of COLUMNS) c[col] = (columns[col] || []).length
    return c
  }, [columns])

  // All task IDs for select-all
  const allTaskIds = useMemo(() => {
    const ids = new Set()
    for (const col of COLUMNS) {
      for (const t of (columns[col] || [])) ids.add(t.id)
    }
    return ids
  }, [columns])

  const allSelected = allTaskIds.size > 0 && selectedIds.size === allTaskIds.size
  const selectCount = selectedIds.size

  const toggleSelectAll = useCallback(() => {
    if (allSelected) setSelectedIds(new Set())
    else setSelectedIds(new Set(allTaskIds))
  }, [allSelected, allTaskIds])

  // ---- Error state ----
  if (error) {
    return h('div', { className: 'kanban-page' },
      h(ErrorState, { error: String(error.message || error), onRetry: refetch })
    )
  }

  // ---- Loading ----
  if (isLoading) {
    return h('div', { className: 'kanban-page kanban-loading' },
      h(GlyphSpinner, null)
    )
  }

  return h('div', { className: 'kanban-page' },

    // Header
    h('div', { className: 'kanban-header' },

      // Board selector
      h('div', { className: 'kanban-header-left' },
        h('div', { className: 'kanban-board-selector' },
          h(Select, {
            value: board || 'default',
            onValueChange: (v) => setBoard(v === 'default' ? null : v),
          },
            h(SelectTrigger, { className: 'kanban-board-trigger' },
              h(SelectValue, { placeholder: 'Select board' })
            ),
            h(SelectContent, null,
              h(SelectItem, { value: 'default' }, 'default'),
              boardList.filter(b => b.slug !== 'default').map(b =>
                h(SelectItem, { key: b.slug, value: b.slug }, b.name || b.slug)
              ),
            ),
          ),
        ),
        h(Button, {
          variant: 'ghost',
          size: 'icon-sm',
          onClick: () => setShowNewBoard(true),
          title: 'New board',
        }, h(icons.Plus, { size: 14 })),
      ),

      // Filters
      h('div', { className: 'kanban-filters' },
        h(SearchField, {
          placeholder: 'Search…',
          value: search,
          onChange: (e) => setSearch(e && e.target ? e.target.value : (e || '')),
          className: 'kanban-search',
        }),
        h(Select, {
          value: tenantFilter || '__all__',
          onValueChange: (v) => setTenantFilter(v === '__all__' ? '' : v),
        },
          h(SelectTrigger, { className: 'kanban-filter-select' },
            h(SelectValue, { placeholder: 'Tenant' })
          ),
          h(SelectContent, null,
            h(SelectItem, { value: '__all__' }, 'All tenants'),
            tenants.map(t => h(SelectItem, { key: t, value: t }, t)),
          ),
        ),
        h(Select, {
          value: assigneeFilter || '__all__',
          onValueChange: (v) => setAssigneeFilter(v === '__all__' ? '' : v),
        },
          h(SelectTrigger, { className: 'kanban-filter-select' },
            h(SelectValue, { placeholder: 'Assignee' })
          ),
          h(SelectContent, null,
            h(SelectItem, { value: '__all__' }, 'Everyone'),
            assignees.map(a => h(SelectItem, { key: a, value: a }, a)),
          ),
        ),
        h('label', { className: 'kanban-archived-toggle' },
          h(Checkbox, {
            checked: includeArchived,
            onCheckedChange: (v) => setIncludeArchived(!!v),
          }),
          h('span', null, 'Archived'),
        ),
        h(Button, { variant: 'ghost', size: 'icon-sm', onClick: refetch, title: 'Refresh' },
          h(icons.RefreshCw, { size: 14 })
        ),
      ),

      // New task
      h('div', { className: 'kanban-header-right' },
        h(Button, { variant: 'default', size: 'sm', onClick: () => setShowCreate(true) },
          h(icons.Plus, { size: 14, style: { marginRight: 4 } }),
          'New Task'
        ),
      ),
    ),

    // Bulk action bar
    selectCount > 0 && h('div', { className: 'kanban-bulk-bar' },
      h('span', { className: 'kanban-bulk-label' }, `${selectCount} selected`),
      h('div', { className: 'kanban-bulk-actions' },
        COLUMNS.map(s =>
          h(Button, {
            key: s,
            variant: 'outline',
            size: 'xs',
            onClick: () => bulkPatch.mutate({ ids: selectedIds, patch: { status: s } }),
          }, `→ ${colLabel(s)}`)
        ),
      ),
    ),

    // Columns
    h('div', {
      className: cn('kanban-columns', draggingId && 'kanban-columns--dragging'),
    },
      COLUMNS.map(status =>
        h(KanbanColumn, {
          key: status,
          status,
          tasks: columns[status] || [],
          label: colLabel(status),
          color: colColor(status),
          Icon: colIcon(status),
          count: taskCounts[status],
          onDrop: handleDrop,
          onDelete: handleDelete,
          onSelect: setSelectedTaskId,
          selectedIds,
          onToggleSelect: (id, multi) => {
            setSelectedIds(prev => {
              const next = new Set(prev)
              if (multi) { if (next.has(id)) next.delete(id); else next.add(id) }
              else { next.clear(); next.add(id) }
              return next
            })
          },
          draggingId,
          onDragStart: setDraggingId,
          onDragEnd: () => setDraggingId(null),
        })
      ),

      // Trash
      h('div', {
        className: cn('kanban-trash', draggingId && 'kanban-trash--visible'),
        onDragOver: (e) => e.preventDefault(),
        onDrop: (e) => {
          e.preventDefault()
          const id = e.dataTransfer.getData('text/plain')
          if (id) handleDelete(id)
        },
      },
        h(icons.Trash2, { size: 16 }),
        h('span', null, 'Delete'),
      ),
    ),

    // Task drawer
    selectedTaskId && h(KanbanDrawer, {
      taskId: selectedTaskId,
      onClose: () => setSelectedTaskId(null),
      eventTick: eventTick[selectedTaskId] || 0,
    }),

    // Dialogs
    showCreate && h(CreateTaskDialog, {
      board,
      onClose: () => setShowCreate(false),
      onCreate: () => { setShowCreate(false); refetch() },
    }),

    showNewBoard && h(NewBoardDialog, {
      onClose: () => setShowNewBoard(false),
      onCreated: (slug) => { setShowNewBoard(false); setBoard(slug); refetch() },
    }),

    deleteConfirm && h(ConfirmDialog, {
      open: true,
      onOpenChange: (open) => { if (!open) setDeleteConfirm(null) },
      title: 'Delete task',
      message: 'Permanently delete this task? This cannot be undone.',
      confirmLabel: 'Delete',
      onConfirm: () => { deleteTask.mutate(deleteConfirm); setDeleteConfirm(null) },
    }),
  )
}

// ---------------------------------------------------------------------------
// KanbanColumn
// ---------------------------------------------------------------------------

function KanbanColumn({ status, tasks, label, color, Icon, count, onDrop, onSelect, selectedIds, onToggleSelect, draggingId, onDragStart, onDragEnd }) {
  const handleDragOver = useCallback((e) => { e.preventDefault(); e.dataTransfer.dropEffect = 'move' }, [])
  const handleDropCb = useCallback((e) => {
    e.preventDefault()
    const id = e.dataTransfer.getData('text/plain')
    if (id) onDrop(id, status)
    onDragEnd()
  }, [status, onDrop, onDragEnd])

  return h('div', {
    className: cn('kanban-column', `kanban-column--${status}`),
    'data-status': status,
    onDragOver: handleDragOver,
    onDrop: handleDropCb,
  },
    h('div', { className: 'kanban-column-header' },
      h('div', { className: 'kanban-column-title' },
        h(Icon, { size: 14, style: { color, marginRight: 6 } }),
        h('span', null, label),
        h(Badge, { variant: 'secondary', className: 'kanban-column-count' }, count),
      ),
    ),
    h('div', { className: 'kanban-column-cards' },
      tasks.length === 0
        ? h('div', { className: 'kanban-column-empty' }, 'Drop cards here')
        : tasks.map(task =>
            h(TaskCard, {
              key: task.id,
              task,
              onSelect,
              selected: selectedIds && selectedIds.has(task.id),
              onToggleSelect: (multi) => onToggleSelect(task.id, multi),
              onDragStart,
              onDragEnd,
            })
          ),
    ),
  )
}

// ---------------------------------------------------------------------------
// TaskCard
// ---------------------------------------------------------------------------

function TaskCard({ task, onSelect, selected, onToggleSelect, onDragStart, onDragEnd }) {
  const handleDragStart = useCallback((e) => {
    e.dataTransfer.setData('text/plain', task.id)
    e.dataTransfer.effectAllowed = 'move'
    onDragStart(task.id)
  }, [task.id, onDragStart])

  const handleClick = useCallback((e) => {
    if (e.target.closest('.kanban-card-select')) return
    onSelect(task.id)
  }, [task.id, onSelect])

  const priColor = task.priority === 'high' ? 'var(--ui-kanban-blocked, #ef4444)'
    : task.priority === 'medium' ? 'var(--ui-kanban-running, #f59e0b)'
    : 'var(--ui-text-quaternary)'

  const age = taskAge(task)

  return h('div', {
    className: cn('kanban-card', selected && 'kanban-card--selected', task.status === 'blocked' && 'kanban-card--blocked'),
    draggable: true,
    onDragStart: handleDragStart,
    onDragEnd,
    onClick: handleClick,
  },
    h('div', { className: 'kanban-card-top' },
      h('div', { className: 'kanban-card-select' },
        h(Checkbox, {
          checked: !!selected,
          onCheckedChange: () => onToggleSelect(true),
        }),
      ),
      h('div', { style: { width: 6, height: 6, borderRadius: '50%', backgroundColor: priColor, flexShrink: 0 } }),
      task.tenant && h(Badge, { variant: 'outline', className: 'kanban-card-tenant' }, task.tenant),
    ),
    h('div', { className: 'kanban-card-title' }, task.title || '(untitled)'),
    h('div', { className: 'kanban-card-footer' },
      task.assignee
        ? h('div', { className: 'kanban-card-assignee', style: { color: asgnColor(task.assignee) } }, task.assignee)
        : h('span', { className: 'kanban-card-unassigned' }, 'unassigned'),
      age && h('span', { className: 'kanban-card-age' }, age),
    ),
  )
}

// ---------------------------------------------------------------------------
// KanbanDrawer — task detail side panel
// ---------------------------------------------------------------------------

function KanbanDrawer({ taskId, onClose, eventTick }) {
  const { data: task, isLoading, error } = useKanbanTask(taskId)
  const { data: attachments } = useKanbanAttachments(taskId)
  const patchTask = usePatchTask()
  const addComment = useAddCommentMut()
  const qc = useQueryClient()

  const [commentText, setCommentText] = useState('')
  const [editingBody, setEditingBody] = useState(false)

  useEffect(() => {
    if (eventTick > 0) qc.invalidateQueries({ queryKey: ['kanban', 'task', taskId] })
  }, [eventTick, taskId, qc])

  const handleStatusChange = useCallback((val) => {
    patchTask.mutate({ taskId, patch: { status: val } })
  }, [taskId, patchTask])

  const handleComment = useCallback(() => {
    if (!commentText.trim()) return
    addComment.mutate({ taskId, body: commentText.trim() }, { onSuccess: () => setCommentText('') })
  }, [taskId, commentText, addComment])

  const handleTitleBlur = useCallback((e) => {
    const val = e.target ? e.target.value : e
    if (val && val !== (task && task.title)) patchTask.mutate({ taskId, patch: { title: val } })
  }, [taskId, task, patchTask])

  const handleBodyBlur = useCallback((e) => {
    const val = e.target ? e.target.value : e
    patchTask.mutate({ taskId, patch: { body: val } })
  }, [taskId, patchTask])

  const comments = (task && task.comments) || []
  const events = (task && task.events) || []

  return h('div', { className: 'kanban-drawer-overlay', onClick: onClose },
    h('div', { className: 'kanban-drawer', onClick: (e) => e.stopPropagation() },

      // Header
      h('div', { className: 'kanban-drawer-header' },
        h('div', { className: 'kanban-drawer-title-row' },
          h('div', { className: 'kanban-drawer-status-control' },
            h(Select, {
              value: (task && task.status) || 'triage',
              onValueChange: handleStatusChange,
            },
              h(SelectTrigger, { className: 'kanban-drawer-status-trigger' },
                h(SelectValue, null)
              ),
              h(SelectContent, null,
                COLUMNS.map(s => h(SelectItem, { key: s, value: s }, colLabel(s)))
              ),
            ),
          ),
          h(Button, { variant: 'ghost', size: 'icon-sm', onClick: onClose },
            h(icons.X, { size: 16 })
          ),
        ),
        h(Input, {
          className: 'kanban-drawer-title-input',
          defaultValue: task && task.title,
          onBlur: handleTitleBlur,
          placeholder: 'Task title…',
        }),
      ),

      isLoading && h('div', { className: 'kanban-drawer-loading' }, h(GlyphSpinner, null)),

      error && h(ErrorState, { error: 'Failed to load task' }),

      task && h('div', { className: 'kanban-drawer-content' },

        // Meta
        h('div', { className: 'kanban-drawer-meta' },
          task.assignee && h('div', { className: 'kanban-drawer-meta-item' },
            h('span', { className: 'kanban-drawer-meta-label' }, 'Assignee:'),
            h('span', { style: { color: asgnColor(task.assignee) } }, task.assignee),
          ),
          task.priority && h('div', { className: 'kanban-drawer-meta-item' },
            h('span', { className: 'kanban-drawer-meta-label' }, 'Priority:'),
            h('span', null, task.priority),
          ),
          task.tenant && h('div', { className: 'kanban-drawer-meta-item' },
            h('span', { className: 'kanban-drawer-meta-label' }, 'Tenant:'),
            h('span', null, task.tenant),
          ),
          task.created_by && h('div', { className: 'kanban-drawer-meta-item' },
            h('span', { className: 'kanban-drawer-meta-label' }, 'By:'),
            h('span', null, task.created_by, task.created_at ? ' ' + fmtDateTime(task.created_at * 1000) : ''),
          ),
          taskAge(task) && h('div', { className: 'kanban-drawer-meta-item' },
            h('span', { className: 'kanban-drawer-meta-label' }, 'Age:'),
            h('span', null, taskAge(task)),
          ),
          task.result && h('div', { className: 'kanban-drawer-meta-item' },
            h('span', { className: 'kanban-drawer-meta-label' }, 'Result:'),
            h('span', null, String(task.result).substring(0, 100)),
          ),
        ),

        h(Separator, null),

        // Description
        h('div', { className: 'kanban-drawer-section' },
          h('div', { className: 'kanban-drawer-section-header' },
            h('span', { className: 'kanban-drawer-section-title' }, 'Description'),
            h(Button, { variant: 'ghost', size: 'icon-xs', onClick: () => setEditingBody(!editingBody) },
              editingBody ? h(icons.Eye, { size: 12 }) : h(icons.Pencil, { size: 12 }),
            ),
          ),
          editingBody
            ? h(Textarea, {
                className: 'kanban-drawer-body-edit',
                defaultValue: task.body || '',
                onBlur: handleBodyBlur,
                placeholder: 'Markdown description…',
                rows: 5,
              })
            : h('div', {
                className: 'kanban-drawer-body',
                dangerouslySetInnerHTML: { __html: renderMd(task.body || '*No description*') },
              }),
        ),

        h(Separator, null),

        // Attachments
        attachments && attachments.length > 0 && h('div', { className: 'kanban-drawer-section' },
          h('div', { className: 'kanban-drawer-section-header' },
            h('span', { className: 'kanban-drawer-section-title' }, `Attachments (${attachments.length})`),
          ),
          h('div', { className: 'kanban-drawer-attachments' },
            attachments.map(a =>
              h('div', { key: a.id, className: 'kanban-drawer-attachment' },
                h(icons.Link, { size: 12 }),
                h('span', null, a.filename || 'file'),
                a.size && h('span', { className: 'kanban-drawer-attachment-size' }, fmtSize(a.size)),
              )
            ),
          ),
        ),

        h(Separator, null),

        // Events
        events.length > 0 && h('div', { className: 'kanban-drawer-section' },
          h('div', { className: 'kanban-drawer-section-header' },
            h('span', { className: 'kanban-drawer-section-title' }, `Events (${events.length})`),
          ),
          h('div', { className: 'kanban-drawer-events' },
            events.slice(-20).map(ev =>
              h('div', { key: ev.id, className: 'kanban-drawer-event' },
                h('span', { className: 'kanban-drawer-event-kind' }, ev.kind || '?'),
                ev.created_at && h('span', { className: 'kanban-drawer-event-time' }, relativeTime(ev.created_at * 1000)),
              )
            ),
          ),
        ),

        h(Separator, null),

        // Comments
        h('div', { className: 'kanban-drawer-section' },
          h('div', { className: 'kanban-drawer-section-header' },
            h('span', { className: 'kanban-drawer-section-title' }, `Comments (${comments.length})`),
          ),
          h('div', { className: 'kanban-drawer-comments' },
            comments.length === 0 && h('div', { className: 'kanban-drawer-no-comments' }, 'No comments'),
            comments.map(c =>
              h('div', { key: c.id, className: 'kanban-drawer-comment' },
                h('div', { className: 'kanban-drawer-comment-author', style: { color: asgnColor(c.author) } },
                  c.author || '?',
                  c.created_at && h('span', { className: 'kanban-drawer-comment-time' }, relativeTime(c.created_at * 1000)),
                ),
                h('div', {
                  className: 'kanban-drawer-comment-body',
                  dangerouslySetInnerHTML: { __html: renderMd(c.body) },
                }),
              )
            ),
          ),
          h('div', { className: 'kanban-drawer-comment-input-row' },
            h(Textarea, {
              className: 'kanban-drawer-comment-input',
              placeholder: 'Add a comment…',
              value: commentText,
              onChange: (e) => setCommentText(e && e.target ? e.target.value : (e || '')),
              rows: 2,
            }),
            h(Button, {
              variant: 'default',
              size: 'sm',
              onClick: handleComment,
              disabled: !commentText.trim() || addComment.isPending,
            }, 'Comment'),
          ),
        ),
      ),
    ),
  )
}

function fmtSize(bytes) {
  if (!bytes) return ''
  if (bytes > 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  if (bytes > 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${bytes} B`
}

// ---------------------------------------------------------------------------
// CreateTaskDialog
// ---------------------------------------------------------------------------

function CreateTaskDialog({ board, onClose, onCreate }) {
  const createTask = useCreateTaskMut()
  const [title, setTitle] = useState('')
  const [body, setBody] = useState('')
  const [status, setStatus] = useState('triage')
  const [priority, setPriority] = useState('medium')
  const [assignee, setAssignee] = useState('')
  const [tenant, setTenant] = useState('')
  const [err, setErr] = useState('')

  const handleSubmit = useCallback(() => {
    if (!title.trim()) { setErr('Title is required'); return }
    setErr('')
    createTask.mutate({
      title: title.trim(),
      body: body.trim(),
      status,
      priority,
      assignee: assignee.trim() || null,
      tenant: tenant.trim() || null,
    }, {
      onSuccess: onCreate,
      onError: (e) => setErr(String(e.message || e)),
    })
  }, [title, body, status, priority, assignee, tenant, createTask, onCreate])

  return h(Dialog, { open: true, onOpenChange: (open) => { if (!open) onClose() } },
    h(DialogContent, null,
      h(DialogHeader, null,
        h(DialogTitle, null, 'Create Task'),
        h(DialogDescription, null, 'Add a new task to the kanban board'),
      ),
      h('div', { className: 'kanban-create-form' },
        h('label', null, 'Title *'),
        h(Input, { value: title, onChange: (e) => setTitle(e && e.target ? e.target.value : (e || '')), placeholder: 'Task title…', autoFocus: true }),
        h('label', null, 'Description (markdown)'),
        h(Textarea, { value: body, onChange: (e) => setBody(e && e.target ? e.target.value : (e || '')), placeholder: 'Optional description…', rows: 4 }),
        h('div', { className: 'kanban-create-row' },
          h('div', null,
            h('label', null, 'Status'),
            h(Select, { value: status, onValueChange: setStatus },
              h(SelectTrigger, null, h(SelectValue, null)),
              h(SelectContent, null,
                COLUMNS.filter(s => s !== 'done').map(s => h(SelectItem, { key: s, value: s }, colLabel(s))),
              ),
            ),
          ),
          h('div', null,
            h('label', null, 'Priority'),
            h(Select, { value: priority, onValueChange: setPriority },
              h(SelectTrigger, null, h(SelectValue, null)),
              h(SelectContent, null,
                h(SelectItem, { value: 'low' }, 'Low'),
                h(SelectItem, { value: 'medium' }, 'Medium'),
                h(SelectItem, { value: 'high' }, 'High'),
              ),
            ),
          ),
        ),
        h('div', { className: 'kanban-create-row' },
          h('div', null,
            h('label', null, 'Assignee'),
            h(Input, { value: assignee, onChange: (e) => setAssignee(e && e.target ? e.target.value : (e || '')), placeholder: 'Profile name…' }),
          ),
          h('div', null,
            h('label', null, 'Tenant'),
            h(Input, { value: tenant, onChange: (e) => setTenant(e && e.target ? e.target.value : (e || '')), placeholder: 'Optional tenant…' }),
          ),
        ),
        err && h('div', { className: 'kanban-create-error' }, err),
      ),
      h(DialogFooter, null,
        h(Button, { variant: 'outline', onClick: onClose }, 'Cancel'),
        h(Button, { variant: 'default', onClick: handleSubmit, disabled: createTask.isPending || !title.trim() },
          createTask.isPending ? 'Creating…' : 'Create'
        ),
      ),
    ),
  )
}

// ---------------------------------------------------------------------------
// NewBoardDialog
// ---------------------------------------------------------------------------

function NewBoardDialog({ onClose, onCreated }) {
  const createBoard = useCreateBoardMut()
  const switchBoard = useSwitchBoardMut()
  const [slug, setSlug] = useState('')
  const [name, setName] = useState('')
  const [switchAfter, setSwitchAfter] = useState(true)
  const [err, setErr] = useState('')

  const norm = (s) => s.trim().toLowerCase().replace(/[^a-z0-9-]/g, '-')

  const handleCreate = useCallback(() => {
    if (!slug.trim()) { setErr('Slug is required'); return }
    setErr('')
    const ns = norm(slug)
    createBoard.mutate({ slug: ns, name: name.trim() || null }, {
      onSuccess: () => {
        if (switchAfter) {
          switchBoard.mutate(ns, { onSuccess: () => onCreated(ns) })
        } else {
          onCreated(ns)
        }
      },
      onError: (e) => setErr(String(e.message || e)),
    })
  }, [slug, name, switchAfter, createBoard, switchBoard, onCreated])

  return h(Dialog, { open: true, onOpenChange: (open) => { if (!open) onClose() } },
    h(DialogContent, null,
      h(DialogHeader, null,
        h(DialogTitle, null, 'Create Board'),
        h(DialogDescription, null, 'Create a new kanban board for a separate workstream'),
      ),
      h('div', { className: 'kanban-create-form' },
        h('label', null, 'Slug *'),
        h(Input, { value: slug, onChange: (e) => setSlug(e && e.target ? e.target.value : (e || '')), placeholder: 'my-project', autoFocus: true }),
        h('label', null, 'Display name'),
        h(Input, { value: name, onChange: (e) => setName(e && e.target ? e.target.value : (e || '')), placeholder: 'My Project' }),
        h('label', { className: 'kanban-switch-label' },
          h(Checkbox, { checked: switchAfter, onCheckedChange: (v) => setSwitchAfter(!!v) }),
          ' Switch to this board after creation',
        ),
        err && h('div', { className: 'kanban-create-error' }, err),
      ),
      h(DialogFooter, null,
        h(Button, { variant: 'outline', onClick: onClose }, 'Cancel'),
        h(Button, { variant: 'default', onClick: handleCreate, disabled: createBoard.isPending || !slug.trim() },
          createBoard.isPending ? 'Creating…' : 'Create'
        ),
      ),
    ),
  )
}

// ---------------------------------------------------------------------------
// Plugin registration
// ---------------------------------------------------------------------------

export default {
  id: 'kanban',
  name: 'Kanban Board',
  defaultEnabled: true,

  register(ctx) {
    _ctx = ctx
    injectStyles()

    // Sidebar nav
    ctx.register({
      id: 'nav',
      area: SIDEBAR_NAV_AREA,
      data: {
        path: '/kanban',
        label: 'Kanban',
        codicon: 'project',
      },
    })

    // Route — full-page board
    ctx.register({
      id: 'page',
      area: ROUTES_AREA,
      data: { path: '/kanban' },
      render: () => jsx(KanbanPage, {}),
    })

    // Command palette
    ctx.register({
      id: 'open',
      area: PALETTE_AREA,
      data: {
        id: 'kanban:open',
        label: 'Open Kanban Board',
        codicon: 'project',
        action: () => host.navigate('/kanban'),
      },
    })

    // Keybind: Cmd+Shift+K
    ctx.register({
      id: 'keybind',
      area: KEYBINDS_AREA,
      data: {
        id: 'kanban:open',
        keys: ['mod+shift+k'],
        action: () => host.navigate('/kanban'),
      },
    })
  },
}
