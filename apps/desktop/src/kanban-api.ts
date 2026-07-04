import type {
  KanbanAssignee,
  KanbanAttachment,
  KanbanAttachmentUploadResponse,
  KanbanBoardPayload,
  KanbanBoardsResponse,
  KanbanCreateTaskPayload,
  KanbanCreateTaskResponse,
  KanbanDispatchResult,
  KanbanTaskDetail,
  KanbanTaskLog,
  KanbanUpdateTaskPayload,
  KanbanUpdateTaskResponse
} from '@/types/kanban'

export type {
  KanbanAssignee,
  KanbanAttachment,
  KanbanAttachmentUploadResponse,
  KanbanBoardInfo,
  KanbanBoardPayload,
  KanbanBoardsResponse,
  KanbanCard,
  KanbanColumn,
  KanbanComment,
  KanbanCreateTaskPayload,
  KanbanCreateTaskResponse,
  KanbanDiagnostic,
  KanbanDispatchResult,
  KanbanEvent,
  KanbanLinks,
  KanbanRun,
  KanbanTask,
  KanbanTaskDetail,
  KanbanTaskLog,
  KanbanUpdateTaskPayload,
  KanbanUpdateTaskResponse,
  KanbanWarningsSummary
} from '@/types/kanban'

// Kanban REST wrappers, deliberately split out of hermes.ts: kanban is
// HOME-scoped, not profile-scoped. Boards are the shared cross-profile
// coordination bus (profiles are the assignees/workers), so these calls must
// NEVER carry profileScoped() / ?profile= — every profile sees the same board.
// Board selection rides the explicit ?board= query param instead; omitting it
// falls through to the backend's active board (env → current file → default).

const KANBAN_BASE = '/api/plugins/kanban'

function kanbanPath(path: string, board?: null | string): string {
  if (!board) {
    return `${KANBAN_BASE}${path}`
  }

  const sep = path.includes('?') ? '&' : '?'

  return `${KANBAN_BASE}${path}${sep}board=${encodeURIComponent(board)}`
}

export function getKanbanBoards(): Promise<KanbanBoardsResponse> {
  return window.hermesDesktop.api<KanbanBoardsResponse>({
    path: kanbanPath('/boards')
  })
}

export function getKanbanBoard(board?: null | string): Promise<KanbanBoardPayload> {
  return window.hermesDesktop.api<KanbanBoardPayload>({
    path: kanbanPath('/board', board)
  })
}

export function getKanbanTask(taskId: string, board?: null | string): Promise<KanbanTaskDetail> {
  return window.hermesDesktop.api<KanbanTaskDetail>({
    path: kanbanPath(`/tasks/${encodeURIComponent(taskId)}`, board)
  })
}

export function createKanbanTask(
  payload: KanbanCreateTaskPayload,
  board?: null | string
): Promise<KanbanCreateTaskResponse> {
  return window.hermesDesktop.api<KanbanCreateTaskResponse>({
    path: kanbanPath('/tasks', board),
    method: 'POST',
    body: payload
  })
}

export function updateKanbanTask(
  taskId: string,
  updates: KanbanUpdateTaskPayload,
  board?: null | string
): Promise<KanbanUpdateTaskResponse> {
  return window.hermesDesktop.api<KanbanUpdateTaskResponse>({
    path: kanbanPath(`/tasks/${encodeURIComponent(taskId)}`, board),
    method: 'PATCH',
    body: updates
  })
}

export function addKanbanComment(
  taskId: string,
  body: string,
  board?: null | string,
  author = 'desktop'
): Promise<{ ok: boolean }> {
  return window.hermesDesktop.api<{ ok: boolean }>({
    path: kanbanPath(`/tasks/${encodeURIComponent(taskId)}/comments`, board),
    method: 'POST',
    body: { body, author }
  })
}

/** POST /dispatch — run one dispatcher pass now instead of waiting for the
 *  ~60s gateway tick. This is the only sanctioned path into 'running'
 *  ('ready → running' happens via dispatcher claim; a direct PATCH
 *  status=running is rejected with 400 — 409 is reserved for invalid
 *  transitions / blocking parents). */
export function kanbanDispatchNudge(board?: null | string): Promise<KanbanDispatchResult> {
  return window.hermesDesktop.api<KanbanDispatchResult>({
    path: kanbanPath('/dispatch', board),
    method: 'POST',
    body: {},
    // Spawning workers (git worktrees, process launch) can be slow.
    timeoutMs: 60_000
  })
}

/** Union of on-disk profiles and board assignees — feeds the assignee pickers
 *  so a fresh profile is selectable before it holds any task. */
export async function getKanbanAssignees(board?: null | string): Promise<KanbanAssignee[]> {
  const { assignees } = await window.hermesDesktop.api<{ assignees: KanbanAssignee[] }>({
    path: kanbanPath('/assignees', board)
  })

  return assignees ?? []
}

export function getKanbanTaskLog(taskId: string, board?: null | string, tailBytes = 20_000): Promise<KanbanTaskLog> {
  return window.hermesDesktop.api<KanbanTaskLog>({
    path: kanbanPath(`/tasks/${encodeURIComponent(taskId)}/log?tail=${tailBytes}`, board)
  })
}

// Attachment bytes ride a dedicated binary IPC (hermes:kanban:attachment:*):
// the hermes:api pipe is JSON-only, and the blob must cross the same authed
// HTTP channel as REST in remote mode. Both throw a plain Error when the
// preload bridge predates the channel (old app shell + new renderer).

function kanbanAttachmentBridge(): NonNullable<(typeof window.hermesDesktop)['kanbanAttachment']> {
  const bridge = window.hermesDesktop.kanbanAttachment

  if (!bridge) {
    throw new Error('Attachment transfer requires an updated desktop app.')
  }

  return bridge
}

/** GET /attachments/:id → saved into the OS Downloads dir; resolves with the
 *  written path (collision-resolved, so it may differ from the filename). */
export function downloadKanbanAttachment(
  attachment: Pick<KanbanAttachment, 'filename' | 'id'>,
  board?: null | string
): Promise<{ path: string }> {
  return kanbanAttachmentBridge().download({
    path: kanbanPath(`/attachments/${attachment.id}`, board),
    filename: attachment.filename
  })
}

/** Multipart POST /tasks/:id/attachments from a local file path. */
export function uploadKanbanAttachment(
  taskId: string,
  filePath: string,
  board?: null | string
): Promise<KanbanAttachmentUploadResponse> {
  return kanbanAttachmentBridge().upload<KanbanAttachmentUploadResponse>({
    path: kanbanPath(`/tasks/${encodeURIComponent(taskId)}/attachments`, board),
    filePath,
    uploadedBy: 'desktop'
  })
}

export interface KanbanApiError {
  status: null | number
  detail: string
}

// Electron's fetchJson rejects HTTP errors as `Error("<status>: <body>")` where
// the body is FastAPI JSON like {"detail": "..."}. By the time that rejection
// crosses ipcRenderer.invoke it arrives wrapped as
// `Error invoking remote method 'hermes:api': Error: <status>: <body>`, so
// unwrap the IPC prefix first (same pattern as store/notifications.ts). 409
// payloads carry actionable transition messages (blocking parents, invalid
// transitions), so surface the detail text and keep the status for revert
// decisions.
export function parseKanbanApiError(error: unknown): KanbanApiError {
  const raw = error instanceof Error ? error.message : String(error ?? '')
  const unwrapped = /Error invoking remote method '[^']+': Error: ([\s\S]+)$/.exec(raw)?.[1] ?? raw
  const message = unwrapped.replace(/^Error:\s*/, '').trim()
  const match = /^(\d{3}):\s*([\s\S]*)$/.exec(message)

  if (!match) {
    return { status: null, detail: message }
  }

  const status = Number(match[1])
  const body = match[2].trim()

  try {
    const parsed = JSON.parse(body) as { detail?: unknown }

    if (parsed && typeof parsed.detail === 'string' && parsed.detail) {
      return { status, detail: parsed.detail }
    }
  } catch {
    // Non-JSON body — fall through to the raw text.
  }

  return { status, detail: body || message }
}
