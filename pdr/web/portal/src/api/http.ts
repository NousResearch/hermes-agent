import type { AskEvent, Attachment, PortalApi, PortalUser, SavedAnswer, Scope, SessionDetail, SessionSummary, SuggestedQuestion } from './types'

const BASE = import.meta.env.VITE_PORTAL_API_BASE || '/api/portal/v1'
let token = ''

export function setHttpToken(value: string): void {
  token = value
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message)
  }
}

async function call<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers)
  if (token) headers.set('Authorization', `Bearer ${token}`)
  if (init.body && !(init.body instanceof FormData)) headers.set('Content-Type', 'application/json')
  const res = await fetch(BASE + path, { ...init, headers })
  const body = res.headers.get('content-type')?.includes('json') ? await res.json() : null
  // 统一响应：{ code, message, data }；code 非 0 为业务错误
  if (!res.ok || (body && body.code !== 0)) {
    throw new ApiError(res.status, body?.message || `请求失败（${res.status}）`)
  }
  return body?.data as T
}

const json = (data: unknown): RequestInit => ({ method: 'POST', body: JSON.stringify(data) })

/** 读取 SSE 流：每个事件是一行 `data: {json}`，空行分隔 */
async function readSse(res: Response, onEvent: (e: AskEvent) => void): Promise<void> {
  const reader = res.body!.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let idx: number
    while ((idx = buffer.indexOf('\n\n')) >= 0) {
      const chunk = buffer.slice(0, idx)
      buffer = buffer.slice(idx + 2)
      const data = chunk.split('\n').filter((l) => l.startsWith('data:')).map((l) => l.slice(5).trim()).join('')
      if (data) onEvent(JSON.parse(data) as AskEvent)
    }
  }
}

export const httpApi: PortalApi = {
  mode: 'http',
  login: (account, password) => call<{ token: string; user: PortalUser }>('/auth/login', json({ account, password })),
  me: () => call<PortalUser>('/me'),
  logout: () => call<void>('/auth/logout', { method: 'POST' }),
  scopes: () => call<Scope[]>('/scopes'),
  suggestedQuestions: () => call<SuggestedQuestion[]>('/suggested-questions'),
  sessions: () => call<SessionSummary[]>('/sessions'),
  session: (id) => call<SessionDetail>(`/sessions/${encodeURIComponent(id)}`),
  createSession: (title) => call<SessionSummary>('/sessions', json({ title })),
  renameSession: (id, title) => call<void>(`/sessions/${encodeURIComponent(id)}`, { method: 'PATCH', body: JSON.stringify({ title }) }),
  deleteSession: (id) => call<void>(`/sessions/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  async ask(sessionId, question, opts, onEvent, signal) {
    const res = await fetch(`${BASE}/sessions/${encodeURIComponent(sessionId)}/ask/stream`, {
      method: 'POST',
      signal,
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream', Authorization: `Bearer ${token}` },
      body: JSON.stringify({ question, ...opts })
    })
    if (!res.ok || !res.body) throw new ApiError(res.status, `提问失败（${res.status}）`)
    await readSse(res, onEvent)
  },
  feedback: (messageId, value) => call<void>(`/messages/${encodeURIComponent(messageId)}/feedback`, json({ value })),
  setSaved: (messageId, saved) => call<void>(`/messages/${encodeURIComponent(messageId)}/saved`, json({ saved })),
  savedAnswers: () => call<SavedAnswer[]>('/saved-answers'),
  uploadAttachment(file) {
    const form = new FormData()
    form.append('file', file)
    return call<Attachment>('/attachments', { method: 'POST', body: form })
  }
}
