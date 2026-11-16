/** 真实接口：/api/admin/v1，令牌受众 aud=admin（用户端令牌调用这里一律 401） */
import { ApiError } from './errors'
import type { AdminApi } from './types'

const BASE = import.meta.env.VITE_ADMIN_API_BASE || '/api/admin/v1'
let token = ''

export function setHttpToken(value: string): void {
  token = value
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

const send = (method: string, data?: unknown): RequestInit => ({ method, body: data === undefined ? undefined : JSON.stringify(data) })
const q = (params: Record<string, string | undefined>) => {
  const s = new URLSearchParams(Object.entries(params).filter(([, v]) => v) as [string, string][]).toString()
  return s ? '?' + s : ''
}
const id = encodeURIComponent

export const httpApi: AdminApi = {
  mode: 'http',
  demoAccounts: async () => [],
  login: (account, password) => call('/auth/login', send('POST', { account, password })),
  logout: () => call('/auth/logout', send('POST')),
  notices: () => call('/notices'),

  dashboard: () => call('/dashboard'),
  assets: (query) => call('/assets' + q({ stage: query.stage, modality: query.modality, keyword: query.keyword })),
  upload: (files) => call('/assets/upload', send('POST', { files })),
  retry: (no) => call(`/assets/${id(no)}/retry`, send('POST')),
  contentBoard: () => call('/content/board'),
  reviewDoc: (no) => call(`/review/docs/${id(no)}`),
  submitReview: (no, blocks, action, tier) => call(`/review/docs/${id(no)}`, send('POST', { blocks, action, tier })),
  mediaReview: (no) => call(`/review/media/${id(no)}`),
  submitMedia: (no, segments) => call(`/review/media/${id(no)}`, send('POST', { segments })),
  reviewQueue: () => call('/review/queue'),
  saveLabel: (labelId, label) => call(`/review/labels/${id(labelId)}`, send('POST', { label })),
  confirmCategory: (no, code) => call(`/review/classify/${id(no)}`, send('POST', { code })),

  knowledgeBases: () => call('/knowledge'),
  knowledgeDocs: (code) => call(`/knowledge/${id(code)}/docs`),
  toggleKnowledgeBase: (code, enabled) => call(`/knowledge/${id(code)}`, send('PATCH', { enabled })),

  configSchema: (scope) => call('/config/schema' + q({ scope })),
  saveConfig: (scope, changes) => call('/config', send('POST', { scope, changes })),
  configLog: () => call('/config/log'),
  toolsets: () => call('/agent/toolsets'),
  toggleToolset: (name, enabled) => call(`/agent/toolsets/${id(name)}`, send('PATCH', { enabled })),
  toggleSkill: (name, enabled) => call(`/agent/skills/${id(name)}`, send('PATCH', { enabled })),
  memory: () => call('/agent/memory'),
  savePersona: (text) => call('/agent/persona', send('PUT', { text })),
  decideMemory: (memoryId, approve) => call(`/agent/memory/${id(memoryId)}/decision`, send('POST', { approve })),
  deleteMemory: (memoryId) => call(`/agent/memory/${id(memoryId)}`, send('DELETE')),
  cronJobs: () => call('/agent/cron'),
  toggleCron: (jobId, enabled) => call(`/agent/cron/${id(jobId)}`, send('PATCH', { enabled })),
  runCron: (jobId) => call(`/agent/cron/${id(jobId)}/run`, send('POST')),
  qaSessions: (keyword) => call('/agent/sessions' + q({ keyword })),

  permCatalog: () => call('/perms'),
  roles: () => call('/roles'),
  saveRole: (role) => call(`/roles/${id(role.code)}`, send('PUT', role)),
  regions: () => call('/regions'),
  users: () => call('/users'),
  setUserStatus: (account, status) => call(`/users/${id(account)}`, send('PATCH', { status })),
  tenants: () => call('/tenants'),
  auditLogs: (module) => call('/audit-logs' + q({ module }))
}
