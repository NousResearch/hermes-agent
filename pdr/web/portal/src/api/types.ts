/**
 * 用户端接口约定（/api/portal/v1，令牌受众 aud=portal）。
 * 模拟实现（mock.ts）与真实实现（http.ts）都遵守这里的类型，后端按此实现即可无缝切换。
 */

export interface PortalUser {
  userCode: string
  userName: string
  account: string
  tenantName: string
}

/** 资料范围：用户在其数据权限内可选的知识库 */
export interface Scope {
  code: string
  name: string
  desc: string
}

export interface SuggestedQuestion {
  id: string
  text: string
  category: string
  scopeCode?: string
}

export interface SessionSummary {
  id: string
  title: string
  updatedAt: string
}

export type CitationKind = 'pdf' | 'word' | 'excel' | 'markdown' | 'image' | 'video' | 'audio'

export interface Citation {
  /** 回答正文里的角标序号 [n] */
  n: number
  chunkNo: string
  title: string
  kind: CitationKind
  /** 位置说明，如“第 3 页 · 线路价格表”“01:04–01:28” */
  location: string
  quote: string
  startMs?: number
  endMs?: number
}

export type MessageStatus = 'searching' | 'streaming' | 'done' | 'error' | 'stopped'

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  createdAt: string
  status?: MessageStatus
  retrievedQty?: number
  citations?: Citation[]
  followUps?: string[]
  feedback?: -1 | 0 | 1
  saved?: boolean
}

export interface SessionDetail {
  summary: SessionSummary
  messages: ChatMessage[]
}

export interface SavedAnswer {
  messageId: string
  sessionId: string
  question: string
  answer: string
  savedAt: string
}

export interface AskOptions {
  scopeCode: string
  /** 深度检索：召回更多资料、回答更完整，耗时更长 */
  deep: boolean
  /** 本次对话的临时附件 ID（不进入内容流水线） */
  attachmentIds: string[]
}

/**
 * 流式回答事件（SSE：每个事件一行 `data: {json}`）
 *   retrieval → citations → delta… → follow_ups → done
 */
export type AskEvent =
  | { type: 'retrieval'; qty: number }
  | { type: 'citations'; citations: Citation[] }
  | { type: 'delta'; text: string }
  | { type: 'follow_ups'; items: string[] }
  | { type: 'done'; messageId: string }
  | { type: 'error'; message: string }

export interface Attachment {
  id: string
  name: string
  size: number
}

export interface PortalApi {
  readonly mode: 'mock' | 'http'
  login(account: string, password: string): Promise<{ token: string; user: PortalUser }>
  me(): Promise<PortalUser>
  logout(): Promise<void>
  scopes(): Promise<Scope[]>
  suggestedQuestions(): Promise<SuggestedQuestion[]>
  sessions(): Promise<SessionSummary[]>
  session(id: string): Promise<SessionDetail>
  createSession(title: string): Promise<SessionSummary>
  renameSession(id: string, title: string): Promise<void>
  deleteSession(id: string): Promise<void>
  ask(sessionId: string, question: string, opts: AskOptions, onEvent: (e: AskEvent) => void, signal: AbortSignal): Promise<void>
  feedback(messageId: string, value: -1 | 0 | 1): Promise<void>
  setSaved(messageId: string, saved: boolean): Promise<void>
  savedAnswers(): Promise<SavedAnswer[]>
  uploadAttachment(file: File): Promise<Attachment>
}
