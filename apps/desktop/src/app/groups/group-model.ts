import { assistantTextPart, type ChatMessagePart, type GatewayEventPayload, upsertToolPart } from '@/lib/chat-messages'
import type { ApprovalRequest } from '@/store/prompts'

export type GroupMessageStatus = 'streaming' | 'complete' | 'error' | 'approval'

export interface GroupMessage {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  parts: ChatMessagePart[]
  approval?: Pick<ApprovalRequest, 'allowPermanent' | 'choices' | 'command' | 'description' | 'smartDenied'>
  profile?: string
  runtimeSessionId?: string
  seq?: number
  status: GroupMessageStatus
  createdAt?: number
}

export interface GroupRoom {
  id: string
  name: string
  profiles: string[]
  messages: GroupMessage[]
  runningProfiles: string[]
  running?: boolean
  workspace?: string
  summary?: string
  contextStatus?: string
}

export interface GroupState {
  rooms: Record<string, GroupRoom>
}

export interface GroupEvent {
  type?: string
  payload?: unknown
}

const record = (value: unknown): Record<string, unknown> =>
  value && typeof value === 'object' ? (value as Record<string, unknown>) : {}

const text = (value: unknown): string => (typeof value === 'string' ? value : '')

function normalizeApproval(value: unknown): GroupMessage['approval'] {
  const raw = record(value)

  const choices = Array.isArray(raw.choices)
    ? raw.choices.filter(choice => choice === 'once' || choice === 'session' || choice === 'always' || choice === 'deny')
    : undefined

  if (!text(raw.command) && !text(raw.description) && !choices) {return undefined}

  return {
    allowPermanent: raw.allowPermanent === false || raw.allow_permanent === false ? false : undefined,
    choices,
    command: text(raw.command),
    description: text(raw.description),
    smartDenied: Boolean(raw.smartDenied || raw.smart_denied)
  }
}

function normalizeSnapshotParts(value: unknown, content: string): ChatMessagePart[] {
  let parts: ChatMessagePart[] = content ? [assistantTextPart(content)] : []

  if (!Array.isArray(value)) {return parts}

  for (const item of value) {
    const tool = record(item)

    const payload = {
      ...tool,
      tool_id: tool.tool_id || tool.toolCallId,
      name: tool.name || tool.toolName
    } as GatewayEventPayload

    parts = upsertToolPart(parts, payload, tool.status === 'complete' ? 'complete' : 'running')
  }

  return parts
}

export function emptyGroupState(): GroupState {
  return { rooms: {} }
}

function normalizeMessage(value: unknown): GroupMessage | null {
  const raw = record(value)
  const id = text(raw.id || raw.message_id)

  if (!id) {return null}
  const role = raw.role === 'user' || raw.role === 'system' ? raw.role : 'assistant'
  const content = text(raw.content || raw.text)

  return {
    id,
    role,
    content,
    parts: normalizeSnapshotParts(raw.tools, content),
    approval: normalizeApproval(raw.approval),
    profile: text(raw.profile || raw.agent || raw.agent_id) || undefined,
    runtimeSessionId: text(raw.runtime_session_id || raw.session_id) || undefined,
    status: raw.status === 'streaming' || raw.status === 'error' || raw.status === 'approval' ? raw.status : 'complete',
    createdAt: typeof raw.created_at === 'number' ? raw.created_at : undefined,
    seq: typeof raw.seq === 'number' ? raw.seq : undefined
  }
}

export function normalizeGroupRoom(value: unknown): GroupRoom | null {
  const raw = record(value)
  const id = text(raw.id || raw.room_id)

  if (!id) {return null}

  const profiles = Array.isArray(raw.profiles)
    ? raw.profiles.flatMap(profile => typeof profile === 'string' ? [profile] : [text(record(profile).name || record(profile).id)]).filter(Boolean)
    : []

  const messages = Array.isArray(raw.messages) ? raw.messages.map(normalizeMessage).filter((m): m is GroupMessage => Boolean(m)) : []

  const running = Boolean(raw.running || raw.is_running)

  return {
    id, name: text(raw.name) || id, profiles, messages, runningProfiles: running ? ['*'] : [], running,
    workspace: text(raw.workspace) || undefined,
    summary: text(raw.summary) || undefined,
    contextStatus: text(raw.context_status || raw.contextStatus) || undefined
  }
}

export function mergeGroupRoom(state: GroupState, roomValue: unknown): GroupState {
  const incoming = normalizeGroupRoom(roomValue)

  if (!incoming) {return state}
  const current = state.rooms[incoming.id]
  const byId = new Map<string, GroupMessage>()

  for (const message of current?.messages ?? []) {byId.set(message.id, message)}

  for (const message of incoming.messages) {byId.set(message.id, { ...byId.get(message.id), ...message })}

  const messages = [...byId.values()].sort((a, b) => {
    if (a.seq !== undefined && b.seq !== undefined) {return a.seq - b.seq}

    if (a.seq !== undefined) {return -1}

    if (b.seq !== undefined) {return 1}

    return 0
  })

  return { rooms: { ...state.rooms, [incoming.id]: { ...current, ...incoming, messages } } }
}

export function applyGroupEvent(state: GroupState, event: GroupEvent): GroupState {
  if (event.type === 'group.event') {
    const envelope = record(event.payload)
    const rawType = text(envelope.type)

    const projectedType = rawType.startsWith('message.')
      ? `group.${rawType}`
      : rawType === 'approval.request'
        ? 'group.message.approval'
        : rawType === 'error'
          ? 'group.message.error'
          : rawType.startsWith('agent.')
            ? `group.run.${rawType.slice('agent.'.length)}`
            : `group.${rawType}`

    return applyGroupEvent(state, {
      type: projectedType,
      payload: {
        ...record(envelope.payload),
        created_at: envelope.created_at,
        profile: envelope.member_profile,
        room_id: envelope.room_id,
        seq: envelope.seq
      }
    })
  }

  if (!event.type?.startsWith('group.')) {return state}
  const payload = record(event.payload)
  const roomId = text(payload.room_id || payload.roomId)

  if (!roomId) {return state}
  const current = state.rooms[roomId] ?? { id: roomId, name: roomId, profiles: [], messages: [], runningProfiles: [] }

  if (event.type === 'group.room.deleted') {
    const rooms = { ...state.rooms }
    delete rooms[roomId]

    return { rooms }
  }

  if (event.type === 'group.room.updated' || event.type === 'group.room.created' || event.type === 'group.room.snapshot') {
    return mergeGroupRoom(state, payload.room ?? payload)
  }

  const messageId = text(payload.message_id || payload.id) ||
    (event.type.startsWith('group.message.') ? `group-${text(payload.profile) || 'agent'}-${text(payload.seq) || Date.now()}` : '')

  if (event.type.startsWith('group.tool.')) {
    const active = [...current.messages].reverse().find(message =>
      message.role === 'assistant' && message.profile === text(payload.profile) && message.status === 'streaming'
    )

    const activeId = active?.id || messageId || `group-${text(payload.profile) || 'agent'}-${text(payload.seq) || Date.now()}`
    const index = current.messages.findIndex(message => message.id === activeId)
    const previous = index >= 0 ? current.messages[index] : undefined
    const phase = event.type === 'group.tool.complete' ? 'complete' : 'running'

    const next: GroupMessage = {
      id: activeId,
      role: 'assistant',
      content: previous?.content || '',
      parts: upsertToolPart(previous?.parts ?? [], payload as GatewayEventPayload, phase),
      profile: text(payload.profile) || previous?.profile,
      runtimeSessionId: text(payload.runtime_session_id || payload.session_id) || previous?.runtimeSessionId,
      status: previous?.status ?? 'streaming',
      createdAt: typeof payload.created_at === 'number' ? payload.created_at : previous?.createdAt
    }

    const messages = [...current.messages]

    if (index >= 0) {messages[index] = next}
    else {messages.push(next)}

    return { rooms: { ...state.rooms, [roomId]: { ...current, messages, running: true } } }
  }

  const profile = text(payload.profile || payload.agent || payload.agent_id)
  const terminal = event.type.endsWith('.complete') || event.type.endsWith('.error') || event.type.includes('run.interrupt')
  let runningProfiles = current.runningProfiles ?? []

  if (event.type.includes('run.start') && profile) {
    runningProfiles = [...new Set([...runningProfiles.filter(item => item !== '*'), profile])]
  } else if (terminal) {
    runningProfiles = profile ? runningProfiles.filter(item => item !== profile && item !== '*') : []
  }

  if (!messageId) {
    const running = runningProfiles.length > 0

    return { rooms: { ...state.rooms, [roomId]: { ...current, runningProfiles, running } } }
  }

  const index = current.messages.findIndex(message => message.id === messageId)
  const previous = index >= 0 ? current.messages[index] : undefined
  const isDelta = event.type.endsWith('.delta')
  const content = isDelta ? `${previous?.content ?? ''}${text(payload.text || payload.delta)}` : text(payload.content || payload.text) || previous?.content || ''

  const status: GroupMessageStatus = event.type.includes('approval')
    ? 'approval'
    : event.type.endsWith('.error')
      ? 'error'
      : event.type.endsWith('.complete')
        ? 'complete'
        : 'streaming'

  const next: GroupMessage = {
    id: messageId,
    role: payload.role === 'user' || payload.role === 'system' ? payload.role : previous?.role ?? 'assistant',
    content,
    parts: content ? [assistantTextPart(content)] : previous?.parts ?? [],
    approval: event.type.includes('approval')
      ? {
          allowPermanent: payload.allow_permanent !== false,
          choices: Array.isArray(payload.choices)
            ? payload.choices.filter(choice => choice === 'once' || choice === 'session' || choice === 'always' || choice === 'deny')
            : undefined,
          command: text(payload.command),
          description: text(payload.description),
          smartDenied: Boolean(payload.smart_denied)
        }
      : previous?.approval,
    profile: text(payload.profile || payload.agent || payload.agent_id) || previous?.profile,
    runtimeSessionId: text(payload.runtime_session_id || payload.session_id) || previous?.runtimeSessionId,
    status,
    createdAt: typeof payload.created_at === 'number' ? payload.created_at : previous?.createdAt
  }

  const messages = [...current.messages]

  if (index >= 0) {messages[index] = next}
  else {messages.push(next)}

  if (status === 'streaming' && profile) {
    runningProfiles = [...new Set([...runningProfiles.filter(item => item !== '*'), profile])]
  }

  return { rooms: { ...state.rooms, [roomId]: { ...current, messages, runningProfiles, running: runningProfiles.length > 0 } } }
}
