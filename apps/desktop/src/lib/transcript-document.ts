export interface TranscriptOptions {
  include_tools: boolean
  include_reasoning: boolean
  include_images: boolean
  include_videos: boolean
  include_files: boolean
  include_links: boolean
  include_asides: boolean
  collapse_details: boolean
}
export interface TranscriptAttachment {
  kind: 'image' | 'video' | 'file'
  name: string
  reference?: string
  local?: boolean
  mime_type?: string
  data_url?: string
  unavailable?: string
}
export interface TranscriptTurn {
  id?: string
  role: string
  speaker: string
  timestamp?: string
  kind?: 'message' | 'aside' | 'notice'
  text: string
  tool_calls?: Array<{ name: string; arguments: string; id?: string; call_id?: string; uid?: string }>
  tool_call_id?: string
  tool_call_uid?: string
  reasoning?: string
  attachments?: TranscriptAttachment[]
}
export interface TranscriptDocument {
  version: 1
  session_id: string
  title: string
  created_at?: string
  exported_at: string
  model?: string
  options: TranscriptOptions
  messages: TranscriptTurn[]
  warnings: string[]
}
export const DEFAULT_TRANSCRIPT_OPTIONS: TranscriptOptions = {
  include_tools: false,
  include_reasoning: false,
  include_images: true,
  include_videos: true,
  include_files: true,
  include_links: true,
  include_asides: true,
  collapse_details: true
}

function object(value: unknown, keys: string[]): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('Invalid transcript object')
  }

  const result = value as Record<string, unknown>

  if (Object.keys(result).some(key => !keys.includes(key))) {
    throw new Error('Unknown transcript field')
  }

  return result
}

function strings(o: Record<string, unknown>, required: string[], optional: string[] = []): void {
  for (const key of required) {
    if (typeof o[key] !== 'string') {
      throw new Error(`Invalid ${key}`)
    }
  }

  for (const key of optional) {
    if (key in o && typeof o[key] !== 'string') {
      throw new Error(`Invalid ${key}`)
    }
  }
}

function validateAttachments(value: unknown, options: Record<string, unknown>): void {
  if (value === undefined) {
    return
  }

  if (!Array.isArray(value)) {
    throw new Error('Invalid attachments')
  }

  const flags: Record<string, string> = { image: 'include_images', video: 'include_videos', file: 'include_files' }

  for (const raw of value) {
    const a = object(raw, ['kind', 'name', 'reference', 'local', 'mime_type', 'data_url', 'unavailable'])
    strings(a, ['kind', 'name'], ['reference', 'mime_type', 'data_url', 'unavailable'])

    if (!['image', 'video', 'file'].includes(a.kind as string)) {
      throw new Error('Invalid attachment kind')
    }

    if (!options[flags[a.kind as string]]) {
      throw new Error('Excluded attachment present')
    }

    if ('local' in a && typeof a.local !== 'boolean') {
      throw new Error('Invalid attachment local flag')
    }
  }
}

function validateIdentities(value: Record<string, unknown>, keys: string[]): void {
  for (const key of keys) {
    if (!(key in value)) {
      continue
    }

    const id = value[key]

    if (typeof id !== 'string' || !id.length || Array.from(id).length > 512 ||
      Array.from(id).some(char => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127)) {
      throw new Error(`Invalid ${key}`)
    }
  }
}

function validateToolCalls(value: unknown): void {
  if (value === undefined) {
    return
  }

  if (!Array.isArray(value)) {
    throw new Error('Invalid tool calls')
  }

  for (const call of value) {
    const item = object(call, ['name', 'arguments', 'id', 'call_id', 'uid'])
    strings(item, ['name', 'arguments'])
    validateIdentities(item, ['id', 'call_id', 'uid'])
  }
}

function validateTurn(raw: unknown, options: Record<string, unknown>): void {
  const turn = object(raw, [
    'id',
    'role',
    'speaker',
    'timestamp',
    'kind',
    'text',
    'tool_calls',
    'tool_call_id',
    'tool_call_uid',
    'reasoning',
    'attachments'
  ])

  strings(turn, ['role', 'speaker', 'text'], ['id', 'timestamp', 'reasoning'])

  if (turn.kind !== undefined && !['message', 'aside', 'notice'].includes(turn.kind as string)) {
    throw new Error('Invalid turn kind')
  }

  if (!['user', 'assistant', 'tool'].includes(turn.role as string)) {
    throw new Error('Invalid or private turn role')
  }

  if (!options.include_tools && (turn.role === 'tool' || turn.tool_calls !== undefined ||
    'tool_call_id' in turn || 'tool_call_uid' in turn)) {
    throw new Error('Excluded tools present')
  }

  if (turn.role !== 'tool' && ('tool_call_id' in turn || 'tool_call_uid' in turn)) {
    throw new Error('Invalid tool result identity')
  }

  validateIdentities(turn, ['tool_call_id', 'tool_call_uid'])

  if (!options.include_asides && turn.kind === 'aside') {
    throw new Error('Excluded aside present')
  }

  if (!options.include_reasoning && turn.reasoning !== undefined) {
    throw new Error('Excluded reasoning present')
  }

  validateToolCalls(turn.tool_calls)
  validateAttachments(turn.attachments, options)
}

export function parseTranscriptDocument(content: string): TranscriptDocument {
  if (content.length > 32 * 1024 * 1024) {
    throw new Error('Transcript too large')
  }

  const doc = object(JSON.parse(content), [
    'version',
    'session_id',
    'title',
    'created_at',
    'exported_at',
    'model',
    'options',
    'messages',
    'warnings'
  ])

  if (doc.version !== 1) {
    throw new Error('Unsupported transcript version')
  }

  strings(doc, ['session_id', 'title', 'exported_at'], ['created_at', 'model'])
  const options = object(doc.options, Object.keys(DEFAULT_TRANSCRIPT_OPTIONS))

  for (const key of Object.keys(DEFAULT_TRANSCRIPT_OPTIONS)) {
    if (typeof options[key] !== 'boolean') {
      throw new Error(`Invalid option ${key}`)
    }
  }

  if (!Array.isArray(doc.messages) || !Array.isArray(doc.warnings) || !doc.warnings.every(x => typeof x === 'string')) {
    throw new Error('Invalid transcript arrays')
  }

  for (const raw of doc.messages) {
    validateTurn(raw, options)
  }

  return doc as unknown as TranscriptDocument
}
