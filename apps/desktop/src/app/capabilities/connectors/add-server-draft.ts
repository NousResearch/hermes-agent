import { type McpServerEntry, normalizeEntry } from '@/lib/mcp-servers'

export type AddServerAuth = 'bearer' | 'none' | 'oauth'

export type AddServerTransport = 'http' | 'stdio'

export interface DraftValue {
  id: number
  value: string
}

export interface DraftPair {
  id: number
  key: string
  value: string
}

export interface AddServerDraft {
  args: DraftValue[]
  auth: AddServerAuth
  bearer: string
  command: string
  cwd: string
  env: DraftPair[]
  headers: DraftPair[]
  name: string
  passthrough: DraftValue[]
  transport: AddServerTransport
  url: string
}

let rowCounter = 0

export const nextRowId = (): number => ++rowCounter

export const emptyValue = (value = ''): DraftValue => ({ id: nextRowId(), value })

export const emptyPair = (key = '', value = ''): DraftPair => ({ id: nextRowId(), key, value })

export const EMPTY_ADD_DRAFT: AddServerDraft = {
  args: [],
  auth: 'none',
  bearer: '',
  command: '',
  cwd: '',
  env: [],
  headers: [],
  name: '',
  passthrough: [],
  transport: 'stdio',
  url: ''
}

export const isDraftComplete = (draft: AddServerDraft): boolean =>
  draft.name.trim() !== '' && (draft.transport === 'stdio' ? draft.command.trim() !== '' : draft.url.trim() !== '')

const filledPairs = (rows: readonly DraftPair[]): Record<string, string> =>
  Object.fromEntries(rows.filter(row => row.key.trim() !== '').map(row => [row.key.trim(), row.value]))

const filledValues = (rows: readonly DraftValue[]): string[] =>
  rows.map(row => row.value.trim()).filter(value => value !== '')

const envReference = (name: string): string => `\${${name}}`

function httpEntry(draft: AddServerDraft): McpServerEntry {
  const headers = filledPairs(draft.headers)
  const entry: McpServerEntry = { url: draft.url.trim() }

  if (Object.keys(headers).length > 0) {
    entry.headers = headers
  }

  if (draft.auth === 'oauth') {
    entry.auth = 'oauth'
  }

  return entry
}

function stdioEntry(draft: AddServerDraft): McpServerEntry {
  const env = { ...filledPairs(draft.env) }

  for (const name of filledValues(draft.passthrough)) {
    env[name] = envReference(name)
  }

  const args = filledValues(draft.args)
  const entry: McpServerEntry = { command: draft.command.trim() }

  if (args.length > 0) {
    entry.args = args
  }

  if (Object.keys(env).length > 0) {
    entry.env = env
  }

  if (draft.cwd.trim() !== '') {
    entry.cwd = draft.cwd.trim()
  }

  return entry
}

export const entryOfDraft = (draft: AddServerDraft): McpServerEntry =>
  draft.transport === 'http' ? httpEntry(draft) : stdioEntry(draft)

type EntryValue = McpServerEntry[string]

const isString = (value: EntryValue): value is string => Object.prototype.toString.call(value) === '[object String]'

const asString = (value: EntryValue): string => (isString(value) ? value : '')

const asRecord = (value: EntryValue): McpServerEntry =>
  value instanceof Object && !Array.isArray(value) ? Object.fromEntries(Object.entries(value)) : {}

const asValues = (value: EntryValue): DraftValue[] =>
  Array.isArray(value) ? value.map((entry: EntryValue) => emptyValue(asString(entry))) : []

const forwarded = (key: string, value: EntryValue): boolean => value === envReference(key)

export function draftFromEntry(name: string, raw: McpServerEntry, previous: AddServerDraft): AddServerDraft {
  const entry = normalizeEntry(raw)
  const url = asString(entry.url)
  const named = name || previous.name

  if (url !== '') {
    return {
      ...EMPTY_ADD_DRAFT,
      auth: asString(entry.auth) === 'oauth' ? 'oauth' : 'none',
      bearer: previous.bearer,
      headers: Object.entries(asRecord(entry.headers)).map(([key, value]) => emptyPair(key, asString(value))),
      name: named,
      transport: 'http',
      url
    }
  }

  const env = Object.entries(asRecord(entry.env))

  return {
    ...EMPTY_ADD_DRAFT,
    args: asValues(entry.args),
    command: asString(entry.command),
    cwd: asString(entry.cwd),
    env: env.filter(([key, value]) => !forwarded(key, value)).map(([key, value]) => emptyPair(key, asString(value))),
    name: named,
    passthrough: env.filter(([key, value]) => forwarded(key, value)).map(([key]) => emptyValue(key)),
    transport: 'stdio'
  }
}
