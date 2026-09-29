import { capabilityScoped, type ProfileScope } from './client'

export interface VaultNoteNode {
  excerpt: string
  folder: string
  id: string
  label: string
  links: number
  size: number
  tags: string[]
  timestamp: number
}

export interface VaultGraph {
  edges: { source: string; target: string }[]
  nodes: VaultNoteNode[]
  vault: { exists: boolean; path: string }
}

export interface VaultNote {
  content: string
  id: string
  label: string
}

const call = <T>(profile: ProfileScope | undefined, path: string, method?: string, body?: unknown) =>
  window.hermesDesktop.api<T>({ ...capabilityScoped(profile), body, method, path })

export const getVaultGraph = (profile?: ProfileScope) => call<VaultGraph>(profile, '/api/vault/graph')

export const getVaultNote = (id: string, profile?: ProfileScope) =>
  call<VaultNote>(profile, `/api/vault/note?id=${encodeURIComponent(id)}`)

export const saveVaultNote = (id: string, content: string, profile?: ProfileScope) =>
  call<{ id: string; ok: boolean }>(profile, '/api/vault/note', 'PUT', { content, id })

/** Query-key root of every rail query that shows vault state; invalidate it after any vault write. */
export const VAULT_RAIL_KEY = 'jarvis-vault-rail'

export interface CreateVaultNoteOptions {
  /** `suffix` picks "Title (2)" on a name clash instead of failing. Default: fail. */
  conflict?: 'error' | 'suffix'
  content?: string
  /** Makes the call idempotent: a note in this name family that already contains the key is returned as `existed`. */
  dedupeKey?: string
  /** The text was written by someone else (a news feed, a page): stored fenced as untrusted data. */
  externalSource?: string
  folder?: string
}

export const createVaultNote = (title: string, options: CreateVaultNoteOptions = {}, profile?: ProfileScope) =>
  call<{ existed: boolean; id: string; ok: boolean }>(profile, '/api/vault/note', 'POST', {
    conflict: options.conflict,
    content: options.content,
    dedupe_key: options.dedupeKey,
    external_source: options.externalSource,
    folder: options.folder ?? '',
    title
  })

export const deleteVaultNote = (id: string, profile?: ProfileScope) =>
  call<{ id: string; ok: boolean }>(profile, '/api/vault/note', 'DELETE', { id })
