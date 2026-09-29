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

export const createVaultNote = (title: string, folder = '', content?: string, profile?: ProfileScope) =>
  call<{ id: string; ok: boolean }>(profile, '/api/vault/note', 'POST', { content, folder, title })

export const deleteVaultNote = (id: string, profile?: ProfileScope) =>
  call<{ id: string; ok: boolean }>(profile, '/api/vault/note', 'DELETE', { id })
