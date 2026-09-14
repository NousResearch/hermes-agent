import { atom, type PluginRestOptions } from '@hermes/plugin-sdk'
import type { VaultGraph, VaultNote, VaultNoteSummary, VaultSuggestion } from './types'

type Rest = <T>(path: string, opts?: PluginRestOptions) => Promise<T>

let rest: null | Rest = null

export function bindApi(ctx: { rest: Rest }) {
  rest = ctx.rest
}

function req<T>(path: string, opts?: PluginRestOptions): Promise<T> {
  if (!rest) {
    return Promise.reject(new Error('Vault API not bound yet (plugin context missing)'))
  }
  return rest<T>(path, opts)
}

export const $selectedNoteTitle = atom<string>('')

export async function fetchVaultNotes(): Promise<{ notes: VaultNoteSummary[]; count: number }> {
  return req<{ notes: VaultNoteSummary[]; count: number }>('/notes')
}

export async function fetchVaultNote(title: string): Promise<{ note: VaultNote }> {
  return req<{ note: VaultNote }>(`/notes/${encodeURIComponent(title)}`)
}

export async function saveVaultNote(payload: {
  title: string
  content: string
  tags?: string[]
  frontmatter?: Record<string, unknown>
  subfolder?: string
}): Promise<{ status: string; note: VaultNote }> {
  return req<{ status: string; note: VaultNote }>('/notes', {
    method: 'POST',
    body: payload
  })
}

export async function appendVaultNote(title: string, content: string): Promise<{ status: string; note: VaultNote }> {
  return req<{ status: string; note: VaultNote }>(`/notes/${encodeURIComponent(title)}/append`, {
    method: 'POST',
    body: { content }
  })
}

export async function deleteVaultNote(title: string): Promise<{ status: string; deleted: string }> {
  return req<{ status: string; deleted: string }>(`/notes/${encodeURIComponent(title)}`, {
    method: 'DELETE'
  })
}

export async function searchVault(q: string): Promise<{ results: Array<{ score: number; title: string; rel_path: string; tags: string[]; preview: string }>; count: number }> {
  return req<{ results: Array<{ score: number; title: string; rel_path: string; tags: string[]; preview: string }>; count: number }>(`/search?q=${encodeURIComponent(q)}`)
}

export async function fetchVaultGraph(): Promise<{ graph: VaultGraph }> {
  return req<{ graph: VaultGraph }>('/graph')
}

export async function fetchVaultSuggestions(prefix = ''): Promise<{ suggestions: VaultSuggestion[] }> {
  return req<{ suggestions: VaultSuggestion[] }>(`/suggest?prefix=${encodeURIComponent(prefix)}`)
}
