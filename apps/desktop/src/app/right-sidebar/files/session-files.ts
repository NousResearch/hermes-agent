// Which files the active session created or edited, for the file tree's
// "This session" markers and filter (#133481). Pure derivation: no React, no IPC.
//
// Same rule as the per-turn "N files changed" card (deriveChangedFiles): only a
// landed file-edit call with a diff counts. A call still running has no result
// yet, and a failed or no-op edit changed nothing.

import {
  fileEditPath,
  inlineDiffFromResult,
  isFileEditTool,
  parseMaybeObject
} from '@/components/assistant-ui/tool/fallback-model'
import { type ToolResultMetadata, toolResultRecord } from '@/lib/tool-result-metadata'

import type { TreeNode } from './use-project-tree'

interface SessionFilePart {
  args?: unknown
  result?: unknown
  toolName?: unknown
  toolResultMetadata?: ToolResultMetadata
  type?: unknown
}

interface SessionFileMessage {
  parts?: readonly unknown[]
}

/**
 * Comparison key for a path: forward slashes, no trailing slash, and Windows
 * drive / UNC paths lower-cased because that file system ignores case. Tree
 * node ids and tool-reported paths go through this before they are compared.
 */
export function sessionFileKey(path: string): string {
  const slashed = path
    .trim()
    .replace(/\\/g, '/')
    .replace(/(.)\/+$/, '$1')

  return /^[a-z]:\//i.test(slashed) || slashed.startsWith('//') ? slashed.toLowerCase() : slashed
}

function isAbsolutePath(path: string): boolean {
  return path.startsWith('/') || path.startsWith('\\\\') || /^[a-z]:[\\/]/i.test(path)
}

function joinPath(base: string, relative: string): string {
  return `${base.replace(/[\\/]+$/, '')}/${relative.replace(/^\.[\\/]+/, '')}`
}

function stringList(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((v): v is string => typeof v === 'string' && v.trim() !== '') : []
}

// Prefer the absolute paths the tool reports (`files_modified`, `resolved_path`)
// over the path the model typed, which may be relative. A V4A patch can touch
// several files at once, so `files_modified` may hold more than one.
function editedPaths(part: SessionFilePart, result: Record<string, unknown>): string[] {
  const modified = stringList(result.files_modified)

  if (modified.length > 0) {
    return modified
  }

  if (typeof result.resolved_path === 'string' && result.resolved_path.trim()) {
    return [result.resolved_path]
  }

  const reported = fileEditPath(parseMaybeObject(part.args), result)

  return reported ? [reported] : []
}

/**
 * Every file a landed file edit in `messages` created or edited, as
 * comparison key (see `sessionFileKey`) -> the path as the tool reported it.
 * `cwd` resolves the rare relative path a tool reports.
 */
export function deriveSessionFiles(messages: readonly SessionFileMessage[], cwd?: string): Map<string, string> {
  const files = new Map<string, string>()

  for (const message of messages) {
    for (const raw of message.parts ?? []) {
      const part = (raw ?? {}) as SessionFilePart

      if (part.type !== 'tool-call' || typeof part.toolName !== 'string' || !isFileEditTool(part.toolName)) {
        continue
      }

      const result = toolResultRecord(part)

      if (!inlineDiffFromResult(result)) {
        continue
      }

      for (const reported of editedPaths(part, result)) {
        const path = isAbsolutePath(reported) || !cwd ? reported : joinPath(cwd, reported)
        const key = sessionFileKey(path)

        if (!files.has(key)) {
          files.set(key, path)
        }
      }
    }
  }

  return files
}

/** Comparison keys only; see `deriveSessionFiles`. */
export function deriveSessionFileKeys(messages: readonly SessionFileMessage[], cwd?: string): Set<string> {
  return new Set(deriveSessionFiles(messages, cwd).keys())
}

export interface SessionFileTree {
  /** Folders + files under `cwd`, folders first, each level sorted by name. */
  data: TreeNode[]
  /** Number of files in `data` (session files outside `cwd` are left out). */
  fileCount: number
  /** Every folder open, so the whole filtered tree is visible at once. */
  openState: Record<string, boolean>
}

function sortLevel(nodes: TreeNode[]): TreeNode[] {
  nodes.sort((a, b) => Number(b.isDirectory) - Number(a.isDirectory) || a.name.localeCompare(b.name))

  for (const node of nodes) {
    if (node.children) {
      sortLevel(node.children)
    }
  }

  return nodes
}

/**
 * The filtered tree for the "This session" toggle: only the session's files
 * under `cwd`, inside their folders. Built from paths rather than by pruning
 * the live tree, because the live tree loads folders lazily and would miss
 * files in folders that were never opened.
 */
export function buildSessionFileTree(paths: Iterable<string>, cwd: string): SessionFileTree {
  const root = cwd.trim().replace(/[\\/]+$/, '')
  const rootKey = sessionFileKey(root)
  const sep = root.includes('\\') ? '\\' : '/'
  const data: TreeNode[] = []
  const folders = new Map<string, TreeNode>()
  const openState: Record<string, boolean> = {}
  let fileCount = 0

  if (!root) {
    return { data, fileCount, openState }
  }

  for (const path of paths) {
    const key = sessionFileKey(path)

    if (!key.startsWith(`${rootKey}/`)) {
      continue
    }

    const segments = path
      .trim()
      .replace(/\\/g, '/')
      .slice(root.replace(/\\/g, '/').length)
      .split('/')
      .filter(Boolean)

    if (segments.length === 0) {
      continue
    }

    let level = data
    let folderId = root

    for (const segment of segments.slice(0, -1)) {
      folderId = `${folderId}${sep}${segment}`
      const folderKey = sessionFileKey(folderId)
      let folder = folders.get(folderKey)

      if (!folder) {
        folder = { children: [], id: folderId, isDirectory: true, name: segment }
        folders.set(folderKey, folder)
        openState[folderId] = true
        level.push(folder)
      }

      level = folder.children ?? []
    }

    level.push({ id: path, isDirectory: false, name: segments.at(-1) ?? path })
    fileCount += 1
  }

  return { data: sortLevel(data), fileCount, openState }
}
