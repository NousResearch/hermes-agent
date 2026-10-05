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
 * Keys (see `sessionFileKey`) of every file a landed file edit in `messages`
 * created or edited. `cwd` resolves the rare relative path a tool reports.
 */
export function deriveSessionFileKeys(messages: readonly SessionFileMessage[], cwd?: string): Set<string> {
  const keys = new Set<string>()

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

      for (const path of editedPaths(part, result)) {
        keys.add(sessionFileKey(isAbsolutePath(path) || !cwd ? path : joinPath(cwd, path)))
      }
    }
  }

  return keys
}
