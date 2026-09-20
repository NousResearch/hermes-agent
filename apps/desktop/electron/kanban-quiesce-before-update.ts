/**
 * Quiesce kanban workers before a Windows in-place update.
 *
 * The Desktop writes the shared live-update marker first, then invokes this
 * helper through the install venv's Python.  Python owns board discovery,
 * dispatch locking, task reclaim semantics, and PID fingerprint checks; the
 * Electron side only orchestrates the bounded subprocess and records evidence.
 */

import { execFileSync, type ExecFileSyncOptionsWithStringEncoding } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

export interface KanbanQuiesceResult {
  ok: boolean
  reclaimed: Array<{ board: string; task_id: string }>
  failed: Array<{ board: string; task_id?: string; error: string }>
  error?: string
}

export interface KanbanQuiesceDeps {
  isWindows?: boolean
  existsSync?: (p: string) => boolean
  execFileSync?: (command: string, args: string[], options: ExecFileSyncOptionsWithStringEncoding) => Buffer | string
}

export const KANBAN_QUIESCE_TIMEOUT_MS = 60_000

function parseQuiesceResult(raw: unknown): KanbanQuiesceResult | undefined {
  try {
    const parsed = JSON.parse(String(raw))

    if (!parsed || typeof parsed !== 'object' || typeof parsed.ok !== 'boolean') {
      return undefined
    }

    return {
      ok: parsed.ok,
      reclaimed: Array.isArray(parsed.reclaimed) ? parsed.reclaimed : [],
      failed: Array.isArray(parsed.failed) ? parsed.failed : [],
      ...(typeof parsed.error === 'string' ? { error: parsed.error } : {})
    }
  } catch {
    return undefined
  }
}

export function quiesceKanbanWorkersForUpdate(
  updateRoot: string,
  hermesHome: string,
  deps: KanbanQuiesceDeps = {}
): KanbanQuiesceResult {
  if (!(deps.isWindows ?? process.platform === 'win32')) {
    return { ok: true, reclaimed: [], failed: [] }
  }

  const python = path.join(updateRoot, 'venv', 'Scripts', 'python.exe')
  const exists = deps.existsSync ?? fs.existsSync

  if (!exists(python)) {
    return { ok: false, reclaimed: [], failed: [], error: 'venv python not found' }
  }

  try {
    const raw = (deps.execFileSync ?? execFileSync)(python, ['-m', 'hermes_cli.kanban_update_coordination'], {
      cwd: updateRoot,
      env: { ...process.env, HERMES_HOME: hermesHome },
      timeout: KANBAN_QUIESCE_TIMEOUT_MS,
      windowsHide: true,
      stdio: ['ignore', 'pipe', 'pipe'],
      encoding: 'utf8'
    } as ExecFileSyncOptionsWithStringEncoding)

    const parsed = parseQuiesceResult(raw)

    if (!parsed) {
      throw new Error('invalid quiesce response')
    }

    return parsed
  } catch (error: any) {
    const parsed = parseQuiesceResult(error?.stdout)

    if (parsed) {
      return parsed
    }

    return {
      ok: false,
      reclaimed: [],
      failed: [],
      error: error?.message || 'kanban quiesce failed'
    }
  }
}
