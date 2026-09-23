import fs from 'node:fs'
import path from 'node:path'

import { formatDesktopLogLine } from './desktop-log-line'
import { ACTIVE_LOG_POLL_MS, planLogRotation, reclaimActiveLogIfOversized } from './log-rotation'

const DESKTOP_LOG_FLUSH_MS = 120
const DESKTOP_LOG_BUFFER_MAX_CHARS = 64 * 1024

export function rotateLogIfNeededSync(base) {
  let size

  try {
    size = fs.statSync(base).size
  } catch {
    return // No live file yet — the append (re)creates it.
  }

  for (const [op, src, dst] of planLogRotation(size, base)) {
    try {
      if (op === 'rm') {
        fs.rmSync(src, { force: true })
      } else {
        fs.renameSync(src, dst)
      }
    } catch {
      // Best-effort — logging must never block startup/shutdown.
    }
  }
}

export function createDesktopLogRuntime(DESKTOP_LOG_PATH: string) {
  const hermesLog: string[] = []
  let desktopLogBuffer = ''
  let desktopLogFlushTimer = null
  let desktopLogFlushPromise = Promise.resolve()

  // Chromium owns its --log-file for the life of the process, so the startup
  // reclaim above cannot bound a shell that stays up for days writing errors.
  // Poll and truncate in place; renaming would leave Chromium appending to the
  // renamed inode. Unref'd so it never holds the process open.
  function startChromiumLogWatcher(file) {
    const io = {
      size: f => {
        try {
          return fs.statSync(f).size
        } catch {
          return null // Not created yet — nothing has been logged.
        }
      },
      truncate: f => fs.truncateSync(f, 0)
    }

    const timer = setInterval(() => {
      try {
        if (reclaimActiveLogIfOversized(file, io)) {
          rememberLog(`[diagnostics] truncated oversized Chromium log ${file}`)
        }
      } catch {
        // Best-effort — an unbounded log beats a crashed shell.
      }
    }, ACTIVE_LOG_POLL_MS)

    timer.unref?.()
  }

  async function rotateDesktopLogIfNeededAsync() {
    let size

    try {
      size = (await fs.promises.stat(DESKTOP_LOG_PATH)).size
    } catch {
      return // No live file yet — the append (re)creates it.
    }

    for (const [op, src, dst] of planLogRotation(size, DESKTOP_LOG_PATH)) {
      try {
        if (op === 'rm') {
          await fs.promises.rm(src, { force: true })
        } else {
          await fs.promises.rename(src, dst)
        }
      } catch {
        // Best-effort — logging must never crash the shell.
      }
    }
  }

  function flushDesktopLogBufferSync() {
    if (!desktopLogBuffer) {
      return
    }

    const chunk = desktopLogBuffer
    desktopLogBuffer = ''

    try {
      fs.mkdirSync(path.dirname(DESKTOP_LOG_PATH), { recursive: true })
      rotateLogIfNeededSync(DESKTOP_LOG_PATH)
      fs.appendFileSync(DESKTOP_LOG_PATH, chunk)
    } catch {
      // Logging must never block app startup/shutdown.
    }
  }

  function flushDesktopLogBufferAsync() {
    if (!desktopLogBuffer) {
      return desktopLogFlushPromise
    }

    const chunk = desktopLogBuffer
    desktopLogBuffer = ''

    desktopLogFlushPromise = desktopLogFlushPromise
      .then(async () => {
        await fs.promises.mkdir(path.dirname(DESKTOP_LOG_PATH), { recursive: true })
        await rotateDesktopLogIfNeededAsync()
        await fs.promises.appendFile(DESKTOP_LOG_PATH, chunk)
      })
      .catch(() => {
        // Logging must never crash the desktop shell.
      })

    return desktopLogFlushPromise
  }

  function scheduleDesktopLogFlush() {
    if (desktopLogFlushTimer) {
      return
    }

    desktopLogFlushTimer = setTimeout(() => {
      desktopLogFlushTimer = null
      void flushDesktopLogBufferAsync()
    }, DESKTOP_LOG_FLUSH_MS)
  }

  function rememberLog(chunk) {
    const text = String(chunk || '').trim()

    if (!text) {
      return
    }

    // One timestamp per chunk: lines arriving in the same event happened
    // at the same moment.  ISO-8601 UTC, matching agent.log/gateway.log.
    const stamp = new Date().toISOString()
    const lines = text.split(/\r?\n/).map(line => formatDesktopLogLine(line, stamp))
    hermesLog.push(...lines)

    if (hermesLog.length > 300) {
      hermesLog.splice(0, hermesLog.length - 300)
    }

    desktopLogBuffer += `${lines.join('\n')}\n`

    if (desktopLogBuffer.length >= DESKTOP_LOG_BUFFER_MAX_CHARS) {
      if (desktopLogFlushTimer) {
        clearTimeout(desktopLogFlushTimer)
        desktopLogFlushTimer = null
      }

      void flushDesktopLogBufferAsync()

      return
    }

    scheduleDesktopLogFlush()
  }

  function stopDesktopLogFlushTimer() {
    if (desktopLogFlushTimer) {
      clearTimeout(desktopLogFlushTimer)
      desktopLogFlushTimer = null
    }
  }

  return { hermesLog, flushDesktopLogBufferSync, rememberLog, startChromiumLogWatcher, stopDesktopLogFlushTimer }
}
