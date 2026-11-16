import { appendFileSync, mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { describe, expect, it, vi } from 'vitest'

import { formatDesktopLogLine } from './desktop-log-line'
import { attachRendererConsoleCapture, formatRendererBoundaryReport, formatRendererConsoleLine } from './renderer-log'

type ConsoleMessageEvent = {
  level?: unknown
  message?: unknown
  sourceId?: unknown
  lineNumber?: unknown
}

type ConsoleMessageHandler = (event: ConsoleMessageEvent) => void

function createWindowHarness() {
  let handler: ConsoleMessageHandler | undefined

  const win = {
    webContents: {
      on: (_event: 'console-message', listener: ConsoleMessageHandler) => {
        handler = listener
      }
    }
  }

  return { win, getHandler: () => handler }
}

describe('formatRendererConsoleLine', () => {
  it('formats the canonical Electron console-message event at error level', () => {
    const line = formatRendererConsoleLine('hud', {
      level: 'error',
      message: 'Minified React error #310',
      sourceId: 'file:///app/index.js',
      lineNumber: 13
    })

    expect(line).toBe('[renderer console:hud] Minified React error #310 (file:///app/index.js:13)')
  })

  it('drops a canonical non-error string level', () => {
    expect(formatRendererConsoleLine('main', { level: 'info', message: 'x', sourceId: 's', lineNumber: 1 })).toBeNull()
  })

  it('drops malformed event objects', () => {
    expect(formatRendererConsoleLine('main', {})).toBeNull()
  })
})

describe('attachRendererConsoleCapture', () => {
  it('writes serialized wake diagnostics to desktop.log without admitting other warnings', () => {
    const directory = mkdtempSync(join(tmpdir(), 'wake-log-'))
    const file = join(directory, 'desktop.log')
    const capture = createWindowHarness()

    const diagnostic = {
      attempts: 3,
      phase: 'hydration',
      profile: 'tom',
      runtimeBound: false,
      selectionSettled: true,
      storedSessionId: 'bot-chat-tom',
      transcriptPainted: false
    }

    try {
      attachRendererConsoleCapture(capture.win, 'session-window', line => {
        appendFileSync(file, `${formatDesktopLogLine(line)}\n`)
      })
      capture.getHandler()?.({
        level: 'warning',
        message: `[bot-wake] hydration timed out ${JSON.stringify(diagnostic)}`,
        sourceId: 'file:///app/index.js',
        lineNumber: 42
      })
      capture.getHandler()?.({ level: 'warning', message: 'unrelated warning', sourceId: 'src', lineNumber: 1 })
      capture.getHandler()?.({ level: 'info', message: '[bot-wake] quiet', sourceId: 'src', lineNumber: 1 })

      const lines = readFileSync(file, 'utf8').trim().split('\n')

      expect(lines).toHaveLength(1)
      expect(lines[0]).toContain('[renderer console:session-window] [bot-wake] hydration timed out ')
      expect(JSON.parse(lines[0]!.split('hydration timed out ')[1]!.split(' (file:')[0]!)).toEqual(diagnostic)
    } finally {
      rmSync(directory, { recursive: true, force: true })
    }
  })

  it('registers a single-argument listener, logs errors, and skips the rest', () => {
    const log = vi.fn()
    const capture = createWindowHarness()

    attachRendererConsoleCapture(capture.win, 'quick-entry', log)

    const handler = capture.getHandler()
    expect(handler).toHaveLength(1)

    handler?.({ level: 'error', message: 'crash', sourceId: 'src', lineNumber: 2 })
    handler?.({ level: 'debug', message: 'debug', sourceId: 'src', lineNumber: 3 })

    expect(log).toHaveBeenCalledTimes(1)
    expect(log).toHaveBeenCalledWith('[renderer console:quick-entry] crash (src:2)')
  })

  it('reports signature drift once across renderer windows', () => {
    const log = vi.fn()
    const firstCapture = createWindowHarness()
    const secondCapture = createWindowHarness()

    attachRendererConsoleCapture(firstCapture.win, 'main', log)
    attachRendererConsoleCapture(secondCapture.win, 'hud', log)

    firstCapture.getHandler()?.({})
    secondCapture.getHandler()?.({})

    expect(log).toHaveBeenCalledTimes(1)
    expect(log).toHaveBeenCalledWith(
      '[renderer console] Electron console-message signature drift detected; renderer errors may not be captured'
    )
  })
})

describe('formatRendererBoundaryReport', () => {
  it('carries window label, boundary label, message, and component stack', () => {
    const report = formatRendererBoundaryReport(
      'main',
      'root',
      'Minified React error #310',
      '\n    at Gde (index.js:13)\n    at C_ (index.js:13)'
    )

    expect(report).toContain('[renderer crash:main] [error-boundary:root] Minified React error #310')
    expect(report).toContain('at Gde (index.js:13)')
  })

  it('survives a malformed payload and clamps oversized fields', () => {
    const report = formatRendererBoundaryReport(undefined, null, 'x'.repeat(10_000), 'y'.repeat(10_000))

    expect(report).toContain('[renderer crash:unknown] [error-boundary:unknown]')
    expect(report.length).toBeLessThan(7_000)
  })
})
