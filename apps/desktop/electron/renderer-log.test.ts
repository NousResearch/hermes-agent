import { describe, expect, it, vi } from 'vitest'

import { attachRendererConsoleCapture, formatRendererBoundaryReport, formatRendererConsoleLine } from './renderer-log'

describe('formatRendererConsoleLine', () => {
  it('formats an error-level Electron console-message event', () => {
    const line = formatRendererConsoleLine('hud', {
      level: 'error',
      message: 'Minified React error #310',
      sourceId: 'file:///app/index.js',
      lineNumber: 13
    })

    expect(line).toBe('[renderer console:hud] Minified React error #310 (file:///app/index.js:13)')
  })

  it('drops non-error levels', () => {
    expect(
      formatRendererConsoleLine('main', { level: 'debug', message: 'x', sourceId: 's', lineNumber: 1 })
    ).toBeNull()
    expect(
      formatRendererConsoleLine('main', { level: 'info', message: 'x', sourceId: 's', lineNumber: 1 })
    ).toBeNull()
    expect(
      formatRendererConsoleLine('main', { level: 'warning', message: 'warn', sourceId: 's', lineNumber: 1 })
    ).toBeNull()
  })
})

describe('attachRendererConsoleCapture', () => {
  it('logs error-level messages and skips the rest', () => {
    const log = vi.fn()
    let handler: ((...args: any[]) => void) | undefined

    const win = {
      webContents: {
        on: (_event: string, listener: (...args: any[]) => void) => {
          handler = listener
        }
      }
    }

    attachRendererConsoleCapture(win, 'quick-entry', log)

    handler?.({ level: 'error', message: 'crash', sourceId: 'src', lineNumber: 2 })
    handler?.({ level: 'debug', message: 'debug message', sourceId: 'src', lineNumber: 3 })

    expect(log).toHaveBeenCalledTimes(1)
    expect(log).toHaveBeenCalledWith('[renderer console:quick-entry] crash (src:2)')
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

  it('omits the stack block when there is no component stack', () => {
    const report = formatRendererBoundaryReport('main', 'root', 'boom', '')

    expect(report).toBe('[renderer crash:main] [error-boundary:root] boom')
    expect(report).not.toContain('\n')
  })
})
