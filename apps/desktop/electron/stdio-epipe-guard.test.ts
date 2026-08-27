import { EventEmitter } from 'node:events'

import { describe, expect, it, vi } from 'vitest'

import { installStdioEpipeGuard } from './stdio-epipe-guard'

describe('installStdioEpipeGuard', () => {
  it('ignores only broken-pipe errors from standard streams', () => {
    const stdout = new EventEmitter()
    const stderr = new EventEmitter()
    const onUnexpected = vi.fn()

    installStdioEpipeGuard({ onUnexpected, stderr, stdout })

    stdout.emit('error', Object.assign(new Error('broken pipe'), { code: 'EPIPE' }))
    stderr.emit('error', Object.assign(new Error('broken pipe'), { code: 'EPIPE' }))

    expect(onUnexpected).not.toHaveBeenCalled()
  })

  it('forwards errors other than EPIPE to the existing crash path', () => {
    const stdout = new EventEmitter()
    const stderr = new EventEmitter()
    const onUnexpected = vi.fn()
    const error = Object.assign(new Error('access denied'), { code: 'EACCES' })

    installStdioEpipeGuard({ onUnexpected, stderr, stdout })
    stdout.emit('error', error)

    expect(onUnexpected).toHaveBeenCalledWith(error)
  })
})
