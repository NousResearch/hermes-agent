import { describe, expect, it } from 'vitest'

import { classifyLocalBootFailure, localBootFailureCopy } from './boot-failure-cause'

const CAUSES = {
  diskFull: 'disk full copy',
  exitedEarly: 'exited early copy',
  installMissing: 'install missing copy',
  permission: 'permission copy',
  portInUse: 'port copy',
  timedOut: 'timed out copy'
}

// The red box on the boot-failure overlay must lead with ONE plain sentence;
// exit codes, millisecond values and Python tracebacks belong under Details.
describe('local boot failure classification', () => {
  it('classifies the raw main-process failures into plain causes', () => {
    expect(
      classifyLocalBootFailure(
        'Hermes backend exited before it became ready (1).\nRecent backend output:\nTraceback (most recent call last):\n  File "x.py"'
      )
    ).toBe('exitedEarly')
    expect(classifyLocalBootFailure('Timed out connecting to Hermes backend after 45000ms')).toBe('timedOut')
    expect(classifyLocalBootFailure("EACCES: permission denied, open '/home/x/.hermes/state.db'")).toBe('permission')
    expect(classifyLocalBootFailure('OSError: [Errno 28] No space left on device')).toBe('diskFull')
    expect(classifyLocalBootFailure('listen EADDRINUSE: address already in use 127.0.0.1:9191')).toBe('portInUse')
    expect(classifyLocalBootFailure(null)).toBeNull()
  })

  it('keeps the raw output out of the headline and behind details', () => {
    const raw =
      'Hermes backend exited before it became ready (1).\nRecent backend output:\nTraceback (most recent call last):'

    const copy = localBootFailureCopy(raw, CAUSES)

    expect(copy.headline).toBe(CAUSES.exitedEarly)
    expect(copy.headline).not.toMatch(/\(1\)|Traceback|ms\b/)
    expect(copy.rawDetail).toBe(raw)
  })

  it('falls back to the first raw line for an unknown failure, never a dump', () => {
    const copy = localBootFailureCopy('Something odd happened\nline 2\nline 3', CAUSES)

    expect(copy.headline).toBe('Something odd happened')
    expect(copy.rawDetail).toContain('line 3')
  })
})
