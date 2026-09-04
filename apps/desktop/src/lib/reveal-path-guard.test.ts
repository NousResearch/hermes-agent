import { describe, expect, it } from 'vitest'

import { isUnsafeRevealPath } from './reveal-path-guard'

describe('isUnsafeRevealPath', () => {
  it.each(['\\\\server\\share\\report.pdf', '//server/share/report.pdf', '\\\\?\\C:\\secret.txt', '\\\\.\\pipe\\name'])(
    'rejects a network or device path: %s',
    value => expect(isUnsafeRevealPath(value)).toBe(true)
  )

  it.each(['C:\\Users\\alex\\report.pdf', '/home/alex/report.pdf', '/mnt/c/Users/alex/report.pdf'])(
    'allows a local path: %s',
    value => expect(isUnsafeRevealPath(value)).toBe(false)
  )
})
