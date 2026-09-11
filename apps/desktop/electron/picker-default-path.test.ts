/**
 * Regression: the native picker's defaultPath must be validated against the
 * LOCAL filesystem before the dialog is seeded with it.
 *
 * `hermes:selectPaths` receives the backend session's cwd as defaultPath. In
 * remote mode that backend is another machine (often as another user), so the
 * path can be unreadable here — a remote gateway running as root yields
 * `/root`, and a dialog seeded with it failed wholesale on Linux with
 * "Could not read the contents of root: Permission denied". The renderer-side
 * file-read path is already local-first; the picker never validated, so the
 * gate lives at the Electron boundary (picker-default-path.ts) and covers
 * every picker caller at once.
 */
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterAll, beforeAll, describe, expect, test } from 'vitest'

import { locallyReadable } from './picker-default-path'

describe('picker defaultPath local-readability gate', () => {
  const scratch = path.join(os.tmpdir(), 'hermes-picker-gate-test')

  beforeAll(() => {
    fs.mkdirSync(scratch, { recursive: true })
  })

  afterAll(() => {
    fs.rmSync(scratch, { recursive: true, force: true })
  })

  test('keeps a locally readable directory', () => {
    expect(locallyReadable(scratch)).toBe(scratch)
  })

  test('drops a path this process cannot read (chmod 000)', () => {
    const sealed = path.join(scratch, 'sealed')
    fs.mkdirSync(sealed, { recursive: true })
    fs.chmodSync(sealed, 0o000)

    try {
      expect(locallyReadable(sealed)).toBeNull()
    } finally {
      fs.chmodSync(sealed, 0o700)
    }
  })

  test('drops a path that does not exist locally (remote cwd)', () => {
    const absent = path.join(scratch, 'only-on-the-remote-backend')
    expect(locallyReadable(absent)).toBeNull()
  })

  test('empty and nullish hints stay null (dialog opens at its native default)', () => {
    expect(locallyReadable(undefined)).toBeNull()
    expect(locallyReadable(null)).toBeNull()
    expect(locallyReadable('')).toBeNull()
  })

  // The bug report's exact shape: a superuser-owned, 0700 home directory
  // (e.g. a remote root backend's cwd rendered on a non-root laptop). Skipped
  // only when the test runner itself has root privileges, where the dialog
  // would not have failed in the first place.
  test.skipIf(process.getuid?.() === 0)('drops a foreign 0700 home like /root', () => {
    expect(locallyReadable('/root')).toBeNull()
  })
})
