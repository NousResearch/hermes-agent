import assert from 'node:assert/strict'

import { test } from 'vitest'

import { nativeWindowsDrivePath } from './native-windows-path'

// The ONE rule both seams share: a delivered Windows drive spelling
// (`/C:/Users/...`) is a Windows path, and the target must receive it in the
// Windows dialect.
//
// The two seams differ in WHOSE platform decides:
//   - the gateway seam transcodes for the TARGET (a macOS desktop talking to
//     a Windows gateway over SSH sends Windows paths, and the gateway must
//     receive them in its own dialect), so `isWindowsHost` follows the target;
//   - the local preview seam resolves on the LOCAL host, so a POSIX host must
//     leave `/C:/...` alone — there it is a legitimate POSIX path and the
//     host's own `path.resolve` handles it correctly.
//
// The platform is a parameter, not read from `process`, so both branches are
// reachable from any runner.

test('the mapping applies when the target is Windows, whatever the host is', () => {
  assert.equal(nativeWindowsDrivePath('/C:/Users/me/notes.md', 'win32', true), 'C:\\Users\\me\\notes.md')
  assert.equal(nativeWindowsDrivePath('C:/Users/me/notes.md', 'win32', true), 'C:\\Users\\me\\notes.md')
  assert.equal(nativeWindowsDrivePath('/C:\\Users/me/notes.md', 'win32', true), 'C:\\Users\\me\\notes.md')
})

test('a POSIX host leaves the delivered spelling alone', () => {
  // On a POSIX host `/C:/Users/...` is a legitimate absolute path; rewriting it
  // into a drive path would point at a file that does not exist.
  assert.equal(nativeWindowsDrivePath('/C:/Users/me/notes.md', 'win32', false), '/C:/Users/me/notes.md')
  assert.equal(nativeWindowsDrivePath('/home/me/notes.md', 'linux', false), '/home/me/notes.md')
})

test('a non-Windows target is never rewritten', () => {
  assert.equal(nativeWindowsDrivePath('/C:/Users/me/notes.md', 'darwin', true), '/C:/Users/me/notes.md')
  assert.equal(nativeWindowsDrivePath('/C:/Users/me/notes.md', 'linux', true), '/C:/Users/me/notes.md')
})

test('POSIX, MSYS and file: spellings are left alone', () => {
  for (const candidate of [
    '/home/me/notes.md',
    // An MSYS drive mount is NOT a native Windows path.
    '/c/Users/me/notes.md',
    '~/notes.md',
    'relative/notes.md',
    'file:///C:/Users/me/a%20b.md'
  ]) {
    assert.equal(nativeWindowsDrivePath(candidate, 'win32', true), candidate)
  }
})

test('an empty value stays empty', () => {
  assert.equal(nativeWindowsDrivePath('', 'win32', true), '')
  assert.equal(nativeWindowsDrivePath('   ', 'win32', true), '')
})
