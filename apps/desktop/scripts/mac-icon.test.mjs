import { createRequire } from 'node:module'
import path from 'node:path'
import { expect, it, vi } from 'vitest'

const require = createRequire(import.meta.url)
const desktopRoot = path.resolve(import.meta.dirname, '..')
const {
  ICON_COMPOSER,
  LEGACY_ICNS,
  installedActoolVersion,
  macIconResource,
  parseActoolVersion
} = require('./mac-icon.cjs')

// actool answers --version with a plist whose top-level key itself contains dots.
const ACTOOL_PLIST = `<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0">
<dict>
\t<key>com.apple.actool.version</key>
\t<dict>
\t\t<key>bundle-version</key>
\t\t<string>25098</string>
\t\t<key>short-bundle-version</key>
\t\t<string>27.0</string>
\t</dict>
</dict>
</plist>
`

it('packages the Icon Composer layers only where actool can compile them', () => {
  const log = vi.fn()
  expect(macIconResource(desktopRoot, { platform: 'darwin', actoolVersion: '26.0', log })).toBe(ICON_COMPOSER)
  expect(macIconResource(desktopRoot, { platform: 'darwin', actoolVersion: '27.0', log })).toBe(ICON_COMPOSER)
  expect(log).not.toHaveBeenCalled()
  // Xcode 16's actool throws inside electron-builder; the .icns-only app must still build.
  expect(macIconResource(desktopRoot, { platform: 'darwin', actoolVersion: '16.4', log })).toBe(LEGACY_ICNS)
  expect(macIconResource(desktopRoot, { platform: 'darwin', actoolVersion: null, log })).toBe(LEGACY_ICNS)
  expect(log).toHaveBeenCalledTimes(2)
  expect(log.mock.calls[1][0]).toContain('not found')
  expect(log.mock.calls[0][0]).toContain(path.join(desktopRoot, ICON_COMPOSER))
})

it('never packages the layered icon from a non-mac host', () => {
  for (const platform of ['linux', 'win32']) {
    expect(macIconResource(desktopRoot, { platform })).toBe(LEGACY_ICNS)
  }
})

it('reads the version out of the actool plist and treats any failure as absent', () => {
  expect(parseActoolVersion(ACTOOL_PLIST)).toBe('27.0')
  expect(parseActoolVersion('xcrun: error: unable to find utility "actool"')).toBeNull()
  expect(installedActoolVersion(() => ({ status: 0, stdout: ACTOOL_PLIST }))).toBe('27.0')
  expect(installedActoolVersion(() => ({ status: 72, stdout: '', stderr: 'no developer tools' }))).toBeNull()
  expect(installedActoolVersion(() => { throw new Error('ENOENT') })).toBeNull()
})
