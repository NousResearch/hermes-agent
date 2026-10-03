/**
 * Regression for #126231: the Linux package manifests must declare the
 * XDG desktop portal Electron's file dialogs need on a Wayland session.
 *
 * `apps/desktop/electron-builder.config.cjs` sets `linux.target: ['AppImage']`,
 * but `npm run dist:linux` also builds `deb` and `rpm` (the target list on the
 * command line widens it), so those two packages are real artifacts of this
 * config. Neither had a `depends` block, which meant both inherited
 * electron-builder's defaults verbatim -- and those defaults list GTK, NSS,
 * libnotify, at-spi2 and xdg-utils, but no XDG desktop portal at all. On a
 * Wayland session Electron routes native file dialogs through
 * `org.freedesktop.portal.FileChooser`, and that interface is only exported
 * when a portal *backend* advertises
 * `org.freedesktop.impl.portal.FileChooser`. The frontend package
 * `xdg-desktop-portal` implements no portal and depends on no backend, so
 * declaring it guarantees nothing; `xdg-desktop-portal-gtk` is the package
 * that depends on the frontend AND ships the FileChooser impl, so it is what
 * these lists must name. A session whose only backend is FileChooser-less (the
 * reporter had `xdg-desktop-portal-hyprland`, which does not implement it)
 * fails the dialog with
 * `No such interface "org.freedesktop.portal.FileChooser"` /
 * `Failed to read portal version property` from
 * `electron/shell/browser/ui/file_dialog_linux_portal.cc`.
 *
 * Setting `depends` REPLACES the defaults rather than extending them, so this
 * file also pins every default the config must keep. Dropping one (say
 * `libsecret-1-0`, which `electron/hardening.ts` needs for safeStorage, or
 * `libatspi2.0-0`, which screen readers need) would otherwise be a silent
 * packaging regression that no other test catches.
 *
 * Why this test lives in tests-js/, not tests/*.py
 * -------------------------------------------------
 *
 * Same reason as `tests-js/desktop-mac-usage-descriptions.test.ts`: the CI
 * change classifier can skip the Python suite on a JS-only PR, so a
 * packaging assertion parked in Python would go green on the PR and red on
 * `main`. The packaging config is a JS artifact; its assertions belong here.
 *
 * What this test does NOT prove
 * -----------------------------
 *
 * That Electron launches on a live Hyprland session. Nothing here observes a
 * running compositor or session bus. These assertions hold that the declared
 * dependency set is right; the end-to-end behaviour still needs a live
 * Wayland host.
 */

import assert from 'node:assert/strict'
import { createRequire } from 'node:module'

import { test } from 'vitest'

const require: NodeJS.Require = createRequire(import.meta.url)

// electron-builder's FpmTarget.getDefaultDepends(), verbatim. Listed here so
// the config's explicit arrays can be diffed against them by eye, and so the
// "keeps the defaults" test below fails loudly if a future electron-builder
// bump changes them out from under us.
const ELECTRON_BUILDER_DEFAULT_DEB_DEPENDS: readonly string[] = [
  'libgtk-3-0',
  'libnotify4',
  'libnss3',
  'libxss1',
  'libxtst6',
  'xdg-utils',
  'libatspi2.0-0',
  'libuuid1',
  'libsecret-1-0'
]

const ELECTRON_BUILDER_DEFAULT_RPM_DEPENDS: readonly string[] = [
  'gtk3',
  'libnotify',
  'nss',
  'libXScrnSaver',
  '(libXtst or libXtst6)',
  'xdg-utils',
  'at-spi2-core',
  '(libuuid or libuuid1)'
]

// The backend package, not the `xdg-desktop-portal` frontend: only a backend
// that advertises org.freedesktop.impl.portal.FileChooser puts
// org.freedesktop.portal.FileChooser on the session bus. `xdg-desktop-portal`
// itself ships no portal impl and depends on no backend, so naming it satisfies
// the installer without putting FileChooser on the bus.
const PORTAL_BACKEND_PACKAGE = 'xdg-desktop-portal-gtk'

interface LinuxPackagingConfig {
  deb?: { depends?: string[] | null }
  rpm?: { depends?: string[] | null }
}

function linuxPackagingConfig(): LinuxPackagingConfig {
  const config = require('../apps/desktop/electron-builder.config.cjs')

  assert.ok(config, 'electron-builder.config.cjs must be requirable')

  return config as LinuxPackagingConfig
}

test('the deb package depends on a portal backend that implements FileChooser', () => {
  const depends = linuxPackagingConfig().deb?.depends

  assert.ok(Array.isArray(depends), 'deb.depends must be an explicit list on Wayland sessions')
  assert.ok(
    depends.includes(PORTAL_BACKEND_PACKAGE),
    `deb.depends must include ${PORTAL_BACKEND_PACKAGE}; the xdg-desktop-portal frontend alone implements no portal, so a .deb install on a Wayland session whose backend lacks FileChooser keeps failing Electron's file dialog with "No such interface" (#126231)`
  )
})

test('the rpm package depends on a portal backend that implements FileChooser', () => {
  const depends = linuxPackagingConfig().rpm?.depends

  assert.ok(Array.isArray(depends), 'rpm.depends must be an explicit list on Wayland sessions')
  assert.ok(
    depends.includes(PORTAL_BACKEND_PACKAGE),
    `rpm.depends must include ${PORTAL_BACKEND_PACKAGE}; the xdg-desktop-portal frontend alone implements no portal, so an .rpm install on a Wayland session whose backend lacks FileChooser keeps failing Electron's file dialog with "No such interface" (#126231)`
  )
})

test('declaring depends keeps every electron-builder default dependency', () => {
  const { deb, rpm } = linuxPackagingConfig()

  for (const dep of ELECTRON_BUILDER_DEFAULT_DEB_DEPENDS) {
    assert.ok(
      deb?.depends?.includes(dep),
      `deb.depends must keep electron-builder's default ${dep}: an explicit depends list REPLACES the defaults, not extends them`
    )
  }

  for (const dep of ELECTRON_BUILDER_DEFAULT_RPM_DEPENDS) {
    assert.ok(
      rpm?.depends?.includes(dep),
      `rpm.depends must keep electron-builder's default ${dep}: an explicit depends list REPLACES the defaults, not extends them`
    )
  }
})
