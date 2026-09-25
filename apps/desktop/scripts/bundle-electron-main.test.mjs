import { execFileSync } from 'node:child_process'
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

import { afterAll, beforeAll, expect, test } from 'vitest'

import { bundleElectronMain } from './bundle-electron-main.mjs'

const repo = resolve(import.meta.dirname, '../../..')
let root
let bundle
let preload

// Run the shipped electron-main.mjs under plain Node as if on Linux. The entry
// relaunches through process.execPath; the preload (inherited through
// NODE_OPTIONS) makes that child print its argv instead of starting the app.
beforeAll(async () => {
  root = mkdtempSync(join(tmpdir(), 'hermes-electron-entry-'))
  await bundleElectronMain({ source: repo, out: join(root, 'dist'), dev: true })
  bundle = join(root, 'dist/electron-main.mjs')
  preload = join(root, 'preload.mjs')
  writeFileSync(
    preload,
    `
    import { readFileSync } from 'node:fs'
    import { registerHooks } from 'node:module'
    Object.defineProperty(process, 'platform', { value: 'linux' })
    if (process.argv.some(arg => arg.startsWith('--ozone-platform='))) {
      console.log(JSON.stringify({ relaunched: process.argv.slice(2) }))
      process.exit(0)
    }
    // Externals the bundle leaves to Electron's node_modules. Stub each with
    // every name the bundle imports from it; only app.exit is ever called.
    const bundled = readFileSync(${JSON.stringify(bundle)}, 'utf8')
    const stub = specifier => {
      const names = new Set(specifier === 'electron' ? ['app'] : [])
      for (const [, list, from] of bundled.matchAll(/import\\s*\\{([^}]*)\\}\\s*from\\s*"([^"]+)"/g)) {
        if (from !== specifier) continue
        for (const name of list.split(',')) names.add(name.trim().split(/\\s+as\\s+/)[0])
      }
      names.delete('')
      return [...names].map(name => name === 'app'
        ? 'export const app = { exit: code => process.exit(code) }'
        : 'export const ' + name + ' = {}').join('\\n') + '\\nexport default {}'
    }
    const externals = new Set(['electron', 'node-pty', 'get-windows'])
    registerHooks({
      resolve: (specifier, context, next) => externals.has(specifier)
        ? { url: 'hermes-stub:' + specifier, shortCircuit: true } : next(specifier, context),
      load: (url, context, next) => url.startsWith('hermes-stub:')
        ? { format: 'module', source: stub(url.slice('hermes-stub:'.length)), shortCircuit: true } : next(url, context),
    })
    `
  )
}, 120_000)

afterAll(() => rmSync(root, { recursive: true, force: true }))

function launch(env, config) {
  const home = mkdtempSync(join(root, 'home-'))

  if (config) {
    mkdirSync(home, { recursive: true })
    writeFileSync(join(home, 'config.yaml'), config)
  }

  const out = execFileSync(process.execPath, [bundle, '.'], {
    cwd: root,
    encoding: 'utf8',
    env: {
      PATH: process.env.PATH,
      HERMES_HOME: home,
      NODE_OPTIONS: `--import=${pathToFileURL(preload).href}`,
      ...env
    }
  })

  return JSON.parse(out.trim().split('\n').at(-1))
}

test('the bundled entry relaunches native Wayland sessions with the wayland ozone platform', () => {
  expect(launch({ XDG_SESSION_TYPE: 'wayland', WAYLAND_DISPLAY: 'wayland-0', DISPLAY: ':0' })).toEqual({
    relaunched: ['.', '--ozone-platform=wayland']
  })
})

test('the bundled entry keeps the WSLg Wayland relaunch', () => {
  expect(launch({ WSL_DISTRO_NAME: 'Ubuntu', WAYLAND_DISPLAY: 'wayland-0', DISPLAY: ':0' })).toEqual({
    relaunched: ['.', '--ozone-platform=wayland']
  })
})

test('the bundled entry lets desktop.electron_flags choose the ozone platform', () => {
  const config = 'desktop:\n  electron_flags:\n    - --ozone-platform=x11\n'

  expect(launch({ XDG_SESSION_TYPE: 'wayland', WAYLAND_DISPLAY: 'wayland-0', DISPLAY: ':0' }, config)).toEqual({
    relaunched: ['.', '--ozone-platform=x11']
  })
})
