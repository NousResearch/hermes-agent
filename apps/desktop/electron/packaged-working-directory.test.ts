import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { once } from 'node:events'
import { mkdir, mkdtemp, rename, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { createInterface } from 'node:readline'
import { fileURLToPath } from 'node:url'

import { build } from 'esbuild'
import { test } from 'vitest'

import { packagedWorkingDirectoryTarget } from './packaged-working-directory'

// The resident helper (an IME cloud service, a WSL/ssh helper) that the app spawns
// WITHOUT a cwd of its own, so it inherits the app's. It outlives the app and holds
// that directory as its cwd for as long as it lives. `detached` keeps it out of
// libuv's kill-on-close job, as a process created natively (by an IME inside the
// app) never joins it.
const HELPER = 'console.log(JSON.stringify({ pid: process.pid, cwd: process.cwd() })); setInterval(() => {}, 1000)'

function killPid(pid: number | undefined): void {
  if (!pid) {
    return
  }

  try {
    process.kill(pid)
  } catch {
    // already gone
  }
}

// The decision itself, on every CI host (the host is data); the live EBUSY proof below needs Windows.
test('a packaged Windows app leaves a CWD inside its install folder and keeps one outside it', () => {
  const app = { platform: 'win32', isPackaged: true, execPath: 'C:\\Hermes\\win-unpacked\\Hermes.exe' } as const
  const home = 'C:\\Users\\ada'

  const target = (cwd: string, facts: Partial<Parameters<typeof packagedWorkingDirectoryTarget>[0]> = {}) =>
    packagedWorkingDirectoryTarget({ ...app, cwd, homedir: home, ...facts })

  assert.equal(target('C:\\Hermes\\win-unpacked'), home, 'the install folder (shortcut "Start in")')
  assert.equal(target('c:\\hermes\\WIN-UNPACKED\\resources'), home, 'below it, any case')
  assert.equal(target('C:\\Hermes'), null, 'the parent folder is a workspace, not the app tree')
  assert.equal(target('D:\\work\\repo'), null, 'a workspace on another drive')
  assert.equal(target('C:\\Hermes\\win-unpacked-old'), null, 'a sibling that shares a name prefix')
  assert.equal(target('C:\\Hermes\\win-unpacked', { isPackaged: false }), null, 'dev runs keep their CWD')

  for (const platform of ['linux', 'darwin'] as const) {
    assert.equal(target('/opt/Hermes', { platform, execPath: '/opt/Hermes/hermes' }), null, platform)
  }
})

// Exercise the real entry point before main loads: the main stand-in spawns the
// helper, the app is killed, and the directory swap runs with only the orphaned
// helper (exe outside the app tree) still alive.
test.skipIf(process.platform !== 'win32')(
  'packaged startup releases the install CWD before main can pass it to resident helpers',
  async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), 'hermes-entry-cwd-'))
    const install = path.join(root, 'win-unpacked')
    const workspace = path.join(root, 'workspace')
    await mkdir(install)
    await mkdir(workspace)

    try {
      for (const [packaged, cwd, expected] of [
        [false, install, install],
        [true, install, os.homedir()],
        [true, workspace, workspace]
      ] as const) {
        const fixture = path.join(root, 'entry.mjs')
        await build({
          entryPoints: [fileURLToPath(new URL('./entry.ts', import.meta.url))],
          outfile: fixture,
          bundle: true,
          platform: 'node',
          format: 'esm',
          define: {
            'process.execPath': JSON.stringify(path.join(install, 'Hermes.exe')),
            'process.env.HERMES_DESKTOP_IS_PACKAGED': 'undefined'
          },
          plugins: [
            {
              name: 'entry-cwd-fixture',
              setup(builder) {
                builder.onResolve({ filter: /^electron$/ }, () => ({ path: 'electron', namespace: 'fixture' }))
                builder.onResolve({ filter: /^\.\/main$/ }, () => ({ path: 'main', namespace: 'fixture' }))
                builder.onLoad({ filter: /.*/, namespace: 'fixture' }, ({ path: name }) => ({
                  contents:
                    name === 'electron'
                      ? `export const app = { isPackaged: ${packaged} }`
                      : [
                          `import { spawn } from 'node:child_process'`,
                          `spawn(process.env.FIXTURE_NODE, ['-e', ${JSON.stringify(HELPER)}], { detached: true, stdio: ['ignore', 'inherit', 'ignore'] })`,
                          `setInterval(() => {}, 1000)`
                        ].join('\n')
                }))
              }
            }
          ]
        })

        const app = spawn(process.execPath, [fixture], {
          cwd,
          env: { ...process.env, FIXTURE_NODE: process.execPath },
          stdio: ['ignore', 'pipe', 'ignore']
        })

        const appExited = once(app, 'exit')
        let helperPid: number | undefined

        try {
          const [line] = await once(createInterface({ input: app.stdout! }), 'line')
          const helper = JSON.parse(String(line)) as { pid: number; cwd: string }
          helperPid = helper.pid
          assert.equal(helper.cwd.toLowerCase(), expected.toLowerCase())

          // Quit the app: the helper is orphaned and keeps its inherited cwd.
          app.kill()
          await appExited
          process.kill(helperPid, 0)
          const moved = `${install}.previous`

          if (!packaged) {
            // Reproduce the original Windows sharing violation on an isolated tree.
            await assert.rejects(rename(install, moved), { code: 'EBUSY' })
          } else {
            await rename(install, moved)
            await rename(moved, install)
          }
        } finally {
          app.kill()
          killPid(helperPid)
        }
      }
    } finally {
      await rm(root, { recursive: true, force: true })
    }
  }
)
