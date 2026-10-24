import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { once } from 'node:events'
import { mkdir, mkdtemp, rename, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { build } from 'esbuild'
import { test } from 'vitest'

// Exercise the real entry point before main loads. The fixture models a resident
// helper outside the app tree; only its inherited CWD can block the directory swap.
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
                      : `console.log(process.cwd()); setInterval(() => {}, 1000)`
                }))
              }
            }
          ]
        })
        const child = spawn(process.execPath, [fixture], { cwd, stdio: ['ignore', 'pipe', 'pipe'] })
        const exited = once(child, 'exit')

        try {
          const [output] = await once(child.stdout!, 'data')
          assert.equal(String(output).trim().toLowerCase(), expected.toLowerCase())
          const moved = `${install}.previous`

          if (!packaged) {
            // Reproduce the original Windows sharing violation on an isolated tree.
            await assert.rejects(rename(install, moved), { code: 'EBUSY' })
          } else {
            await rename(install, moved)
            await rename(moved, install)
          }
        } finally {
          child.kill()
          await exited
        }
      }
    } finally {
      await rm(root, { recursive: true, force: true })
    }
  }
)
