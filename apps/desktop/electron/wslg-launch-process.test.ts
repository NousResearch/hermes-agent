import assert from 'node:assert/strict'
import { type ChildProcess, spawn } from 'node:child_process'
import { once } from 'node:events'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { createInterface } from 'node:readline'
import type { Readable } from 'node:stream'
import { fileURLToPath } from 'node:url'

import { build } from 'esbuild'
import { afterAll, beforeAll, test } from 'vitest'

let fixtureDir: string

beforeAll(async () => {
  fixtureDir = await mkdtemp(path.join(tmpdir(), 'hermes-wslg-process-'))
  await build({
    entryPoints: [
      fileURLToPath(new URL('./wslg-launch-process.ts', import.meta.url)),
      fileURLToPath(new URL('./linux-launcher-ready.ts', import.meta.url))
    ],
    outdir: fixtureDir,
    outExtension: { '.js': '.mjs' },
    bundle: true,
    platform: 'node',
    format: 'esm'
  })
  await writeFile(
    path.join(fixtureDir, 'app.mjs'),
    `
    import inspector from 'node:inspector'
    import { notifyLauncherWindowRevealed } from './linux-launcher-ready.mjs'
    console.log(JSON.stringify({
      kind: 'app', pid: process.pid, url: inspector.url(),
      ready: notifyLauncherWindowRevealed(), argv: process.execArgv
    }))
    setInterval(() => {}, 1000)
  `
  )
  await writeFile(
    path.join(fixtureDir, 'launcher.mjs'),
    `
    import { fstatSync } from 'node:fs'
    import inspector from 'node:inspector'
    import { spawnWslgLaunch } from './wslg-launch-process.mjs'
    const fd = Number(process.env.HERMES_DESKTOP_READY_FD)
    const originalUrl = inspector.url()
    const args = originalUrl ? ['--inspect=' + new URL(originalUrl).host] : []
    args.push(new URL('./app.mjs', import.meta.url).pathname)
    if (process.env.TEST_FAIL_SPAWN) process.execPath += '.missing'
    const child = spawnWslgLaunch(args)
    let fdClosed = false
    try { fstatSync(fd) } catch { fdClosed = true }
    console.log(JSON.stringify({ kind: 'launcher', pid: process.pid,
      originalUrl, url: inspector.url(), fdClosed,
      readyEnv: process.env.HERMES_DESKTOP_READY_FD
    }))
    child.once('error', error => { console.error(error); process.exit(1) })
    child.once('exit', (code, signal) => {
      console.log(JSON.stringify({ kind: 'exit', code, signal }))
      process.exit(code ?? 1)
    })
    process.stdin.on('data', () => child.kill('SIGTERM'))
  `
  )
})

afterAll(async () => {
  if (fixtureDir) {
    await rm(fixtureDir, { recursive: true, force: true })
  }
})

async function withLauncher(
  inspect: boolean,
  run: (launcher: ChildProcess, messages: AsyncIterator<string>, stderr: () => string) => Promise<void>,
  readyFd: string | undefined = '7',
  failSpawn = false
) {
  const launcher = spawn(
    process.execPath,
    [...(inspect ? ['--inspect=127.0.0.1:0'] : []), path.join(fixtureDir, 'launcher.mjs')],
    {
      detached: true,
      env: {
        ...process.env,
        NODE_OPTIONS: '',
        HERMES_DESKTOP_READY_FD: readyFd,
        TEST_FAIL_SPAWN: failSpawn ? '1' : ''
      },
      stdio: ['pipe', 'pipe', 'pipe', 'ignore', 'ignore', 'ignore', 'ignore', 'pipe']
    }
  )

  let stderr = ''
  launcher.stderr!.on('data', chunk => {
    stderr += chunk
  })
  const lines = createInterface({ input: launcher.stdout! })

  let timer: ReturnType<typeof setTimeout>

  try {
    await Promise.race([
      run(launcher, lines[Symbol.asyncIterator](), () => stderr),
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new Error('Launcher lifecycle timed out: ' + stderr)), 10_000)
      })
    ])
  } finally {
    clearTimeout(timer!)
    lines.close()

    // Include the app child even if an assertion fails before orderly shutdown.
    try {
      process.kill(-launcher.pid!, 'SIGKILL')
    } catch (error) {
      assert.equal((error as NodeJS.ErrnoException).code, 'ESRCH')
    }

    if (launcher.exitCode === null && launcher.signalCode === null) {
      await once(launcher, 'exit')
    }
  }
}

async function startup(messages: AsyncIterator<string>) {
  const results: Record<string, any> = {}

  for (let i = 0; i < 2; i++) {
    const line = await messages.next()
    assert.equal(line.done, false, 'launcher exited before app startup')
    const message = JSON.parse(line.value)
    results[message.kind] = message
  }

  return results
}

test.skipIf(process.platform !== 'linux')(
  'hands the ready pipe to the app and closes the waiting parent copy',
  async () => {
    await withLauncher(false, async (launcher, messages) => {
      let ready = ''
      const pipe = (launcher.stdio as Array<Readable | null>)[7]!
      pipe.on('data', chunk => {
        ready += chunk
      })
      const eof = once(pipe, 'end')
      const state = await startup(messages)
      assert.equal(state.app.ready, true)
      assert.equal(state.launcher.fdClosed, true)
      assert.equal(state.launcher.readyEnv, undefined)
      await eof
      assert.equal(ready, 'r')
      // EOF must arrive while both the waiting launcher and app still live.
      assert.equal(launcher.exitCode, null)
      process.kill(state.app.pid, 0)
      const exited = once(launcher, 'exit')
      launcher.stdin!.write('stop')
      assert.deepEqual(await exited, [1, null])
      const exit = JSON.parse((await messages.next()).value)
      assert.equal(exit.signal, 'SIGTERM')
    })
  },
  15_000
)

test.skipIf(process.platform !== 'linux')(
  'releases the launcher inspector so the app owns the requested port',
  async () => {
    await withLauncher(true, async (launcher, messages, stderr) => {
      ;(launcher.stdio as Array<Readable | null>)[7]!.resume()
      const state = await startup(messages)
      assert.ok(state.launcher.originalUrl)
      assert.equal(state.launcher.url, undefined)
      assert.ok(state.app.url, stderr())
      const requested = new URL(state.launcher.originalUrl).host
      assert.equal(new URL(state.app.url).host, requested)
      assert.deepEqual(state.app.argv, ['--inspect=' + requested])
      const targets = await fetch('http://' + requested + '/json/list').then(response => response.json())
      assert.equal(targets[0].webSocketDebuggerUrl, state.app.url)
      assert.notEqual(targets[0].webSocketDebuggerUrl, state.launcher.originalUrl)
      assert.doesNotMatch(stderr(), /address already in use/i)
      // The supervisor remains alive but is no longer the debug target.
      assert.equal(launcher.exitCode, null)
      process.kill(state.app.pid, 0)
    })
  },
  15_000
)

test
  .skipIf(process.platform !== 'linux')
  .each(['', 'seven', '0', '1', '2', '0x7', '100000'])(
  'ignores invalid or closed ready fd %j without breaking child stdio',
  async readyFd => {
    await withLauncher(
      false,
      async (launcher, messages) => {
        const state = await startup(messages)
        assert.equal(state.app.ready, false)
        assert.equal(state.launcher.readyEnv, undefined)
        const exited = once(launcher, 'exit')
        launcher.stdin!.write('stop')
        assert.deepEqual(await exited, [1, null])
      },
      readyFd
    )
  },
  15_000
)

test.skipIf(process.platform !== 'linux')(
  'closes the ready pipe even when the app cannot be spawned',
  async () => {
    await withLauncher(
      false,
      async (launcher, messages, stderr) => {
        const pipe = (launcher.stdio as Array<Readable | null>)[7]!
        let ready = ''
        pipe.on('data', chunk => {
          ready += chunk
        })
        const eof = once(pipe, 'end')
        const exited = once(launcher, 'exit')
        const state = JSON.parse((await messages.next()).value)
        assert.equal(state.kind, 'launcher')
        assert.equal(state.fdClosed, true)
        assert.equal(state.readyEnv, undefined)
        await eof
        assert.equal(ready, '')
        assert.deepEqual(await exited, [1, null])
        assert.match(stderr(), /ENOENT/)
      },
      '7',
      true
    )
  },
  15_000
)
