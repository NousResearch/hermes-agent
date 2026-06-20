const assert = require('node:assert/strict')
const { EventEmitter } = require('node:events')
const { PassThrough } = require('node:stream')
const test = require('node:test')

const {
  buildAugmentedPath,
  buildLangflowLaunch,
  createWorkflowBackendManager,
  getWorkflowAuthStatus,
  killProcessTree,
  resolveLangflowRoot
} = require('./workflow-backend.cjs')

function fakeChild(pid = 4242) {
  const child = new EventEmitter()
  child.pid = pid
  child.killed = false
  child.stdout = new PassThrough()
  child.stderr = new PassThrough()
  child.kill = signal => {
    child.killed = true
    child.signal = signal
    child.emit('exit', null, signal)
    return true
  }

  return child
}

function eventually(assertion, timeoutMs = 250) {
  const deadline = Date.now() + timeoutMs

  return new Promise((resolve, reject) => {
    const tick = () => {
      Promise.resolve()
        .then(assertion)
        .then(resolve)
        .catch(error => {
          if (Date.now() >= deadline) {
            reject(error)
            return
          }

          setTimeout(tick, 10)
        })
    }

    tick()
  })
}

test('buildLangflowLaunch starts our Langflow checkout without opening an external browser', () => {
  const launch = buildLangflowLaunch({
    configDir: '/tmp/hermes/langflow',
    root: '/workspace/langflow',
    host: '127.0.0.1',
    port: 7860
  })

  assert.equal(launch.command, 'make')
  assert.deepEqual(launch.args, ['run_cli', 'host=127.0.0.1', 'port=7860', 'open_browser=false'])
  assert.equal(launch.options.cwd, '/workspace/langflow')
  assert.equal(launch.options.detached, true)
  assert.equal(launch.options.shell, false)
  assert.equal(launch.options.windowsHide, true)
  assert.equal(launch.options.env.LANGFLOW_CONFIG_DIR, '/tmp/hermes/langflow')
  assert.equal(launch.options.env.KARI_PERMS_DB, '/tmp/hermes/langflow/.kari_perms.sqlite')
  assert.equal(launch.options.env.PYTHONUNBUFFERED, '1')
})

test('buildLangflowLaunch forces embedded Langflow to use EasyHermes outer login state', () => {
  const launch = buildLangflowLaunch({
    env: {
      KARI_HUB_URL: 'https://cloud.example',
      LANGFLOW_AUTO_LOGIN: 'false',
      LANGFLOW_SKIP_AUTH_AUTO_LOGIN: 'false',
      OPENAI_API_KEY: 'llm-from-hub'
    },
    root: '/workspace/langflow'
  })

  assert.equal(launch.options.env.LANGFLOW_AUTO_LOGIN, 'true')
  assert.equal(launch.options.env.LANGFLOW_SKIP_AUTH_AUTO_LOGIN, 'true')
  assert.equal(launch.options.env.KARI_HUB_URL, 'https://cloud.example')
  assert.equal(launch.options.env.OPENAI_API_KEY, 'llm-from-hub')
})

test('buildLangflowLaunch maps local workflow secrets into Langflow runtime env', () => {
  const launch = buildLangflowLaunch({
    env: {
      OPENAI_API_KEY: 'old-openai',
      OPENAI_BASE_URL: 'https://old.example/v1'
    },
    readFile: filePath => {
      assert.equal(filePath, '/tmp/hermes/workflow-secrets.json')
      return JSON.stringify({
        env: {
          ANTHROPIC_API_KEY: 'anthropic-from-local'
        },
        kari: {
          token: 'tok-from-local',
          cloudBaseURL: 'https://cloud.example'
        },
        openai: {
          apiKey: 'openai-from-local',
          baseURL: 'https://local.example/v1'
        },
        anthropic: {
          apiKey: 'anthropic-object-key',
          baseURL: 'https://anthropic.example/v1'
        }
      })
    },
    root: '/workspace/langflow',
    secretsPath: '/tmp/hermes/workflow-secrets.json'
  })

  // 云端中枢地址 + per-user token 注入;真实 KIE_API_KEY 绝不在本地出现。
  assert.equal(launch.options.env.KARI_HUB_URL, 'https://cloud.example')
  assert.equal(launch.options.env.KARI_WORKSPACE_TOKEN, 'tok-from-local')
  assert.equal(launch.options.env.KIE_API_KEY, undefined)
  assert.equal(launch.options.env.OPENAI_API_KEY, 'openai-from-local')
  assert.equal(launch.options.env.OPENAI_BASE_URL, 'https://local.example/v1')
  assert.equal(launch.options.env.ANTHROPIC_API_KEY, 'anthropic-object-key')
  assert.equal(launch.options.env.ANTHROPIC_BASE_URL, 'https://anthropic.example/v1')
  assert.equal(launch.options.env.KARI_LLM_PERFORMANCE_API_KEY, 'openai-from-local')
  assert.equal(launch.options.env.KARI_LLM_PERFORMANCE_BASE_URL, 'https://local.example/v1')
  assert.equal(launch.options.env.KARI_LLM_EXTREME_API_KEY, 'anthropic-object-key')
  assert.equal(launch.options.env.KARI_LLM_EXTREME_BASE_URL, 'https://anthropic.example/v1')
})

test('getWorkflowAuthStatus treats an unreachable saved Kari hub as logged out', async () => {
  const status = await getWorkflowAuthStatus({
    probe: async url => {
      assert.equal(url, 'http://127.0.0.1:8900')
      return false
    },
    readFile: () =>
      JSON.stringify({
        kari: {
          token: 'workspace-token',
          cloudBaseURL: 'http://127.0.0.1:8900/'
        }
      }),
    secretsPath: '/tmp/hermes/workflow-secrets.json'
  })

  assert.equal(status.loggedIn, false)
  assert.equal(status.cloudBaseUrl, 'http://127.0.0.1:8900')
  assert.equal(status.cloudReachable, false)
  assert.match(status.error, /Kari hub is not reachable/)
})

test('buildAugmentedPath prepends dev tool dirs so a GUI minimal PATH can find uv', () => {
  const home = '/Users/dev'
  const existingDirs = new Set([`${home}/.local/bin`, `${home}/.cargo/bin`, '/opt/homebrew/bin', '/usr/local/bin'])
  const augmented = buildAugmentedPath({
    env: { HOME: home, PATH: '/usr/bin:/bin' },
    exists: dir => existingDirs.has(dir),
    platform: 'darwin'
  })

  const parts = augmented.split(':')

  assert.ok(parts.includes(`${home}/.local/bin`), 'expected uv dir on PATH')
  assert.ok(parts.includes('/opt/homebrew/bin'), 'expected homebrew dir on PATH')
  assert.ok(parts.indexOf(`${home}/.local/bin`) < parts.indexOf('/usr/bin'), 'dev dirs should come first')
  assert.equal(parts.indexOf('/usr/bin'), parts.lastIndexOf('/usr/bin'), 'PATH entries should be de-duplicated')
})

test('buildLangflowLaunch augments PATH with dev tool dirs for the spawned backend', () => {
  const launch = buildLangflowLaunch({
    env: { HOME: '/Users/dev', PATH: '/usr/bin:/bin' },
    exists: dir => dir === '/Users/dev/.local/bin' || dir.startsWith('/Volumes'),
    root: '/Volumes/langflow'
  })

  assert.ok(launch.options.env.PATH.split(':').includes('/Users/dev/.local/bin'))
})

test('buildAugmentedPath leaves Windows PATH untouched', () => {
  const augmented = buildAugmentedPath({
    env: { Path: 'C:\\Windows\\System32' },
    exists: () => true,
    platform: 'win32'
  })

  assert.equal(augmented, 'C:\\Windows\\System32')
})

test('buildLangflowLaunch uses direct uv run when the frontend is already built', () => {
  const root = '/workspace/langflow'
  const launch = buildLangflowLaunch({
    configDir: '/tmp/hermes/langflow',
    exists: filePath => filePath === `${root}/src/backend/base/langflow/frontend/index.html`,
    root,
    host: '127.0.0.1',
    port: 7860
  })

  assert.equal(launch.command, 'uv')
  assert.deepEqual(launch.args, [
    'run',
    'langflow',
    'run',
    '--frontend-path',
    'src/backend/base/langflow/frontend',
    '--log-level',
    'debug',
    '--host',
    '127.0.0.1',
    '--port',
    '7860',
    '--env-file',
    '.env',
    '--no-open-browser'
  ])
  assert.equal(launch.options.cwd, root)
  assert.equal(launch.options.detached, true)
  assert.equal(launch.options.env.LANGFLOW_CONFIG_DIR, '/tmp/hermes/langflow')
})

test('resolveLangflowRoot avoids a personal absolute default and picks configured or valid candidates', () => {
  const valid = '/repo/sibling/langflow'
  const invalid = '/repo/missing/langflow'
  const exists = path => path === `${valid}/Makefile` || path === `${valid}/pyproject.toml`

  assert.equal(
    resolveLangflowRoot({
      candidates: [invalid, valid],
      env: {},
      exists
    }),
    valid
  )
  assert.equal(
    resolveLangflowRoot({
      candidates: [valid],
      env: { HERMES_DESKTOP_LANGFLOW_ROOT: '/custom/langflow' },
      exists: () => false
    }),
    '/custom/langflow'
  )
  assert.equal(resolveLangflowRoot({ candidates: [], env: {}, exists: () => false }), '')
})

test('killProcessTree terminates the process group on POSIX', () => {
  const child = fakeChild()
  const killed = []

  killProcessTree(child, {
    kill: (pid, signal) => killed.push({ pid, signal }),
    platform: 'darwin'
  })

  assert.deepEqual(killed, [{ pid: -child.pid, signal: 'SIGTERM' }])
})

test('killProcessTree uses taskkill for Windows process trees', () => {
  const child = fakeChild()
  const calls = []

  killProcessTree(child, {
    execFileSync: (command, args, options) => calls.push({ args, command, options }),
    platform: 'win32'
  })

  assert.equal(calls[0].command, 'taskkill')
  assert.deepEqual(calls[0].args, ['/PID', String(child.pid), '/T', '/F'])
  assert.equal(calls[0].options.windowsHide, true)
})

test('workflow backend start spawns Langflow once and reuses an in-flight process', async () => {
  const child = fakeChild()
  const spawnCalls = []
  const manager = createWorkflowBackendManager({
    exists: () => true,
    killTree: target => {
      target.kill('SIGTERM')
    },
    pollIntervalMs: 10,
    probe: async () => false,
    readyTimeoutMs: 100,
    root: '/tmp/langflow',
    spawn: (command, args, options) => {
      spawnCalls.push({ args, command, options })
      return child
    }
  })

  const first = await manager.start()
  const second = await manager.start()

  assert.equal(spawnCalls.length, 1)
  assert.equal(first.state, 'starting')
  assert.equal(second.state, 'starting')
  assert.equal(second.pid, child.pid)

  await manager.stop()
})

test('workflow backend stop waits for the process to exit before resolving', async () => {
  const child = fakeChild()
  // Simulate a real process that does not die the instant SIGTERM is sent.
  child.kill = () => {
    child.killed = true
    return true
  }

  const manager = createWorkflowBackendManager({
    exists: () => true,
    killTree: target => {
      target.kill('SIGTERM')
    },
    pollIntervalMs: 10,
    probe: async () => false,
    readyTimeoutMs: 100,
    root: '/tmp/langflow',
    spawn: () => child,
    stopTimeoutMs: 1_000
  })

  await manager.start()

  let resolved = false
  const stopping = manager.stop().then(() => {
    resolved = true
  })

  // stop() must not resolve while the old process is still alive — otherwise a
  // restart would attach to the still-listening port instead of respawning.
  await new Promise(resolve => setTimeout(resolve, 20))
  assert.equal(resolved, false)

  child.emit('exit', null, 'SIGTERM')
  await stopping
  assert.equal(resolved, true)
})

test('workflow backend falls back to make run_cli when direct uv run exits before ready', async () => {
  const root = '/tmp/langflow'
  const directChild = fakeChild(1111)
  const makeChild = fakeChild(2222)
  const spawnCalls = []
  const exists = filePath =>
    [
      `${root}/Makefile`,
      `${root}/pyproject.toml`,
      `${root}/src/backend/base/langflow/frontend/index.html`
    ].includes(filePath)
  const manager = createWorkflowBackendManager({
    exists,
    killTree: target => {
      target.kill('SIGTERM')
    },
    pollIntervalMs: 10,
    probe: async () => false,
    readyTimeoutMs: 250,
    root,
    spawn: (command, args, options) => {
      spawnCalls.push({ args, command, options })
      return spawnCalls.length === 1 ? directChild : makeChild
    }
  })

  const starting = await manager.start()

  assert.equal(starting.state, 'starting')
  assert.equal(starting.pid, directChild.pid)
  assert.equal(spawnCalls[0].command, 'uv')

  directChild.emit('exit', 1, null)

  await eventually(async () => {
    const status = await manager.status()
    assert.equal(spawnCalls.length, 2)
    assert.equal(spawnCalls[1].command, 'make')
    assert.equal(status.state, 'starting')
    assert.equal(status.pid, makeChild.pid)
  })

  await manager.stop()
})

test('workflow backend start reuses an already reachable Langflow server', async () => {
  const manager = createWorkflowBackendManager({
    exists: () => true,
    killTree: target => {
      target.kill('SIGTERM')
    },
    probe: async () => true,
    root: '/tmp/langflow',
    spawn: () => {
      throw new Error('must not spawn when Langflow is already reachable')
    }
  })

  const status = await manager.start()

  assert.equal(status.state, 'ready')
  assert.equal(status.external, true)
  assert.equal(status.pid, null)
})

test('workflow backend transitions to ready when the spawned server answers', async () => {
  const child = fakeChild()
  let probes = 0
  const manager = createWorkflowBackendManager({
    exists: () => true,
    killTree: target => {
      target.kill('SIGTERM')
    },
    pollIntervalMs: 10,
    probe: async () => {
      probes += 1
      return probes >= 2
    },
    readyTimeoutMs: 250,
    root: '/tmp/langflow',
    spawn: () => child
  })

  const starting = await manager.start()

  assert.equal(starting.state, 'starting')

  await eventually(async () => {
    const status = await manager.status()
    assert.equal(status.state, 'ready')
    assert.equal(status.pid, child.pid)
  })

  await manager.stop()
})

test('workflow backend retry reuses a live child after a ready timeout', async () => {
  const child = fakeChild()
  const spawnCalls = []
  const manager = createWorkflowBackendManager({
    exists: () => true,
    pollIntervalMs: 5,
    probe: async () => false,
    readyTimeoutMs: 5,
    root: '/tmp/langflow',
    spawn: () => {
      spawnCalls.push(child)
      return child
    }
  })

  await manager.start()

  await eventually(async () => {
    const status = await manager.status()
    assert.equal(status.state, 'error')
  })

  const retry = await manager.start()

  assert.equal(spawnCalls.length, 1)
  assert.equal(retry.state, 'starting')
  assert.equal(retry.pid, child.pid)

  await manager.stop()
})

test('workflow backend stop ignores a delayed exit from the killed child', async () => {
  const child = fakeChild()
  child.kill = signal => {
    child.killed = true
    child.signal = signal
    return true
  }
  const manager = createWorkflowBackendManager({
    exists: () => true,
    killTree: target => {
      target.kill('SIGTERM')
    },
    pollIntervalMs: 10,
    probe: async () => false,
    readyTimeoutMs: 100,
    root: '/tmp/langflow',
    spawn: () => child,
    // The child never exits during stop(); fall through the bounded wait fast
    // so this exercises the "exit arrives after stop() resolved" path.
    stopTimeoutMs: 50
  })

  await manager.start()
  await manager.stop()
  child.emit('exit', null, 'SIGTERM')

  const status = await manager.status()

  assert.equal(status.state, 'stopped')
  assert.equal(status.error, null)
})
