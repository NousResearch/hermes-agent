import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import {
  appendUniquePathEntries,
  buildDesktopBackendEnv,
  normalizeRabbitHomeRoot,
  pathEnvKey,
  POSIX_SANE_PATH_ENTRIES,
  profileBackendParentEnv
} from './backend-env'
import { applyLoginShellPath } from './shell-path'

test('backend env scrubs PYTHONPATH and PYTHONHOME', () => {
  const env = buildDesktopBackendEnv({
    currentEnv: {
      PATH: '/usr/bin:/bin',
      PYTHONPATH: '/leaked/other/checkout',
      PYTHONHOME: '/leaked/python'
    },
    platform: 'darwin'
  })

  assert.equal(env.PYTHONPATH, '')
  assert.equal(env.PYTHONHOME, '')
})

test('POSIX backend PATH keeps the inherited PATH first and appends missing sane entries', () => {
  const env = buildDesktopBackendEnv({
    currentEnv: { PATH: '/opt/homebrew/bin:/usr/bin:/bin' },
    platform: 'darwin'
  })

  const entries = env.PATH.split(':')
  assert.equal(entries[0], '/opt/homebrew/bin', 'inherited PATH keeps precedence')
  assert.equal(entries.filter(entry => entry === '/opt/homebrew/bin').length, 1, 'no duplicates')

  for (const expected of POSIX_SANE_PATH_ENTRIES) {
    assert.ok(entries.includes(expected), `${expected} should be present`)
  }
})

test('backend runs the store toolchain even after the login-shell PATH is merged in front of it', async () => {
  // `rabbit desktop` hands Electron a PATH with the PM store first; the
  // login-shell merge then puts nvm/Homebrew ahead of it in process.env.
  const env: Record<string, string> = {
    PATH: '/Users/u/.rabbit/tools/node-26.7.0-darwin-arm64/bin:/Users/u/.rabbit/tools/uv-0.12.3-darwin-arm64:/usr/bin:/bin'
  }

  const loginPath = '/Users/u/.nvm/versions/node/v20.0.0/bin:/opt/homebrew/bin:/usr/bin'

  const execFileFn = (_file, _args, _options, callback) => {
    queueMicrotask(() => callback(null, `__RABBIT_LOGIN_PATH_START__${loginPath}__RABBIT_LOGIN_PATH_END__`, ''))

    return { stdin: { end() {} } }
  }

  await applyLoginShellPath({ env, platform: 'darwin', execFileFn })
  assert.equal(env.PATH.split(':')[0], '/Users/u/.nvm/versions/node/v20.0.0/bin', 'user-facing env keeps login order')

  const backend = buildDesktopBackendEnv({ currentEnv: env, platform: 'darwin', homedir: '/Users/u' })

  assert.deepEqual(backend.PATH.split(':').slice(0, 5), [
    '/Users/u/.rabbit/tools/node-26.7.0-darwin-arm64/bin',
    '/Users/u/.rabbit/tools/uv-0.12.3-darwin-arm64',
    '/Users/u/.nvm/versions/node/v20.0.0/bin',
    '/opt/homebrew/bin',
    '/usr/bin'
  ])
})

test('RABBIT_RUNTIME_DIR names the store; look-alike prefixes are not Rabbit-owned', () => {
  const store = '/Applications/Rabbit.app/Contents/Resources/agent-payload/tools'

  const backend = buildDesktopBackendEnv({
    currentEnv: {
      RABBIT_RUNTIME_DIR: store,
      PATH: `/opt/homebrew/bin:/Users/u/.rabbit/tools-old/bin:${store}/npm-12.0.2-darwin-arm64/bin:/usr/bin`
    },
    platform: 'darwin',
    homedir: '/Users/u'
  })

  assert.deepEqual(backend.PATH.split(':').slice(0, 4), [
    `${store}/npm-12.0.2-darwin-arm64/bin`,
    '/opt/homebrew/bin',
    '/Users/u/.rabbit/tools-old/bin',
    '/usr/bin'
  ])
})

test('Windows PATH casing and delimiter are preserved without POSIX sane entries', () => {
  const env = buildDesktopBackendEnv({
    currentEnv: { Path: 'C:\\Windows\\System32;C:\\Windows' },
    platform: 'win32'
  })

  assert.equal(env.Path, 'C:\\Windows\\System32;C:\\Windows')
  assert.equal(env.PATH, undefined)
})

test('buildDesktopBackendEnv forces PYTHONUTF8 unless the user set it explicitly', () => {
  const defaulted = buildDesktopBackendEnv({
    currentEnv: { PATH: '/usr/bin' },
    platform: 'darwin'
  })

  assert.equal(defaulted.PYTHONUTF8, '1')

  const optedOut = buildDesktopBackendEnv({
    currentEnv: { PATH: '/usr/bin', PYTHONUTF8: '0' },
    platform: 'darwin'
  })

  assert.equal(optedOut.PYTHONUTF8, '0')
})

test('normalizeRabbitHomeRoot expands a literal leading ~ against the home directory, not cwd', () => {
  assert.equal(
    normalizeRabbitHomeRoot('~/.rabbit', { pathModule: path.posix, homedir: '/Users/test' }),
    '/Users/test/.rabbit'
  )
  assert.equal(
    normalizeRabbitHomeRoot('~/.rabbit/profiles/oracle', { pathModule: path.posix, homedir: '/Users/test' }),
    '/Users/test/.rabbit'
  )
  assert.equal(
    normalizeRabbitHomeRoot('~\\.rabbit', { pathModule: path.win32, homedir: 'C:\\Users\\test' }),
    'C:\\Users\\test\\.rabbit'
  )
  assert.equal(normalizeRabbitHomeRoot('~', { pathModule: path.posix, homedir: '/Users/test' }), '/Users/test')
})

test('normalizeRabbitHomeRoot maps profile homes back to the global Rabbit root', () => {
  assert.equal(
    normalizeRabbitHomeRoot('/Users/test/.rabbit/profiles/oracle', { pathModule: path.posix }),
    '/Users/test/.rabbit'
  )
  assert.equal(
    normalizeRabbitHomeRoot('C:\\Users\\test\\AppData\\Local\\rabbit\\profiles\\oracle', { pathModule: path.win32 }),
    'C:\\Users\\test\\AppData\\Local\\rabbit'
  )
  assert.equal(normalizeRabbitHomeRoot('/Users/test/.rabbit', { pathModule: path.posix }), '/Users/test/.rabbit')
})

test('pathEnvKey finds the platform-cased PATH key', () => {
  assert.equal(pathEnvKey({ Path: 'x' }, 'win32'), 'Path')
  assert.equal(pathEnvKey({ PATH: 'x' }, 'win32'), 'PATH')
  assert.equal(pathEnvKey({}, 'win32'), 'PATH')
  assert.equal(pathEnvKey({ Path: 'x' }, 'darwin'), 'PATH')
})

test('appendUniquePathEntries flattens, dedupes, and preserves first occurrence', () => {
  assert.equal(appendUniquePathEntries(['/a:/b', ['/b', '/c'], '', null], { delimiter: ':' }), '/a:/b:/c')
})

// `rabbit desktop` loads its launch profile's .env/.op.env into os.environ and
// hands that env to Electron; these cover what a profile backend inherits (#68367).
function withRabbitRoot(files: Record<string, string>, run: (root: string) => void) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'rabbit-profile-env-'))

  try {
    for (const [rel, contents] of Object.entries(files)) {
      fs.mkdirSync(path.dirname(path.join(root, rel)), { recursive: true })
      fs.writeFileSync(path.join(root, rel), contents)
    }

    run(root)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
}

const ROOT_SCOPE_FILES = {
  '.env': '\uFEFFTLON_SHIP_URL=https://moon.invalid\nexport TLON_SHIP_CODE = "root-code" # moon\nPATH=/root/bin\n',
  '.op.env': 'OP_SERVICE_ACCOUNT_TOKEN=root-op\n',
  'profiles/urbot/.env': 'ANTHROPIC_API_KEY=urbot-key\n'
}

const ROOT_LAUNCHED_ENV = {
  HOME: '/Users/test',
  PATH: '/usr/bin:/bin',
  TLON_SHIP_URL: 'https://moon.invalid',
  TLON_SHIP_CODE: 'root-code',
  OP_SERVICE_ACCOUNT_TOKEN: 'root-op',
  OPENROUTER_API_KEY: 'shell-key'
}

test('a named profile backend does not inherit secrets the root .env/.op.env loaded into Desktop', () => {
  withRabbitRoot(ROOT_SCOPE_FILES, root => {
    const env = profileBackendParentEnv({
      rabbitHome: root,
      profile: 'urbot',
      currentEnv: ROOT_LAUNCHED_ENV,
      platform: 'linux'
    })

    // Shell exports the root dotenv never declared, and OS names, still reach the child.
    assert.deepEqual(env, { HOME: '/Users/test', PATH: '/usr/bin:/bin', OPENROUTER_API_KEY: 'shell-key' })
  })
})

test('the launch profile backend inherits the Desktop env unchanged', () => {
  withRabbitRoot(ROOT_SCOPE_FILES, root => {
    for (const profile of ['default', null, undefined]) {
      assert.deepEqual(
        profileBackendParentEnv({ rabbitHome: root, profile, currentEnv: ROOT_LAUNCHED_ENV, platform: 'linux' }),
        ROOT_LAUNCHED_ENV
      )
    }
  })
})

test('a primary backend without an explicit profile follows the sticky active_profile', () => {
  withRabbitRoot({ ...ROOT_SCOPE_FILES, active_profile: 'urbot\n' }, root => {
    const env = profileBackendParentEnv({ rabbitHome: root, profile: null, currentEnv: ROOT_LAUNCHED_ENV })

    assert.equal(env.TLON_SHIP_CODE, undefined)
    assert.equal(env.OP_SERVICE_ACCOUNT_TOKEN, undefined)
    assert.equal(env.OPENROUTER_API_KEY, 'shell-key')
  })
})

test('Desktop launched from a named profile keeps that profile out of the default backend', () => {
  withRabbitRoot(
    {
      '.env': 'OPENAI_API_KEY=root-key\n',
      'profiles/work/.env': 'TLON_SHIP_CODE=work-code\nOP_SERVICE_ACCOUNT_TOKEN=work-op\n'
    },
    root => {
      const currentEnv = {
        RABBIT_HOME: path.join(root, 'profiles', 'work'),
        TLON_SHIP_CODE: 'work-code',
        OP_SERVICE_ACCOUNT_TOKEN: 'work-op',
        OPENAI_API_KEY: 'shell-key'
      }

      assert.deepEqual(profileBackendParentEnv({ rabbitHome: root, profile: 'default', currentEnv }), {
        RABBIT_HOME: currentEnv.RABBIT_HOME,
        OPENAI_API_KEY: 'shell-key'
      })
      assert.deepEqual(profileBackendParentEnv({ rabbitHome: root, profile: 'work', currentEnv }), currentEnv)
    }
  )
})

test('Windows matches profile homes and dotenv names case-insensitively', () => {
  const root = 'C:\\Users\\test\\AppData\\Local\\rabbit'
  const files = { [`${root}\\.env`]: 'TELEGRAM_BOT_TOKEN=root-token\r\n' }

  const fsModule = {
    readFileSync: file => {
      if (!(file in files)) {
        throw Object.assign(new Error('ENOENT'), { code: 'ENOENT' })
      }

      return files[file]
    }
  }

  const currentEnv = {
    RABBIT_HOME: 'c:\\users\\test\\appdata\\local\\RABBIT',
    Path: 'C:\\Windows',
    Telegram_Bot_Token: 'root-token'
  }

  const scoped = (profile: string) =>
    profileBackendParentEnv({ rabbitHome: root, profile, currentEnv, platform: 'win32', fsModule })

  assert.deepEqual(scoped('default'), currentEnv)
  assert.deepEqual(scoped('urbot'), { RABBIT_HOME: currentEnv.RABBIT_HOME, Path: 'C:\\Windows' })
})
