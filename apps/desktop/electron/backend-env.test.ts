import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import {
  appendUniquePathEntries,
  buildDesktopBackendEnv,
  buildDesktopBackendPath,
  hermesManagedNodePathEntries,
  normalizeHermesHomeRoot,
  pathEnvKey,
  POSIX_SANE_PATH_ENTRIES,
  profileBackendParentEnv
} from './backend-env'

test('desktop backend PATH adds Hermes-managed bins and missing POSIX sane entries', () => {
  const result = buildDesktopBackendPath({
    hermesHome: '/Users/test/.hermes',
    venvRoot: '/Users/test/.hermes/hermes-agent/venv',
    currentPath: '/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin',
    platform: 'darwin',
    pathModule: path.posix
  })

  const entries = result.split(':')
  // Both managed-Node layouts lead, POSIX-native shape first, then the venv.
  assert.deepEqual(entries.slice(0, 3), [
    '/Users/test/.hermes/node/bin',
    '/Users/test/.hermes/node',
    '/Users/test/.hermes/hermes-agent/venv/bin'
  ])
  assert.ok(entries.includes('/opt/homebrew/bin'), 'Apple Silicon Homebrew bin is added')
  assert.ok(entries.includes('/opt/homebrew/sbin'), 'Apple Silicon Homebrew sbin is added')
  assert.ok(entries.includes('/usr/local/sbin'), 'missing standard sbin is added')

  for (const expected of POSIX_SANE_PATH_ENTRIES) {
    assert.ok(entries.includes(expected), `${expected} should be present`)
  }
})

test('managed Node dirs lead with the platform-native layout but always offer both', () => {
  const posix = hermesManagedNodePathEntries('/Users/test/.hermes', {
    platform: 'darwin',
    pathModule: path.posix
  })

  const windows = hermesManagedNodePathEntries('C:\\Users\\test\\AppData\\Local\\hermes', {
    platform: 'win32',
    pathModule: path.win32
  })

  // install.sh uses node/bin; install.ps1 unpacks node.exe into node\ itself.
  // Both shapes are always emitted so migrated installs keep resolving.
  assert.deepEqual(posix, ['/Users/test/.hermes/node/bin', '/Users/test/.hermes/node'])
  assert.deepEqual(windows, [
    'C:\\Users\\test\\AppData\\Local\\hermes\\node',
    'C:\\Users\\test\\AppData\\Local\\hermes\\node\\bin'
  ])
})

test('managed Node dirs are empty without a Hermes home', () => {
  assert.deepEqual(hermesManagedNodePathEntries(undefined, { platform: 'darwin', pathModule: path.posix }), [])
  assert.deepEqual(hermesManagedNodePathEntries('', { platform: 'win32', pathModule: path.win32 }), [])
})

test('every managed Node dir outranks the inherited PATH on both platforms', () => {
  for (const [platform, pathModule, home, inherited, delimiter] of [
    ['darwin', path.posix, '/Users/test/.hermes', '/usr/local/bin:/usr/bin', ':'],
    ['win32', path.win32, 'C:\\hermes', 'C:\\Program Files\\nodejs;C:\\Windows\\System32', ';']
  ] as const) {
    const entries = buildDesktopBackendPath({
      hermesHome: home,
      venvRoot: null,
      currentPath: inherited,
      platform,
      pathModule
    }).split(delimiter)

    const managed = hermesManagedNodePathEntries(home, { platform, pathModule })
    const firstInherited = Math.min(...inherited.split(delimiter).map(entry => entries.indexOf(entry)))

    for (const dir of managed) {
      assert.ok(
        entries.indexOf(dir) >= 0 && entries.indexOf(dir) < firstInherited,
        `${dir} must precede the inherited PATH on ${platform}`
      )
    }
  }
})

test('desktop backend PATH preserves first occurrence and avoids duplicates', () => {
  const result = buildDesktopBackendPath({
    hermesHome: '/Users/test/.hermes',
    venvRoot: '/Users/test/.hermes/hermes-agent/venv',
    currentPath: '/opt/homebrew/bin:/usr/bin:/opt/homebrew/bin:/bin',
    platform: 'darwin',
    pathModule: path.posix
  })

  const entries = result.split(':')
  assert.equal(entries.filter(entry => entry === '/opt/homebrew/bin').length, 1)
  assert.ok(
    entries.indexOf('/opt/homebrew/bin') < entries.indexOf('/opt/homebrew/sbin'),
    'existing Homebrew bin keeps its precedence over appended missing sane entries'
  )
})

test('buildDesktopBackendEnv extends PYTHONPATH and backend PATH together', () => {
  const env = buildDesktopBackendEnv({
    hermesHome: '/Users/test/.hermes',
    pythonPathEntries: ['/repo/hermes-agent'],
    venvRoot: '/Users/test/.hermes/hermes-agent/venv',
    currentEnv: {
      PATH: '/usr/bin:/bin',
      PYTHONPATH: '/existing/pythonpath'
    },
    platform: 'darwin',
    pathModule: path.posix
  })

  assert.equal(env.PYTHONPATH, '/repo/hermes-agent:/existing/pythonpath')
  assert.ok(
    env.PATH.startsWith(
      '/Users/test/.hermes/node/bin:/Users/test/.hermes/node:/Users/test/.hermes/hermes-agent/venv/bin:'
    )
  )
  assert.ok(env.PATH.includes('/opt/homebrew/bin'))
})

test('buildDesktopBackendEnv forces PYTHONUTF8 unless the user set it explicitly', () => {
  const defaulted = buildDesktopBackendEnv({
    hermesHome: '/Users/test/.hermes',
    currentEnv: { PATH: '/usr/bin' },
    platform: 'darwin',
    pathModule: path.posix
  })

  assert.equal(defaulted.PYTHONUTF8, '1')

  const optedOut = buildDesktopBackendEnv({
    hermesHome: '/Users/test/.hermes',
    currentEnv: { PATH: '/usr/bin', PYTHONUTF8: '0' },
    platform: 'darwin',
    pathModule: path.posix
  })

  assert.equal(optedOut.PYTHONUTF8, '0')
})

test('normalizeHermesHomeRoot expands a literal leading ~ against the home directory, not cwd', () => {
  assert.equal(
    normalizeHermesHomeRoot('~/.hermes', { pathModule: path.posix, homedir: '/Users/test' }),
    '/Users/test/.hermes'
  )
  assert.equal(
    normalizeHermesHomeRoot('~/.hermes/profiles/oracle', { pathModule: path.posix, homedir: '/Users/test' }),
    '/Users/test/.hermes'
  )
  assert.equal(
    normalizeHermesHomeRoot('~\\.hermes', { pathModule: path.win32, homedir: 'C:\\Users\\test' }),
    'C:\\Users\\test\\.hermes'
  )
  assert.equal(normalizeHermesHomeRoot('~', { pathModule: path.posix, homedir: '/Users/test' }), '/Users/test')
})

test('normalizeHermesHomeRoot maps profile homes back to the global Hermes root', () => {
  assert.equal(
    normalizeHermesHomeRoot('/Users/test/.hermes/profiles/oracle', { pathModule: path.posix }),
    '/Users/test/.hermes'
  )
  assert.equal(
    normalizeHermesHomeRoot('C:\\Users\\test\\AppData\\Local\\hermes\\profiles\\oracle', { pathModule: path.win32 }),
    'C:\\Users\\test\\AppData\\Local\\hermes'
  )
  assert.equal(normalizeHermesHomeRoot('/Users/test/.hermes', { pathModule: path.posix }), '/Users/test/.hermes')
})

test('Windows PATH casing and delimiter are preserved without POSIX sane entries', () => {
  const env = buildDesktopBackendEnv({
    hermesHome: 'C:\\Users\\test\\AppData\\Local\\hermes',
    pythonPathEntries: ['C:\\repo\\hermes-agent'],
    venvRoot: 'C:\\Users\\test\\AppData\\Local\\hermes\\hermes-agent\\venv',
    currentEnv: {
      Path: 'C:\\Windows\\System32;C:\\Windows',
      PYTHONPATH: 'C:\\existing\\pythonpath'
    },
    platform: 'win32',
    pathModule: path.win32
  })

  assert.equal(pathEnvKey({ Path: 'x' }, 'win32'), 'Path')
  assert.equal(env.PATH, undefined)
  // Windows leads with the portable layout (install.ps1 unpacks node.exe
  // straight into node\, no bin\), then the POSIX shape for migrated installs.
  assert.ok(
    env.Path.startsWith(
      'C:\\Users\\test\\AppData\\Local\\hermes\\node;C:\\Users\\test\\AppData\\Local\\hermes\\node\\bin;'
    )
  )
  assert.ok(env.Path.includes('\\venv\\Scripts;'))
  assert.ok(env.Path.includes(';C:\\Windows\\System32;C:\\Windows'))
  assert.equal(env.Path.includes('/opt/homebrew/bin'), false)
})

test('appendUniquePathEntries drops empty entries and keeps first occurrence', () => {
  assert.equal(appendUniquePathEntries([':/a::/b', ['/a', '/c']], { delimiter: ':' }), '/a:/b:/c')
})

// `hermes desktop` loads its launch profile's .env/.op.env into os.environ and
// hands that env to Electron; these cover what a profile backend inherits (#68367).
function withHermesRoot(files: Record<string, string>, run: (root: string) => void) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-profile-env-'))

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
  withHermesRoot(ROOT_SCOPE_FILES, root => {
    const env = profileBackendParentEnv({
      hermesHome: root,
      profile: 'urbot',
      currentEnv: ROOT_LAUNCHED_ENV,
      platform: 'linux'
    })

    // Shell exports the root dotenv never declared, and OS names, still reach the child.
    assert.deepEqual(env, { HOME: '/Users/test', PATH: '/usr/bin:/bin', OPENROUTER_API_KEY: 'shell-key' })
  })
})

test('the launch profile backend inherits the Desktop env unchanged', () => {
  withHermesRoot(ROOT_SCOPE_FILES, root => {
    for (const profile of ['default', null, undefined]) {
      assert.deepEqual(
        profileBackendParentEnv({ hermesHome: root, profile, currentEnv: ROOT_LAUNCHED_ENV, platform: 'linux' }),
        ROOT_LAUNCHED_ENV
      )
    }
  })
})

test('a primary backend without an explicit profile follows the sticky active_profile', () => {
  withHermesRoot({ ...ROOT_SCOPE_FILES, active_profile: 'urbot\n' }, root => {
    const env = profileBackendParentEnv({ hermesHome: root, profile: null, currentEnv: ROOT_LAUNCHED_ENV })

    assert.equal(env.TLON_SHIP_CODE, undefined)
    assert.equal(env.OP_SERVICE_ACCOUNT_TOKEN, undefined)
    assert.equal(env.OPENROUTER_API_KEY, 'shell-key')
  })
})

test('Desktop launched from a named profile keeps that profile out of the default backend', () => {
  withHermesRoot(
    {
      '.env': 'OPENAI_API_KEY=root-key\n',
      'profiles/work/.env': 'TLON_SHIP_CODE=work-code\nOP_SERVICE_ACCOUNT_TOKEN=work-op\n'
    },
    root => {
      const currentEnv = {
        HERMES_HOME: path.join(root, 'profiles', 'work'),
        TLON_SHIP_CODE: 'work-code',
        OP_SERVICE_ACCOUNT_TOKEN: 'work-op',
        OPENAI_API_KEY: 'shell-key'
      }

      assert.deepEqual(profileBackendParentEnv({ hermesHome: root, profile: 'default', currentEnv }), {
        HERMES_HOME: currentEnv.HERMES_HOME,
        OPENAI_API_KEY: 'shell-key'
      })
      assert.deepEqual(profileBackendParentEnv({ hermesHome: root, profile: 'work', currentEnv }), currentEnv)
    }
  )
})

test('Windows matches profile homes and dotenv names case-insensitively', () => {
  const root = 'C:\\Users\\test\\AppData\\Local\\hermes'
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
    HERMES_HOME: 'c:\\users\\test\\appdata\\local\\HERMES',
    Path: 'C:\\Windows',
    Telegram_Bot_Token: 'root-token'
  }

  const scoped = (profile: string) =>
    profileBackendParentEnv({ hermesHome: root, profile, currentEnv, platform: 'win32', fsModule })

  assert.deepEqual(scoped('default'), currentEnv)
  assert.deepEqual(scoped('urbot'), { HERMES_HOME: currentEnv.HERMES_HOME, Path: 'C:\\Windows' })
})
