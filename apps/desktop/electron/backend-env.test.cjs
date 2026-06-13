const test = require('node:test')
const assert = require('node:assert/strict')
const path = require('node:path')

const {
  POSIX_SANE_PATH_ENTRIES,
  appendUniquePathEntries,
  buildDesktopBackendEnv,
  buildDesktopBackendPath,
  pathEnvKey
} = require('./backend-env.cjs')

test('desktop backend PATH adds Hermes-managed bins and missing POSIX sane entries', () => {
  const result = buildDesktopBackendPath({
    hermesHome: '/Users/test/.hermes',
    venvRoot: '/Users/test/.hermes/hermes-agent/venv',
    currentPath: '/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin',
    platform: 'darwin',
    pathModule: path.posix
  })

  const entries = result.split(':')
  assert.equal(entries[0], '/Users/test/.hermes/node/bin')
  assert.equal(entries[1], '/Users/test/.hermes/hermes-agent/venv/bin')
  assert.ok(entries.includes('/opt/homebrew/bin'), 'Apple Silicon Homebrew bin is added')
  assert.ok(entries.includes('/opt/homebrew/sbin'), 'Apple Silicon Homebrew sbin is added')
  assert.ok(entries.includes('/usr/local/sbin'), 'missing standard sbin is added')

  for (const expected of POSIX_SANE_PATH_ENTRIES) {
    assert.ok(entries.includes(expected), `${expected} should be present`)
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
  assert.ok(env.PATH.startsWith('/Users/test/.hermes/node/bin:/Users/test/.hermes/hermes-agent/venv/bin:'))
  assert.ok(env.PATH.includes('/opt/homebrew/bin'))
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
  assert.ok(env.Path.startsWith('C:\\Users\\test\\AppData\\Local\\hermes\node\\bin;'))
  assert.ok(env.Path.includes('\\venv\\Scripts;'))
  assert.ok(env.Path.includes(';C:\\Windows\\System32;C:\\Windows'))
  assert.equal(env.Path.includes('/opt/homebrew/bin'), false)
})

test('appendUniquePathEntries drops empty entries and keeps first occurrence', () => {
  assert.equal(
    appendUniquePathEntries([':/a::/b', ['/a', '/c']], { delimiter: ':' }),
    '/a:/b:/c'
  )
})

const fs = require('node:fs')
const os = require('node:os')

const {
  readDotenvFile
} = require('./backend-env.cjs')

function withTempEnv(content, fn) {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-dotenv-test-'))
  const envPath = path.join(tmpDir, '.env')
  fs.writeFileSync(envPath, content, 'utf8')
  try {
    fn(tmpDir)
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true })
  }
}

test('readDotenvFile returns empty object when .env is missing', () => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-dotenv-test-'))
  try {
    const result = readDotenvFile(tmpDir)
    assert.deepEqual(result, {})
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true })
  }
})

test('readDotenvFile parses KEY=VALUE pairs from .env', () => {
  withTempEnv(
    "OPENROUTER_API_KEY=***\nHERMES_HOME=/custom/path",
    (dir) => {
      const result = readDotenvFile(dir)
      assert.equal(result.OPENROUTER_API_KEY, '***')
      assert.equal(result.HERMES_HOME, '/custom/path')
    }
  )
})

test('readDotenvFile strips matching quotes around values', () => {
  withTempEnv(
    'SECRET="double-quoted"\nTOKEN=\'single-quoted\'\nPLAIN=unquoted',
    (dir) => {
      const result = readDotenvFile(dir)
      assert.equal(result.SECRET, 'double-quoted')
      assert.equal(result.TOKEN, 'single-quoted')
      assert.equal(result.PLAIN, 'unquoted')
    }
  )
})

test('readDotenvFile skips comments and blank lines', () => {
  withTempEnv(
    '# This is a comment\n\nKEY=value\n# Another comment\n\nFOO=bar',
    (dir) => {
      const result = readDotenvFile(dir)
      assert.equal(result.KEY, 'value')
      assert.equal(result.FOO, 'bar')
      assert.equal(Object.keys(result).length, 2)
    }
  )
})

test('readDotenvFile handles Windows-style line endings (CRLF)', () => {
  withTempEnv(
    'KEY=value\r\nFOO=bar\r\n',
    (dir) => {
      const result = readDotenvFile(dir)
      assert.equal(result.KEY, 'value')
      assert.equal(result.FOO, 'bar')
    }
  )
})

test('readDotenvFile skips lines without = sign', () => {
  withTempEnv(
    'KEY=value\nNOEQUALS\nANOTHER=42',
    (dir) => {
      const result = readDotenvFile(dir)
      assert.equal(result.KEY, 'value')
      assert.equal(result.ANOTHER, '42')
      assert.ok(!('NOEQUALS' in result))
    }
  )
})

test('readDotenvFile trims leading/trailing whitespace from keys', () => {
  withTempEnv(
    '  KEY  =  spaced-value  \n\nTIGHT=exact',
    (dir) => {
      const result = readDotenvFile(dir)
      assert.equal(result.KEY, 'spaced-value')
      assert.equal(result.TIGHT, 'exact')
    }
  )
})
