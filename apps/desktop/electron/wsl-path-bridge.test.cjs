const assert = require('node:assert/strict')
const test = require('node:test')

const {
  resolveLocalReadPath,
  resolvePickerDefaultPath,
  resolvePickerResultForRemoteBackend,
  windowsPathToWslMount,
  wslPosixToWindowsAccessible,
  wslUncToPosix
} = require('./wsl-path-bridge.cjs')

test('windowsPathToWslMount converts drive paths', () => {
  assert.equal(windowsPathToWslMount('C:\\Users\\don\\projects'), '/mnt/c/Users/don/projects')
  assert.equal(windowsPathToWslMount('D:/work/project'), '/mnt/d/work/project')
  assert.equal(windowsPathToWslMount('/home/user'), null)
})

test('wslUncToPosix converts WSL UNC paths', () => {
  assert.equal(
    wslUncToPosix('\\\\wsl.localhost\\Ubuntu\\home\\don\\projects'),
    '/home/don/projects'
  )
  assert.equal(wslUncToPosix('\\\\wsl$\\Ubuntu\\home\\don\\projects'), '/home/don/projects')
})

test('wslPosixToWindowsAccessible maps mount and home paths', () => {
  assert.equal(
    wslPosixToWindowsAccessible('/mnt/c/Users/don/projects', 'Ubuntu'),
    'C:\\Users\\don\\projects'
  )
  assert.equal(
    wslPosixToWindowsAccessible('/home/don/projects', 'Ubuntu'),
    '\\\\wsl.localhost\\Ubuntu\\home\\don\\projects'
  )
})

test('resolvePickerDefaultPath exposes WSL paths in the Windows picker', () => {
  assert.equal(
    resolvePickerDefaultPath('/home/don/projects', 'Ubuntu'),
    '\\\\wsl.localhost\\Ubuntu\\home\\don\\projects'
  )
})

test('resolvePickerResultForRemoteBackend normalizes picker output for WSL backends', () => {
  assert.equal(
    resolvePickerResultForRemoteBackend('C:\\Users\\don\\projects'),
    '/mnt/c/Users/don/projects'
  )
  assert.equal(
    resolvePickerResultForRemoteBackend('\\\\wsl.localhost\\Ubuntu\\home\\don\\projects'),
    '/home/don/projects'
  )
})

test('resolveLocalReadPath maps WSL cwd values for Windows readDir', () => {
  const original = process.platform
  Object.defineProperty(process, 'platform', { value: 'win32' })

  try {
    assert.equal(
      resolveLocalReadPath('/home/don/projects', 'Ubuntu'),
      '\\\\wsl.localhost\\Ubuntu\\home\\don\\projects'
    )
    assert.equal(resolveLocalReadPath('/mnt/c/Users/don', 'Ubuntu'), 'C:\\Users\\don')
  } finally {
    Object.defineProperty(process, 'platform', { value: original })
  }
})