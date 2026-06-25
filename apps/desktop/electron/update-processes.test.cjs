const assert = require('node:assert/strict')
const test = require('node:test')

const {
  collectHermesRuntimeProcessIds,
  isHermesRuntimeProcess,
  killHermesRuntimeProcessesForUpdate,
  listWindowsProcesses,
  parseWindowsProcessList
} = require('./update-processes.cjs')

const ROOT = String.raw`G:\hermes\hermes-agent`
const POSIX_ROOT = '/Users/willie/hermes-agent'

test('targets a stray venv python gateway that would lock installed packages', () => {
  assert.equal(
    isHermesRuntimeProcess(
      {
        pid: 4242,
        name: 'pythonw.exe',
        executablePath: String.raw`G:\hermes\hermes-agent\venv\Scripts\pythonw.exe`,
        commandLine: String.raw`"G:\hermes\hermes-agent\venv\Scripts\pythonw.exe" -m hermes_cli.main gateway run`
      },
      { currentPid: 1, updateRoot: ROOT }
    ),
    true
  )
})

test('matches Hermes runtime processes with POSIX-shaped paths too', () => {
  assert.equal(
    isHermesRuntimeProcess(
      {
        pid: 5252,
        name: 'python',
        executablePath: '/Users/willie/hermes-agent/venv/bin/python',
        commandLine: '/Users/willie/hermes-agent/venv/bin/python -m hermes_cli.main gateway run'
      },
      { currentPid: 1, updateRoot: POSIX_ROOT }
    ),
    true
  )
})

test('targets any hermes_cli module command that references the target checkout', () => {
  assert.equal(
    isHermesRuntimeProcess(
      {
        pid: 5353,
        name: 'python.exe',
        executablePath: String.raw`C:\Python311\python.exe`,
        commandLine: String.raw`"C:\Python311\python.exe" -m hermes_cli.cron --root G:\hermes\hermes-agent`
      },
      { currentPid: 1, updateRoot: ROOT }
    ),
    true
  )
})

test('does not target unrelated python or an editor merely mentioning the checkout', () => {
  assert.equal(
    isHermesRuntimeProcess(
      {
        pid: 5001,
        name: 'python.exe',
        executablePath: String.raw`C:\Python311\python.exe`,
        commandLine: String.raw`"C:\Python311\python.exe" C:\other\script.py`
      },
      { currentPid: 1, updateRoot: ROOT }
    ),
    false
  )

  assert.equal(
    isHermesRuntimeProcess(
      {
        pid: 5002,
        name: 'Code.exe',
        executablePath: String.raw`C:\Users\willi\AppData\Local\Programs\Microsoft VS Code\Code.exe`,
        commandLine: String.raw`"Code.exe" "G:\hermes\hermes-agent\README.md"`
      },
      { currentPid: 1, updateRoot: ROOT }
    ),
    false
  )

  assert.equal(
    isHermesRuntimeProcess(
      {
        pid: 5003,
        name: 'node.exe',
        executablePath: String.raw`C:\Program Files\nodejs\node.exe`,
        commandLine: String.raw`"C:\Program Files\nodejs\node.exe" C:\tools\lint.js G:\hermes\hermes-agent`
      },
      { currentPid: 1, updateRoot: ROOT }
    ),
    false
  )
})

test('does not target a different Hermes checkout with a sibling path prefix', () => {
  assert.equal(
    isHermesRuntimeProcess(
      {
        pid: 5050,
        name: 'pythonw.exe',
        executablePath: String.raw`G:\hermes\hermes-agent-old\venv\Scripts\pythonw.exe`,
        commandLine: String.raw`"G:\hermes\hermes-agent-old\venv\Scripts\pythonw.exe" -m hermes_cli.main gateway run`
      },
      { currentPid: 1, updateRoot: ROOT }
    ),
    false
  )
})

test('collects only Hermes-owned runtime process ids and excludes this process', () => {
  const ids = collectHermesRuntimeProcessIds(
    [
      {
        pid: 111,
        name: 'pythonw.exe',
        executablePath: String.raw`G:\hermes\hermes-agent\venv\Scripts\pythonw.exe`,
        commandLine: String.raw`"G:\hermes\hermes-agent\venv\Scripts\pythonw.exe" -m hermes_cli.main gateway run`
      },
      {
        pid: 222,
        name: 'node.exe',
        executablePath: String.raw`C:\Program Files\nodejs\node.exe`,
        commandLine: String.raw`"node.exe" G:\hermes\hermes-agent\scripts\whatsapp-bridge\bridge.js`
      },
      {
        pid: 333,
        name: 'python.exe',
        executablePath: String.raw`G:\hermes\hermes-agent\venv\Scripts\python.exe`,
        commandLine: String.raw`"G:\hermes\hermes-agent\venv\Scripts\python.exe" -m pip install -e .[all]`
      },
      {
        pid: 444,
        name: 'python.exe',
        executablePath: String.raw`G:\other\venv\Scripts\python.exe`,
        commandLine: String.raw`"G:\other\venv\Scripts\python.exe" -m pip install something`
      }
    ],
    { currentPid: 333, updateRoot: ROOT }
  )

  assert.deepEqual(ids, [111, 222])
})

test('parses PowerShell ConvertTo-Json process output for one or many rows', () => {
  assert.deepEqual(parseWindowsProcessList(''), [])
  assert.deepEqual(
    parseWindowsProcessList(
      '{"ProcessId":9512,"Name":"pythonw.exe","ExecutablePath":"G:\\\\hermes\\\\hermes-agent\\\\venv\\\\Scripts\\\\pythonw.exe","CommandLine":"pythonw -m hermes_cli.main gateway run"}'
    ),
    [
      {
        pid: 9512,
        name: 'pythonw.exe',
        executablePath: String.raw`G:\hermes\hermes-agent\venv\Scripts\pythonw.exe`,
        commandLine: 'pythonw -m hermes_cli.main gateway run'
      }
    ]
  )
  assert.equal(parseWindowsProcessList('[{"ProcessId":1},{"ProcessId":2}]').length, 2)
})

test('lists Windows processes from PowerShell with a valid command', { skip: process.platform !== 'win32' }, () => {
  const rows = listWindowsProcesses()

  assert.ok(rows.some(row => row.pid === process.pid), 'expected current Node process in Win32_Process output')
})

test('kill helper uses injected process listing and tree kill function', () => {
  const killed = []
  const pids = killHermesRuntimeProcessesForUpdate(ROOT, {
    currentPid: 1,
    killTree: pid => killed.push(pid),
    listProcesses: () => [
      {
        pid: 777,
        name: 'pythonw.exe',
        executablePath: String.raw`G:\hermes\hermes-agent\venv\Scripts\pythonw.exe`,
        commandLine: String.raw`"G:\hermes\hermes-agent\venv\Scripts\pythonw.exe" -m hermes_cli.main gateway run`
      },
      {
        pid: 888,
        name: 'python.exe',
        executablePath: String.raw`C:\Python311\python.exe`,
        commandLine: String.raw`"C:\Python311\python.exe" C:\other\script.py`
      }
    ]
  })

  assert.deepEqual(pids, [777])
  assert.deepEqual(killed, [777])
})

test('kill helper throws when matching processes are found without a tree killer', () => {
  assert.throws(
    () =>
      killHermesRuntimeProcessesForUpdate(ROOT, {
        currentPid: 1,
        listProcesses: () => [
          {
            pid: 999,
            name: 'pythonw.exe',
            executablePath: String.raw`G:\hermes\hermes-agent\venv\Scripts\pythonw.exe`,
            commandLine: String.raw`"G:\hermes\hermes-agent\venv\Scripts\pythonw.exe" -m hermes_cli.main gateway run`
          }
        ]
      }),
    /killTree/
  )
})

test('kill helper reports process enumeration failures', () => {
  const errors = []
  const pids = killHermesRuntimeProcessesForUpdate(ROOT, {
    currentPid: 1,
    killTree: () => {
      throw new Error('should not be called')
    },
    listProcesses: () => {
      throw new Error('process listing denied')
    },
    onListError: err => errors.push(err.message)
  })

  assert.deepEqual(pids, [])
  assert.deepEqual(errors, ['process listing denied'])
})
