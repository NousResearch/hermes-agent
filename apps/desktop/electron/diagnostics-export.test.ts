import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  alignEvent,
  electronProcessTree,
  executableBasename,
  exportDiagnosticsBundle,
  sanitizeProcessTree
} from './diagnostics-export'

// In-memory fs: the exporter's whole contract is "which files, with what
// bytes", so the test asserts on a map rather than a temp directory (and stays
// as fast and platform-neutral as the rest of the `electron` project).
function makeFakeFs() {
  const dirs: string[] = []
  const files = new Map<string, string>()

  return {
    dirs,
    files,
    fs: {
      async mkdir(dir: string) {
        dirs.push(dir)
      },
      async writeFile(file: string, data: string) {
        files.set(file, data)
      }
    },
    // POSIX join regardless of host, so the assertions read the same on Windows.
    join: (...parts: string[]) => parts.join('/')
  }
}

function makeBundle(overrides: any = {}) {
  return {
    captureId: 'cap-1234',
    anchors: { wallClockAnchorMs: 1_700_000_000_000, mainMonotonicAnchorMs: 1_000 },
    renderer: [
      {
        windowId: 7,
        captureId: 'cap-1234',
        wallClockAnchorMs: 1_700_000_000_000,
        monotonicAnchorMs: 500,
        droppedEvents: 2,
        events: [
          { type: 'long_frame', t: 700, ms: 120, styleMs: 4, blockingMs: 90, scripts: [] },
          { type: 'memory_sample', t: 900, usedMb: 300, totalMb: 400, limitMb: 4_096 }
        ]
      }
    ],
    main: [{ type: 'transport_error', t: 1_200, channel: 'hermes:api', route: '/api/sessions', durationMs: 60_000, errorClass: 'timeout' }],
    mainDropped: 0,
    gateway: {
      captureId: 'cap-1234',
      monotonicAnchorMs: 42_000,
      dropped: 1,
      events: [{ kind: 'loop_drift', t_monotonic: 43.5, drift_s: 0.9 }]
    },
    ...overrides
  } as any
}

function parseJsonl(text: string) {
  return text
    .split('\n')
    .filter(Boolean)
    .map(line => JSON.parse(line))
}

// ── happy path ────────────────────────────────────────────────────────

test('export writes a manifest whose streams parse and share the capture id', async () => {
  const { fs, join, files, dirs } = makeFakeFs()

  const result = await exportDiagnosticsBundle({
    bundle: makeBundle(),
    userDataPath: '/userData',
    appVersion: '9.9.9',
    platform: 'win32',
    wallClock: () => 1_700_000_123_456,
    fs,
    join
  })

  assert.equal(result.directory, '/userData/diagnostics/cap-1234')
  assert.deepEqual(dirs, ['/userData/diagnostics/cap-1234'])

  const manifest = JSON.parse(files.get('/userData/diagnostics/cap-1234/manifest.json')!)

  assert.equal(manifest.capture_id, 'cap-1234')
  assert.equal(manifest.app_version, '9.9.9')
  assert.equal(manifest.platform, 'win32')
  assert.equal(manifest.exported_at_ms, 1_700_000_123_456)
  assert.equal(manifest.anchors.wall_clock_anchor_ms, 1_700_000_000_000)

  // Every non-absent stream file named by the manifest exists and parses.
  for (const stream of manifest.streams) {
    if (!stream.file) {
      continue
    }

    const body = files.get(`/userData/diagnostics/cap-1234/${stream.file}`)

    assert.ok(body !== undefined, `missing ${stream.file}`)
    assert.equal(parseJsonl(body!).length, stream.events)
  }

  assert.deepEqual(
    manifest.streams.map((stream: any) => stream.name),
    ['renderer-7', 'main', 'gateway']
  )
  assert.equal(manifest.streams.find((s: any) => s.name === 'renderer-7').dropped, 2)
  assert.equal(manifest.streams.find((s: any) => s.name === 'gateway').dropped, 1)

  // Classification rides along in its own file, keyed to the same capture.
  const classification = JSON.parse(files.get('/userData/diagnostics/cap-1234/classification.json')!)

  assert.ok(Array.isArray(classification.labels))
})

test('export aligns each stream onto the shared wall clock (KTD3)', async () => {
  const { fs, join, files } = makeFakeFs()

  await exportDiagnosticsBundle({ bundle: makeBundle(), userDataPath: '/u', fs, join })

  // renderer: anchor 500ms, event t=700 -> +200ms past the wall-clock anchor.
  const renderer = parseJsonl(files.get('/u/diagnostics/cap-1234/renderer-7.jsonl')!)

  assert.equal(renderer[0].wall_clock_ms, 1_700_000_000_200)
  assert.equal(renderer[0].t, 700, 'raw monotonic value must survive untouched')

  // main: anchor 1000ms, event t=1200 -> +200ms.
  const main = parseJsonl(files.get('/u/diagnostics/cap-1234/main.jsonl')!)

  assert.equal(main[0].wall_clock_ms, 1_700_000_000_200)

  // gateway: anchor 42000ms, event t_monotonic 43.5s = 43500ms -> +1500ms.
  const gateway = parseJsonl(files.get('/u/diagnostics/cap-1234/gateway.jsonl')!)

  assert.equal(gateway[0].wall_clock_ms, 1_700_000_001_500)
})

// ── error path: absent gateway ────────────────────────────────────────

test('an absent gateway stream still exports and is marked absent', async () => {
  const { fs, join, files } = makeFakeFs()

  const result = await exportDiagnosticsBundle({
    bundle: makeBundle({ gateway: { absent: 'remote-gateway' } }),
    userDataPath: '/u',
    fs,
    join
  })

  assert.equal(files.has('/u/diagnostics/cap-1234/gateway.jsonl'), false)

  const gateway = result.manifest.streams.find(stream => stream.name === 'gateway')!

  assert.equal(gateway.file, null)
  assert.equal(gateway.absent, 'remote-gateway')
  assert.equal(gateway.events, 0)
  // The rest of the bundle is unaffected.
  assert.ok(files.has('/u/diagnostics/cap-1234/main.jsonl'))
  assert.ok(files.has('/u/diagnostics/cap-1234/manifest.json'))
})

test('an empty gateway ring exports as a present but empty stream', async () => {
  const { fs, join, files } = makeFakeFs()

  const result = await exportDiagnosticsBundle({
    bundle: makeBundle({ gateway: { captureId: 'cap-1234', monotonicAnchorMs: 10, dropped: 0, events: [] } }),
    userDataPath: '/u',
    fs,
    join
  })

  assert.equal(files.get('/u/diagnostics/cap-1234/gateway.jsonl'), '')
  assert.equal(result.manifest.streams.find(stream => stream.name === 'gateway')!.absent, undefined)
})

test('a capture with no renderer streams at all still exports', async () => {
  const { fs, join } = makeFakeFs()

  const result = await exportDiagnosticsBundle({
    bundle: makeBundle({ renderer: [], gateway: { absent: 'unavailable' } }),
    userDataPath: '/u',
    fs,
    join
  })

  assert.deepEqual(
    result.manifest.streams.map(stream => stream.name),
    ['main', 'gateway']
  )
})

// ── sanitization ──────────────────────────────────────────────────────

test('executableBasename strips directories and any argv tail', () => {
  assert.equal(executableBasename('C:\\Users\\me\\AppData\\Local\\hermes\\Hermes.exe'), 'Hermes.exe')
  assert.equal(executableBasename('/opt/hermes/bin/hermes'), 'hermes')
  assert.equal(executableBasename('/opt/hermes/bin/hermes --token=SECRET123'), 'hermes')
  assert.equal(executableBasename('Hermes.exe'), 'Hermes.exe')
  assert.equal(executableBasename(''), 'unknown')
  assert.equal(executableBasename(undefined), 'unknown')
})

test('sanitizeProcessTree rebuilds entries, dropping every unlisted field', () => {
  const tree = sanitizeProcessTree([
    {
      pid: 4242,
      ppid: 1,
      name: 'C:\\Program Files\\Hermes\\Hermes.exe',
      // Fields a future probe might add. None may survive.
      cmdline: '--api-key=sk-MARKER --home=C:\\Users\\me',
      exe: 'C:\\Program Files\\Hermes\\Hermes.exe',
      environ: { HERMES_TOKEN: 'MARKER' }
    } as never
  ])

  assert.deepEqual(tree, [{ pid: 4242, ppid: 1, name: 'Hermes.exe' }])
})

test('a marker string in a process command line never reaches the bundle', async () => {
  const { fs, join, files } = makeFakeFs()
  const MARKER = 'HERMES-SANITIZATION-CANARY-9f3a'

  await exportDiagnosticsBundle({
    bundle: makeBundle(),
    userDataPath: '/u',
    fs,
    join,
    processes: [
      { pid: 11, ppid: 1, name: `/opt/hermes/bin/hermes --secret=${MARKER}` } as never,
      { pid: 12, ppid: 11, name: 'C:\\hermes\\Hermes.exe', cmdline: MARKER, argv: ['hermes', MARKER] } as never
    ]
  })

  for (const [file, body] of files) {
    assert.equal(body.includes(MARKER), false, `${MARKER} leaked into ${file}`)
  }

  const manifest = JSON.parse(files.get('/u/diagnostics/cap-1234/manifest.json')!)

  assert.deepEqual(manifest.process_tree, [
    { pid: 11, ppid: 1, name: 'hermes' },
    { pid: 12, ppid: 11, name: 'Hermes.exe' }
  ])

  // No entry carries an argv-ish or path-ish key, whatever it is called.
  for (const entry of manifest.process_tree) {
    assert.deepEqual(Object.keys(entry).sort(), ['name', 'pid', 'ppid'])
    assert.equal(/[\\/]/.test(entry.name), false, 'process name must be a basename, not a path')
  }
})

test('electronProcessTree parents every metrics process on main', () => {
  const tree = electronProcessTree({
    mainPid: 100,
    mainPpid: 4,
    execPath: 'C:\\Program Files\\Hermes\\Hermes.exe',
    metrics: [{ pid: 100, type: 'Browser' }, { pid: 101, type: 'Tab' }, { pid: 102, type: 'Utility', name: 'Network Service' }]
  })

  assert.deepEqual(sanitizeProcessTree(tree), [
    { pid: 100, ppid: 4, name: 'Hermes.exe' },
    { pid: 101, ppid: 100, name: 'Tab' },
    // Whitespace in a utility's role name is not a path tail; the basename rule
    // keeps the first token, which is enough to tell the roles apart.
    { pid: 102, ppid: 100, name: 'Network' }
  ])
})

// ── alignment helper ──────────────────────────────────────────────────

test('alignEvent leaves events without a monotonic stamp alone', () => {
  assert.deepEqual(alignEvent({ type: 'weird' }, 1_000, 0), { type: 'weird' })
  assert.equal(alignEvent(null, 1_000, 0), null)
  assert.equal(alignEvent('scalar', 1_000, 0), 'scalar')
})
