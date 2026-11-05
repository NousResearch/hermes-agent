import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'
import { fr } from '@/i18n/fr'
import { setRuntimeI18nLocale } from '@/i18n/runtime'
import { toChatMessages } from '@/lib/chat-messages'
import { resolveDesktopCommand } from '@/lib/desktop-slash-commands'
import { $connection } from '@/store/session'
import { dropSessionState } from '@/store/session-states'

import { useHermesConfig } from '../use-hermes-config'

import { actRender, Harness, type HarnessHandle, RUNTIME_SESSION_ID } from './index-test-utils'
import { clearSingleFlightSessionResumeState } from './single-flight-resume'
import { formatHistory, saveRequest } from './slash-desktop-commands'

// T11 (#134889): Desktop commands the shared gateway refused as `slash.exec`, or that Desktop never
// offered, run on the owner's existing verbs instead.

vi.mock('@/hermes', () => ({
  getLatestSessionMessages: vi.fn(async () => ({ messages: [], session_id: 'session' })),
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  getHermesConfig: vi.fn(async () => ({ display: { timestamps: false } })),
  getHermesConfigDefaults: vi.fn(async () => ({})),
  getSession: vi.fn(),
  PROMPT_SUBMIT_REQUEST_TIMEOUT_MS: 1_800_000,
  setApiRequestProfile: vi.fn(),
  transcribeAudio: vi.fn()
}))

vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestGatewayForAgent: vi.fn()
}))

const copy = en.promptNotices.desktopCommands

type Responder = (method: string, params?: Record<string, unknown>) => unknown

function gateway(respond: Responder) {
  return vi.fn(async (method: string, params?: Record<string, unknown>) => (respond(method, params) ?? {}) as never)
}

async function mount(requestGateway: ReturnType<typeof gateway>, seedMessages?: unknown[]) {
  let handle: HarnessHandle | null = null
  await actRender(
    <Harness
      onReady={h => (handle = h)}
      refreshSessions={async () => undefined}
      requestGateway={requestGateway}
      seedMessages={seedMessages}
    />
  )

  return handle!
}

const callsOf = (requestGateway: ReturnType<typeof gateway>, method: string) =>
  requestGateway.mock.calls.filter(([name]) => name === method).map(([, params]) => params)

const renderedText = (handle: HarnessHandle) =>
  handle
    .state()
    .messages.flatMap(message => message.parts.map(part => ('text' in part ? String(part.text) : '')))
    .join('\n')

beforeEach(() => {
  clearSingleFlightSessionResumeState()
  $connection.set({ mode: 'local', wsUrl: 'ws://localhost/api/ws?native_dial=unminted' } as never)
})

afterEach(() => {
  cleanup()
  $connection.set(null)
  dropSessionState(RUNTIME_SESSION_ID)
  vi.restoreAllMocks()
})

describe('desktop surfaces for /rollback /agents /config /history', () => {
  it('routes each to a local action (never slash.exec, never "terminal only")', () => {
    for (const [name, action] of Object.entries({
      '/rollback': 'rollback',
      '/agents': 'agents',
      '/tasks': 'agents',
      '/config': 'config',
      '/history': 'history'
    })) {
      expect(resolveDesktopCommand(name)?.surface).toEqual({ kind: 'action', action })
    }
  })
})

const CHECKPOINTS = {
  enabled: true,
  checkpoints: [
    { hash: 'aaaa1111', timestamp: '2026-10-18T10:00:00', message: 'before write_file' },
    { hash: 'bbbb2222', timestamp: '2026-10-18T09:00:00', message: 'before patch' }
  ]
}

describe('/rollback', () => {
  it('lists checkpoints through rollback.list', async () => {
    const requestGateway = gateway(method => (method === 'rollback.list' ? CHECKPOINTS : undefined))
    const handle = await mount(requestGateway)

    await handle.submitText('/rollback')

    expect(callsOf(requestGateway, 'rollback.list')).toEqual([{ session_id: RUNTIME_SESSION_ID }])
    expect(callsOf(requestGateway, 'slash.exec')).toEqual([])
    const text = renderedText(handle)
    expect(text).toContain('1. aaaa1111')
    expect(text).toContain('before patch')
  })

  it('says checkpoints are off when the owner reports them disabled', async () => {
    const requestGateway = gateway(method =>
      method === 'rollback.list' ? { enabled: false, checkpoints: [] } : undefined
    )

    const handle = await mount(requestGateway)

    await handle.submitText('/rollback list')

    expect(renderedText(handle)).toMatch(/checkpoints are not enabled/i)
  })

  it('restores checkpoint N (with --all and a file) through rollback.restore', async () => {
    const requestGateway = gateway(method =>
      method === 'rollback.restore' ? { success: true, restored_to: 'aaaa1111', history_removed: 2 } : undefined
    )

    const handle = await mount(requestGateway)

    await handle.submitText('/rollback 1 --all')
    await handle.submitText('/rollback 2 src/app.py')

    expect(callsOf(requestGateway, 'rollback.restore')).toEqual([
      { session_id: RUNTIME_SESSION_ID, hash: '1', all: true },
      { session_id: RUNTIME_SESSION_ID, hash: '2', file_path: 'src/app.py' }
    ])
    expect(renderedText(handle)).toContain('aaaa1111')
  })

  it('shows a failed restore as an error, and diffs through rollback.diff', async () => {
    const requestGateway = gateway(method => {
      if (method === 'rollback.restore') {
        return { success: false, error: 'Invalid checkpoint number. Use 1-2.' }
      }

      if (method === 'rollback.diff') {
        return { stat: ' a.py | 2 +-', diff: '-old\n+new' }
      }
    })

    const handle = await mount(requestGateway)

    await handle.submitText('/rollback 9')
    await handle.submitText('/rollback diff 1')

    expect(callsOf(requestGateway, 'rollback.diff')).toEqual([{ session_id: RUNTIME_SESSION_ID, hash: '1' }])
    const text = renderedText(handle)
    expect(text).toContain('Invalid checkpoint number')
    expect(text).toContain('+new')
  })

  it('reloads the transcript after a restore that rewound history, but not after a file-only restore', async () => {
    const rows = [
      { role: 'user', content: 'KEPT QUESTION', row_id: 1 },
      { role: 'assistant', content: 'KEPT ANSWER', row_id: 2 },
      { role: 'user', content: 'REMOVED QUESTION', row_id: 3 },
      { role: 'assistant', content: 'REMOVED ANSWER', row_id: 4 }
    ]

    const requestGateway = gateway((method, params) => {
      if (method === 'rollback.restore') {
        return params?.file_path
          ? { success: true, restored_to: 'bbbb2222' }
          : { success: true, restored_to: 'aaaa1111', history_removed: 2 }
      }

      if (method === 'session.resume') {
        return { messages: rows.slice(0, 2) }
      }
    })

    const handle = await mount(requestGateway, toChatMessages(rows as never))

    await handle.submitText('/rollback 1 src/app.py')
    expect(callsOf(requestGateway, 'session.resume')).toEqual([])
    expect(renderedText(handle)).toContain('REMOVED QUESTION')

    await handle.submitText('/rollback 1')
    expect(callsOf(requestGateway, 'session.resume')).toEqual([{ session_id: RUNTIME_SESSION_ID }])
    const text = renderedText(handle)
    expect(text).toContain('KEPT ANSWER')
    expect(text).not.toContain('REMOVED QUESTION')
    expect(text).not.toContain('REMOVED ANSWER')
    expect(text).toContain(copy.rewound('2'))
  })
})

describe('/agents', () => {
  it("reads this session's subagents and background processes from the owner", async () => {
    const requestGateway = gateway(method => {
      if (method === 'subagent.list') {
        return {
          subagents: [{ subagent_id: 'sa-1', depth: 1, goal: 'research the API', status: 'running', tool_count: 3 }]
        }
      }

      if (method === 'process.list') {
        return { processes: [{ session_id: 'proc_1', command: 'npm run dev', status: 'running' }] }
      }
    })

    const handle = await mount(requestGateway)

    await handle.submitText('/tasks')

    expect(callsOf(requestGateway, 'subagent.list')).toEqual([{ session_id: RUNTIME_SESSION_ID }])
    expect(callsOf(requestGateway, 'process.list')).toEqual([{ session_id: RUNTIME_SESSION_ID }])
    expect(callsOf(requestGateway, 'slash.exec')).toEqual([])
    const text = renderedText(handle)
    expect(text).toContain('research the API')
    expect(text).toContain('npm run dev')
  })

  it('says nothing is running when both lists are empty', async () => {
    const requestGateway = gateway(method =>
      method === 'subagent.list' ? { subagents: [] } : method === 'process.list' ? { processes: [] } : undefined
    )

    const handle = await mount(requestGateway)

    await handle.submitText('/agents')

    expect(renderedText(handle)).toMatch(/no active subagents or background processes/i)
  })
})

describe('/config', () => {
  it("shows the profile's client config projection from config.get", async () => {
    const requestGateway = gateway(method =>
      method === 'config.get' ? { config: { display: { skin: 'mono', timestamps: true }, voice: {} } } : undefined
    )

    const handle = await mount(requestGateway)

    await handle.submitText('/config')

    expect(callsOf(requestGateway, 'config.get')).toEqual([{ session_id: RUNTIME_SESSION_ID, key: 'full' }])
    const text = renderedText(handle)
    expect(text).toContain('skin: mono')
    // An empty section (the live projection always carries `voice: {}`) prints no dangling header.
    expect(text).not.toContain('voice:')
    expect(text).toContain(copy.configFooter)
  })

  it('renders the canonical projection in full', async () => {
    const requestGateway = gateway(method =>
      method === 'config.get'
        ? {
            config: {
              display: { skin: 'mono', timestamps: true, details_mode: 'collapsed' },
              approvals: { mode: 'manual', timeout: 60 },
              paste_collapse_threshold: 5,
              paste_collapse_char_threshold: 800,
              voice: { record_key: 'ctrl+b', submit_mode: 'auto' }
            }
          }
        : undefined
    )

    const handle = await mount(requestGateway)

    await handle.submitText('/config')

    const text = renderedText(handle)

    for (const line of [
      'skin: mono',
      'timestamps: true',
      'details_mode: collapsed',
      'mode: manual',
      'timeout: 60',
      'paste_collapse_threshold: 5',
      'paste_collapse_char_threshold: 800',
      'record_key: ctrl+b',
      'submit_mode: auto'
    ]) {
      expect(text).toContain(line)
    }
  })

  it('fails closed on a legacy raw config: only the canonical allowlist renders', async () => {
    // A standalone `hermes serve` answers config.get key=full with the raw effective config.
    const S = 'PROBE-SECRET'

    const requestGateway = gateway(method =>
      method === 'config.get'
        ? {
            config: {
              model: { default: 'm', api_key: S },
              providers: { p: { 'api-key': S } },
              custom_providers: [{ name: 'c', api_key: S }],
              mcp_servers: {
                s: { headers: { Authorization: `Bearer ${S}`, 'X-Api-Key': S, Cookie: `sid=${S}` }, env: { X: S } }
              },
              FAL_KEY: S,
              browser: { cdp_url: `ws://u:${S}@127.0.0.1:9222/`, session_key: S },
              terminal: {
                ssh_private_key: S,
                docker_shared_container_key: S,
                docker_env: { DB_PASS: S, MY_CREDENTIALS: S }
              },
              tools: { web: { key: S, bearer: S, jwt: S, passwd: S, backend: 'x' } },
              platforms: { hooks: [{ url: 'https://h', auth: `Bearer ${S}` }] },
              display: {
                skin: 'mono',
                'api-key': S,
                key: S,
                bearer: S,
                jwt: S,
                passwd: S,
                session_key: S,
                headers: { Authorization: S, 'X-Api-Key': S, Cookie: S },
                image_url: `https://user:${S}@example.com/x`,
                list: [{ auth: `Bearer ${S}`, keep: 'ok', url: `https://u:${S}@h.example/` }]
              },
              voice: { record_key: 'ctrl+b', api_key: S, provider: 'openai' }
            }
          }
        : undefined
    )

    const handle = await mount(requestGateway)

    await handle.submitText('/config')

    const text = renderedText(handle)
    expect(text).toContain('skin: mono')
    expect(text).toContain('record_key: ctrl+b')
    expect(text).toContain('https://example.com/x')
    expect(text).toContain('"keep":"ok"')
    expect(text).not.toContain(S)
    expect(text).not.toContain('provider: openai')
    expect(text).not.toContain('backend: x')
    expect(text).not.toMatch(
      /^\s*(model|providers|custom_providers|mcp_servers|FAL_KEY|browser|terminal|tools|platforms):/m
    )
  })
})

describe('/history', () => {
  it("renders the owner's transcript snapshot", async () => {
    const requestGateway = gateway(method =>
      method === 'session.resume'
        ? {
            messages: [
              { role: 'user', content: 'first question', row_id: 1 },
              { role: 'assistant', content: [{ type: 'text', text: 'first answer' }], row_id: 2 },
              { role: 'tool', content: 'tool noise', row_id: 3 }
            ]
          }
        : undefined
    )

    const handle = await mount(requestGateway)

    await handle.submitText('/history')

    expect(callsOf(requestGateway, 'session.resume')).toEqual([{ session_id: RUNTIME_SESSION_ID }])
    expect(callsOf(requestGateway, 'slash.exec')).toEqual([])
    const text = renderedText(handle)
    expect(text).toContain('You: first question')
    expect(text).toContain('Hermes: first answer')
    expect(text).not.toContain('tool noise')
  })

  it('uses the transcript projection: archived turns shown, hidden rows skipped, display_content preferred', () => {
    const history = formatHistory([
      { role: 'user', content: 'ARCHIVED QUESTION', archived: true, row_id: 1 },
      { role: 'assistant', content: 'ARCHIVED ANSWER', archived: true, row_id: 2 },
      { role: 'user', content: 'MODEL ONLY HEARTBEAT', display_kind: 'hidden', row_id: 3 },
      { role: 'user', content: 'MODEL ONLY SKILL BODY', display_content: '/skill question', row_id: 4 },
      { role: 'assistant', content: 'VISIBLE ANSWER', row_id: 5 }
    ] as never)

    expect(history).toContain('You: ARCHIVED QUESTION')
    expect(history).toContain('Hermes: ARCHIVED ANSWER')
    expect(history).toContain('You: /skill question')
    expect(history).toContain('Hermes: VISIBLE ANSWER')
    expect(history).not.toContain('MODEL ONLY')
  })
})

describe('/save writes on the Electron side', () => {
  it('asks the owner for the rendered file (session.export) and saves it through the main process', async () => {
    const saveTextFileAs = vi.fn(async () => ({ saved: true, path: '/home/u/Documents/notes.md' }))

    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = { saveTextFileAs }

    const requestGateway = gateway(method =>
      method === 'session.export'
        ? { format: 'md', filename: 'hermes_session_s1.md', content: '# transcript' }
        : undefined
    )

    const handle = await mount(requestGateway)

    await handle.submitText('/save md notes.md redact')

    expect(callsOf(requestGateway, 'session.export')).toEqual([
      { session_id: RUNTIME_SESSION_ID, format: 'md', redact: true }
    ])
    expect(callsOf(requestGateway, 'session.save')).toEqual([])
    expect(saveTextFileAs).toHaveBeenCalledWith(
      expect.objectContaining({ content: '# transcript', defaultPath: 'notes.md' })
    )
    expect(renderedText(handle)).toContain(copy.savedTo('/home/u/Documents/notes.md'))
  })

  it('defaults to json and the owner filename, and reports a cancelled dialog', async () => {
    const saveTextFileAs = vi.fn(async () => ({ saved: false, canceled: true }))

    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = { saveTextFileAs }

    const requestGateway = gateway(method =>
      method === 'session.export' ? { format: 'json', filename: 'hermes_session_s1.json', content: '{}' } : undefined
    )

    const handle = await mount(requestGateway)

    await handle.submitText('/save')

    expect(callsOf(requestGateway, 'session.export')).toEqual([{ session_id: RUNTIME_SESSION_ID, format: 'json' }])
    expect(saveTextFileAs).toHaveBeenCalledWith(expect.objectContaining({ defaultPath: 'hermes_session_s1.json' }))
    expect(renderedText(handle)).toContain(copy.saveCancelled)
  })

  it('falls back to the legacy host save only for a plain /save, and says where it wrote', async () => {
    const requestGateway = gateway(method => {
      if (method === 'session.export') {
        throw Object.assign(new Error('Method not found'), { code: -32601 })
      }

      if (method === 'session.save') {
        return { file: '/gateway/saved/raw.json' }
      }
    })

    const handle = await mount(requestGateway)

    await handle.submitText('/save html report.html redact')
    await handle.submitText('/save json redact')
    await handle.submitText('/save md')
    await handle.submitText('/save json mine.json')
    expect(callsOf(requestGateway, 'session.save')).toEqual([])
    expect(renderedText(handle)).not.toContain('/gateway/saved/raw.json')
    expect(renderedText(handle)).toContain(copy.saveLegacyRefused)

    await handle.submitText('/save')
    expect(callsOf(requestGateway, 'session.save')).toEqual([{ session_id: RUNTIME_SESSION_ID }])
    expect(renderedText(handle)).toContain(copy.savedOnHost('/gateway/saved/raw.json'))
  })

  it('rejects an unknown format without a gateway call', async () => {
    const requestGateway = gateway(() => undefined)
    const handle = await mount(requestGateway)

    await handle.submitText('/save pdf')

    expect(callsOf(requestGateway, 'session.export')).toEqual([])
    expect(renderedText(handle)).toContain(copy.saveUsage)
  })
})

describe('/timestamps, /palette, /worktree, /snapshot', () => {
  it('/timestamps toggles the transcript timestamps locally (UI state, no gateway call)', async () => {
    const { $displayTimestamps } = await import('@/store/display-timestamps')
    $displayTimestamps.set(false)
    const requestGateway = gateway(() => undefined)
    const handle = await mount(requestGateway)

    await handle.submitText('/timestamps on')
    expect($displayTimestamps.get()).toBe(true)
    await handle.submitText('/ts')
    expect($displayTimestamps.get()).toBe(false)
    await handle.submitText('/timestamps status')

    expect(requestGateway.mock.calls.filter(([m]) => m === 'slash.exec' || m === 'command.dispatch')).toEqual([])
    expect(renderedText(handle)).toContain(copy.timestampsOff)
  })

  it('/timestamps is a window override that survives an ordinary config refresh', async () => {
    const { $displayTimestamps } = await import('@/store/display-timestamps')
    $displayTimestamps.set(false)
    const config = renderHook(() => useHermesConfig({ activeSessionIdRef: { current: RUNTIME_SESSION_ID } }))
    const handle = await mount(gateway(() => undefined))

    await handle.submitText('/timestamps on')
    await act(async () => {
      await config.result.current.refreshHermesConfig()
    })

    expect($displayTimestamps.get()).toBe(true)
    await handle.submitText('/timestamps off')
  })

  it('/palette opens the desktop command palette', async () => {
    const { $commandPaletteOpen } = await import('@/store/command-palette')
    $commandPaletteOpen.set(false)
    const handle = await mount(gateway(() => undefined))

    await handle.submitText('/palette')

    expect($commandPaletteOpen.get()).toBe(true)
  })

  it('/worktree and /snapshot are named refusals that say what to use instead', () => {
    expect(resolveDesktopCommand('/worktree')?.surface.kind).toBe('unavailable')
    expect(resolveDesktopCommand('/snapshot')?.surface.kind).toBe('unavailable')
  })
})

describe('/export and /import (profile archives)', () => {
  it('route to the desktop profile share flows on a local gateway', () => {
    expect(resolveDesktopCommand('/export')?.surface).toEqual({ kind: 'action', action: 'export' })
    expect(resolveDesktopCommand('/import')?.surface).toEqual({ kind: 'action', action: 'import' })
  })

  it('refuse by name on a remote gateway, whose filesystem is not this machine', async () => {
    $connection.set({ mode: 'remote', wsUrl: 'wss://example.test/api/ws' } as never)
    const requestGateway = gateway(() => undefined)
    const handle = await mount(requestGateway)

    await handle.submitText('/export')
    await handle.submitText('/import')

    expect(callsOf(requestGateway, 'slash.exec')).toEqual([])
    expect(renderedText(handle)).toContain(copy.exportRemote)
    expect(renderedText(handle)).toContain(copy.importRemote)
  })
})

describe('desktop command notices follow the app language', () => {
  afterEach(() => setRuntimeI18nLocale('en'))

  it('renders /timestamps, /rollback usage and /history in the active locale', async () => {
    setRuntimeI18nLocale('fr')
    const { $displayTimestamps } = await import('@/store/display-timestamps')
    $displayTimestamps.set(false)
    const handle = await mount(gateway(method => (method === 'session.resume' ? { messages: [] } : undefined)))

    await handle.submitText('/timestamps status')
    await handle.submitText('/rollback diff')
    await handle.submitText('/history')

    const text = renderedText(handle)
    const fr_ = fr.promptNotices.desktopCommands
    expect(fr_.timestampsOff).not.toBe(copy.timestampsOff)
    expect(text).toContain(fr_.timestampsOff)
    expect(text).toContain(fr_.rollbackUsage)
    expect(text).toContain(fr_.historyEmpty)
    expect(text).not.toContain(copy.timestampsOff)
  })
})

describe('/save --concise', () => {
  it.each([
    ['md --concise', { concise: true, format: 'md', redact: false }],
    [
      'md conversation.md --concise --redact',
      { concise: true, filename: 'conversation.md', format: 'md', redact: true }
    ],
    ['--concise md notes.md redact', { concise: true, filename: 'notes.md', format: 'md', redact: true }],
    ['md notes.md', { concise: false, filename: 'notes.md', format: 'md', redact: false }],
    ['md redact.md', { concise: false, filename: 'redact.md', format: 'md', redact: false }],
    ['', { concise: false, format: 'json', redact: false }]
  ])('parses "%s" without taking a flag for the file name', (arg, expected) => {
    expect(saveRequest(arg)).toEqual(expected)
  })

  it.each(['json --concise', 'html --concise', 'pdf --concise', 'md a.md b.md --concise'])('"%s" is usage', arg => {
    expect(saveRequest(arg)).toBeNull()
  })

  it('asks the owner for the concise transcript and saves it on this machine', async () => {
    const saveTextFileAs = vi.fn(async () => ({ saved: true, path: '/home/u/c.md' }))

    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = { saveTextFileAs }

    const requestGateway = gateway(method =>
      method === 'session.export'
        ? { format: 'md', filename: 'hermes_session_s1.md', content: '# Conversation: x' }
        : undefined
    )

    const handle = await mount(requestGateway)

    await handle.submitText('/save md c.md --concise')

    expect(callsOf(requestGateway, 'session.export')).toEqual([
      { concise: true, format: 'md', session_id: RUNTIME_SESSION_ID }
    ])
    expect(saveTextFileAs).toHaveBeenCalledWith(
      expect.objectContaining({ content: '# Conversation: x', defaultPath: 'c.md' })
    )
  })

  it('says so when the gateway predates --concise, and never falls back to a host-side save', async () => {
    const requestGateway = gateway(method => {
      if (method === 'session.export') {
        throw new Error('invalid_params')
      }
    })

    const handle = await mount(requestGateway)

    await handle.submitText('/save md --concise')

    expect(renderedText(handle)).toContain(copy.conciseOutdated)
    expect(callsOf(requestGateway, 'session.save')).toEqual([])
  })

  it('a gateway without session.export refuses --concise rather than writing raw JSON on its host', async () => {
    const requestGateway = gateway(method => {
      if (method === 'session.export') {
        throw Object.assign(new Error('Method not found'), { code: -32601 })
      }

      if (method === 'session.save') {
        return { file: '/gateway/saved/raw.json' }
      }
    })

    const handle = await mount(requestGateway)

    await handle.submitText('/save md --concise')

    expect(callsOf(requestGateway, 'session.save')).toEqual([])
    expect(renderedText(handle)).toContain(copy.saveLegacyRefused)
  })
})
