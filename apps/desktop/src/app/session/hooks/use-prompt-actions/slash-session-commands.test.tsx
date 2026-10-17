import { cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { CanonicalDesktopProtocol } from '@/api/canonical-protocol'
import { en } from '@/i18n/en'
import { textPart } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { resolveDesktopCommand } from '@/lib/desktop-slash-commands'
import { $composerDraft, setComposerDraft } from '@/store/composer'
import { $queuedPromptsBySession } from '@/store/composer-queue'
import { $connection } from '@/store/session'
import { dropSessionState, publishSessionState } from '@/store/session-states'

import { actRender, Harness, type HarnessHandle, RUNTIME_SESSION_ID } from './index-test-utils'
import { clearSingleFlightSessionResumeState } from './single-flight-resume'

// /undo, /retry, /steer and /queue run against the session owner's canonical verbs (the Ink TUI's
// behaviour on the same gateway), never as `slash.exec`, which the shared gateway refuses.

vi.mock('@/hermes', () => ({
  getLatestSessionMessages: vi.fn(async () => ({ messages: [], session_id: 'session' })),
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  getSession: vi.fn(),
  PROMPT_SUBMIT_REQUEST_TIMEOUT_MS: 1_800_000,
  setApiRequestProfile: vi.fn(),
  transcribeAudio: vi.fn()
}))

vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestGatewayForAgent: vi.fn()
}))

const copy = en.promptNotices.sessionCommands

const SNAPSHOT = {
  revision: 4,
  execution_generation: 2,
  messages: [
    { role: 'user', content: 'first question', row_id: 11 },
    { role: 'assistant', content: 'first answer', row_id: 12 },
    { role: 'user', content: '[CONTEXT COMPACTION] summary', row_id: 13 },
    { role: 'user', content: 'second question', row_id: 14 },
    { role: 'assistant', content: 'second answer', row_id: 15 }
  ]
}

type Responder = (method: string, params?: Record<string, unknown>) => unknown

function gateway(respond: Responder) {
  return vi.fn(async (method: string, params?: Record<string, unknown>) => (respond(method, params) ?? {}) as never)
}

async function mount(requestGateway: ReturnType<typeof gateway>) {
  let handle: HarnessHandle | null = null
  await actRender(
    <Harness onReady={h => (handle = h)} refreshSessions={async () => undefined} requestGateway={requestGateway} />
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

const queuedTexts = () =>
  Object.values($queuedPromptsBySession.get())
    .flat()
    .map(entry => entry.text)

const canonical = () => $connection.set({ mode: 'local', wsUrl: 'ws://localhost/api/ws?native_dial=unminted' } as never)

beforeEach(() => {
  clearSingleFlightSessionResumeState()
  window.localStorage.removeItem('hermes.desktop.composerQueue.v1')
  $queuedPromptsBySession.set({})
  setComposerDraft('')
})

afterEach(() => {
  cleanup()
  $connection.set(null)
  dropSessionState(RUNTIME_SESSION_ID)
  vi.restoreAllMocks()
})

describe('desktop surfaces for the canonical session commands', () => {
  it('routes /undo, /retry, /steer (/s) and /queue (/q) to local actions, not slash.exec', () => {
    const expected = {
      '/undo': 'undo',
      '/retry': 'retry',
      '/steer': 'steer',
      '/s': 'steer',
      '/queue': 'queue',
      '/q': 'queue'
    }

    for (const [name, action] of Object.entries(expected)) {
      expect(resolveDesktopCommand(name)?.surface).toEqual({ kind: 'action', action })
    }
  })
})

describe('/undo on a canonical session', () => {
  beforeEach(canonical)

  it('rewinds the newest user turn by its durable row id and puts its text in the composer', async () => {
    const requestGateway = gateway(method => {
      if (method === 'session.resume') {
        return SNAPSHOT
      }

      if (method === 'session.rewind') {
        return { rewound_count: 2, target_message: { content: 'second question' } }
      }
    })

    const handle = await mount(requestGateway)
    await handle.submitText('/undo')

    expect(callsOf(requestGateway, 'session.rewind')).toEqual([
      { session_id: RUNTIME_SESSION_ID, target_message_id: 14, derived_target: true }
    ])
    expect(callsOf(requestGateway, 'slash.exec')).toEqual([])
    expect($composerDraft.get()).toBe('second question')
    expect(renderedText(handle)).toContain(copy.undidOther('2'))
  })

  it('/undo N targets the Nth newest user turn, skipping compaction carriers', async () => {
    const requestGateway = gateway(method => {
      if (method === 'session.resume') {
        return SNAPSHOT
      }

      if (method === 'session.rewind') {
        return { rewound_count: 5, target_message: { content: 'first question' } }
      }
    })

    const handle = await mount(requestGateway)
    await handle.submitText('/undo 2')

    expect(callsOf(requestGateway, 'session.rewind')).toEqual([
      { session_id: RUNTIME_SESSION_ID, target_message_id: 11, derived_target: true }
    ])
  })

  it('says nothing to undo when there are fewer user turns, without a rewind', async () => {
    const requestGateway = gateway(method => (method === 'session.resume' ? SNAPSHOT : undefined))

    const handle = await mount(requestGateway)
    await handle.submitText('/undo 3')

    expect(callsOf(requestGateway, 'session.rewind')).toEqual([])
    expect(renderedText(handle)).toContain(copy.undoNothing)
  })
})

describe('/retry on a canonical session', () => {
  beforeEach(canonical)

  it('rewinds the newest user turn, then resubmits its text as a new prompt', async () => {
    const requestGateway = gateway(method => {
      if (method === 'session.resume') {
        return SNAPSHOT
      }

      if (method === 'session.rewind') {
        return { rewound_count: 2, target_message: { content: [{ type: 'text', text: 'second question' }] } }
      }
    })

    const handle = await mount(requestGateway)
    await handle.submitText('/retry')

    expect(callsOf(requestGateway, 'session.rewind')).toEqual([
      { session_id: RUNTIME_SESSION_ID, target_message_id: 14, derived_target: true }
    ])
    expect(callsOf(requestGateway, 'prompt.submit')).toEqual([
      expect.objectContaining({ session_id: RUNTIME_SESSION_ID, text: 'second question' })
    ])
    expect(callsOf(requestGateway, 'slash.exec')).toEqual([])
  })

  it('refuses a media turn before archiving anything', async () => {
    const requestGateway = gateway(method =>
      method === 'session.resume'
        ? { ...SNAPSHOT, messages: [{ role: 'user', content: [{ type: 'image_url', image_url: 'x' }], row_id: 3 }] }
        : undefined
    )

    const handle = await mount(requestGateway)
    await handle.submitText('/retry')

    expect(callsOf(requestGateway, 'session.rewind')).toEqual([])
    expect(callsOf(requestGateway, 'prompt.submit')).toEqual([])
    expect(renderedText(handle)).toContain(copy.retryUnsupported)
  })
})

// The real CanonicalDesktopProtocol in front of a revision-fenced owner that answers an exact retry with
// its original receipt (the authority's durable receipt by request_id). The first rewind commits and its
// reply is lost; the user repeats the command.
function lostRewindReply() {
  const protocol = new CanonicalDesktopProtocol()

  const rows = [
    { role: 'user', content: 'first question', row_id: 11, active: true },
    { role: 'assistant', content: 'first answer', row_id: 12, active: true },
    { role: 'user', content: 'second question', row_id: 13, active: true },
    { role: 'assistant', content: 'second answer', row_id: 14, active: true },
    { role: 'user', content: 'third question', row_id: 15, active: true },
    { role: 'assistant', content: 'third answer', row_id: 16, active: true }
  ]

  const receipts = new Map<unknown, Record<string, unknown>>()
  let revision = 4
  let loseReply = true

  const owner = (wire: string, params: Record<string, unknown>) => {
    if (wire === 'session.resume') {
      const messages = rows.filter(row => row.active).map(({ active: _active, ...row }) => row)

      return { session_id: RUNTIME_SESSION_ID, revision, execution_generation: 2, messages }
    }

    const replay = receipts.get(params.request_id)

    if (replay) {
      return replay
    }

    if (params.expected_revision !== revision) {
      throw Object.assign(new Error('revision_conflict'), { data: { reason: 'revision_conflict', revision } })
    }

    const target = (params.payload as { target_message_id: number }).target_message_id
    const rewound = rows.filter(row => row.active && row.row_id >= target)
    rewound.forEach(row => (row.active = false))
    revision += 1

    const receipt = {
      session_id: RUNTIME_SESSION_ID,
      revision,
      rewound_count: rewound.length,
      target_message: { id: target, content: rewound[0].content }
    }

    receipts.set(params.request_id, receipt)

    if (loseReply) {
      loseReply = false

      throw new Error('Hermes gateway connection lost')
    }

    return receipt
  }

  const requestGateway = vi.fn(async (method: string, params: Record<string, unknown> = {}) => {
    if (method !== 'session.resume' && method !== 'session.rewind') {
      return {} as never
    }

    const prepared = protocol.prepare(method, params)

    try {
      return protocol.result(method, prepared, owner(protocol.wire(method, prepared), prepared)) as never
    } catch (error) {
      protocol.failure(prepared, error)

      throw error
    }
  })

  return { requestGateway, active: () => rows.filter(row => row.active).map(row => row.content) }
}

describe('a repeat after a lost rewind reply on a canonical session', () => {
  beforeEach(canonical)

  it('/undo removes only the newest exchange and restores its text', async () => {
    const owner = lostRewindReply()
    let handle: HarnessHandle | null = null

    const seedMessages = ['first question', 'first answer', 'second question', 'second answer', 'third question', 'third answer'].map(
      (text, index) => ({
        id: `m${index}`,
        parts: [textPart(text)],
        role: index % 2 ? ('assistant' as const) : ('user' as const),
        rowId: 11 + index,
        timestamp: index
      })
    )

    await actRender(
      <Harness
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={owner.requestGateway}
        seedMessages={seedMessages}
      />
    )

    await handle!.submitText('/undo')
    expect(renderedText(handle!)).toContain('Hermes gateway connection lost')

    await handle!.submitText('/undo')

    expect(owner.active()).toEqual(['first question', 'first answer', 'second question', 'second answer'])
    expect($composerDraft.get()).toBe('third question')
    expect(renderedText(handle!)).toContain(copy.undidOther('2'))
    // The view drops the rewound rows the owner named, not the turn this repeat's snapshot pointed at.
    expect(renderedText(handle!)).toContain('second answer')
    expect(renderedText(handle!)).not.toContain('third question')
  })

  it('/retry resends the newest question, not the one before it', async () => {
    const owner = lostRewindReply()
    const handle = await mount(owner.requestGateway)

    await handle.submitText('/retry')
    await handle.submitText('/retry')

    expect(owner.active()).toEqual(['first question', 'first answer', 'second question', 'second answer'])
    expect(callsOf(owner.requestGateway, 'prompt.submit')).toEqual([
      expect.objectContaining({ session_id: RUNTIME_SESSION_ID, text: 'third question' })
    ])
  })
})

describe('/steer', () => {
  it('steers the running turn through session.steer', async () => {
    publishSessionState(RUNTIME_SESSION_ID, { ...createClientSessionState(RUNTIME_SESSION_ID), busy: true })
    const requestGateway = gateway(method => (method === 'session.steer' ? { status: 'queued' } : undefined))

    const handle = await mount(requestGateway)
    await handle.submitText('/s check the tests too')

    expect(callsOf(requestGateway, 'session.steer')).toEqual([
      { session_id: RUNTIME_SESSION_ID, text: 'check the tests too' }
    ])
    expect(callsOf(requestGateway, 'slash.exec')).toEqual([])
    expect(queuedTexts()).toEqual([])
    expect(renderedText(handle)).toContain(copy.steerQueued('check the tests too'))
  })

  it('queues the text as the next turn when the steer is not accepted', async () => {
    publishSessionState(RUNTIME_SESSION_ID, { ...createClientSessionState(RUNTIME_SESSION_ID), busy: true })
    const requestGateway = gateway(method => (method === 'session.steer' ? { status: 'rejected' } : undefined))

    const handle = await mount(requestGateway)
    await handle.submitText('/steer late words')

    expect(queuedTexts()).toEqual(['late words'])
    expect(renderedText(handle)).toContain(copy.steerRejected)
  })

  it('queues the text without a steer when no turn is running', async () => {
    const requestGateway = gateway(() => undefined)

    const handle = await mount(requestGateway)
    await handle.submitText('/steer next thing')

    expect(callsOf(requestGateway, 'session.steer')).toEqual([])
    expect(queuedTexts()).toEqual(['next thing'])
    expect(renderedText(handle)).toContain(copy.steerNoActiveTurn('next thing'))
  })
})

describe('/queue', () => {
  it('enqueues the text on the composer queue and counts it with a bare /queue', async () => {
    const requestGateway = gateway(() => undefined)

    const handle = await mount(requestGateway)
    await handle.submitText('/q do this after')

    expect(queuedTexts()).toEqual(['do this after'])
    expect(callsOf(requestGateway, 'slash.exec')).toEqual([])

    await handle.submitText('/queue')

    expect(renderedText(handle)).toContain(copy.queued('do this after'))
    expect(renderedText(handle)).toContain(copy.countOne('1'))
  })

  it('admits the text into the owner FIFO on a canonical session', async () => {
    canonical()
    const requestGateway = gateway(() => undefined)

    const handle = await mount(requestGateway)
    await handle.submitText('/queue owner fifo')

    expect(callsOf(requestGateway, 'prompt.submit')).toEqual([
      expect.objectContaining({ queued: true, session_id: RUNTIME_SESSION_ID, text: 'owner fifo' })
    ])
    expect(callsOf(requestGateway, 'slash.exec')).toEqual([])
  })
})
