import { beforeEach, describe, expect, it, vi } from 'vitest'

const { sdkMock } = vi.hoisted(() => {
  const atom = <T,>(initial: T) => {
    let value = initial

    return {
      get: () => value,
      listen: () => () => undefined,
      set: (next: T) => {
        value = next
      }
    }
  }

  const component = () => null
  const host: Record<string, unknown> = { state: {} }

  return {
    sdkMock: {
      atom,
      blobatarSvg: vi.fn(),
      Button: component,
      Checkbox: component,
      cn: (...values: unknown[]) => values.filter(Boolean).join(' '),
      Codicon: component,
      COMPOSER_AREAS: {},
      createBudgetedLoop: vi.fn(),
      ConfirmDialog: component,
      ContextMenu: component,
      ContextMenuContent: component,
      ContextMenuItem: component,
      ContextMenuSeparator: component,
      ContextMenuTrigger: component,
      CopyButton: component,
      Dialog: component,
      DialogContent: component,
      DialogDescription: component,
      DialogFooter: component,
      DialogHeader: component,
      DialogTitle: component,
      DropdownMenu: component,
      DropdownMenuContent: component,
      DropdownMenuItem: component,
      DropdownMenuSeparator: component,
      DropdownMenuTrigger: component,
      EmptyState: component,
      GlyphSpinner: component,
      haptic: vi.fn(),
      host,
      Input: component,
      PALETTE_AREA: 'palette',
      profileColor: () => '#000',
      queryClient: { invalidateQueries: vi.fn() },
      relativeTime: () => 'now',
      ScrollArea: component,
      SearchField: component,
      Select: component,
      SelectContent: component,
      SelectItem: component,
      SelectTrigger: component,
      SelectValue: component,
      SkillsView: component,
      Streamdown: component,
      Switch: component,
      Textarea: component,
      Tip: component,
      ToolsetConfigPanel: component,
      McpTab: component,
      useQuery: vi.fn(),
      useValue: <T,>(store: { get: () => T }) => store.get()
    }
  }
})

vi.mock('@hermes/plugin-sdk', () => sdkMock)

type Scheduler = {
  close: (message?: string) => void
  enqueue: <T>(key: string, run: () => Promise<T> | T) => Promise<T>
}

async function scheduler(): Promise<Scheduler> {
  // @ts-expect-error Bundled plugin remains plain JavaScript for disk-plugin compatibility.
  const mod = await import('./plugin.js')

  return mod.default.createRelayDeliveryScheduler() as Scheduler
}

async function botPlugin() {
  // @ts-expect-error Bundled plugin remains plain JavaScript for disk-plugin compatibility.
  const mod = await import('./plugin.js')

  return mod.default as { drainRelayOutboxes: () => Promise<void> }
}

beforeEach(() => {
  vi.clearAllMocks()
})

describe('Bot relay delivery scheduler', () => {
  it('runs independent targets concurrently while preserving FIFO per target', async () => {
    const relay = await scheduler()
    const started: string[] = []
    const completed: string[] = []
    let releaseA!: () => void

    const aGate = new Promise<void>(resolve => {
      releaseA = resolve
    })

    const a = relay.enqueue('connection-a\u0000alpha', async () => {
      started.push('a')
      await aGate
      completed.push('a')

      return 'a'
    })

    const b = relay.enqueue('connection-b\u0000beta', async () => {
      started.push('b')
      completed.push('b')

      return 'b'
    })

    const c = relay.enqueue('connection-a\u0000alpha', async () => {
      started.push('c')
      completed.push('c')

      return 'c'
    })

    await vi.waitFor(() => expect(started).toEqual(['a', 'b']))
    await expect(b).resolves.toBe('b')

    expect(started).toEqual(['a', 'b'])

    releaseA()

    await expect(Promise.all([a, c])).resolves.toEqual(['a', 'c'])

    expect(started).toEqual(['a', 'b', 'c'])
    expect(completed).toEqual(['b', 'a', 'c'])
  })

  it('rejects queued deliveries on stop so their caller can notify the sender', async () => {
    const relay = await scheduler()
    let releaseRunning!: () => void

    const runningGate = new Promise<void>(resolve => {
      releaseRunning = resolve
    })

    const running = relay.enqueue('connection-a\u0000alpha', async () => {
      await runningGate

      return 'running'
    })

    const queued = relay.enqueue('connection-a\u0000alpha', () => 'queued')

    relay.close('relay disposed')

    await expect(queued).rejects.toThrow('relay disposed')
    releaseRunning()

    await expect(running).resolves.toBe('running')
  })

  it('drains a sender outbox through target delivery and posts one terminal reply', async () => {
    const sender = { connectionId: 'sender', profile: 'default', targetProfile: 'default' }
    const target = { connectionId: 'target', profile: 'ops', targetProfile: 'ops' }
    const replies: Array<Record<string, unknown>> = []
    sdkMock.host.profileRoutes = vi.fn(async () => [sender, target])
    sdkMock.host.requestProfile = vi.fn(async (route: { connectionId: string }, method: string, params: Record<string, unknown>) => {
      if (route.connectionId === 'sender' && method === 'bot_relay.outbox.drain') {
        return { envelopes: [{ id: 'a'.repeat(32), target_connection: 'target', target_profile: 'ops', message: 'hello' }] }
      }

      if (route.connectionId === 'target' && method === 'bot_relay.deliver') {
        expect(params).toMatchObject({ profile: 'ops', message: 'hello' })

        return { reply: 'world' }
      }

      if (route.connectionId === 'sender' && method === 'bot_relay.reply') {
        replies.push(params)

        return { ok: true }
      }

      throw new Error(`unexpected RPC: ${route.connectionId} ${method}`)
    })

    await (await botPlugin()).drainRelayOutboxes()

    expect(replies).toEqual([{ id: 'a'.repeat(32), reply: 'world' }])
  })
})