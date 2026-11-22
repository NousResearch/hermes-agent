import { expect, test, type WebSocketRoute } from '@playwright/test'

import {
  coreAppEnv,
  createCoreSandbox,
  currentSessionId,
  launchCoreApp,
  recordWebSockets,
  send,
  waitForInteractive,
  writeProviderHome
} from './harness'
import { startScriptedProvider } from './provider'

interface StoredRow {
  id: number
  role: 'assistant' | 'user' | 'tool'
  content: string
}

/** Replay a synthetic machine turn through the real socket listener and IPC history reader. */
test('a clipped Desktop refresh shows a backend-started tool turn once', async () => {
  const provider = await startScriptedProvider()
  const sandbox = createCoreSandbox('machine-turn-refresh')
  writeProviderHome(sandbox.hermesHome, provider.url)
  const { app, page } = await launchCoreApp(coreAppEnv(sandbox))
  const ws = recordWebSockets(page)

  const viewport = page.locator('[data-slot="aui_thread-viewport"]').filter({ visible: true }).first()
  let socket: WebSocketRoute | undefined
  let priorExecution: { authority_epoch: number; execution_generation: number; session_id: string } | undefined
  const old = 'U501-machineold'
  const sentinel = 'U502-machinesentinel'
  const wake = 'WITNESS_DELEGATION_START'
  const interim = ['WITNESS_MACHINE_INTERIM_A', 'WITNESS_MACHINE_INTERIM_B']
  const final = 'WITNESS_MACHINE_FINAL'
  const turnId = 'machine-fixture-execution'
  const count = async (marker: string) => ((await viewport.textContent()) ?? '').split(marker).length - 1

  try {
    await waitForInteractive(app, page)
    provider.script(old, [
      { text: ['WITNESS_OLD_PASS_A'], toolCalls: [{ name: 'terminal', args: { command: 'echo old-pass-a' } }] },
      { text: ['WITNESS_OLD_PASS_B'], toolCalls: [{ name: 'terminal', args: { command: 'echo old-pass-b' } }] },
      { text: ['WITNESS_OLD_FINAL'] }
    ])
    await send(page, `${old} seed older folded history`, 'Enter', ws)
    await expect(viewport).toContainText('WITNESS_OLD_FINAL')
    await expect.poll(() => currentSessionId(page)).not.toBe('')
    const stored = await currentSessionId(page)

    // Cold hydration paints the old tool round under one fold; the following send gives this cached
    // window another common stored anchor after the cut and identifies the current runtime socket.
    await page.routeWebSocket(/\/api\/ws/, client => {
      socket = client
      const server = client.connectToServer()

      server.onMessage(raw => {
        const frame = JSON.parse(String(raw))
        const params = frame.params

        if (
          frame.method === 'event' &&
          params?.type === 'message.complete' &&
          params.payload?.text === 'WITNESS_SENTINEL_FINAL'
        ) {
          priorExecution = params
        }

        client.send(raw)
      })
    })
    await page.reload()
    await waitForInteractive(app, page)
    await expect.poll(() => currentSessionId(page)).toBe(stored)
    await expect(viewport).toContainText('WITNESS_OLD_FINAL')
    provider.script(sentinel, [{ text: ['WITNESS_SENTINEL_FINAL'] }])
    await send(page, `${sentinel} identify the reopened runtime`, 'Enter', ws)
    await expect(viewport).toContainText('WITNESS_SENTINEL_FINAL')
    await expect
      .poll(() => priorExecution)
      .toMatchObject({
        authority_epoch: expect.any(Number),
        execution_generation: expect.any(Number),
        session_id: expect.any(String)
      })

    const execution = {
      authority_epoch: priorExecution!.authority_epoch,
      execution_generation: priorExecution!.execution_generation + 1,
      admission_id: turnId
    }

    const history = await page.evaluate(async id => {
      const bridge = (
        window as typeof window & {
          hermesDesktop: { api: (options: { path: string }) => Promise<{ messages: StoredRow[] }> }
        }
      ).hermesDesktop

      return (await bridge.api({ path: `/api/sessions/${id}/messages?order=oldest&limit=500` })).messages
    }, stored)

    const cut = history.findIndex(row => row.content?.includes('WITNESS_OLD_PASS_B'))
    expect(cut).toBeGreaterThan(1)
    const first = Math.max(...history.map(row => row.id)) + 1
    const metadata = { display_text: wake }
    const turnMetadata = { stream_turn_id: turnId }
    const at = Date.now() / 1000

    const starter = {
      id: first,
      role: 'user',
      content: wake,
      display_kind: 'async_delegation_complete',
      display_metadata: metadata,
      timestamp: at
    }

    const rows = interim.flatMap((text, index) => {
      const id = first + 1 + index * 2
      const toolId = `machine-fixture-tool-${index}`

      return [
        {
          id,
          role: 'assistant',
          content: text,
          display_metadata: turnMetadata,
          timestamp: at + index,
          tool_calls: [{ id: toolId, type: 'function', function: { name: 'terminal', arguments: '{}' } }]
        },
        {
          id: id + 1,
          role: 'tool',
          content: `machine result ${index}`,
          display_metadata: turnMetadata,
          timestamp: at + index,
          tool_call_id: toolId,
          tool_name: 'terminal'
        }
      ]
    })

    const end = first + rows.length + 1
    const finalRow = { id: end, role: 'assistant', content: final, display_metadata: turnMetadata, timestamp: at + 3 }
    const pageRows = [...history.slice(cut), starter, ...rows, finalRow]

    const emit = (type: string, payload: Record<string, unknown>) => {
      expect(socket).toBeDefined()
      socket!.send(
        JSON.stringify({
          jsonrpc: '2.0',
          method: 'event',
          params: { type, session_id: priorExecution!.session_id, payload, ...execution }
        })
      )
    }

    emit('message.start', { turn_id: turnId })

    for (const [index, text] of interim.entries()) {
      emit('message.delta', { text, turn_id: turnId, segment: index, offset: 0 })
      await expect(viewport).toContainText(text)
      emit('message.interim', { text, already_streamed: true, turn_id: turnId, segment: index })
      emit('tool.start', { name: 'terminal', tool_id: `machine-fixture-tool-${index}`, args: {} })
      emit('tool.complete', {
        name: 'terminal',
        tool_id: `machine-fixture-tool-${index}`,
        result: `machine result ${index}`
      })
    }

    emit('message.delta', { text: final, turn_id: turnId, segment: interim.length, offset: 0 })
    await expect(viewport).toContainText(final)
    emit('message.complete', {
      status: 'complete',
      text: final,
      persisted_turn: {
        complete: true,
        user_row_id: first,
        user_row_ids: [first],
        final_assistant_row_id: end,
        row_ids: [starter.id, ...rows.map(row => row.id), end]
      }
    })
    await expect.poll(() => count(interim[0])).toBe(1)

    // Respond to the production refresh read with a leading boundary-cut fold and the newly
    // committed machine turn. Let the app run its own merge; the system marker proves publication.
    await app.evaluate(
      ({ ipcMain }, { stored, pageRows }) => {
        const handlers = (
          ipcMain as unknown as { _invokeHandlers: Map<string, (event: unknown, request: unknown) => Promise<unknown>> }
        )._invokeHandlers

        const original = handlers.get('hermes:api')!
        ipcMain.removeHandler('hermes:api')
        ipcMain.handle('hermes:api', (event, request: { path: string }) => {
          if (request.path.startsWith(`/api/sessions/${stored}/messages?`) && request.path.includes('order=latest')) {
            return { session_id: stored, messages: pageRows }
          }

          return original(event, request)
        })
      },
      { stored, pageRows }
    )
    emit('sessions.changed', {})
    await expect(viewport).toContainText(wake)

    for (const marker of [...interim, final]) {
      expect(await count(marker), marker).toBe(1)
    }

    await expect(viewport).toContainText('WITNESS_OLD_FINAL')
  } finally {
    await app.close().catch(() => undefined)
    await provider.close()
    sandbox.cleanup()
  }
})
