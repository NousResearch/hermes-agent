// @vitest-environment node
import { readFileSync } from 'node:fs'
import { join } from 'node:path'

import { expect, test } from 'vitest'

import { HermesGateway } from './client'

// B09 / C-hx-1: a local Desktop new chat sends `desktopSessionCreateParams` (use-session-actions)
// verbatim. The fixture owner refuses exactly what the generated canonical contract refuses, so a
// frame that reaches it as `session.create` is one the real owner admits.
const doc = JSON.parse(
  readFileSync(join(process.cwd(), '..', 'shared', 'src', 'gateway-contract.openrpc.json'), 'utf8')
)
const createRef = doc['x-canonical-methods'].find((m: { name: string }) => m.name === 'session.create').params[0].schema
  .$ref
const CREATE_KEYS = new Set(Object.keys(doc.components.schemas[createRef.split('/').at(-1)].properties))

const BASE = { cols: 96, source: 'desktop', profile: 'default' }

// [label, desktopSessionCreateParams output, expected owner frame subset | named refusal]
// RT-26 (#139298): `profile: 'default'` rides the frame. Omitting it let a backend launched as a
// named profile (active-profile.json) create the chat in that profile instead.
const CASES: Array<[string, Record<string, unknown>, Record<string, unknown> | RegExp]> = [
  [
    'inherited workspace',
    { ...BASE, cwd: '/w', cwd_explicit: false, fast: false },
    { profile: 'default', cwd: '/w', cwd_explicit: false, service_tier: 'normal' }
  ],
  [
    'picked workspace',
    { ...BASE, cwd: '/w', cwd_explicit: true, fast: false },
    { profile: 'default', cwd: '/w', cwd_explicit: true, service_tier: 'normal' }
  ],
  [
    'composer effort',
    { ...BASE, cwd: '/w', cwd_explicit: false, reasoning_effort: 'high', fast: false },
    { profile: 'default', cwd: '/w', reasoning: 'high', service_tier: 'normal' }
  ],
  [
    'onboarding minimal effort',
    { ...BASE, reasoning_effort: 'minimal', fast: false, title: 'Welcome' },
    { profile: 'default', reasoning: 'minimal', title: 'Welcome', service_tier: 'normal' }
  ],
  [
    'Bot Chat',
    { ...BASE, title: 'Bot Chat', hidden: true, follow_profile_config: true },
    { profile: 'default', follow_profile_config: true, hidden: true }
  ],
  [
    'Fast speed',
    { ...BASE, cwd: '/w', cwd_explicit: false, fast: true },
    { profile: 'default', cwd: '/w', service_tier: 'priority' }
  ],
  [
    'Ultrafast speed',
    { ...BASE, fast: true, service_tier: 'ultrafast' },
    { profile: 'default', service_tier: 'ultrafast' }
  ],
  // KoNit-K P1: a manual model pick ships its provider; the owner's create contract admits both.
  [
    'manual model + provider',
    { ...BASE, model: 'gpt-5', provider: 'openrouter', fast: false },
    { profile: 'default', model: 'gpt-5', provider: 'openrouter', service_tier: 'normal' }
  ]
]

test.each(CASES)(
  '%s: the Desktop create frame is admitted by the owner contract or refused by name',
  async (_label, params, expected) => {
    const wsPackage = 'ws'
    const { WebSocketServer } = await import(wsPackage)
    const server = new WebSocketServer({ host: '127.0.0.1', port: 0 })
    await new Promise<void>(resolve => server.once('listening', resolve))
    const sent: any[] = []
    server.on('connection', (socket: any) => {
      socket.on('message', (bytes: Buffer) => {
        const frame = JSON.parse(bytes.toString())

        if (frame.method === 'client.capabilities') {
          socket.send(JSON.stringify({ jsonrpc: '2.0', id: frame.id, result: { server_requests: [] } }))

          return
        }

        sent.push(frame)
        const fields = Object.keys(frame.params).filter(key => !CREATE_KEYS.has(key))

        socket.send(
          JSON.stringify(
            fields.length
              ? {
                  jsonrpc: '2.0',
                  id: frame.id,
                  error: { code: 4001, message: 'invalid_params', data: { reason: 'invalid_params', fields } }
                }
              : {
                  jsonrpc: '2.0',
                  id: frame.id,
                  result: {
                    session_id: 's',
                    stored_session_id: 's',
                    revision: 1,
                    execution_generation: 0,
                    messages: [],
                    prompts: []
                  }
                }
          )
        )
      })
    })
    const client = new HermesGateway()

    try {
      const address = server.address() as { port: number }
      await client.connect(`ws://127.0.0.1:${address.port}/api/ws?native_dial=fixture&ticket=one-use`)
      const call = client.request('session.create', { ...params, request_id: 'intent' })

      if (expected instanceof RegExp) {
        await expect(call).rejects.toThrow(expected)
        expect(sent).toEqual([])
      } else {
        await expect(call).resolves.toMatchObject({ session_id: 's' })
        expect(sent.map(frame => frame.method)).toEqual(['session.create'])
        expect(sent[0].params).toMatchObject({ ...expected, source: 'gui', request_id: 'intent' })
      }
    } finally {
      client.close()

      for (const socket of server.clients) {
        socket.terminate()
      }
      await new Promise<void>(resolve => server.close(() => resolve()))
    }
  }
)
