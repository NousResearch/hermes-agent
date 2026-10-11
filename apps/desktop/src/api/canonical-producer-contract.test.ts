// @vitest-environment node
import { readFileSync } from 'node:fs'
import { join } from 'node:path'

import { expect, test } from 'vitest'

import { HermesGateway } from './client'

// Class sweep: every Desktop producer frame that reaches the shared owner must fit the closed
// (`additionalProperties: false`) canonical contract generated from tui_gateway/contracts. The
// fixture owner refuses exactly what `canonical_param_problems` refuses: unknown or missing keys.
interface Schema {
  properties: Record<string, unknown>
  required?: string[]
}

function canonicalContract(): Map<string, Schema> {
  const doc = JSON.parse(
    readFileSync(join(process.cwd(), '..', 'shared', 'src', 'gateway-contract.openrpc.json'), 'utf8')
  )

  const schemas = doc.components.schemas as Record<string, Schema>

  return new Map(
    (doc['x-canonical-methods'] as Array<{ name: string; params: Array<{ schema: { $ref: string } }> }>).map(method => [
      method.name,
      schemas[method.params[0].schema.$ref.split('/').at(-1)!]
    ])
  )
}

const CONTRACT = canonicalContract()

function problems(method: string, params: Record<string, unknown>): string[] {
  const schema = CONTRACT.get(method)

  if (!schema) {
    return []
  }

  return [
    ...Object.keys(params).filter(key => !(key in schema.properties)),
    ...(schema.required ?? []).filter(key => !(key in params))
  ]
}

const SNAPSHOT = {
  session_id: 's',
  stored_session_id: 's',
  revision: 4,
  execution_generation: 2,
  running: false,
  messages: [],
  prompts: []
}

function answer(frame: { method: string; params: Record<string, unknown> }): Record<string, unknown> {
  const p = frame.params

  if (frame.method === 'session.resume') {
    return SNAPSHOT
  }

  if (frame.method === 'session.mutate') {
    return { session_id: p.session_id, revision: 5, rewound_count: 2 }
  }

  if (frame.method === 'prompt.submit') {
    return {
      admission_id: p.submission_id,
      ref: { session_id: p.session_id },
      sequence: 1,
      status: 'queued',
      outcome: null,
      authority_epoch: 1,
      execution_generation: 2
    }
  }

  return { status: 'settled', delivery_id: p.id, value: 'x' }
}

// [label, Desktop method, exact producer params, expected: wire methods sent (after attach) or a named refusal]
const SITES: Array<[string, string, Record<string, unknown>, string[] | RegExp]> = [
  [
    'hermes-bots/relay.ts relayDeliverParams',
    'bot_relay.deliver',
    {
      id: 'e',
      profile: 'ops',
      message: 'Message from 🤖 S (@s): hi',
      from_profile: 's',
      from_handle: 's',
      from_connection: 'c'
    },
    ['bot_relay.deliver']
  ],
  [
    'reasoning-slash.ts /reasoning --global',
    'config.set',
    { key: 'reasoning', session_id: 's', value: 'high', scope: 'global' },
    /reasoning .*not available/
  ],
  [
    'reasoning-step.ts keybind step',
    'config.set',
    { key: 'reasoning', session_id: 's', value: 'high' },
    /reasoning is not available/
  ],
  ['model-presets.ts fast', 'config.set', { key: 'fast', session_id: 's', value: 'fast' }, /fast is not available/],
  [
    'approval-mode.ts menu',
    'config.set',
    { key: 'approvals.mode', value: 'manual', profile: 'default' },
    /approvals\.mode is not available/
  ],
  [
    'voice-live.ts engine',
    'config.set',
    { key: 'voice.voice_chat_mode', value: 'chained' },
    /voice\.voice_chat_mode is not available/
  ],
  [
    'yolo-session.ts Shift+click zap',
    'config.set',
    { key: 'yolo', scope: 'global', value: '1' },
    /yolo \(global\) is not available/
  ],
  ['yolo-session.ts session toggle', 'config.set', { key: 'yolo', session_id: 's', value: '1' }, ['config.set']],
  [
    'session-tile-delegate / quick entry / bot chats (identityless)',
    'prompt.submit',
    { session_id: 's', text: 'hi' },
    ['prompt.submit']
  ],
  [
    'rewind.ts edit / regenerate / restore',
    'prompt.submit',
    {
      session_id: 's',
      text: 'again',
      confirm_truncate: true,
      truncate_before_row_id: 9,
      confirm_empty_truncate: true,
      rebind_survivor_row_ids: [3]
    },
    ['session.mutate', 'prompt.submit']
  ],
  [
    'use-prompt-actions /handoff',
    'handoff.request',
    { session_id: 's', platform: 'telegram' },
    /handoff\.request is not available/
  ],
  ['use-prompt-actions handoff poll', 'handoff.state', { session_id: 's' }, /handoff\.state is not available/],
  [
    'use-preview-routing restart',
    'preview.restart',
    { session_id: 's', url: 'http://x' },
    /preview\.restart is not available/
  ],
  [
    'connector-tool.tsx in-chat connect',
    'connectors.connect',
    { connectors: ['gh'], owner: { session_id: 's', type: 'session' } },
    /connectors\.connect is not available/
  ],
  [
    'connectors rpc.ts account connect',
    'connectors.connect',
    { connectors: ['gh'], owner: { type: 'account' } },
    ['connectors.connect']
  ]
]

test.each(SITES)('%s: the canonical frame fits the generated contract', async (_label, method, params, expected) => {
  const wsPackage = 'ws'
  const { WebSocketServer } = await import(wsPackage)
  const server = new WebSocketServer({ host: '127.0.0.1', port: 0 })
  await new Promise<void>(resolve => server.once('listening', resolve))
  const sent: any[] = []
  server.on('connection', (socket: any) => {
    socket.on('message', (bytes: Buffer) => {
      const frame = JSON.parse(bytes.toString())
      sent.push(frame)
      const fields = problems(frame.method, frame.params)
      const error = { code: 4001, message: 'invalid_params', data: { reason: 'invalid_params', fields } }
      socket.send(
        JSON.stringify(
          fields.length
            ? { jsonrpc: '2.0', id: frame.id, error }
            : { jsonrpc: '2.0', id: frame.id, result: answer(frame) }
        )
      )
    })
  })
  const client = new HermesGateway()

  try {
    const address = server.address() as { port: number }
    await client.connect(`ws://127.0.0.1:${address.port}/api/ws?native_dial=fixture&ticket=one-use`)
    const call = client.request(method, params)

    if (expected instanceof RegExp) {
      await expect(call).rejects.toThrow(expected)
      expect(sent.filter(frame => frame.method !== 'session.resume')).toEqual([])
    } else {
      await call
      expect(sent.map(frame => frame.method).filter(name => name !== 'session.resume')).toEqual(expected)
      expect(sent.map(frame => [frame.method, problems(frame.method, frame.params)])).toEqual(
        sent.map(frame => [frame.method, []])
      )
    }

    if (method === 'prompt.submit' && Array.isArray(expected)) {
      const submit = sent.at(-1).params
      expect(submit).toMatchObject({ session_id: 's', text: params.text })
      expect(typeof submit.submission_id).toBe('string')
    }

    if (Array.isArray(expected) && expected[0] === 'session.mutate') {
      expect(sent.find(frame => frame.method === 'session.mutate').params).toMatchObject({
        session_id: 's',
        operation: 'rewind',
        payload: { target_message_id: 9 },
        expected_revision: 4,
        expected_generation: 2
      })
    }
  } finally {
    client.close()

    for (const socket of server.clients) {
      socket.terminate()
    }

    await new Promise<void>(resolve => server.close(() => resolve()))
  }
})
