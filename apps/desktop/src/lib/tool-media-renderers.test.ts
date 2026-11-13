import { describe, expect, it, vi } from 'vitest'

import {
  claimPatternOwned,
  effectiveToolCall,
  findProducingToolCall,
  namesWholePath,
  ownerOfTool,
  refusedClaimPatterns,
  resolveToolMediaClaim,
  type ToolOwnership
} from './tool-media-renderers'

const call = (toolName: string, result: unknown, extra: Record<string, unknown> = {}) => ({
  type: 'tool-call',
  toolCallId: `${toolName}-id`,
  toolName,
  args: {},
  result,
  ...extra
})

const turn = (...content: unknown[]) => [
  { id: 'u', role: 'user', content: [{ type: 'text', text: 'make it' }] },
  { id: 'a', role: 'assistant', content }
]

describe('findProducingToolCall', () => {
  const path = '/tmp/out/model.glb'

  it('picks the earliest call whose result names the path (a later ls or a plugin reading it only mentions it)', () => {
    const messages = [
      { id: 'u0', role: 'user', content: [] },
      { id: 'a0', role: 'assistant', content: [call('mcp__meshy__download', { path })] },
      { id: 'u1', role: 'user', content: [] },
      { id: 'a1', role: 'assistant', content: [call('sfx_analyze', `read ${path}`)] },
      { id: 'a2', role: 'assistant', content: [call('terminal', { output: path }), { type: 'text', text: 'MEDIA' }] }
    ]

    // Re-delivered in a later turn, the file still belongs to the call that made it.
    expect(findProducingToolCall(messages, 'a2', path)?.toolName).toBe('mcp__meshy__download')
    expect(findProducingToolCall(messages.slice(2), 'a2', path)?.toolName).toBe('sfx_analyze')
    expect(findProducingToolCall([messages[2], messages[4]], 'a2', path)?.toolName).toBe('terminal')
    // Nothing after the delivering message counts.
    expect(findProducingToolCall(messages, 'u1', path)?.toolName).toBe('mcp__meshy__download')
  })

  it('matches whole paths only: a longer sibling file never claims this one', () => {
    const messages = turn(
      call('first', { path: `${path}.bak` }),
      call('nested', `copied /x${path}`),
      call('url', `https://cdn.example${path}`),
      call('second', `saved to ${path}.`)
    )

    expect(findProducingToolCall(messages, 'a', path)?.toolName).toBe('second')
    expect(namesWholePath(`MEDIA:${path}`, path)).toBe(true)
    expect(namesWholePath(`${path}: 3 MB`, path)).toBe(true)
    expect(namesWholePath(`${path}.bak`, path)).toBe(false)
  })

  it('reads every file of a result that joins several MCP media blocks with newlines', () => {
    // An MCP result is text inside a JSON string: the first path is followed by
    // the escaped `\n`, which must still end the path.
    const result = JSON.stringify({ result: 'Saved /out/a.glb\nSaved /out/b.glb\nSaved /out/c.png' })
    const messages = turn(call('mcp__meshy__download', result))

    for (const file of ['/out/a.glb', '/out/b.glb', '/out/c.png']) {
      expect(findProducingToolCall(messages, 'a', file)?.toolName).toBe('mcp__meshy__download')
    }

    // Stored again as an object (a reloaded transcript), the same holds.
    expect(
      findProducingToolCall(turn(call('mcp__meshy__download', JSON.parse(result))), 'a', '/out/a.glb')
    ).not.toBeNull()
  })

  it('keeps a core tool the producer when its output names the file before a newline', () => {
    const messages = turn(
      call('terminal', JSON.stringify({ output: 'wrote /out/a.mp3\nDone' })),
      call('sfx_analyze', { path: '/out/a.mp3' })
    )

    expect(findProducingToolCall(messages, 'a', '/out/a.mp3')?.toolName).toBe('terminal')
  })

  it('never credits a failed call, live or reloaded (a stored result is never marked isError)', () => {
    const messages = turn(
      call('sfx', { path }, { isError: true }),
      call('meshy_x', JSON.stringify({ error: `exists: ${path}` })),
      call('meshy_y', { success: false, message: `cannot write ${path}` }),
      call('image_generate', { path })
    )

    expect(findProducingToolCall(messages, 'a', path)?.toolName).toBe('image_generate')
    expect(findProducingToolCall(turn(call('ok', { error: null, path })), 'a', path)?.toolName).toBe('ok')
  })

  it('gives a file the user named first (an upload) no producer, even when a plugin echoes it', () => {
    const upload = '/home/u/uploads/voice.ogg'

    const messages = [
      { id: 'u', role: 'user', content: [{ type: 'text', text: `transcode ${upload}` }] },
      { id: 'a', role: 'assistant', content: [call('sfx_transcode', { input: upload })] }
    ]

    expect(findProducingToolCall(messages, 'a', upload)).toBeNull()
  })

  it('finds Windows, file URL and escaped non-ASCII spellings of the same path', () => {
    const win = 'C:\\Users\\me\\out\\clip.wav'

    expect(findProducingToolCall(turn(call('sfx', JSON.stringify({ win }))), 'a', win)?.toolName).toBe('sfx')
    expect(findProducingToolCall(turn(call('sfx', 'saved c:/users/ME/out/clip.wav')), 'a', win)?.toolName).toBe('sfx')
    expect(findProducingToolCall(turn(call('sfx', { win })), 'a', 'file:///C:/Users/me/out/clip.wav')).not.toBeNull()
    // Python's json.dumps escapes non-ASCII by default.
    expect(
      findProducingToolCall(turn(call('sfx', '{"path": "/out/caf\\u00e9.wav"}')), 'a', '/out/café.wav')
    ).not.toBeNull()
    expect(findProducingToolCall(turn(call('sfx', { win })), 'missing', win)).toBeNull()
  })
})

describe('effectiveToolCall', () => {
  it('unwraps a single-entry deferred-tool bridge call and leaves batches and plain calls alone', () => {
    const bridged = call('tool_call', {}, { args: { calls: [{ name: 'sfx_make', arguments: { prompt: 'zap' } }] } })
    const batch = call('tool_call', {}, { args: { calls: [{ name: 'a' }, { name: 'b' }] } })

    expect(effectiveToolCall(bridged)).toMatchObject({ args: { prompt: 'zap' }, toolName: 'sfx_make' })
    expect(effectiveToolCall(batch).toolName).toBe('tool_call')
    expect(effectiveToolCall(call('terminal', {})).toolName).toBe('terminal')
  })

  it('reads every shape the backend runs: legacy single, JSON-string calls, one object, string arguments', () => {
    const shapes = [
      { name: 'sfx_make', arguments: '{"prompt":"zap"}' },
      { calls: JSON.stringify([{ name: 'sfx_make', arguments: { prompt: 'zap' } }]) },
      { calls: { name: ' sfx_make ', arguments: { prompt: 'zap' } } }
    ]

    for (const args of shapes) {
      expect(effectiveToolCall(call('tool_call', {}, { args }))).toMatchObject({
        args: { prompt: 'zap' },
        toolName: 'sfx_make'
      })
    }

    // The bridge never names itself.
    expect(effectiveToolCall(call('tool_call', {}, { args: { name: 'tool_call' } })).toolName).toBe('tool_call')
  })
})

describe('ownership', () => {
  const owners = new Map<string, ToolOwnership>([
    ['sfx', { tools: new Set(['sfx_make']) }],
    ['meshy', { tools: new Set(['mcp__meshy__download', 'mcp__meshy__status']) }],
    ['nested', { tools: new Set(['mcp__meshy__x__gen']) }]
  ])

  it('resolves a tool to the one package that registered it', () => {
    expect(ownerOfTool('sfx_make', owners)).toBe('sfx')
    expect(ownerOfTool('mcp__meshy__download', owners)).toBe('meshy')
    expect(ownerOfTool('mcp__meshy__x__gen', owners)).toBe('nested')
    expect(ownerOfTool('text_to_speech', owners)).toBeNull()
  })

  it('counts a claim pattern as owned only when it names a tool the package registered', () => {
    expect(claimPatternOwned('mcp__meshy__*', owners.get('meshy'))).toBe(true)
    expect(claimPatternOwned('text_to_speech', owners.get('sfx'))).toBe(false)
    expect(claimPatternOwned('mcp__other__*', owners.get('meshy'))).toBe(false)
    expect(claimPatternOwned('sfx_make', undefined)).toBe(false)
  })

  it('serves the first claim whose package owns the tool and reports every other match', () => {
    const render = () => null

    const packages: Record<string, string | null> = {
      'plugin:meshy': 'meshy',
      'plugin:thief': 'thief',
      'plugin:solo': null
    }

    const claims = [
      { id: 'thief:x', source: 'plugin:thief', data: { tools: ['mcp__meshy__*'], render } },
      { id: 'solo:x', source: 'plugin:solo', data: { tools: ['mcp__meshy__download'], render } },
      { id: 'meshy:x', source: 'plugin:meshy', data: { tools: ['mcp__meshy__*'], render } }
    ]

    const onRefused = vi.fn()
    const packageOf = (source?: string) => (source ? (packages[source] ?? null) : null)

    expect(resolveToolMediaClaim(claims, 'mcp__meshy__download', { onRefused, owners, packageOf })?.id).toBe('meshy:x')
    expect(onRefused.mock.calls.map(([c]) => c.id)).toEqual(['thief:x', 'solo:x'])
    // Another package's tool is never served through meshy's wildcard.
    expect(resolveToolMediaClaim(claims, 'mcp__meshy__x__gen', { owners, packageOf })).toBeNull()
    expect(refusedClaimPatterns(claims, owners, packageOf).map(r => r.claim.id)).toEqual(['thief:x', 'solo:x'])
  })
})
