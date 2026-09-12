import { expect, test } from 'vitest'

import { CanonicalDesktopProtocol } from './canonical-protocol'

test('corrections preserve intent and fence the observed session generation', () => {
  const protocol = new CanonicalDesktopProtocol()

  for (const method of ['session.redirect', 'session.steer']) {
    expect(() => protocol.prepare(method, { session_id: 's', text: 'correction' })).toThrow('execution identity')
  }

  protocol.event({ type: 'message.start', session_id: 's', payload: { execution_generation: 4 } })
  protocol.event({ type: 'message.start', session_id: 'other', payload: { execution_generation: 9 } })

  for (const method of ['session.redirect', 'session.steer']) {
    const params = protocol.prepare(method, { session_id: 's', text: 'correction' })
    expect(params).toEqual({ session_id: 's', text: 'correction', execution_generation: 4 })
    expect(protocol.wire(method, params)).toBe(method)
    expect(protocol.prepare(method, { ...params, execution_generation: 2 }).execution_generation).toBe(2)
  }
})

test('metadata writes retain the original CAS revision and request identity until acknowledgement', () => {
  const protocol = new CanonicalDesktopProtocol()
  protocol.result('session.resume', { session_id: 's' }, { session_id: 's', revision: 7 })
  const first = protocol.prepare('session.title', { session_id: 's', title: 'new title' })
  expect(first).toEqual({ session_id: 's', request_id: expect.any(String), expected_revision: 7, operation: 'rename', payload: { title: 'new title' } })
  protocol.result('session.resume', { session_id: 's' }, { session_id: 's', revision: 8 })
  expect(protocol.prepare('session.title', { session_id: 's', title: 'new title' })).toEqual(first)
  protocol.failure(first, { data: { reason: 'revision_conflict' } })
  const retry = protocol.prepare('session.title', { session_id: 's', title: 'new title' })
  expect(retry.expected_revision).toBe(8)
  expect(retry.request_id).not.toBe(first.request_id)
  protocol.result('session.title', first, { session_id: 's', revision: 8, title: 'new title' })
  expect(protocol.prepare('session.archive', { session_id: 's', archived: true })).toMatchObject({ expected_revision: 8, operation: 'archive', payload: { archived: true } })
})

test('canonical create retries retain identity and reject unsupported explicit intent', () => {
  const protocol = new CanonicalDesktopProtocol()
  const params = { source: 'desktop', profile: 'default', cols: 96, fast: false, cwd: '/tmp' }
  const first = protocol.prepare('session.create', params)
  expect(first).toMatchObject({ source: 'gui', cwd: '/tmp', request_id: expect.any(String) })
  expect(protocol.prepare('session.create', { ...params })).toEqual(first)
  expect(first).not.toHaveProperty('profile')
  expect(() => protocol.prepare('session.create', { ...params, fast: true })).toThrow('fast')
  expect(() => protocol.prepare('session.create', { ...params, provider: 'custom' })).toThrow('provider')
  protocol.result('session.create', first, { session_id: 'canonical', stored_session_id: 'canonical' })
  expect(protocol.prepare('session.create', params).request_id).not.toBe(first.request_id)
})

test('canonical receipts preserve admission identity and replay restores fenced controls', () => {
  const protocol = new CanonicalDesktopProtocol()
  const receipt = protocol.result('prompt.submit', { session_id: 's', submission_id: 'input' }, { admission_id: 'admission', ref: { session_id: 's', profile_id: '/tmp/profile' }, status: 'queued' })
  expect(receipt).toMatchObject({ admission_id: 'admission', submission_id: 'input', session_id: 's' })
  expect(() => protocol.result('prompt.submit', { session_id: 'other', submission_id: 'input' }, receipt)).toThrow('destination')
  const resumed = protocol.result('session.resume', { session_id: 's' }, { session_id: 's', stored_session_id: 's', execution_generation: 3, prompts: [{ kind: 'approval', prompt_id: 'p', execution_generation: 3, command: 'test' }] })
  expect(resumed.pending_approval).toMatchObject({ request_id: 'p', execution_generation: 3 })
  expect(protocol.prepare('approval.respond', { session_id: 's', request_id: 'p', choice: 'once' })).toEqual({ session_id: 's', execution_generation: 3, prompt_id: 'p', choice: 'once' })
  expect(protocol.prepare('session.interrupt', { session_id: 's' })).toEqual({ session_id: 's', execution_generation: 3 })
  protocol.event({ type: 'message.start', session_id: 's', payload: { execution_generation: 4 } })
  expect(() => protocol.prepare('approval.respond', { session_id: 's', request_id: 'p', choice: 'once' })).toThrow('stale')
})

test('pending resume and live updates share the existing queue projection', () => {
  const protocol = new CanonicalDesktopProtocol()
  const pending = [{ admission_id: 'a', input_id: 'original', text: 'queued text', status: 'queued' }]
  const resumed = protocol.result('session.resume', {}, { session_id: 's', pending })
  const event = { type: 'session.info', session_id: 's', payload: { pending } }
  protocol.event(event)
  expect(resumed.info.pending_submissions).toEqual([{ ...pending[0], user: 'queued text' }])
  expect(event.payload).toHaveProperty('pending_submissions', resumed.info.pending_submissions)
  const cleared = { type: 'session.info', session_id: 's', payload: { pending: [] } }
  protocol.event(cleared)
  expect(cleared.payload).toHaveProperty('pending_submissions', [])
})

test('composer branch and model switch become canonical prepared mutations with identity retained', () => {
  const protocol = new CanonicalDesktopProtocol()
  protocol.result('session.resume', { session_id: 's' }, { session_id: 's', revision: 4, execution_generation: 9 })
  const branch = protocol.prepare('session.branch', { session_id: 's' })
  expect(protocol.wire('session.branch')).toBe('session.mutate')
  expect(branch).toEqual({ session_id: 's', request_id: expect.any(String), expected_revision: 4, expected_generation: 9, operation: 'branch', payload: {} })
  expect(protocol.prepare('session.branch', { session_id: 's' })).toEqual(branch)
  const branched = protocol.result('session.branch', branch, { session_id: 's', revision: 5, operation: 'branch', branched_session_id: 'child', copied_messages: 6 })
  expect(branched).toMatchObject({ session_id: 'child', stored_session_id: 'child', parent_session_id: 's' })
  expect(protocol.prepare('session.branch', { session_id: 's' }).request_id).not.toBe(branch.request_id)

  const model = protocol.prepare('slash.exec', { session_id: 's', command: 'model switched' })
  expect(protocol.wire('slash.exec', model)).toBe('session.mutate')
  expect(model).toMatchObject({ session_id: 's', expected_revision: 5, operation: 'model', payload: { model: 'switched' } })
  expect(protocol.result('slash.exec', model, { session_id: 's', revision: 6, operation: 'model', model: 'switched', provider: 'custom' })).toMatchObject({ type: 'exec', output: expect.stringContaining('switched') })
  expect(protocol.wire('slash.exec', protocol.prepare('slash.exec', { session_id: 's', command: 'help' }))).toBe('slash.exec')
})

test('a session.info fanout after a turn refreshes the CAS revision the next mutation presents', () => {
  const protocol = new CanonicalDesktopProtocol()
  protocol.result('session.resume', { session_id: 's' }, { session_id: 's', revision: 2, execution_generation: 1 })
  protocol.event({ type: 'session.info', session_id: 's', payload: { pending: [], running: false, execution_generation: 2, revision: 5 } })
  expect(protocol.prepare('session.branch', { session_id: 's' })).toMatchObject({ expected_revision: 5, expected_generation: 2 })
})

test('acknowledging a turn lost across a restart presents the unknown row\'s own generation', () => {
  const protocol = new CanonicalDesktopProtocol()
  protocol.result('session.resume', { session_id: 's' }, { session_id: 's', stored_session_id: 's', execution_generation: 9, revision: 2,
    pending: [{ admission_id: 'lost', status: 'unknown', execution_generation: 4, text: 'LOST' }, { admission_id: 'next', status: 'queued', execution_generation: null, text: 'NEXT' }] })
  expect(protocol.prepare('prompt.resolve_unknown', { session_id: 's', admission_id: 'lost' }))
    .toEqual({ session_id: 's', admission_id: 'lost', execution_generation: 4 })
  expect(() => protocol.prepare('prompt.resolve_unknown', { session_id: 's', admission_id: 'next' })).toThrow('unknown')

  const receipt = protocol.result('prompt.resolve_unknown', { session_id: 's', admission_id: 'lost', execution_generation: 4 },
    { admission_id: 'lost', ref: { session_id: 's', profile_id: '/tmp/profile' }, status: 'terminal', outcome: 'interrupted' })

  expect(receipt).toMatchObject({ admission_id: 'lost', session_id: 's', status: 'terminal' })
  protocol.event({ type: 'session.info', session_id: 's', payload: { pending: [{ admission_id: 'next', status: 'started', execution_generation: 10, text: 'NEXT' }], execution_generation: 10 } })
  expect(() => protocol.prepare('prompt.resolve_unknown', { session_id: 's', admission_id: 'lost' })).toThrow('unknown')
})

test('the dedicated compress action becomes a fenced canonical mutation and settles with the resumed transcript', async () => {
  const protocol = new CanonicalDesktopProtocol()
  protocol.result('session.resume', { session_id: 's' }, { session_id: 's', revision: 4, execution_generation: 9 })

  const prepared = protocol.prepare('session.compress', { session_id: 's', focus_topic: 'billing' })
  expect(prepared).toEqual({ session_id: 's', request_id: expect.any(String), expected_revision: 4, expected_generation: 9, operation: 'compress', payload: { focus: 'billing' } })
  expect(protocol.prepare('session.compress', { session_id: 's' }).payload).toEqual({})
  expect(protocol.wire('session.compress', prepared)).toBe('session.mutate')
  expect(() => protocol.result('session.compress', prepared, { session_id: 'other', revision: 5, operation: 'compress' })).toThrow('destination')

  const calls: Array<[string, Record<string, unknown>]> = []
  const resumed = { session_id: 's', messages: [{ role: 'assistant', content: 'summary' }], info: { title: 'T' } }

  const fakeRequest = async (method: string, params: Record<string, unknown>) => { calls.push([method, params]);

 return resumed }

  const receipt = protocol.result('session.compress', prepared, { session_id: 's', revision: 5, operation: 'compress', target_session_id: 's', message_count: 3 })
  const settled = await protocol.settle('session.compress', prepared, receipt, fakeRequest)

  expect(settled.messages).toEqual(resumed.messages)
  expect(settled.info).toEqual(resumed.info)
  expect(settled.host_ack.output).toContain('3')
  expect(calls).toEqual([['session.resume', { session_id: 's' }]])
  // Acknowledged: the next compress presents the refreshed revision under a new request id.
  expect(protocol.prepare('session.compress', { session_id: 's', focus_topic: 'billing' })).toMatchObject({ expected_revision: 5 })
  expect(protocol.prepare('session.compress', { session_id: 's', focus_topic: 'billing' }).request_id).not.toBe(prepared.request_id)

  const untouched = { session_id: 's', ok: true }
  expect(await protocol.settle('session.title', { session_id: 's' }, untouched, fakeRequest)).toBe(untouched)
  expect(calls).toHaveLength(1)
})
