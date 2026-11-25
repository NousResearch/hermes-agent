import { CanonicalBatchAdapter, type ServerRequest } from '@hermes/shared'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { I18nProvider } from '@/i18n'
import { normalizeQuestions, clearClarifyRequest } from '@/store/clarify'
import { $gateway } from '@/store/gateway'
import { resetServerRequestsForTests, rememberServerRequest } from '@/store/server-requests'
import { ClarifyToolPending } from './pending'

afterEach(() => {
  cleanup()
  clearClarifyRequest()
  resetServerRequestsForTests()
  $gateway.set(null)
})

test('existing whole-set panel sends sequential typed locks even when withdrawal precedes final acknowledgement', async () => {
  const identity = { profile_id: '/profiles/work', authority_epoch: 1, stream_id: 'stream', execution_generation: 1 }
  const owner = { sessionId: 's', profile: 'work', profileId: '/profiles/work' }
  const batch = {
    request_id: 'srq-b',
    identity,
    questions: [
      { qid: 'q0', question: 'Which target?', choices: ['staging', 'production'], multi_select: false },
      { qid: 'q1', question: 'Which region?', choices: ['us', 'eu'], multi_select: false }
    ],
    answers: {},
    replayed: false
  }
  let card: ServerRequest | undefined
  const calls: Array<{ method: string; params: Record<string, unknown> }> = []
  const done = vi.fn(),
    withdraw = vi.fn()
  const adapter = new CanonicalBatchAdapter({
    deliver: r => {
      card = r
    },
    withdraw,
    recover: async () => {},
    rpc: async <T,>(method: string, params: Record<string, unknown>): Promise<T> => {
      calls.push({ method, params })
      if (calls.length === 2) adapter.cancel(owner, batch.request_id, identity)
      return { status: 'ok', remaining: calls.length === 1 ? ['q1'] : [] } as T
    }
  })
  adapter.open(owner, batch, false)
  rememberServerRequest(card!)
  $gateway.set({
    request: (_method: string, p: Record<string, unknown>) => adapter.lock(p.request_id, p.question_id, p.answer)
  } as never)
  render(
    <I18nProvider configClient={null} initialLocale="en">
      <ClarifyToolPending
        onAnswered={done}
        undelivered={false}
        request={{ requestId: card!.id, sessionId: 's', questions: normalizeQuestions(card!.params.questions) }}
      />
    </I18nProvider>
  )
  expect(screen.getAllByRole('button', { name: /Confirm and continue/ })).toHaveLength(1)
  fireEvent.click(screen.getByRole('button', { name: /staging/ }))
  fireEvent.click(screen.getByRole('button', { name: /eu/ }))
  fireEvent.click(screen.getByRole('button', { name: /Confirm and continue/ }))
  await waitFor(() => expect(done).toHaveBeenCalledTimes(1))
  expect(calls.map(c => [c.method, c.params.question_id, c.params.answer])).toEqual([
    ['clarify.batch.lock', 'q0', 'staging'],
    ['clarify.batch.lock', 'q1', 'eu']
  ])
  expect(
    calls.every(c => c.params.request_id === 'srq-b' && JSON.stringify(c.params.identity) === JSON.stringify(identity))
  ).toBe(true)
})


test.each(['reconnect', 'replacement', 'adapter replacement'])('retained existing panel submit is fenced after %s; replacement panel remains usable', async mode => {
  const owner = {sessionId:'s',profile:'work',profileId:'/profiles/work'}
  const batch = {request_id:'retained',identity:{profile_id:owner.profileId,authority_epoch:1,stream_id:'stream',execution_generation:1},
    questions:[{qid:'q0',question:'Target?',choices:['staging','production'],multi_select:false}],answers:{},replayed:false}
  const cards: ServerRequest[] = [], writes: unknown[] = []
  const makeAdapter = () => new CanonicalBatchAdapter({deliver:r => {cards.push(r);rememberServerRequest(r)},withdraw:()=>{},recover:async()=>{},
    rpc:async <T,>(method:string,params:Record<string,unknown>):Promise<T> => {writes.push({method,params});return {status:'ok',remaining:[]} as T}})
  let adapter = makeAdapter()
  $gateway.set({request:(_method:string,p:Record<string,unknown>)=>adapter.lock(p.request_id,p.question_id,p.answer)} as never)
  adapter.open(owner,batch,false)
  const mount = (card:ServerRequest) => render(<I18nProvider configClient={null} initialLocale="en"><ClarifyToolPending
    onAnswered={()=>{}} undelivered={false} request={{requestId:card.id,sessionId:'s',questions:normalizeQuestions(card.params.questions)}} /></I18nProvider>)
  const old = mount(cards[0])
  fireEvent.click(screen.getByRole('button',{name:/staging/}))
  // Keep this exact component mounted, with its real captured submit callback.
  if(mode==='reconnect') adapter.reset()
  if(mode==='adapter replacement') adapter = makeAdapter()
  adapter.snapshot(owner,[{...batch,answers:{q0:'production'},replayed:true}],adapter.checkpoint())
  expect(cards).toHaveLength(2)
  await act(async () => {
    fireEvent.click(screen.getByRole('button',{name:/Confirm and continue/}))
    await new Promise(r=>setTimeout(r,30))
  })
  expect(writes).toHaveLength(0)
  old.unmount()
  mount(cards[1])
  fireEvent.click(screen.getByRole('button',{name:/staging/}))
  fireEvent.click(screen.getByRole('button',{name:/Confirm and continue/}))
  await waitFor(()=>expect(writes).toHaveLength(1))
})
