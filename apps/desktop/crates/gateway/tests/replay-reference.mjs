// Generate or verify vectors by EXECUTING the original shared client, using only
// synthetic events and credentials. No server, account or provider is contacted.
import { readFileSync, writeFileSync } from 'node:fs'
import { createHash } from 'node:crypto'
import { pathToFileURL } from 'node:url'
const [sourcePath, fixturePath, mode = '--verify'] = process.argv.slice(2)
const { JsonRpcGatewayClient } = await import(pathToFileURL(sourcePath).href)
const fixtures = JSON.parse(readFileSync(fixturePath, 'utf8'))
class FakeSocket extends EventTarget {
  static OPEN = 1
  static instances = []
  readyState = 0
  sent = []
  answered = new Set()
  constructor(url) { super(); this.url = url; FakeSocket.instances.push(this) }
  send(text) { this.sent.push(JSON.parse(text)) }
  close() { this.readyState = 3; this.dispatchEvent(new Event('close')) }
  open() { this.readyState = 1; this.dispatchEvent(new Event('open')) }
  frame(frame) { this.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(frame) })) }
}
globalThis.WebSocket = FakeSocket
const flush = async () => { for (let i = 0; i < 16; i++) await Promise.resolve() }
const lastLabel = event => `${event?.session_id ?? ''}:${event?.type ?? ''}:${event?.seq ?? ''}`
async function run(vector) {
  FakeSocket.instances = []
  const events = [], barriers = [], openRequests = []
  const client = new JsonRpcGatewayClient({ socketFactory: url => new FakeSocket(url), heartbeatIntervalMs: 0, heartbeatDeadlineMs: 0, connectTimeoutMs: 1000, replay: vector.replay !== false })
  client.onAny(event => events.push(JSON.parse(JSON.stringify(event))))
  client.onRequest(request => { openRequests.push({ id: request.id, method: request.method, params: request.params, replayed: request.replayed === true }) })
  for (const step of vector.steps) {
    const sock = FakeSocket.instances.at(step.socket ?? -1)
    if (step.op === 'open') { const opening = client.connect('ws://synthetic.invalid'); FakeSocket.instances.at(-1).open(); await opening }
    else if (step.op === 'drop') client.invalidate('synthetic disconnect')
    else if (step.op === 'event') sock.frame({ jsonrpc: '2.0', method: 'event', params: step.event })
    else if (step.op === 'answer' || step.op === 'error') {
      const request = sock.sent.find(request => request.method === 'session.events.since' && request.params.session_id === step.session_id && !sock.answered.has(request.id))
      if (!request) throw new Error(`No replay request for ${vector.name}/${step.session_id}`)
      sock.answered.add(request.id)
      sock.frame({ jsonrpc: '2.0', id: request.id, [step.op === 'error' ? 'error' : 'result']: step.op === 'error' ? step.error : step.result })
    } else if (step.op === 'barrier') {
      const barrier = client.sessionReplayBarrier(step.session_id)
      if (!barrier) barriers.push({ label: step.label, absent: true })
      else void barrier.then(valid => barriers.push({ label: step.label, valid, event_count: events.length, last_event: lastLabel(events.at(-1)) }))
    } else if (step.op === 'flood') {
      for (let index = 0; index < step.count; index++) {
        sock.frame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: `s${index}`, seq: 1, replay_epoch: `e${index}` } })
        if (step.touch_every && index % step.touch_every === 0) sock.frame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: 's0', seq: index + 2, replay_epoch: 'e0' } })
      }
    } else throw new Error(`Unknown step ${step.op}`)
    await flush()
  }
  const result = { events, watermarks: client.getSeqWatermarks(), session_epochs: Object.fromEntries(client.replayEpochBySession), replay_requests: FakeSocket.instances.flatMap(socket => socket.sent.filter(request => request.method === 'session.events.since').map(request => request.params)), barriers, open_requests: openRequests }
  client.close(); await flush()
  return result
}
for (const vector of fixtures.vectors) {
  const result = await run(vector)
  if (mode === '--write') vector.expected = result
  else if (JSON.stringify(result) !== JSON.stringify(vector.expected)) throw new Error(`Original TypeScript behavior changed: ${vector.name}`)
}
const hash = createHash('sha256').update(readFileSync(sourcePath)).digest('hex')
if (mode === '--write') { fixtures.upstream_sha256 = hash; writeFileSync(fixturePath, JSON.stringify(fixtures, null, 2) + '\n') }
else if (fixtures.upstream_sha256 !== hash) throw new Error('Original shared gateway source hash changed; regenerate and review vectors')
console.log(`Verified ${fixtures.vectors.length} original TypeScript replay vectors (${hash.slice(0, 12)}).`)
