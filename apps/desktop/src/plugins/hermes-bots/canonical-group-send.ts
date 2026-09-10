import type { CanonicalGroupBinding } from './canonical-groups'

const STORAGE_KEY = 'hermes.desktop.canonicalGroupSends.v1'

export interface PreparedCanonicalGroupSend {
  binding: CanonicalGroupBinding
  params: {
    room_id: string
    event_id: string
    payload: Record<string, unknown>
  }
}

function journalKey(binding: CanonicalGroupBinding): string {
  if (![binding.connectionId, binding.profile, binding.roomId].every(value => typeof value === 'string' && value.trim())) {
    throw new Error('Canonical group Send requires an explicit connection, profile and room')
  }

  // A separate namespace inside the existing origin-scoped native journal.
  return JSON.stringify(['canonical-group-send-v1', binding.connectionId, binding.profile, binding.roomId])
}

async function readJournal(): Promise<Record<string, PreparedCanonicalGroupSend>> {
  const native = window.hermesDesktop?.preparedSubmissions
  const parsed: unknown = JSON.parse(native
    ? await native.read()
    : window.localStorage.getItem(STORAGE_KEY) || '{}')

  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('Invalid canonical group Send journal')
  }

  return parsed as Record<string, PreparedCanonicalGroupSend>
}

async function update(key: string, entry: PreparedCanonicalGroupSend | null): Promise<void> {
  const native = window.hermesDesktop?.preparedSubmissions

  if (native) {
    // Await private atomic-file publication before the caller may send. Never
    // downgrade a native write failure to Chromium's deferred localStorage.
    await native.update(key, entry === null ? null : JSON.stringify(entry))

    return
  }

  // Browser-only fallback guarantees reload recovery, not process-crash safety.
  const journal = await readJournal()
  if (entry === null) {delete journal[key]}
  else {journal[key] = entry}
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(journal))
}

export async function readCanonicalGroupSend(binding: CanonicalGroupBinding): Promise<PreparedCanonicalGroupSend | undefined> {
  const key = journalKey(binding)
  const entry = (await readJournal())[key]

  if (entry && (journalKey(entry.binding) !== key || entry.params.room_id !== binding.roomId ||
    !entry.params.event_id || !entry.params.payload || typeof entry.params.payload !== 'object')) {
    throw new Error('Invalid canonical group Send entry')
  }

  return entry
}

// One unresolved intent per room. This is not an automatic outbox: callers show
// the recovered payload and explicitly retry it rather than silently replacing it.
export async function prepareCanonicalGroupSend(
  binding: CanonicalGroupBinding,
  payload: Record<string, unknown>
): Promise<PreparedCanonicalGroupSend> {
  const existing = await readCanonicalGroupSend(binding)
  if (existing) {return existing}

  const eventId = crypto.randomUUID()
  const entry: PreparedCanonicalGroupSend = JSON.parse(JSON.stringify({
    binding,
    params: {
      room_id: binding.roomId,
      event_id: eventId,
      payload: { ...payload, thread_id: payload.thread_id ?? eventId }
    }
  }))
  await update(journalKey(binding), entry)

  return entry
}

// Call only after a definitive groups.send ACK; timeout/unknown retains the
// exact event and payload. A delayed ACK cannot retire a newer intent.
export async function retireCanonicalGroupSend(binding: CanonicalGroupBinding, eventId: string): Promise<void> {
  const entry = await readCanonicalGroupSend(binding)
  if (entry?.params.event_id === eventId) {await update(journalKey(binding), null)}
}
