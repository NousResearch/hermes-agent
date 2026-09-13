import { getQueuedPrompts, type QueuedPromptEntry, writeSessionQueue } from './composer-queue'

const STORAGE_KEY = 'hermes.desktop.pendingSubmissions.v1'
interface PendingSubmission { id: string; text: string; displayText?: string; status?: string }
type Journal = Record<string, Record<string, PendingSubmission>>

const readJournal = (): Journal => {
  try { return JSON.parse(window.localStorage.getItem(STORAGE_KEY) || '{}') as Journal }
  catch { return {} }
}

// Not an outbox: an uncertain accepted input must never be replayed automatically.
export function trackPendingSubmission(key: string, entry: PendingSubmission): void {
  const journal = readJournal()
  journal[key] = { ...journal[key], [entry.id]: entry }
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(journal))
}

export function reconcilePendingSubmissions(key: string, value: unknown): void {
  if (!Array.isArray(value)) { return }
  const journal = readJournal()
  const known = journal[key] ?? {}
  const receipts = new Map<string, PendingSubmission>()
  const admissionByInput = new Map<string, string>()

  for (const raw of value) {
    if (!raw || typeof raw.admission_id !== 'string' || !['queued', 'started', 'unknown'].includes(raw.status)) { continue }
    const id = raw.admission_id

    if (typeof raw.input_id === 'string') { admissionByInput.set(raw.input_id, id) }
    receipts.set(id, { ...known[id], id, text: typeof raw.user === 'string' ? raw.user : known[id]?.text ?? '', status: raw.status })
  }

  const current = getQueuedPrompts(key)
  const next: QueuedPromptEntry[] = []

  for (const entry of current) {
    const receipt = receipts.get(admissionByInput.get(entry.id) ?? entry.id)

    if (receipt) {
      if (receipt.status !== 'started') { next.push({ ...entry, id: receipt.id, serverStatus: receipt.status }) }
      receipts.delete(receipt.id)
    } else if (!entry.serverStatus) { next.push(entry) }
  }

  for (const receipt of receipts.values()) {
    if (receipt.status !== 'started') {
      next.push({ id: receipt.id, text: receipt.text, displayText: receipt.displayText, attachments: [], queuedAt: Date.now(), serverStatus: receipt.status })
    }
  }

  // Only observed server records may be retired by their later absence.
  for (const [id, entry] of Object.entries(known)) {
    if (entry.status && !value.some(raw => raw?.admission_id === id)) { delete known[id] }
  }

  for (const raw of value) {
    if (typeof raw?.admission_id === 'string') { known[raw.admission_id] = { ...known[raw.admission_id], id: raw.admission_id, text: raw.user ?? known[raw.admission_id]?.text ?? '', status: raw.status } }
  }

  journal[key] = known
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(journal))

  if (JSON.stringify(current) !== JSON.stringify(next)) { writeSessionQueue(key, next) }
}
