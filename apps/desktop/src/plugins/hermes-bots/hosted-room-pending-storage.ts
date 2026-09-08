import type { PluginStorage } from '@hermes/plugin-sdk'

import type { PendingInput } from './hosted-room-client'

interface StoredInput extends Omit<PendingInput, 'attachments'> {
  attachments?: Array<Omit<NonNullable<PendingInput['attachments']>[number], 'data'> & { data: Blob }>
}

// Keep immutable bytes out of subsequent metadata writes (notably the staged manifest).
const blobs = new WeakMap<object, { data: string; blob: Blob }>()

async function transaction<T>(run: (store: IDBObjectStore, done: (value: T) => void) => void): Promise<T> {
  const db = await new Promise<IDBDatabase>((resolve, reject) => {
    const opening = indexedDB.open('hermes.plugin.hermes-bots.hosted-inputs', 1)
    opening.onupgradeneeded = () => opening.result.createObjectStore('pending')
    opening.onerror = () => reject(opening.error)
    let blocked = false

    opening.onblocked = () => {
      blocked = true
      reject(new Error('Hosted input storage is blocked. Close other Desktop windows and retry.'))
    }

    opening.onsuccess = () => blocked ? opening.result.close() : resolve(opening.result)
  })

  try {
    return await new Promise<T>((resolve, reject) => {
      const tx = db.transaction('pending', 'readwrite', { durability: 'strict' })
      let result: T
      tx.oncomplete = () => resolve(result)
      tx.onabort = () => reject(tx.error || new Error('Hosted input persistence aborted'))
      tx.onerror = () => reject(tx.error)

      try { run(tx.objectStore('pending'), value => { result = value }) }
      catch (error) { tx.abort(); reject(error) }
    })
  } finally { db.close() }
}

function validate(pending: PendingInput | StoredInput): void {
  if (!pending || typeof pending.eventId !== 'string' || !pending.eventId || typeof pending.threadId !== 'string' ||
      !pending.threadId || typeof pending.text !== 'string' ||
      (pending.attachments !== undefined && !Array.isArray(pending.attachments))) {
    throw new Error('Invalid saved hosted input. Recovery storage was not overwritten.')
  }
}

async function encodeInWorker<T>(data: string[] | string): Promise<T> {
  const worker = new Worker(new URL('./hosted-room-pending-blobs.worker.ts', import.meta.url), { type: 'module' })

  try {
    return await new Promise<T>((resolve, reject) => {
      worker.onmessage = event => resolve(event.data)
      worker.onerror = () => reject(new Error('Hosted input byte persistence or legacy recovery failed'))
      worker.onmessageerror = () => reject(new Error('Hosted input byte persistence failed'))
      worker.postMessage(data)
    })
  } finally { worker.terminate() }
}

async function encode(pending: PendingInput): Promise<StoredInput> {
  validate(pending)

  const missing = (pending.attachments || []).filter(attachment => {
    if (typeof attachment.data !== 'string') {throw new Error('Saved hosted attachment bytes are unavailable')}

    return blobs.get(attachment)?.data !== attachment.data
  })

  if (missing.length) {
    const encoded = await encodeInWorker<Blob[]>(missing.map(attachment => attachment.data))
    missing.forEach((attachment, index) => blobs.set(attachment, { data: attachment.data, blob: encoded[index] }))
  }

  const { attachments, ...metadata } = pending

  return { ...metadata, ...(attachments ? { attachments: attachments.map(attachment =>
    ({ ...attachment, data: blobs.get(attachment)!.blob })) } : {}) }
}

export async function writePendingInput(key: string, pending: PendingInput | null): Promise<void> {
  const value = pending && await encode(pending)
  await transaction<void>((store, done) => {
    store.put(value, key)
    done(undefined)
  })
}

export async function readPendingInput(key: string, legacy: PluginStorage): Promise<PendingInput | null> {
  let stored = await transaction<StoredInput | null | undefined>((store, done) => {
    const request = store.get(key)
    request.onsuccess = () => done(request.result)
  })

  if (stored === undefined) {
    // localStorage has no asynchronous read API. Read shipped data only until a
    // strict IDB commit succeeds; never delete the only recoverable copy first.
    const raw = legacy.getRaw(`hosted-input:${key}`)
    const migrated = raw === null ? null : await encodeInWorker<StoredInput | null>(raw)

    if (migrated !== null) {validate(migrated)}
    stored = await transaction<StoredInput | null>((store, done) => {
      const request = store.get(key)

      request.onsuccess = () => {
        const value = request.result === undefined ? migrated : request.result

        if (request.result === undefined) {store.put(value, key)}
        done(value)
      }
    })
    // A committed null is a tombstone: even failed legacy cleanup cannot resurrect input.
    await legacy.remove(`hosted-input:${key}`)
  }

  if (stored === null) {return null}

  validate(stored)

  if (!stored.attachments) {return stored as PendingInput}

  return { ...stored, attachments: await Promise.all(stored.attachments.map(async attachment => {
    if (!(attachment.data instanceof Blob)) {throw new Error('Saved hosted attachment bytes are unavailable')}
    const data = await attachment.data.text()
    const restored = { ...attachment, data }
    blobs.set(restored, { data, blob: attachment.data })

    return restored
  })) }
}
