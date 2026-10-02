import { createHash, randomUUID } from 'node:crypto'
import {
  closeSync,
  existsSync,
  fsyncSync,
  mkdirSync,
  openSync,
  readFileSync,
  renameSync,
  unlinkSync,
  writeFileSync
} from 'node:fs'
import path from 'node:path'

export interface ThoughtOwner {
  connectionId: string
  profile: string
}

export interface SavedThought {
  id: string
  text: string
  createdAt: string
}

export interface ThoughtDraft {
  id: string
  text: string
  handoffAttempted?: boolean
}

export interface ThoughtSnapshot {
  owner: ThoughtOwner
  token: string
  draft: ThoughtDraft
  thoughts: SavedThought[]
}

interface ThoughtFile {
  version: 1
  draft: ThoughtDraft
  thoughts: SavedThought[]
}

function validDraft(value: unknown): value is ThoughtDraft {
  if (!value || typeof value !== 'object') {
    return false
  }

  const draft = value as ThoughtDraft

  return (
    typeof draft.id === 'string' &&
    /^[a-zA-Z0-9-]{1,80}$/.test(draft.id) &&
    typeof draft.text === 'string' &&
    (draft.handoffAttempted === undefined || typeof draft.handoffAttempted === 'boolean') &&
    Buffer.byteLength(draft.text, 'utf8') <= 65536
  )
}

/** One writer in Electron; the renderer receives success only after file sync and rename. */
export class ThoughtCaptureStore {
  private readonly retiredOwners = new Set<string>()

  activateProfile(owner: ThoughtOwner): void {
    this.retiredOwners.delete(JSON.stringify(owner))
  }

  blockProfile(owner: ThoughtOwner): void {
    this.retiredOwners.add(JSON.stringify(owner))
  }

  assertActive(owner: ThoughtOwner): void {
    if (this.retiredOwners.has(JSON.stringify(owner))) {
      throw new Error('This profile was deleted. Select an existing profile before capturing thoughts.')
    }
  }

  constructor(private readonly directory: string) {}

  private scopeDirectory(owner: ThoughtOwner): string {
    const key = createHash('sha256')
      .update(JSON.stringify([owner.connectionId, owner.profile]))
      .digest('hex')

    return path.join(this.directory, key)
  }

  read(owner: ThoughtOwner): ThoughtFile {
    const filename = path.join(this.scopeDirectory(owner), 'saved.json')

    const value: ThoughtFile = existsSync(filename)
      ? JSON.parse(readFileSync(filename, 'utf8'))
      : { version: 1, draft: { id: randomUUID(), text: '' }, thoughts: [] }

    if (
      value?.version !== 1 ||
      !validDraft(value.draft) ||
      !Array.isArray(value.thoughts) ||
      value.thoughts.some(item => !validDraft(item) || typeof item.createdAt !== 'string')
    ) {
      throw new Error('Saved thoughts could not be read. Restore the local file before saving again.')
    }

    const draftFile = path.join(this.scopeDirectory(owner), 'draft.json')

    if (existsSync(draftFile)) {
      const draft: unknown = JSON.parse(readFileSync(draftFile, 'utf8'))

      if (!validDraft(draft)) {
        throw new Error('The local thought draft could not be read.')
      }

      // A committed save wins over its older draft, including after a process crash.
      if (!value.thoughts.some(item => item.id === draft.id)) {
        value.draft = draft
      }
    }

    return value
  }

  private write(owner: ThoughtOwner, name: string, value: unknown): void {
    mkdirSync(this.directory, { recursive: true, mode: 0o700 })
    const directory = this.scopeDirectory(owner)
    mkdirSync(directory, { recursive: true, mode: 0o700 })
    const filename = path.join(directory, name)
    const temporary = `${filename}.${randomUUID()}.tmp`

    try {
      const descriptor = openSync(temporary, 'wx', 0o600)

      try {
        writeFileSync(descriptor, JSON.stringify(value), 'utf8')
        fsyncSync(descriptor)
      } finally {
        closeSync(descriptor)
      }

      renameSync(temporary, filename)
    } finally {
      if (existsSync(temporary)) {
        unlinkSync(temporary)
      }
    }
  }

  draft(owner: ThoughtOwner, draft: ThoughtDraft): void {
    if (!validDraft(draft)) {
      throw new Error('Invalid thought or text exceeds 64 KiB.')
    }

    this.write(owner, 'draft.json', draft)
  }

  save(owner: ThoughtOwner, draft: ThoughtDraft): SavedThought {
    if (!validDraft(draft) || !draft.text.trim()) {
      throw new Error('Enter a thought before saving.')
    }

    const state = this.read(owner)
    const previous = state.thoughts.find(item => item.id === draft.id)

    if (previous) {
      if (previous.text !== draft.text) {
        throw new Error('This save identity already belongs to different text.')
      }

      return previous
    }

    const thought = { ...draft, createdAt: new Date().toISOString() }
    this.write(owner, 'saved.json', {
      version: 1,
      draft: { id: randomUUID(), text: '' },
      thoughts: [thought, ...state.thoughts]
    })

    return thought
  }

  retireProfile(owner: ThoughtOwner) {
    const source = this.scopeDirectory(owner)

    if (!existsSync(source)) {
      return null
    }

    const recoveryPath = `${source}.retired-${randomUUID()}`
    renameSync(source, recoveryPath)

    return {
      recoveryPath,
      restore: () => {
        if (existsSync(source)) {
          throw new Error(`Local thoughts remain in ${recoveryPath}; the original inbox already exists.`)
        }

        renameSync(recoveryPath, source)
      }
    }
  }

  renameProfile(connectionId: string, oldName: string, newName: string): void {
    const oldFile = this.scopeDirectory({ connectionId, profile: oldName })

    if (!existsSync(oldFile)) {
      return
    }

    const destination = this.scopeDirectory({ connectionId, profile: newName })

    // Never overwrite an existing inbox if the backend returns an unexpected rename.
    if (existsSync(destination)) {
      throw new Error(
        `Profile renamed, but local thoughts remain in ${oldFile}. Move that folder to ${destination} after resolving the existing inbox.`
      )
    }

    try {
      renameSync(oldFile, destination)
    } catch {
      throw new Error(
        `Profile renamed, but local thoughts could not move. They remain in ${oldFile}; restore this folder to ${destination} before capturing in the renamed profile.`
      )
    }
  }
}

export interface ThoughtCaptureIpc {
  handle(channel: string, listener: (event: { sender: { id: number } }, payload?: any) => unknown): void
}

export function registerThoughtCapture(
  ipc: ThoughtCaptureIpc,
  store: ThoughtCaptureStore,
  owner: () => ThoughtOwner | null,
  senderAllowed: (id: number) => boolean
) {
  let generation = randomUUID()

  const current = () => {
    const value = owner()

    if (!value) {
      throw new Error('Open a connection and profile in Hermes before saving thoughts.')
    }

    store.assertActive(value)

    return value
  }

  const token = (value: ThoughtOwner) => `${generation}:${JSON.stringify(value)}`

  const snapshot = (): ThoughtSnapshot => {
    const value = current()

    return { ...store.read(value), owner: value, token: token(value) }
  }

  const authorize = (id: number, expected?: string) => {
    if (!senderAllowed(id)) {
      throw new Error('Thought capture is available only in Quick Entry.')
    }

    const value = current()

    if (expected !== undefined && expected !== token(value)) {
      throw new Error('The active profile changed. Reopen Quick Entry before saving.')
    }

    return value
  }

  ipc.handle('hermes:thoughts:read', event => {
    authorize(event.sender.id)

    return snapshot()
  })
  ipc.handle('hermes:thoughts:draft', (event, payload) => {
    if (typeof payload?.token !== 'string') {
      throw new Error('Missing thought owner.')
    }

    const value = authorize(event.sender.id, payload.token)
    store.draft(value, payload.draft)
  })
  ipc.handle('hermes:thoughts:save', (event, payload) => {
    if (typeof payload?.token !== 'string') {
      throw new Error('Missing thought owner.')
    }

    const value = authorize(event.sender.id, payload.token)
    store.save(value, payload.draft)

    return snapshot()
  })

  return {
    authorize,
    invalidate: () => {
      generation = randomUUID()
    }
  }
}

const pendingDeletions = new WeakMap<ThoughtCaptureStore, Set<string>>()

/** Retire before DELETE: a timeout may follow success, so only a definite refusal restores the inbox. */
export async function deleteWithThoughtRetirement<T extends { ok?: boolean; success?: boolean; error?: unknown }>(
  store: ThoughtCaptureStore,
  owner: ThoughtOwner,
  invalidate: (retiredOwner: ThoughtOwner) => void,
  remove: () => Promise<T>
): Promise<T> {
  const pending = pendingDeletions.get(store) ?? new Set<string>()
  pendingDeletions.set(store, pending)
  const key = JSON.stringify(owner)

  if (pending.has(key)) {
    throw new Error('This profile deletion is already in progress.')
  }

  pending.add(key)

  try {
    const retired = store.retireProfile(owner)
    store.blockProfile(owner)
    invalidate(owner)

    try {
      const response = await remove()

      if (response?.ok === false || response?.success === false || response?.error) {
        retired?.restore()
        store.activateProfile(owner)
      } else {
        store.blockProfile(owner)
      }

      return response
    } catch (error) {
      store.blockProfile(owner)

      if (!retired) {
        throw error
      }

      throw new Error(
        `${error instanceof Error ? error.message : 'Profile deletion was not confirmed.'} Local thoughts are preserved in ${retired.recoveryPath}.`,
        { cause: error }
      )
    }
  } finally {
    pending.delete(key)
  }
}
