import { atom, type WritableAtom } from 'nanostores'

export interface SavedPrompt {
  id: string
  title: string
  body: string
  updatedAt: number
}
const stores = new Map<string, WritableAtom<SavedPrompt[]>>()
const keyFor = (scope: string) => `agent-czesiek.prompts.v1:${scope}`

export function readPrompts(scope: string, storage: Pick<Storage, 'getItem'> = localStorage): SavedPrompt[] {
  const raw = storage.getItem(keyFor(scope))

  if (!raw) {return []}
  const parsed: unknown = JSON.parse(raw)

  if (!Array.isArray(parsed)) {throw new Error('Invalid prompt library')}

  return parsed.filter(
    (item): item is SavedPrompt =>
      item &&
      typeof item.id === 'string' &&
      typeof item.title === 'string' &&
      typeof item.body === 'string' &&
      typeof item.updatedAt === 'number'
  )
}

export function promptStore(scope: string): WritableAtom<SavedPrompt[]> {
  let store = stores.get(scope)

  if (!store) {
    store = atom(readPrompts(scope))
    stores.set(scope, store)
  }

  return store
}

function commit(scope: string, items: SavedPrompt[]): void {
  localStorage.setItem(keyFor(scope), JSON.stringify(items))
  promptStore(scope).set(items)
}

export function savePrompt(scope: string, draft: Pick<SavedPrompt, 'title' | 'body'> & { id?: string }): SavedPrompt {
  if (!draft.title.trim() || !draft.body.trim()) {throw new Error('A prompt needs a title and content')}

  const prompt = {
    id: draft.id || crypto.randomUUID(),
    title: draft.title.trim(),
    body: draft.body,
    updatedAt: Date.now()
  }

  commit(scope, [
    prompt,
    ...promptStore(scope)
      .get()
      .filter(item => item.id !== prompt.id)
  ])

  return prompt
}

export function removePrompt(scope: string, id: string): void {
  commit(
    scope,
    promptStore(scope)
      .get()
      .filter(item => item.id !== id)
  )
}
