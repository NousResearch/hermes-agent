/**
 * The canvas's line to the real Hermes.
 *
 * This file used to stream a turn by hand — subscribe to message deltas and
 * tool starts, accumulate a reply, render it into bespoke transcript rows. All
 * of that is deleted. The canvas mounts the app's own chat (`SessionChat` from
 * the SDK), which is the same thread, tool cards, streaming indicators,
 * attachments and composer the workspace pane renders. Duplicating that was
 * never going to keep up with it.
 *
 * What's left is the part only this plugin can know: which session the canvas
 * talks to, and how it is created.
 *
 * Each workflow has its own session. Switching graphs without that would keep
 * talking about the previous one. Hidden keeps it out of the sidebar: it
 * belongs to the canvas, not to their list of chats.
 *
 * `source: 'desktop'` is load-bearing and not decoration: the gateway decides
 * whether a session may see the `desktop_ui` toolset — which is where the
 * `workflow` tool lives — from the SESSION's source, never from a process env
 * var. Create this session without it and Hermes answers with its hands tied,
 * having never been offered the one tool the page exists for.
 */

import { host } from '@hermes/plugin-sdk'

import { $workflows, bindWorkflowSession, type WorkflowDoc } from './documents'

const TITLE = 'Workflows'

/** Pre-per-workflow key. Adopted once onto whichever graph opens first without
 *  a session of its own, then forgotten. */
const LEGACY_KEY = 'workflows.session'

interface CreatedSession {
  session_id?: string
  stored_session_id?: string
}

export interface SessionStorage {
  get: <T>(key: string, fallback: T) => T
  set: (key: string, value: unknown) => void
}

const creating = new Map<string, Promise<string>>()
let store: null | SessionStorage = null

/** Hand the plugin's storage over at register, so the page can ask for the
 *  conversation without threading it through the canvas. */
export function bindCanvasSession(storage: SessionStorage): void {
  store = storage
}

async function mintSession(title: string): Promise<string> {
  const created = await host.request<CreatedSession>('session.create', {
    title,
    // The gate on the `workflow` tool. See the module note.
    source: 'desktop',
    // Plumbing, not a conversation the user started.
    hidden: true,
    // Write the row now. session.create is lazy by default (no "Untitled"
    // drafts), but this id is stored on the workflow — a refresh that
    // can't resume it paints "session not found" over the dock.
    persist: true
  })

  // The STORED id is the durable one — a runtime is reaped whenever its
  // socket closes, so remembering that instead would lose the conversation
  // on the first reconnect.
  const id = created?.stored_session_id ?? created?.session_id

  if (!id) {
    throw new Error('Hermes could not start a conversation for the canvas.')
  }

  return id
}

/** This workflow's conversation, minted on first use and reused after.
 *
 *  Deduped per workflow because two callers racing here would mint two
 *  sessions, splitting the conversation in half with no way to tell which half
 *  you were talking to. */
export function ensureCanvasSession(workflowId: string): Promise<string> {
  const doc = $workflows.get().find(d => d.id === workflowId)

  if (!doc) {
    return Promise.reject(new Error('That workflow is gone.'))
  }

  if (doc.sessionId) {
    return Promise.resolve(doc.sessionId)
  }

  const inflight = creating.get(workflowId)

  if (inflight) {
    return inflight
  }

  const work = (async () => {
    const taken = new Set($workflows.get().map(d => d.sessionId).filter((id): id is string => !!id))
    const legacy = store?.get(LEGACY_KEY, '') ?? ''

    // One-time: the old single canvas chat lands on the first workflow that
    // doesn't have its own, so a restart doesn't orphan the conversation.
    const id = legacy && !taken.has(legacy) ? legacy : await mintSession(doc.name || TITLE)

    if (legacy) {
      store?.set(LEGACY_KEY, '')
    }

    bindWorkflowSession(workflowId, id)

    return id
  })().finally(() => {
    creating.delete(workflowId)
  })

  creating.set(workflowId, work)

  return work
}

/** A workflow is born with its conversation. Mint anything that landed
 *  without one (older docs, or a create that hasn't finished yet) so the
 *  canvas never has a "loading the session" state. */
export function watchCanvasSessions(): () => void {
  const kick = (docs: readonly WorkflowDoc[]) => {
    for (const doc of docs) {
      if (!doc.sessionId) {
        void ensureCanvasSession(doc.id)
      }
    }
  }

  kick($workflows.get())

  return $workflows.listen(kick)
}

/** The stored id is gone (Cmd+R reaped a never-persisted draft). Mint another
 *  and bind it — the canvas does not get a "couldn't open" state. */
export function replaceCanvasSession(workflowId: string): Promise<string> {
  bindWorkflowSession(workflowId, '')
  creating.delete(workflowId)

  return ensureCanvasSession(workflowId)
}
