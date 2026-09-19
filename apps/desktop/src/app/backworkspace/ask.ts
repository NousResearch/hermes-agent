import type { GatewayEvent } from '@hermes/shared'

import { onGatewayEvent } from '@/contrib/events'
import { requestGatewayForAgent, retainGatewayForAgent } from '@/store/gateway'

import type { BackworkspaceRoute } from './page'

// One hidden session per profile answers this page, found by NAME. A title
// cannot dangle the way a stored id can (see Bot Mode in src/AGENTS.md), and
// reusing it is what gives the page a memory across questions.
const SESSION_TITLE = 'Back Workspace'
const SESSION_LIST_LIMIT = 200
const REPLY_TIMEOUT_MS = 180_000
// JSON-RPC 4001: the runtime session is gone (reaped, or the backend restarted).
const SESSION_GONE = 4001

interface SessionRow {
  id: string
  resolved_id?: null | string
}

interface MessageCompletePayload {
  error?: string
  rendered?: string
  status?: string
  text?: string
}

/** Runtime session id per owner — the id a turn is submitted against. */
const runtimeSessions = new Map<string, string>()

export interface AskTarget {
  /** `@handle` as it appears in the page. */
  handle: string
  route: BackworkspaceRoute
}

function ownerKey(route: BackworkspaceRoute): string {
  return `${route.connectionId ?? ''}:${route.profile}`
}

function request<T>(route: BackworkspaceRoute, method: string, params: Record<string, unknown>): Promise<T> {
  return requestGatewayForAgent<T>(route.connectionId, route.profile, method, params, undefined, undefined, {
    spawnPriority: 'foreground'
  })
}

function isSessionGone(error: unknown): boolean {
  const code = error && typeof error === 'object' ? (error as { code?: unknown }).code : undefined

  if (typeof code === 'number') {
    return code === SESSION_GONE
  }

  return /session (?:not found|not in memory)/i.test(error instanceof Error ? error.message : String(error))
}

async function findSession(route: BackworkspaceRoute): Promise<SessionRow | undefined> {
  const listed = await request<{ sessions?: SessionRow[] }>(route, 'session.list', {
    // Hidden by design, and a window-free lookup: a page that has not been
    // asked anything for months still resolves.
    include_hidden: true,
    limit: SESSION_LIST_LIMIT,
    title: SESSION_TITLE
  })

  return listed.sessions?.[0]
}

async function resumeSession(route: BackworkspaceRoute, row: SessionRow): Promise<string> {
  const stored = row.resolved_id || row.id

  const resumed = await request<{ session_id?: string }>(route, 'session.resume', {
    omit_messages: true,
    session_id: stored,
    source: 'desktop'
  })

  return resumed.session_id || stored
}

/** The live session for `route`, resolved by title and created only when there is none. */
async function ensureSession(route: BackworkspaceRoute): Promise<string> {
  const key = ownerKey(route)
  const cached = runtimeSessions.get(key)

  if (cached) {
    return cached
  }

  const existing = await findSession(route)

  const runtime = existing
    ? await resumeSession(route, existing)
    : await createSession(route).catch(async error => {
        // Another window minted it between the lookup and the create: adopt
        // the winner rather than forking a second page session.
        const winner = await findSession(route)

        if (!winner) {
          throw error
        }

        return resumeSession(route, winner)
      })

  runtimeSessions.set(key, runtime)

  return runtime
}

async function createSession(route: BackworkspaceRoute): Promise<string> {
  const created = await request<{ session_id: string }>(route, 'session.create', {
    follow_profile_config: true,
    hidden: true,
    source: 'desktop',
    title: SESSION_TITLE
  })

  // session.create is lazy: the row exists once it is titled, and titling it
  // now also beats the auto-titler to the name we look it up by.
  await request(route, 'session.title', { session_id: created.session_id, title: SESSION_TITLE })

  return created.session_id
}

/** The reply for `sessionId`, and a way to stop waiting when the question never left. */
function replyFor(sessionId: string, route: BackworkspaceRoute): { cancel: () => void; promise: Promise<string> } {
  let stop = () => {}

  const promise = new Promise<string>((resolve, reject) => {
    const mine = (event: GatewayEvent) =>
      event.session_id === sessionId &&
      (!event.connectionId || !route.connectionId || event.connectionId === route.connectionId)

    const complete = onGatewayEvent('message.complete', event => {
      if (!mine(event)) {
        return
      }

      const payload = (event.payload ?? {}) as MessageCompletePayload

      stop()

      if (payload.status === 'error') {
        reject(new Error(payload.error || 'turn failed'))

        return
      }

      resolve(String(payload.text || payload.rendered || ''))
    })

    const failed = onGatewayEvent('error', event => {
      if (!mine(event)) {
        return
      }

      stop()
      reject(new Error(String((event.payload as { error?: string } | undefined)?.error || 'turn failed')))
    })

    const timer = setTimeout(() => {
      stop()
      reject(new Error('the agent did not answer in time'))
    }, REPLY_TIMEOUT_MS)

    stop = () => {
      clearTimeout(timer)
      complete()
      failed()
    }
  })

  return { cancel: () => stop(), promise }
}

/**
 * Every turn carries its frame, not just the first.
 *
 * Told once that it could edit the page file, the agent helpfully wrote its
 * own answer into it and then said "Written to the page" — which is what the
 * app pasted in, since the page takes the turn's LAST message. The frame is
 * cheap (a line), it survives a session that already learned the other habit,
 * and it is explicit about who writes: the app does.
 */
function framed(question: string, pagePath: null | string): string {
  const where = pagePath ? ` The page is the file ${pagePath}; read it if you need more context.` : ''

  return [
    `[From my back-workspace page — the note page on the back of the Hermes desktop window.${where}`,
    'Answer the part addressed to you, in plain prose. Do not write to the page or the file: the app pastes your reply in for you.]',
    '',
    question
  ].join('\n')
}

/**
 * Ask `target` the question and return its reply.
 *
 * The socket is held for the whole exchange: between two RPCs the pool would
 * otherwise drop the last reference, close the secondary and let the gateway
 * reap the session mid-question.
 */
export async function askBackworkspace(target: AskTarget, question: string, pagePath: null | string): Promise<string> {
  const { route } = target
  const release = await retainGatewayForAgent(route.connectionId, route.profile, { spawnPriority: 'foreground' })

  try {
    return await submit(route, question, pagePath)
  } catch (error) {
    if (!isSessionGone(error)) {
      throw error
    }

    // The session went away (backend restart, idle reap). Resolve it again and
    // ask once more before giving up.
    runtimeSessions.delete(ownerKey(route))

    return await submit(route, question, pagePath)
  } finally {
    release()
  }
}

async function submit(route: BackworkspaceRoute, question: string, pagePath: null | string): Promise<string> {
  const sessionId = await ensureSession(route)
  const reply = replyFor(sessionId, route)

  try {
    // Subscribed before submitting: a fast turn can complete before an await
    // resumes, and the reply would be missed.
    await request(route, 'prompt.submit', { session_id: sessionId, text: framed(question, pagePath) })
  } catch (error) {
    // The question never left: stop waiting for an answer to it.
    reply.cancel()
    void reply.promise.catch(() => undefined)
    throw error
  }

  return reply.promise
}
