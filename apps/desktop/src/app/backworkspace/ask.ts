import type { GatewayEvent } from '@hermes/shared'
import { atom } from 'nanostores'

import { onGatewayEvent } from '@/contrib/events'
import { requestGatewayForAgent, retainGatewayForAgent } from '@/store/gateway'
import { sessionApprovalRequests } from '@/store/prompts'

import { backworkspaceOwnerKey, type BackworkspaceRoute } from './page'

// One hidden session per profile answers this page, found by NAME. A title
// cannot dangle the way a stored id can (see Bot Mode in src/AGENTS.md), and
// reusing it is what gives the page a memory across questions.
const SESSION_TITLE = 'Back Workspace'
const SESSION_LIST_LIMIT = 200
// How long the page waits with no news before it gives up. An approval waiting
// on the reader buys the longer deadline instead: the agent is not late, it is
// waiting for a person, and `approvals.timeout` (300s by default) is longer
// than the short one on its own. It is a longer deadline rather than none, so a
// queue entry the backend never resolves cannot hold the page forever.
const REPLY_TIMEOUT_MS = 180_000
const REPLY_TIMEOUT_WAITING_MS = 600_000
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

/**
 * Runtime session id per owner — the id a turn is submitted against.
 *
 * Observable, because it is also the id the approval queue files its requests
 * under: the page has to watch for an approval addressed to the session it is
 * talking to, and the titlebar has to know one is waiting while the window is
 * turned to the front.
 */
export const $backworkspaceSessions = atom<Readonly<Record<string, string>>>({})

function rememberSession(key: string, sessionId: string) {
  $backworkspaceSessions.set({ ...$backworkspaceSessions.get(), [key]: sessionId })
}

function forgetSession(key: string) {
  const { [key]: _gone, ...rest } = $backworkspaceSessions.get()

  $backworkspaceSessions.set(rest)
}

/**
 * The session the page belonging to each owner is waiting on, while a question
 * of its own is in flight.
 *
 * Keyed by the ASKING page, not by the agent: the question may have gone to a
 * bot on another profile, and it is the page in front of the reader that has to
 * carry that bot's approval. Emptied when the exchange settles, because an
 * approval only ever arrives inside a turn.
 */
export const $backworkspaceWaiting = atom<Readonly<Record<string, string>>>({})

function markWaiting(owner: string, sessionId: string) {
  $backworkspaceWaiting.set({ ...$backworkspaceWaiting.get(), [owner]: sessionId })
}

function clearWaiting(owner: string) {
  const { [owner]: _done, ...rest } = $backworkspaceWaiting.get()

  $backworkspaceWaiting.set(rest)
}

export interface AskTarget {
  /** The page that is asking, so its own approvals find their way back to it. */
  asker: BackworkspaceRoute
  /** `@handle` as it appears in the page. */
  handle: string
  route: BackworkspaceRoute
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
  const key = backworkspaceOwnerKey(route)
  const cached = $backworkspaceSessions.get()[key]

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

  rememberSession(key, runtime)

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

    let timer: ReturnType<typeof setTimeout> | undefined

    // An approval parked on the page restarts the clock on the longer
    // deadline, and answering it restarts the short one, so a command allowed
    // in the fifth minute still has its reply written in.
    const waiting = sessionApprovalRequests(sessionId).subscribe(requests => {
      clearTimeout(timer)

      timer = setTimeout(
        () => {
          stop()
          reject(new Error('the agent did not answer in time'))
        },
        requests.length ? REPLY_TIMEOUT_WAITING_MS : REPLY_TIMEOUT_MS
      )
    })

    stop = () => {
      clearTimeout(timer)
      waiting()
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
  const { asker, route } = target
  const owner = backworkspaceOwnerKey(asker)
  const release = await retainGatewayForAgent(route.connectionId, route.profile, { spawnPriority: 'foreground' })

  try {
    return await submit(route, question, pagePath, owner)
  } catch (error) {
    if (!isSessionGone(error)) {
      throw error
    }

    // The session went away (backend restart, idle reap). Resolve it again and
    // ask once more before giving up.
    forgetSession(backworkspaceOwnerKey(route))

    return await submit(route, question, pagePath, owner)
  } finally {
    clearWaiting(owner)
    release()
  }
}

async function submit(
  route: BackworkspaceRoute,
  question: string,
  pagePath: null | string,
  owner: string
): Promise<string> {
  const sessionId = await ensureSession(route)
  const reply = replyFor(sessionId, route)

  markWaiting(owner, sessionId)

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

  return await reply.promise
}
