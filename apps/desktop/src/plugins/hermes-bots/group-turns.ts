/**
 * One member's turn: its hidden per-group session, the submit/poll loop that
 * runs it, the pending clarify/approval prompts mirrored out of it, and the
 * late-reply harvest for a turn that timed out.
 *
 * Room-level sequencing lives in group-rounds.ts, which drives these.
 */

import { host } from '@hermes/plugin-sdk'

import { recordGroupActivity } from './group-activity'
import { $groupChats, $groupClarify, $groupNeedsYou, appendGroupChatEntry, updateGroupChat } from './group-chat'
import type { GroupChatRoom } from './group-chat'
import { groupMemberKey, groupSessionOwner } from './group-membership'
import { botConnectionRoute, requestForBot } from './routing'
import type { Attachment, GroupMember, GroupPrompt, GroupPromptQuestion, ProfileRoute } from './types'

/** "(pass)" (loosely: pass / (pass) / pass.) or empty = the member stayed silent. */
export function isGroupPassText(text: unknown) {
  const trimmed = String(text || '').trim()

  if (!trimmed) {
    return true
  }

  return /^\(?\s*pass\s*\)?\.?$/i.test(trimmed)
}

/** One transcript entry in a `session.resume` snapshot, as the turn harvester
 *  reads it — the session's own message shape, not the plugin's GroupMessage.
 *  `content` is a plain string on most providers and a part array on the rest. */
interface GroupTurnTranscriptMessage {
  content?: string | Array<string | { text?: string }>
  role?: string
  text?: string
}

/** #94376: pick the reply a finished turn should surface among the messages
 *  appended since `before`. Scans newest-first and prefers the last
 *  substantive (non-pass) assistant answer over a trailing pass — a Codex
 *  intent-ack continuation nudge can land a complete answer and then get a
 *  synthetic "(pass)" to the nudge itself, which must not hide the answer.
 *  When only pass text exists in range, returns the newest (last
 *  chronological) one rather than the oldest. Returns null only when no
 *  assistant message appears in that range. */
function pickGroupTurnReply(messages: GroupTurnTranscriptMessage[], before: number): null | string {
  let passText: null | string = null

  for (let i = messages.length - 1; i >= before; i--) {
    const msg = messages[i]

    if (msg?.role !== 'assistant') {
      continue
    }

    const text =
      typeof msg.content === 'string'
        ? msg.content
        : Array.isArray(msg.content)
          ? msg.content.map(p => (typeof p === 'string' ? p : p?.text || '')).join('')
          : msg?.text || ''

    const replyText = String(text).trim()

    if (isGroupPassText(replyText)) {
      if (passText === null) {
        passText = replyText
      }

      continue
    }

    return replyText
  }

  return passText
}

/** A clarify question blocking inside a member's session, as `session.resume`
 *  reports it. Older backends omit the field entirely. */
interface GroupPendingClarify {
  choices?: string[]
  multi_select?: unknown
  question?: unknown
  questions?: GroupPromptQuestion[]
  request_id?: string
}

/** A command approval blocking inside a member's session, same wire as the
 *  1:1 approval card. */
interface GroupPendingApproval {
  choices?: string[]
  command?: unknown
  description?: unknown
  request_id?: string
}

/** The `session.resume` fields the room engine reads off a member's hidden
 *  per-group session. */
interface GroupSessionSnapshot {
  inflight?: boolean
  message_count?: number
  messages?: GroupTurnTranscriptMessage[]
  pending_approval?: GroupPendingApproval
  pending_clarify?: GroupPendingClarify
  running?: boolean
  session_id?: string
  session_key?: string
}

/** A member's per-group session, resolved for one turn. */
interface GroupMemberSessionHandle {
  /** Live runtime id every RPC in this turn targets. */
  runtime: null | string
  /** Durable id persisted in `room.sessions`; `true` is the legacy sentinel. */
  stored?: null | string | true
}

/** Ensure the member's per-group session exists and return a LIVE runtime
 *  session id for it. Gateway-native: session.create mints the session
 *  (lazy until its first message), session.resume by stored id — or by
 *  title, which also covers rehydrated rooms whose sid was lost — reopens
 *  it after restarts. Cross-connection members route to their OWN source
 *  via requestForBot; the window's gateway never switches. */
export async function ensureGroupChatSession(group: string, member: GroupMember): Promise<GroupMemberSessionHandle> {
  const room = $groupChats.get()[group] || {}
  // New rooms title member sessions by their immutable roomId so a
  // same-name recreate never resumes the old room's sessions by title;
  // legacy rooms without a roomId fall back to the display name.
  const title = `Group: ${room.roomId || group}`
  const key = groupMemberKey(member)
  const known = room.sessions && room.sessions[key]

  // Try resuming what we know (stored sid first, then title lookup).
  //
  // FAIL CLOSED on a transient lookup failure — mirrors the sibling fix in
  // findExistingCanonicalChat (87b645f52c). session.resume signals "this
  // target genuinely doesn't exist" with JSON-RPC code 4007; every other
  // failure (network blip, the backend still warming up after a restart,
  // an oversized-resume refusal) means the real session might still be
  // there and must not be read as "no session, mint a new one" — that
  // forks the member's real history, and the fork silently overwrites
  // room.sessions[key] so the old session becomes unreachable from the
  // room. Only a genuine 4007 on BOTH targets means there truly is nothing
  // to resume yet, so the loop falls through to session.create below.
  for (const target of [known, title]) {
    if (!target || target === true) {
      continue
    }

    try {
      const res = (await requestForBot(member, 'session.resume', {
        session_id: target,
        profile: member.name,
        omit_messages: true
      })) as GroupSessionSnapshot

      if (res?.session_id) {
        // TODO(bot-mode-types): `known` is `room.sessions[key]`, which the
        // domain model types `string | true` — and the `target === true` skip
        // above shows the legacy `true` sentinel is expected here. A backend
        // that answers the title resume without a `session_key` therefore
        // stores `true` back into room.sessions and hands `true` on as the
        // durable id, which later rides into `session_id` on the recovery
        // resume and on session.interrupt. Typed as-written.
        const stored = res.session_key || known

        if (stored) {
          updateGroupChat(group, (current: GroupChatRoom) => {
            current.sessions = {
              ...(current.sessions || {}),
              [key]: stored
            }
            current.sessionOwners = {
              ...(current.sessionOwners || {}),
              [key]: groupSessionOwner(member)
            }

            return current
          })
        }

        return {
          runtime: res.session_id,
          stored
        }
      }
    } catch (error: any) {
      if (error?.code !== 4007) {
        const detail = error instanceof Error && error.message ? ` (${error.message})` : ''
        throw new Error(`Could not check ${member?.name || 'member'}'s group session${detail} — not starting a new one`)
      }
      /* genuinely doesn't exist (4007) — try the next target / fall through to create */
    }
  }

  const created = (await requestForBot(member, 'session.create', {
    profile: member.name,
    title,
    // Room member sessions are plumbing — always hidden from the sidebar.
    hidden: true,
    // Explicit contracts (PR #97008): room plumbing sessions always rebuild
    // from the member profile's CURRENT config on resume, never a stale
    // stored model/provider pin. Older gateways ignore the unknown params;
    // the server's hidden + "Group: " title fallback then covers legacy.
    room_plumbing: true,
    follow_profile_config: true
  })) as { session_id?: string; stored_session_id?: string }

  const stored = created?.stored_session_id || null

  if (stored) {
    updateGroupChat(group, (r: GroupChatRoom) => {
      r.sessions = {
        ...(r.sessions || {}),
        [key]: stored
      }
      r.sessionOwners = {
        ...(r.sessionOwners || {}),
        [key]: groupSessionOwner(member)
      }

      return r
    })
  }

  return {
    runtime: created?.session_id || null,
    stored
  }
}

const GROUP_TURN_TIMEOUT_MS = 180000
// Backstop cadence only. The turn normally wakes the instant the member's
// session emits its terminal frame (message.complete / error) via host.onEvent;
// this poll exists for hosts without the event tap, sessions whose events
// ride a socket this window doesn't hold, and frames lost to a reconnect.
const GROUP_TURN_POLL_MS = 5000
const GROUP_TURN_SETTLE_RECHECK_MS = 250
const GROUP_TURN_SETTLE_RECHECKS = 8

/** Resolve as soon as the member's session reports a terminal frame for the
 *  turn — or after `ms` as the backstop. Feature-detected: without
 *  `host.onEvent` (older shells, the node test harness) this is a plain sleep.
 *  `error` is terminal too (agent init failed → no message.complete follows). */
function waitForTurnSignal(runtimeIds: string[], ms: number): Promise<boolean> {
  const ids = new Set(runtimeIds.filter(id => id.length > 0))

  return new Promise(resolve => {
    const unsubs: Array<() => void> = []
    let timer: null | ReturnType<typeof setTimeout> = null
    let settled = false

    const done = (signalled: boolean) => {
      if (settled) {
        return
      }

      settled = true

      if (timer !== null) {
        clearTimeout(timer)
      }

      for (const unsub of unsubs) {
        try {
          unsub()
        } catch {
          /* disposer already ran */
        }
      }

      resolve(signalled)
    }

    // (The test harness runs timers inline — `done` may already have fired
    // by the time this assignment lands, hence the `settled` guard below.)
    timer = setTimeout(() => done(false), ms)

    if (settled || typeof host.onEvent !== 'function' || !ids.size) {
      return
    }

    for (const type of ['message.complete', 'error']) {
      try {
        unsubs.push(
          host.onEvent(type, (event: { session_id?: string }) => {
            if (ids.has(String(event?.session_id || ''))) {
              done(true)
            }
          })
        )
      } catch {
        /* event tap unavailable — the timer still resolves */
      }
    }
  })
}

// --- group-turn session-lease helpers (#93602) ------------------------------
// A member turn is a session-scoped RPC SEQUENCE (resume → attach → submit →
// poll) issued with the runtime id its first RPC minted. requestForBot routes
// each RPC through a per-request socket lease (retained:false secondaries in
// store/gateway), so between two RPCs the refcount can hit 0, the leased
// socket closes, the gateway detaches the runtime session on WS disconnect,
// and the orphan reaper frees it — the next RPC then fails 4001 "not in
// memory" and the bot goes silent in the room.

/** A gateway rejection as it reaches the room engine: an `Error`, a raw
 *  JSON-RPC error object, or (across a realm boundary) a bare string. */
interface GatewayErrorLike {
  code?: number
  data?: { reason?: unknown }
  message?: unknown
}

/** 4001-class "the runtime session was reaped" failure. Distinct from 4007
 *  ("genuinely never existed"), which must keep flowing to session.create. */
export function isSessionGoneError(error: GatewayErrorLike | null | undefined): boolean {
  if (!error || error.code === 4007) {
    return false
  }

  if (error.code === 4001) {
    return true
  }

  // Duck-typed (not instanceof): gateway errors can cross realm boundaries.
  const message = typeof error?.message === 'string' ? error.message : typeof error === 'string' ? error : ''

  return message.includes('not in memory') || /session not found/i.test(message)
}
// --- end group-turn session-lease helpers ---

/** Hold the member's pooled socket open for the WHOLE turn. Feature-detected:
 *  hosts without retainProfile (or members on the active gateway, which never
 *  closes mid-turn) get a no-op release. A failed acquire must not kill the
 *  turn — the catch-retry on submit still covers the race. */
async function retainGroupTurnRoute(member: GroupMember): Promise<() => void> {
  const noop = () => undefined
  let route: ProfileRoute | null = null

  try {
    route = botConnectionRoute(member)
  } catch {
    return noop
  }

  if (!route || typeof host.retainProfile !== 'function') {
    return noop
  }

  try {
    const release = await host.retainProfile(route)

    return typeof release === 'function' ? release : noop
  } catch {
    return noop
  }
}

/** prompt.submit with one belt-and-braces retry: when the runtime session was
 *  reaped between minting and submitting (4001 class), re-resume via the
 *  STORED id — the durable identity — to mint a fresh runtime id, and submit
 *  exactly once more. Returns the runtime id the submit actually landed on so
 *  the poll loop keeps a live fallback target. */
async function submitGroupTurnPrompt(
  member: GroupMember,
  runtime: string,
  stored: null | string | true | undefined,
  text: string
): Promise<string> {
  try {
    await requestForBot(member, 'prompt.submit', {
      session_id: runtime,
      text
    })

    return runtime
  } catch (error: any) {
    if (!isSessionGoneError(error) || !stored) {
      throw error
    }

    const res = (await requestForBot(member, 'session.resume', {
      session_id: stored,
      profile: member.name,
      omit_messages: true
    })) as GroupSessionSnapshot

    const fresh = res?.session_id

    if (!fresh) {
      throw error
    }

    await requestForBot(member, 'prompt.submit', {
      session_id: fresh,
      text
    })

    return fresh
  }
}

// A member turn that is VISIBLY still working (session reports
// inflight/running) keeps its slot alive up to this hard cap. The base
// timeout alone silently dropped long real turns: a 7-minute research run
// timed out at 3 minutes, read as a pass, and its finished result never
// reached the room (db's Aug 2026 report).
const GROUP_TURN_HARD_CAP_MS = 20 * 60000

/** Mirror a member's pending prompt — clarify question OR command approval —
 *  from its resume snapshot into the room store, keyed
 *  `${group}::${memberKey}` (#90694). Returns true while a prompt is
 *  blocking, so the turn poll can extend its deadline — a waiting prompt
 *  must not be eaten by the group-turn timeout. Feature-detected: older
 *  backends without `pending_clarify`/`pending_approval` in the resume
 *  payload always sync to "no prompt". Clarify wins when both are somehow
 *  present (approvals resolve inside tool batches; clarify is the outer
 *  blocker). */
export function syncGroupClarify(group: string, member: GroupMember, state: GroupSessionSnapshot | null): boolean {
  const key = `${group}::${groupMemberKey(member)}`
  const clarify = state && typeof state.pending_clarify === 'object' ? state.pending_clarify : null

  // The `!requestId` bail below is what makes the approval branch reachable,
  // so an approval read there is never the null arm of this ternary — a fact
  // control-flow analysis can't carry across the two separate locals.
  const approval = (
    state && typeof state.pending_approval === 'object' ? state.pending_approval : null
  ) as GroupPendingApproval

  const pending = clarify || approval
  const requestId = pending?.request_id || null
  const all = $groupClarify.get()
  const current = all[key]

  if (!requestId) {
    if (current) {
      const next = {
        ...all
      }

      delete next[key]
      $groupClarify.set(next)
    }

    return false
  }

  // Same request already mirrored — keep the object identity so the card
  // doesn't lose its draft to a re-render.
  if (current?.requestId === requestId) {
    return true
  }

  const base = {
    requestId,
    group,
    member: member.name,
    memberKey: groupMemberKey(member),
    // approval.respond keys on the session, not just the request — carry the
    // runtime id the snapshot came from.
    sessionId: state?.session_id || null,
    at: Date.now()
  }

  $groupClarify.set({
    ...all,
    [key]: clarify
      ? {
          ...base,
          kind: 'clarify',
          question: typeof clarify.question === 'string' ? clarify.question : '',
          choices: Array.isArray(clarify.choices) ? clarify.choices.filter(c => typeof c === 'string' && c) : [],
          multiSelect: Boolean(clarify.multi_select),
          // Batch clarifies carry `questions`; the room card answers them
          // one wire call per question, mirroring the 1:1 batch contract.
          questions: Array.isArray(clarify.questions) ? clarify.questions : null
        }
      : {
          ...base,
          kind: 'approval',
          question: typeof approval.description === 'string' ? approval.description : '',
          command: typeof approval.command === 'string' ? approval.command : '',
          // The server precomputes the choice set from allow_permanent
          // (once/session/always/deny); fall back to the minimal pair.
          choices:
            Array.isArray(approval.choices) && approval.choices.length
              ? approval.choices.filter(c => typeof c === 'string' && c)
              : ['once', 'deny'],
          multiSelect: false,
          questions: null
        }
  })
  // A blocked member is a question for the human — badge the room.
  $groupNeedsYou.set({
    ...$groupNeedsYou.get(),
    [group]: true
  })

  return true
}

/** Drop every mirrored clarify belonging to `group` (disband/rename). */
export function clearGroupClarify(group: string) {
  const all = $groupClarify.get()
  const next: Record<string, GroupPrompt> = {}
  let changed = false

  for (const [key, value] of Object.entries<GroupPrompt>(all)) {
    if (value?.group === group) {
      changed = true
    } else {
      next[key] = value
    }
  }

  if (changed) {
    $groupClarify.set(next)
  }
}

/** Answer a member's pending prompt from the room. Routes to the member's
 *  OWN source (requestForBot), so cross-connection members work.
 *  - clarify: `clarify.respond`; batch questions send one respond per
 *    question, sequentially — the LAST lock resolves the blocked tool
 *    server-side (same contract as the 1:1 batch card). allow_expired
 *    server-side makes racing the timeout harmless.
 *  - approval: `approval.respond` with the choice (once/session/always/deny),
 *    keyed by session + request_id — the same wire the 1:1 approval card
 *    and native notifications use. */
export async function answerGroupClarify(
  entry: GroupPrompt,
  member: GroupMember,
  answers: Record<string, string> | string | undefined
) {
  if (entry.kind === 'approval') {
    await requestForBot(member, 'approval.respond', {
      session_id: entry.sessionId || undefined,
      request_id: entry.requestId,
      choice: typeof answers === 'string' && answers ? answers : 'deny'
    })
  } else if (entry.questions && entry.questions.length) {
    for (const question of entry.questions) {
      // Question ids are opaque on the wire (`GroupPrompt.questions` types
      // them `unknown`); the batch card keys its answer bag by exactly them.
      const qid = (question?.qid ?? question?.id) as string
      await requestForBot(member, 'clarify.respond', {
        request_id: entry.requestId,
        question_id: qid,
        answer: (answers as Record<string, string>)?.[qid] ?? ''
      })
    }
  } else {
    await requestForBot(member, 'clarify.respond', {
      request_id: entry.requestId,
      answer: typeof answers === 'string' ? answers : ''
    })
  }

  const all = $groupClarify.get()
  const key = `${entry.group}::${entry.memberKey}`

  if (all[key]?.requestId === entry.requestId) {
    const next = {
      ...all
    }

    delete next[key]
    $groupClarify.set(next)
  }
}

/** One member turn, gateway-native: submit the room delta as a prompt into
 *  the member's per-group session, then poll the session until a NEW
 *  assistant message lands (or timeout → pass). While the session visibly
 *  reports work in flight the deadline extends (bounded by the hard cap),
 *  so slow models aren't cut off mid-run. A turn that still times out
 *  records a stranded marker so the finished reply can be harvested into
 *  the room at the member's next turn instead of being lost. */
export async function runGroupChatMemberTurn(
  group: string,
  member: GroupMember,
  prompt: string,
  thread: string,
  images?: Attachment[]
): Promise<null | string> {
  // #93602: hold the member's route socket for the whole turn. Without the
  // lease, every RPC below rides its own request-scoped socket lease; the
  // socket that minted `runtime` can close between RPCs, the gateway reaps
  // the runtime session, and prompt.submit dies 4001 — the bot goes silent.
  const releaseTurnLease = await retainGroupTurnRoute(member)

  try {
    return await runGroupChatMemberTurnLeased(group, member, prompt, thread, images)
  } finally {
    // A normal return/throw means this renderer observed the terminal state, so
    // its active recovery marker is no longer needed. If the renderer dies,
    // this finally never runs and the durable marker lets the next window
    // harvest the backend reply instead of submitting the work twice.
    const memberKey = groupMemberKey(member)
    updateGroupChat(group, room => {
      const marker = room.stranded?.[memberKey]

      if (typeof marker === 'object' && marker?.active) {
        const next = { ...(room.stranded || {}) }

        delete next[memberKey]
        room.stranded = next
      }

      return room
    })
    releaseTurnLease()
  }
}

async function runGroupChatMemberTurnLeased(
  group: string,
  member: GroupMember,
  prompt: string,
  thread: string,
  images?: Attachment[]
): Promise<null | string> {
  const { runtime, stored } = await ensureGroupChatSession(group, member)

  if (!runtime) {
    return null
  }

  // #91868/#94569: remember the epoch this turn was dispatched under so the
  // poll loop below can tell an explicit stop from ordinary room churn.
  const dispatchEpoch = ($groupChats.get()[group] || {}).epoch || 0
  const memberKey = groupMemberKey(member)
  recordGroupActivity(group, {
    kind: 'working',
    member: member.name,
    thread
  })

  // Baseline: how many messages exist before our submit.
  let before = 0
  // Every runtime id this turn has seen for the member's session. Terminal
  // frames are keyed by runtime id, and a resume can hand back a fresh one.
  const runtimeIds = new Set<string>([runtime])

  try {
    const pre = (await requestForBot(member, 'session.resume', {
      session_id: stored || runtime,
      profile: member.name
    })) as GroupSessionSnapshot

    before = Array.isArray(pre?.messages) ? pre.messages.length : pre?.message_count || 0

    if (pre?.session_id) {
      runtimeIds.add(pre.session_id)
    }
  } catch {
    /* lazy session — zero messages */
  }

  // Persist the in-flight baseline before submitting. Renderer reloads and
  // gateway re-homes destroy the JavaScript poll loop, not the backend turn;
  // this marker lets the next window resume by harvesting that exact session.
  updateGroupChat(group, room => {
    room.stranded = {
      ...(room.stranded || {}),
      [memberKey]: { active: true, before, thread }
    }

    return room
  })

  // Stage this delta's attachments into the member's session so the model
  // receives the actual payload with the prompt — the same attach RPCs the
  // 1:1 chat uses (they also work cross-connection, where the member's
  // gateway can't see this machine's files). Images queue as vision tiles,
  // PDFs render per-page via pdf.attach, and other files materialize in the
  // session workspace (their @file: refs are appended to the prompt so the
  // member's file tools can read them). A failed attach degrades that
  // member to text-only; the transcript line still names the attachment so
  // the member knows something was shared.
  const fileRefs: string[] = []

  for (const img of Array.isArray(images) ? images : []) {
    if (!img || typeof img.data !== 'string' || !img.data) {
      continue
    }

    try {
      if (img.kind === 'pdf') {
        await requestForBot(member, 'pdf.attach', {
          session_id: runtime,
          content_base64: img.data,
          filename: img.name || 'attachment.pdf'
        })
      } else if (img.kind === 'file') {
        const res = (await requestForBot(member, 'file.attach', {
          session_id: runtime,
          data_url: img.data,
          name: img.name || 'attachment'
        })) as { ref_text?: string }

        if (res?.ref_text) {
          fileRefs.push(`${img.name || 'attachment'} → ${res.ref_text}`)
        }
      } else {
        await requestForBot(member, 'image.attach_bytes', {
          session_id: runtime,
          content_base64: img.data,
          filename: img.name || 'attachment.png'
        })
      }
    } catch {
      /* text-only fallback for this member */
    }
  }

  const turnText = fileRefs.length
    ? `${prompt}\n\nAttached files staged in your session workspace:\n${fileRefs.join('\n')}`
    : prompt

  // #93602: one-shot recovery when the runtime session was reaped between
  // minting and submitting. Tracks the runtime id the submit landed on so
  // the poll fallback below targets a live session.
  const liveRuntime = await submitGroupTurnPrompt(member, runtime, stored, turnText)
  runtimeIds.add(liveRuntime)
  const started = Date.now()
  let deadline = started + GROUP_TURN_TIMEOUT_MS
  // After the terminal frame fires, the gateway still has to flip
  // session.running off in its turn `finally` — re-check quickly for a few
  // beats instead of falling back to the slow backstop cadence.
  let quickRechecks = 0

  while (Date.now() < deadline) {
    const signalled = quickRechecks
      ? await waitForTurnSignal([], GROUP_TURN_SETTLE_RECHECK_MS)
      : await waitForTurnSignal([...runtimeIds], GROUP_TURN_POLL_MS)

    quickRechecks = signalled ? GROUP_TURN_SETTLE_RECHECKS : Math.max(0, quickRechecks - 1)

    // #91868/#94569: an explicit stop (stopGroupThread) bumped the epoch AND
    // held this member — the member's session was interrupted, so nothing is
    // coming; abandon the poll instead of grinding until the deadline. Both
    // conditions on purpose: an ordinary newer send bumps the epoch WITHOUT
    // a hold, and that turn must keep polling so finished work can still be
    // delivered (the #93127 commit check decides its fate, not this loop).
    const roomDuringPoll = $groupChats.get()[group] || {}

    if ((roomDuringPoll.epoch || 0) !== dispatchEpoch && (roomDuringPoll.holds || {})[memberKey]) {
      return null
    }

    let state: GroupSessionSnapshot | null = null

    try {
      state = (await requestForBot(member, 'session.resume', {
        session_id: stored || liveRuntime,
        profile: member.name
      })) as GroupSessionSnapshot
    } catch {
      continue
    }

    if (state?.session_id) {
      runtimeIds.add(state.session_id)
    }

    const messages = Array.isArray(state?.messages) ? state.messages : []
    const busy = Boolean(state?.inflight || state?.running)
    // A clarify blocking inside the member's session is a question for the
    // HUMAN (#90694) — mirror it into the room store so a card renders, and
    // hold the turn open: the member isn't stalling, it's waiting on us.
    const awaitingUser = syncGroupClarify(group, member, state)
    const done = !busy && !awaitingUser

    if (messages.length > before && done) {
      const replyText = pickGroupTurnReply(messages, before)

      if (replyText !== null) {
        recordGroupActivity(group, {
          kind: isGroupPassText(replyText) ? 'passed' : 'replied',
          member: member.name,
          thread
        })

        return replyText
      }

      recordGroupActivity(group, {
        kind: 'passed',
        member: member.name,
        thread
      })

      return null
    }

    // Still visibly working — or waiting on the user's answer to a clarify:
    // extend the deadline (never past the hard cap). A pending question must
    // outlive the base turn timeout or it dies unanswered at 3 minutes.
    if (busy || awaitingUser) {
      deadline = Math.min(started + GROUP_TURN_HARD_CAP_MS, Math.max(deadline, Date.now() + GROUP_TURN_TIMEOUT_MS))
    }
  }

  // Timeout — clear any still-mirrored question card (the server-side
  // clarify timeout runs its own course) and read as a pass, but remember the baseline + thread
  // (runtime-only) so the finished reply can be posted late into the RIGHT
  // thread instead of vanishing.
  recordGroupActivity(group, {
    kind: 'timed-out',
    member: member.name,
    thread
  })
  syncGroupClarify(group, member, null)
  updateGroupChat(group, (r: GroupChatRoom) => {
    r.stranded = {
      ...(r.stranded || {}),
      [groupMemberKey(member)]: {
        before,
        thread
      }
    }

    return r
  })

  return null
}

/** Post a timed-out member's finished reply into the room, if it landed
 *  after we stopped waiting. Called at the member's next turn boundary and
 *  on user sends, so long-running work is delivered late rather than lost. */
export async function harvestStrandedGroupReply(group: string, member: GroupMember) {
  const memberKey = groupMemberKey(member)
  const room = $groupChats.get()[group] || {}
  const marker = room.stranded?.[memberKey]
  // Markers were a bare number before threads; normalize both shapes.
  const strandedBefore = typeof marker === 'number' ? marker : marker?.before
  const strandedThread = (typeof marker === 'object' && marker?.thread) || 'legacy'

  if (typeof strandedBefore !== 'number') {
    return
  }

  let state: GroupSessionSnapshot | null = null

  try {
    const stored = room.sessions?.[memberKey]
    state = (await requestForBot(member, 'session.resume', {
      session_id: stored || `Group: ${room.roomId || group}`,
      profile: member.name
    })) as GroupSessionSnapshot
  } catch {
    return // source unreachable — leave the marker for the next boundary
  }

  if (state?.inflight || state?.running) {
    return // still grinding — keep waiting
  }

  // A stranded member blocked on a clarify is not "grinding" — surface the
  // question card (#90694) and keep the marker until it resolves.
  if (syncGroupClarify(group, member, state)) {
    return
  }

  // Done (or dead): the marker is consumed either way.
  updateGroupChat(group, (r: GroupChatRoom) => {
    const next = {
      ...(r.stranded || {})
    }

    delete next[memberKey]
    r.stranded = next

    return r
  })
  const messages = Array.isArray(state?.messages) ? state.messages : []

  if (messages.length <= strandedBefore) {
    return
  }

  const reply = pickGroupTurnReply(messages, strandedBefore)

  if (reply && !isGroupPassText(reply)) {
    recordGroupActivity(group, {
      kind: 'delivered',
      member: member.name,
      thread: strandedThread
    })
    appendGroupChatEntry(
      group,
      {
        kind: 'member',
        name: member.name,
        ...(member.remoteSource
          ? {
              source: member.connectionLabel || member.connectionId
            }
          : {})
      },
      reply,
      strandedThread
    )
    updateGroupChat(group, (r: GroupChatRoom) => {
      r.watermarks[`${strandedThread}::${memberKey}`] = r.log.length

      return r
    })
  }
}
