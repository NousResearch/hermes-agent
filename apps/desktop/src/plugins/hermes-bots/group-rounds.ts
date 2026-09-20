/**
 * Room-level coordination: who speaks, in what order, for how long — the
 * @mention parse, the round-robin driver, the #93129 member holds, the stop
 * path, and the user send that starts it all.
 */
import { host } from '@hermes/plugin-sdk'

import { botFriendlyNames, botHandle, botMentionTag, mentionNameForms } from './data'
import { groupFailureReason, recordGroupActivity } from './group-activity'
import {
  $groupChats,
  $groupNeedsYou,
  appendGroupChatEntry,
  GROUP_CHAT_MAX_CONTINUATIONS,
  GROUP_CHAT_MAX_MESSAGES,
  GROUP_CHAT_MAX_ROUNDS,
  groupChatRoomKey,
  groupThreadOf,
  mintGroupThreadId,
  updateGroupChat
} from './group-chat'
import type { GroupChatRoom, GroupHoldStamp } from './group-chat'
import {
  durableGroupChatMembers,
  followGroupChat,
  groupMemberKey,
  groupSessionKey,
  hasThreadScopedGroupSession
} from './group-membership'
import { runGroupContinuationMembers, runGroupRoundMember } from './group-round-members'
import { rejectGroupSlashCommand } from './group-slash'
import { GROUP_TURN_HARD_CAP_MS, harvestStrandedGroupReply } from './group-turns'
import { botsText } from './i18n'
import { requestForBot } from './routing'
import type { Attachment, GroupMember, GroupMessage } from './types'

// ── group chats: bounded round-robin coordination over a shared room log ─────
//
// Behavioral model (clean-room): a group conversation is ONE ordered room log
// owned by the plugin. A user send triggers at most GROUP_CHAT_MAX_ROUNDS
// serial round-robin rounds over the member roster — never parallel, no LLM
// router. Who speaks each round is a deterministic @mention parse since the
// last user message (mentioned members only, else everyone); whether a member
// actually speaks is its own turn's choice — replying with exactly "(pass)"
// (or nothing, or failing) is silence. Hard caps end every turn; a round in
// which everyone passed means the conversation settled. Each member runs its
// turn in its OWN persistent per-group Hermes session and is fed only the
// room messages that are NEW since it last saw the room.

/** Deterministic @mention parse. Handles @name, @"two words" via display
 *  titles, and @everyone/@all. Names match case-insensitively against member
 *  profile names, display titles, and collapsed no-space forms. */
export function parseGroupChatMentions(text: unknown, members: GroupMember[]) {
  const source = String(text || '')
  const mentioned = new Set<string>()
  let everyone = false
  const handles = new Map<string, string>()

  for (const member of members) {
    const title = String(member.title || '').trim()
    // Normalize legacy "default" handles without aliasing device-qualified
    // defaults to @hermes: that would retarget the primary tag by roster order.
    const handle = String(botHandle(member.name, member) || '').trim()

    const forms = new Set([
      member.name.toLowerCase(),
      member.name.toLowerCase().replace(/[\s_-]+/g, ''),
      ...(handle ? [handle.toLowerCase(), handle.toLowerCase().replace(/[\s_-]+/g, '')] : []),
      ...(title
        ? [title.toLowerCase(), title.toLowerCase().replace(/[\s_-]+/g, ''), title.split(/\s+/)[0].toLowerCase()]
        : [])
    ])

    // Renamed members answer to their friendly names too (profile
    // display_name and Bot Mode title), in slugged and collapsed forms —
    // the same tags the roster autocomplete inserts.
    for (const friendly of botFriendlyNames(member)) {
      for (const form of mentionNameForms(friendly)) {
        forms.add(form)
      }
    }

    // A same-named Connections twin gets `@<name>-<device>` from the registry,
    // but the room's own-source member keeps its bare name and so loses every
    // shared form to the twin (Map last-wins). `@<name>-local` is its
    // always-available unambiguous address.
    if (!member.remoteSource) {
      forms.add(`${member.name.toLowerCase()}-local`)
    }

    for (const form of forms) {
      if (form) {
        handles.set(form, groupMemberKey(member))
      }
    }

    // Normalized slug/collapsed variants of a live identity, gap-filled only — an exact live name elsewhere always wins, so an old handle can never squat (#110200).
    for (const raw of [member.name, handle, title, ...botFriendlyNames(member)]) {
      for (const form of mentionNameForms(raw)) {
        if (form && !handles.has(form)) {
          handles.set(form, groupMemberKey(member))
        }
      }
    }
  }

  // Renamed members answer to previous handles, gap-fill only — every live identity's variants are claimed first, so a live name always wins (#110200).
  for (const member of members) {
    const key = groupMemberKey(member)
    const previous = Array.isArray(member.previous_names) ? member.previous_names : []

    for (const name of previous) {
      for (const form of mentionNameForms(name)) {
        if (form && !handles.has(form)) {
          handles.set(form, key)
        }
      }
    }
  }

  for (const match of source.matchAll(/@([a-z0-9][a-z0-9._-]*)/gi)) {
    const handle = match[1].toLowerCase()

    if (handle === 'everyone' || handle === 'all') {
      everyone = true

      continue
    }

    if (handle === 'user') {
      continue
    }

    const resolved = handles.get(handle) || handles.get(handle.replace(/[._-]+/g, ''))

    if (resolved) {
      mentioned.add(resolved)
    }
  }

  return {
    everyone,
    mentioned
  }
}

/** The `@tag` "Reply to" seeds for one member: its friendly tag when that
 *  routes to this member alone, else the first owner-qualified form that does
 *  (`@<name>-<device>` for a Connections twin, `@<name>-local` for the room's
 *  own-source twin — see #89883). A bare `{ name }` of a member who left the
 *  room resolves to nothing and keeps its friendly tag. */
export function groupReplyMentionTag(member: GroupMember, members: GroupMember[]): string {
  const key = groupMemberKey(member)

  const candidates = [botMentionTag(member), botHandle(member.name, member), `${member.name}-local`]
    .map(tag => String(tag || '').trim())
    .filter(Boolean)

  return (
    candidates.find(tag => {
      const { mentioned } = parseGroupChatMentions(`@${tag}`, members)

      return mentioned.size === 1 && mentioned.has(key)
    }) ||
    candidates[0] ||
    ''
  )
}

/** Members that should take a turn this round: everyone when no member is
 *  @-mentioned in messages since the last user entry (or @everyone appears),
 *  otherwise only the mentioned members. Recomputed every round so a member
 *  pulled in mid-conversation joins the next round. */
export function resolveGroupResponders(log: GroupMessage[], members: GroupMember[]) {
  let sinceLastUser: GroupMessage[] = []

  for (let i = log.length - 1; i >= 0; i--) {
    if (log[i].from.kind === 'user') {
      sinceLastUser = log.slice(i)

      break
    }
  }

  const mentioned = new Set<string>()
  let everyone = false

  for (const entry of sinceLastUser) {
    const parsed = parseGroupChatMentions(entry.text, members)

    if (parsed.everyone) {
      everyone = true
    }

    for (const name of parsed.mentioned) {
      mentioned.add(name)
    }
  }

  if (everyone || mentioned.size === 0) {
    return members
  }

  return members.filter(member => mentioned.has(groupMemberKey(member)))
}

/** Rotate the roster so a different member leads each round. */
export function rotateGroupSpeakers(members: GroupMember[], round: number) {
  if (members.length < 2) {
    return members
  }

  const shift = round % members.length

  return [...members.slice(shift), ...members.slice(0, shift)]
}

// --- member-hold helpers (#93129) — pure, unit-tested ---

/** #93129: classify a USER room message's effect on member holds. Only user
 *  sends ever reach this (bot replies are appended by the round loop, never
 *  through sendToGroupChat), so a bot saying "stopped working on it" can
 *  never set a hold. Conservative on purpose: a standalone stop/halt/pause
 *  word NEXT TO a mention (within two words, #103893) holds those members —
 *  "don't stop @x" therefore also holds, which errs toward the bot staying
 *  quiet until re-addressed (a wrongly-held bot is one mention away from
 *  release; a wrongly-running one keeps doing work it was told to stop).
 *  A stop word far from every mention is ambiguous — "@x go, das ist halt
 *  ein Test" and "@x mach mal Pause" are prose that addresses the bot, so
 *  non-English rooms whose everyday vocabulary overlaps the keyword list
 *  are not silently held — but "@x please just stop now" is a genuine stop,
 *  so the message is NEUTRAL: it neither holds nor releases. A missed hold
 *  is one adjacent "stop @x" away from repair; re-dispatching a bot the user
 *  just told to stop is the one flip this classifier must never make.
 *  A non-stop direct mention releases the mentioned members — the user
 *  addressing a bot directly overrides its hold — and
 *  addressing the whole room (@all / @everyone) without a stop word is the
 *  same intent for every member (#97740): "@all <task>" wakes a stopped
 *  room without the user having to know the literal "resume" incantation. */
export function classifyGroupHoldDirective(
  text: string,
  mentionedKeys: Iterable<string> | null | undefined,
  everyone: boolean
) {
  const value = String(text || '')
  const mentioned = [...(mentionedKeys || [])]
  const stop = stopWordPlacement(value)

  if (stop === 'adjacent') {
    // "@all stop" holds every member — symmetric with "@all resume".
    return {
      hold: mentioned,
      holdAll: Boolean(everyone),
      release: [],
      releaseAll: false
    }
  }

  return {
    hold: [],
    holdAll: false,
    release: stop === 'distant' ? [] : mentioned,
    releaseAll: stop === null && Boolean(everyone)
  }
}

/** #117040: what a fenced block, inline code span, straight-quoted span or
 *  blockquote line says is content the user quotes or pastes, not a room
 *  directive — its stop words must not hold a member. Mask each span down to
 *  the @tokens it contains (mentions resolve from the raw text and must keep
 *  their place in the proximity window, so an address like "…"@impl"…" still
 *  releases a held member) or to one neutral filler word when it has none. */
function maskQuotedAndCodeSpans(value: string): string {
  const mentionsOnly = (span: string): string => (span.match(/@[\p{L}\p{N}._-]+/gu) || []).join(' ') || 'quoted'
  const kept: string[] = []
  let fence = ''

  for (const line of value.split('\n')) {
    if (fence) {
      // Closing fence rows carry no content; an unterminated fence (a
      // cut-short paste) simply swallows the rest of the message.
      const closing = line.trim().startsWith(fence)

      kept.push(closing ? '' : mentionsOnly(line))

      if (closing) {
        fence = ''
      }

      continue
    }

    const opened = line.match(/^\s*(`{3,}|~{3,})/)

    if (opened) {
      fence = opened[1]

      continue
    }

    if (/^\s*>/.test(line)) {
      kept.push(mentionsOnly(line))

      continue
    }

    kept.push(line.replace(/`[^`\n]*`/g, mentionsOnly).replace(/"[^"\n]*"/g, mentionsOnly))
  }

  return kept.join('\n')
}

/** #103893: where the stop/halt/pause tokens sit relative to the @tokens —
 *  `adjacent` when one is within two words of ANY mention, `distant` when
 *  the message carries a stop word but none that close, null without one.
 *  Proximity is measured against the raw @tokens of the DIRECTIVE surface —
 *  quoted/pasted spans are masked first (#117040) —
 *  not the resolved member keys the caller passes (those are roster keys
 *  such as `<connectionId>::<name>`, and a mention resolves through titles
 *  and friendly names too, so the @token text rarely equals the key). The
 *  ≤2 window is fitted to observed directive/filler pairs ("@x please
 *  halt" = 2, "@x go, das ist halt ein Test" = 4); widen only with measured
 *  cases, never by guessing. */
function stopWordPlacement(value: string): 'adjacent' | 'distant' | null {
  const tokens = maskQuotedAndCodeSpans(value).toLowerCase().match(/@[\p{L}\p{N}._-]+|[\p{L}\p{N}_-]+/gu) || []
  const mentionAt: number[] = []
  const stopAt: number[] = []

  tokens.forEach((token, index) => {
    if (token.startsWith('@')) {
      mentionAt.push(index)
    } else if (token === 'stop' || token === 'halt' || token === 'pause') {
      stopAt.push(index)
    }
  })

  if (stopAt.some(stop => mentionAt.some(mention => Math.abs(stop - mention) <= 2))) {
    return 'adjacent'
  }

  return stopAt.length ? 'distant' : null
}

/** What `parseGroupChatMentions` reports for one room message. */
interface GroupMentionParse {
  everyone?: boolean
  mentioned?: Iterable<string>
}

/** #93129: next holds map after one user message. Holds are keyed by
 *  memberKey at ROOM scope (not thread scope): every main-composer send
 *  mints a NEW thread, so a thread-scoped hold would never block the next
 *  send's turns and the stop would not stick. Returns the same object when
 *  nothing changed. */
export function applyGroupHoldDirective(
  holds: Record<string, GroupHoldStamp> | null | undefined,
  mentions: GroupMentionParse | null | undefined,
  text: string,
  stamp: GroupHoldStamp | null | undefined,
  allMemberKeys: string[] = []
): Record<string, GroupHoldStamp> {
  const prior: Record<string, GroupHoldStamp> = holds && typeof holds === 'object' ? holds : {}
  const action = classifyGroupHoldDirective(text, mentions?.mentioned || [], Boolean(mentions?.everyone))

  if (action.releaseAll) {
    return Object.keys(prior).length ? {} : prior
  }

  // "@all stop": expand to every member key the caller knows about.
  const toHold = action.holdAll ? [...allMemberKeys] : action.hold
  let next = prior

  for (const key of toHold) {
    if (next === prior) {
      next = {
        ...prior
      }
    }

    next[key] = {
      at: stamp?.at || Date.now(),
      byMessageId: stamp?.byMessageId || null,
      thread: stamp?.thread || null
    }
  }

  for (const key of action.release) {
    if (Object.prototype.hasOwnProperty.call(next, key)) {
      if (next === prior) {
        next = {
          ...prior
        }
      }

      delete next[key]
    }
  }

  return next
}

// --- end member-hold helpers ---

/** Members cited by @mention in a thread who have not posted any entry after
 *  the citing one — the unresolved-handoff detector for #94478. A mention
 *  inside a member reply is visible to the NEXT round's responder selection,
 *  but the round loop exits first when nobody has new delta to read
 *  (`spokeThisRound === 0`) or a cap lands, so the room settles while a
 *  called bot never answers. Returns member keys still owed a turn. */
export function unaddressedGroupMentions(group: string, members: GroupMember[], thread: string) {
  const room = $groupChats.get()[group] || {
    log: []
  }

  const log = (room.log || []).filter((e: GroupMessage) => groupThreadOf(e) === thread)

  // key → log INDEX of the entry that most recently cited this member.
  // Entry ids are UUIDs (groupChatEntryId), NOT monotonic — index order is
  // the only guaranteed ordering, and it is what "answered after the citing
  // entry" actually means. (#94478 review)
  const citedAt = new Map()

  for (const entry of log) {
    const parsed = parseGroupChatMentions(entry.text || '', members)

    // A user send re-drives everyone anyway; only member-to-member handoffs
    // can strand here.
    if (entry.from.kind !== 'member') {
      continue
    }

    for (const key of parsed.mentioned) {
      const citingMemberKey = (() => {
        const m = members.find((mm: GroupMember) => mm.name === entry.from?.name)

        return m ? groupMemberKey(m) : null
      })()

      // Never count a bot citing itself as a pending handoff.
      if (citingMemberKey && citingMemberKey !== key) {
        citedAt.set(key, log.indexOf(entry))
      }
    }
  }

  // A citation is answered when the cited member posts any entry after the
  // citing one (its turn, whatever the content).
  const lastPostAt = new Map()

  for (const entry of log) {
    if (entry.from.kind !== 'member') {
      continue
    }

    const speakerKey = (() => {
      const m = members.find((mm: GroupMember) => mm.name === entry.from?.name)

      return m ? groupMemberKey(m) : null
    })()

    if (speakerKey) {
      lastPostAt.set(speakerKey, log.indexOf(entry))
    }
  }

  return [...citedAt.keys()].filter(key => {
    const citedIdx = citedAt.get(key)
    const answeredIdx = lastPostAt.get(key)

    return answeredIdx === undefined || answeredIdx <= citedIdx
  })
}

/** #91868/#94569: the REAL stop path for a group round. The round loop's only
 *  cancellation primitives were the epoch bump (checked at member boundaries)
 *  and #93129 holds (skip FUTURE turns) — neither touches the member whose
 *  model call is in flight RIGHT NOW, so "stop" meant "finish this turn
 *  first". This primitive does all three legs atomically enough to matter:
 *
 *  1. Bumps the room epoch — the driving loop bails at its next boundary and
 *     never selects another member (`isCurrent()` in runGroupChatRounds).
 *  2. Sets a #93129 hold for EVERY member — future turns stay skipped until
 *     the user explicitly releases (resume / @all resume / direct mention),
 *     the exact contract user-typed "@all stop" already has.
 *  3. Sends session.interrupt to the member currently ON TURN (room.turn,
 *     runtime-only) via its own route, so the in-flight model call actually
 *     dies instead of grinding to completion in the background. Best-effort:
 *     an unreachable member still leaves the room stopped — the poll loop's
 *     staleness check (epoch moved AND member held) abandons the turn.
 *
 *  `members` is the live roster when the caller has one (the workspace);
 *  falls back to the room's durable roster so a two-arg call still works. */
export async function stopGroupThread(group: string, thread: null | string, members: GroupMember[] | null = null) {
  const room = $groupChats.get()[group] || {}
  const roster = Array.isArray(members) && members.length ? members : room.members || []
  const onTurn = room.turn || null
  groupChatDrives.get(groupChatRoomKey(group, room))?.pending.clear()

  const stamp: GroupHoldStamp = {
    at: Date.now(),
    byMessageId: null,
    thread: thread || null
  }

  updateGroupChat(group, (r: GroupChatRoom) => {
    r.epoch = (r.epoch || 0) + 1
    r.running = false
    r.turn = null

    // Same hold shape applyGroupHoldDirective mints for "@all stop" — the
    // held-skip path (watermark consume + 'held' activity note) and every
    // release gesture apply unchanged. An existing hold keeps its stamp.
    const holds: Record<string, GroupHoldStamp> = {
      ...(r.holds || {})
    }

    for (const member of roster) {
      const key = groupMemberKey(member)

      if (key && !holds[key]) {
        holds[key] = {
          ...stamp
        }
      }
    }

    r.holds = holds

    return r
  })

  // Recorded AFTER the bump so the event is tagged with the new epoch — it
  // stays visible as the current run's outcome instead of dropping out of
  // view with the superseded run's events.
  recordGroupActivity(group, {
    kind: 'stopped',
    member: 'You',
    thread: thread || null
  })

  // The captured descriptor owns routing even if the roster has changed.
  // Sessions are per thread, so a stop targets the session of the thread it
  // was issued from; an unmigrated room still answers on its bare pointer.
  const sessions = room.sessions || {}
  const onTurnKey = onTurn ? groupMemberKey(onTurn) : ''

  const sessionId = onTurn
    ? sessions[groupSessionKey(thread || 'legacy', onTurn)] ||
      (hasThreadScopedGroupSession(sessions, onTurnKey) ? null : sessions[onTurnKey])
    : null

  if (onTurn && sessionId) {
    try {
      await requestForBot(onTurn, 'session.interrupt', {
        session_id: sessionId
      })
    } catch {
      /* best-effort — the epoch/hold legs above already stopped the room;
         the abandoned poll loop exits on its staleness check */
    }
  }
}

/** Drive one bounded round-robin turn for ONE THREAD. Serial — one member at
 *  a time. User follow-ups queue behind this drive; Stop invalidates its
 *  epoch and discards queued continuations.
 *  Watermarks are per thread+member (`${thread}::${memberKey}`), so parallel
 *  topics never eat each other's deltas. */
export async function runGroupChatRounds(group: string, members: GroupMember[], thread: string, failedMembers = new Set<string>()) {
  const binding = followGroupChat(group, name => {
    group = name
  })

  const startEpoch = ($groupChats.get()[group] || {}).epoch || 0
  const isCurrent = () => binding.isLive() && (($groupChats.get()[group] || {}).epoch || 0) === startEpoch

  const context = {
    get group() {
      return group
    },
    members,
    thread,
    startEpoch,
    failedMembers,
    binding,
    isCurrent
  }

  let posted = 0
  let continuations = 0
  // #94478: how this drive ended. 'settled' means quiet consensus (everyone
  // passed with nothing pending); 'capped' means a round/message/continuation
  // cap forced the exit — the activity feed must tell those apart.
  let exitKind: 'capped' | 'settled' = 'settled'

  try {
    for (let round = 0; round < GROUP_CHAT_MAX_ROUNDS; round++) {
      // Deliver any replies that finished after their turn timed out —
      // every member, not just this round's responders, so long work is
      // late, never lost.
      for (const member of members) {
        if (!isCurrent()) {
          recordGroupActivity(group, {
            kind: 'cancelled',
            member: null,
            thread
          })

          return
        }

        await harvestStrandedGroupReply(group, member)

        if (!binding.isLive()) {
          return
        }
      }

      const roomLog = (($groupChats.get()[group] || {}).log || []).filter(
        (e: GroupMessage) => groupThreadOf(e) === thread
      )

      // Exclude members the harvest pass just above confirmed are STILL
      // running (their stranded marker survived harvest because
      // state.inflight/running was true). Re-selecting one here would
      // prompt.submit into their live session — the gateway's default busy
      // policy redirects or hard-interrupts that turn (tui_gateway's
      // _handle_busy_submit), killing exactly the long-running work this
      // stranded/harvest mechanism exists to protect. Skip them; the next
      // harvest pass picks the reply up once it actually lands. A marker's
      // mere presence means "still stranded" (harvestStrandedGroupReply
      // deletes it once the member is confirmed done/dead) — presence, not
      // value shape, since markers are a bare number pre-thread or
      // {before, thread} post-thread.
      const strandedNow = ($groupChats.get()[group] || {}).stranded || {}

      const responders = rotateGroupSpeakers(resolveGroupResponders(roomLog, members), round).filter(
        (member: GroupMember) => !Object.prototype.hasOwnProperty.call(strandedNow, groupMemberKey(member))
      )

      let spokeThisRound = 0

      for (const member of responders) {
        if (!isCurrent() || posted >= GROUP_CHAT_MAX_MESSAGES) {
          if (!isCurrent()) {
            recordGroupActivity(group, {
              kind: 'cancelled',
              member: null,
              thread
            })
          } else {
            exitKind = 'capped' // message cap, not consensus (#94478)
          }

          return
        }

        const result = await runGroupRoundMember(context, member)

        if (!binding.isLive() || result === null) {
          return
        }

        if (result) {
          posted += 1
          spokeThisRound += 1
        }
      }

      if (spokeThisRound === 0) {
        // #94478: "everyone passed" is NOT the only way a round can go quiet —
        // responders can be narrowed to members with no new delta while the
        // thread's tail carries an @mention handoff that was never answered.
        // Before settling, check for cited members still owed a turn and run
        // one bounded continuation round for exactly those members. If none
        // exist (or the continuation also goes quiet), the room genuinely
        // settled.
        const pendingKeys = unaddressedGroupMentions(group, members, thread)

        // #94478 review: bound continuation rounds independently of the
        // message cap so a pathological mention chain can't consume the
        // room's entire budget on back-and-forth handoffs.
        continuations += 1

        const continued = await runGroupContinuationMembers(context, pendingKeys, continuations, posted)

        if (!binding.isLive() || continued === null) {
          return
        }

        posted += continued
        spokeThisRound += continued

        if (spokeThisRound === 0) {
          // Genuinely nothing left to say — including after the continuation
          // attempt above produced no spoken turns. Settle honestly, but if
          // cited members are STILL owed a turn and only the continuation /
          // message caps stopped us from driving them, this is a capped
          // exit, not consensus. (#94478)
          if (
            pendingKeys.length &&
            (continuations > GROUP_CHAT_MAX_CONTINUATIONS || posted >= GROUP_CHAT_MAX_MESSAGES)
          ) {
            exitKind = 'capped'
          }

          return
        }
      }
    }

    // All GROUP_CHAT_MAX_ROUNDS rounds ran with someone still speaking —
    // the round cap ended the drive, not consensus. (#94478)
    exitKind = 'capped'
  } finally {
    if (isCurrent()) {
      recordGroupActivity(group, {
        kind: exitKind,
        member: null,
        thread
      })
      updateGroupChat(group, (r: GroupChatRoom) => {
        r.running = false
        r.turn = null

        return r
      })

      // #89545: the loop's harvest pass only ran at the top of each round of
      // an ACTIVE loop — a member whose turn timed out after the final round
      // stayed stranded until the user's NEXT send. Poll for the late reply
      // in the background (bounded) so long work is late, never lost.
      // (window feature-detect: the engine also runs under node in tests.)
      const strandedLeft = Object.keys(($groupChats.get()[group] || {}).stranded || {})

      if (strandedLeft.length && typeof window !== 'undefined') {
        void harvestStrandedUntilSettled(group, members, thread)
      }
    }

    binding.dispose()
  }
}

/** Bounded background harvest for members whose replies outlived the turn
 *  loop. Watches for a further hard-cap duration plus a minute of grace
 *  after foreground polling ends; stops early when nothing is
 *  stranded, a new loop takes the room over (it harvests on its own), or the
 *  room record disappears (disband). */
async function harvestStrandedUntilSettled(group: string, members: GroupMember[], thread: string) {
  const binding = followGroupChat(group, name => {
    group = name
  })

  try {
    const HARVEST_INTERVAL_MS = 5000
    const HARVEST_MAX_TRIES = Math.ceil((GROUP_TURN_HARD_CAP_MS + 60000) / HARVEST_INTERVAL_MS)

    for (let attempt = 0; attempt < HARVEST_MAX_TRIES; attempt++) {
      await new Promise(resolve => window.setTimeout(resolve, HARVEST_INTERVAL_MS))
      const room = $groupChats.get()[group]

      if (!binding.isLive() || !room || room.running) {
        return
      }

      const stranded = room.stranded || {}

      if (!Object.keys(stranded).length) {
        return
      }

      for (const member of members) {
        if (!binding.isLive()) {
          return
        }

        if (Object.prototype.hasOwnProperty.call(stranded, groupMemberKey(member))) {
          try {
            await harvestStrandedGroupReply(group, member)
          } catch {
            // Best-effort: the next tick retries; the bound stops runaways.
          }
        }
      }
    }

    if (!binding.isLive()) {
      return
    }

    recordGroupActivity(group, {
      kind: 'failed',
      member: null,
      thread
    })
  } finally {
    binding.dispose()
  }
}

/** User send into a group room. `thread` continues that thread (its reply
 *  box); omitted/null mints a NEW thread — the main composer's Slack shape.
 *  Appends and queues the target thread behind the active room drive,
 *  without redirecting a member whose inference is still in flight.
 *  Returns the thread id the message landed in. */
export function sendToGroupChat(
  group: string,
  members: GroupMember[],
  text: string,
  thread?: null | string,
  images?: Attachment[]
): null | string {
  const trimmed = String(text || '').trim()

  if (rejectGroupSlashCommand(trimmed)) {
    return null
  }

  const attached = Array.isArray(images) ? images.filter((img: Attachment) => img && img.data) : []

  if (!trimmed && !attached.length) {
    return null
  }

  // An empty member seat (roster hydration race, meta clobber, legacy room
  // record without member descriptors) used to swallow the send: a fully
  // typed message vanished with no thread and no error. Surface it — the
  // caller keeps the draft, so nothing is lost.
  if (!members.length) {
    host.notify({
      kind: 'error',
      message: botsText().group.noMembersToSend(group)
    })

    return null
  }

  const target = thread || mintGroupThreadId()
  $groupNeedsYou.set({
    ...$groupNeedsYou.get(),
    [group]: false
  })
  // Refresh the durable room roster on every send. This backfills rooms made
  // by older Desktop builds and keeps the gateway mirror complete even when
  // members overlap across multiple groups.
  updateGroupChat(group, (room: GroupChatRoom) => {
    room.members = durableGroupChatMembers(members)

    return room
  })

  const sent = appendGroupChatEntry(
    group,
    {
      kind: 'user',
      name: 'You'
    },
    trimmed,
    target,
    attached
  )

  updateGroupChat(group, (room: GroupChatRoom) => {
    // #93129: user text is the ONLY input that changes member holds. An
    // explicit "stop @member" sets a sticky hold; "@member resume" (or
    // @all resume, or any direct non-stop mention of the held member)
    // releases it. Bot replies never flow through this function.
    room.holds = applyGroupHoldDirective(
      room.holds,
      parseGroupChatMentions(trimmed, members),
      trimmed,
      {
        at: sent?.at,
        byMessageId: sent?.id,
        thread: target
      },
      members.map((member: GroupMember) => groupMemberKey(member))
    )

    return room
  })
  recordGroupActivity(group, {
    kind: 'queued',
    member: 'You',
    thread: target
  })

  queueGroupChatDrive(group, members, target)

  return target
}

interface GroupChatDrive {
  failedMembers: Set<string>
  pending: Map<string, GroupMember[]>
  binding: ReturnType<typeof followGroupChat>
}

// Keep the owner until its awaited member releases, even after Stop. A
// rename follows the room identity; disband retires the binding permanently.
const groupChatDrives = new Map<string, GroupChatDrive>()

function queueGroupChatDrive(group: string, members: GroupMember[], thread: string) {
  let key = groupChatRoomKey(group, $groupChats.get()[group])
  const active = groupChatDrives.get(key)

  if (active?.binding.isLive()) {
    // Only a new user action AFTER failure authorizes another attempt.
    active.failedMembers.clear()
    active.pending.set(thread, members)

    return
  }

  const binding = followGroupChat(group, name => {
    groupChatDrives.delete(key)
    group = name
    key = groupChatRoomKey(group, $groupChats.get()[group])
    groupChatDrives.set(key, drive)
  })

  const drive: GroupChatDrive = { pending: new Map([[thread, members]]), failedMembers: new Set(), binding }
  groupChatDrives.set(key, drive)
  // Queued threads share the activity epoch, so draining one cannot hide
  // unresolved failures from the preceding thread. Stop still invalidates it.
  updateGroupChat(group, room => ({ ...room, epoch: (room.epoch || 0) + 1 }))

  void (async () => {
    let currentThread = thread

    try {
      while (binding.isLive() && drive.pending.size) {
        const [nextThread, nextMembers] = drive.pending.entries().next().value!
        currentThread = nextThread
        drive.pending.delete(nextThread)
        updateGroupChat(group, room => ({ ...room, running: true }))
        await runGroupChatRounds(group, nextMembers, nextThread, drive.failedMembers)
      }
    } catch (error) {
      if (binding.isLive()) {
        const reason = groupFailureReason(error)
        recordGroupActivity(group, { kind: 'failed', member: null, thread: currentThread, ...(reason ? { reason } : {}) })
        updateGroupChat(group, room => ({ ...room, running: false, turn: null }))
      }
    } finally {
      binding.dispose()

      if (groupChatDrives.get(key) === drive) {
        groupChatDrives.delete(key)
      }
    }
  })()
}
