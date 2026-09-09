/**
 * Room-level coordination: who speaks, in what order, for how long — the
 * @mention parse, the round-robin driver, the #93129 member holds, the stop
 * path, and the user send that starts it all.
 */
import { botFriendlyNames, botHandle, mentionNameForms } from './data'
import { desktopRoomIdentity } from './desktop-room-command-client'
import { recordGroupActivity } from './group-activity'
import {
  $groupChats,
  $groupNeedsYou,
  appendGroupChatEntry,
  GROUP_CHAT_MAX_CONTINUATIONS,
  GROUP_CHAT_MAX_MESSAGES,
  GROUP_CHAT_MAX_ROUNDS,
  groupChatHostedGateway,
  groupThreadOf,
  mintGroupThreadId,
  persistGroupChatRoomsRequired,
  updateGroupChat
} from './group-chat'
import type { GroupChatRoom, GroupHoldStamp } from './group-chat'
import {
  bindGroupCommandFence,
  cancelGroupCommandFence,
  groupCommandFenceLive,
  groupCommandFenceMatches
} from './group-command-fence'
import type { GroupCommandFence } from './group-command-fence'
import { desktopCommandResult, settleDesktopCommand } from './group-command-receipts'
import { durableGroupChatMembers, followGroupChat, groupMemberKey } from './group-membership'
import { runGroupContinuationMembers, runGroupRoundMember } from './group-round-members'
import { harvestStrandedGroupReply } from './group-turns'
import {
  beginHostedRoomMutation,
  groupChatContinuityReady,
  hostedRoomMutationIsCurrent,
  queueHostedGroupChat,
  stopHostedGroupChat
} from './hosted-room-runtime'
import { botsText } from './i18n'
import { requestForBot } from './routing'
import type { Attachment, GroupMember, GroupMessage } from './types'

function hostedConnectionName(room: null | Partial<GroupChatRoom> | undefined) {
  return room?.members?.find(member => member.connectionLabel)?.connectionLabel || botsText().group.thisHost
}

interface SendGroupChatOptions {
  commandFence?: GroupCommandFence
  entryId?: string
  userName?: string
}

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
    // Cross-connection members are also addressable by their @name-device
    // handle (the roster's disambiguated form) — same-named agents on two
    // machines resolve to the right one.
    const handle = String(member.handle || botHandle(member.name, member) || '').trim()

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

    for (const form of forms) {
      if (form) {
        handles.set(form, groupMemberKey(member))
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
 *  never set a hold. Conservative on purpose: any standalone stop/halt/pause
 *  word next to a mention holds those members — "don't stop @x" therefore
 *  also holds, which errs toward the bot staying quiet until re-addressed
 *  (a wrongly-held bot is one mention away from release; a wrongly-running
 *  one keeps doing work it was told to stop). A non-stop direct mention
 *  releases the mentioned members — the user addressing a bot directly
 *  overrides its hold. */
export function classifyGroupHoldDirective(
  text: string,
  mentionedKeys: Iterable<string> | null | undefined,
  everyone: boolean
) {
  const value = String(text || '')
  const mentioned = [...(mentionedKeys || [])]
  const stop = /\b(stop|halt|pause)\b/i.test(value)
  const resume = /\b(resume|continue|go|proceed)\b/i.test(value)

  if (stop) {
    // "@all stop" holds every member — symmetric with "@all resume".
    return {
      hold: mentioned,
      holdAll: Boolean(everyone),
      release: [],
      releaseAll: false
    }
  }

  if (resume) {
    return {
      hold: [],
      holdAll: false,
      release: mentioned,
      releaseAll: Boolean(everyone)
    }
  }

  return {
    hold: [],
    holdAll: false,
    release: mentioned,
    releaseAll: false
  }
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

  if (groupChatHostedGateway(room)) {
    if (room.hostedStatus?.state === 'stopping') {
      return
    }

    const connectionName = hostedConnectionName(room)
    const roomId = String(room.roomId || '')
    const generation = beginHostedRoomMutation(roomId)

    updateGroupChat(
      group,
      current => ({
        ...current,
        running: true,
        hostedStatus: {
          state: 'stopping',
          label: botsText().group.hostedStopping
        }
      }),
      {
        sync: false
      }
    )

    try {
      const acknowledged = await stopHostedGroupChat(group)

      if (!hostedRoomMutationIsCurrent(roomId, generation)) {
        return
      }

      updateGroupChat(
        group,
        current => ({
          ...current,
          running: !acknowledged,
          hostedStatus: {
            state: acknowledged ? 'stopped' : 'queued',
            label: acknowledged ? botsText().group.hostedStopped : botsText().group.hostedStopQueued(connectionName)
          },
          continuityIssue: acknowledged ? null : botsText().group.hostedStopQueuedHint(connectionName)
        }),
        {
          sync: false
        }
      )
    } catch {
      if (!hostedRoomMutationIsCurrent(roomId, generation)) {
        return
      }

      updateGroupChat(
        group,
        current => ({
          ...current,
          running: false,
          hostedStatus: {
            state: 'offline',
            label: botsText().group.hostedUnavailable(connectionName)
          },
          continuityIssue: botsText().group.hostedReconnectToStop(connectionName)
        }),
        {
          sync: false
        }
      )
    }

    if (hostedRoomMutationIsCurrent(roomId, generation)) {
      recordGroupActivity(group, {
        kind: 'stopped',
        member: 'You',
        thread: thread || null
      })
    }

    return
  }

  const roster = Array.isArray(members) && members.length ? members : room.members || []
  const turnName = room.turn || null

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

  // Interrupt the member actually mid-turn. room.turn is runtime-only and
  // names exactly one member (the loop is serial); a settled room has none.
  const onTurn = turnName ? roster.find((member: GroupMember) => member?.name === turnName) : null
  const sessionId = onTurn ? (room.sessions || {})[groupMemberKey(onTurn)] : null

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

/** Fence a classic turn after its mailbox lease is lost, without adding the
 * durable holds that belong only to an explicit user Stop. */
export function cancelGroupThreadForLeaseLoss(group: string, members: GroupMember[] | null, fence: GroupCommandFence) {
  cancelGroupCommandFence(fence)
  const room = $groupChats.get()[group] || {}

  // A later user send or replacement room owns a different drive. Cancelling
  // this command must not interrupt it or change its running/hold state.
  if (
    (fence.roomValid && !fence.roomValid()) ||
    desktopRoomIdentity(group, room) !== fence.roomId ||
    fence.epoch === null ||
    (room.epoch || 0) !== fence.epoch
  ) {
    return
  }

  const roster = Array.isArray(members) && members.length ? members : room.members || []
  const turnName = room.turn || null

  updateGroupChat(group, current => ({
    ...current,
    epoch: (current.epoch || 0) + 1,
    running: false,
    turn: null
  }))

  const onTurn = turnName ? roster.find(member => member?.name === turnName) : null
  const sessionId = onTurn ? (room.sessions || {})[groupMemberKey(onTurn)] : null

  if (onTurn && sessionId) {
    void Promise.resolve()
      .then(() => requestForBot(onTurn, 'session.interrupt', { session_id: sessionId }))
      .catch(() => undefined)
  }
}

/** Drive one bounded round-robin turn for ONE THREAD. Serial — one member at
 *  a time. A newer user send bumps the room epoch; this loop notices at the
 *  next member boundary, bails, and the newest send's own loop takes over.
 *  Watermarks are per thread+member (`${thread}::${memberKey}`), so parallel
 *  topics never eat each other's deltas. */
export async function runGroupChatRounds(
  group: string,
  members: GroupMember[],
  thread: string,
  fence?: GroupCommandFence
) {
  const binding = followGroupChat(group, name => {
    group = name
  })

  const leaseLive = () =>
    binding.isLive() && groupCommandFenceMatches(fence, desktopRoomIdentity(group, $groupChats.get()[group]), thread)
  if (!leaseLive() || (fence && fence.epoch !== ($groupChats.get()[group]?.epoch || 0))) {
    binding.dispose()
    return
  }
  const deliveryFailed = new Set<string>()
  const startEpoch = ($groupChats.get()[group] || {}).epoch || 0
  const isCurrent = () => leaseLive() && (($groupChats.get()[group] || {}).epoch || 0) === startEpoch

  const context = {
    get group() {
      return group
    },
    members,
    thread,
    startEpoch,
    binding,
    leaseLive,
    fence,
    deliveryFailed,
    isCurrent
  }

  let posted = 0
  let continuations = 0
  // #94478: how this drive ended. 'settled' means quiet consensus (everyone
  // passed with nothing pending); 'capped' means a round/message/continuation
  // cap forced the exit — the activity feed must tell those apart.
  let exitKind: 'capped' | 'failed' | 'settled' = 'settled'

  try {
    if (fence) {
      try {
        // Persist the command identity before its first Bot can perform work.
        await persistGroupChatRoomsRequired()
      } catch {
        fence.persistenceFailed = true
        cancelGroupThreadForLeaseLoss(group, members, fence)

        return
      }
    }

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

        const memberKey = groupMemberKey(member)
        const marker = $groupChats.get()[group]?.stranded?.[memberKey]
        const commandId = typeof marker === 'object' ? marker?.classicTurn?.mailboxCommandId : undefined

        if (commandId && commandId !== fence?.commandId) {
          // A new input retires unresolved output owned by an older command.
          updateGroupChat(group, current => ({
            ...current,
            desktopCommandSettled: settleDesktopCommand(group, current, commandId, 'send', {
              room_name: group,
              stopped: true
            })
          }))

          try {
            // Persist retirement before a newer Bot turn can perform work.
            await persistGroupChatRoomsRequired()
          } catch {
            exitKind = 'failed'

            if (fence) {
              fence.persistenceFailed = true
              cancelGroupThreadForLeaseLoss(group, members, fence)
            }

            return
          }

          if (!isCurrent()) {
            return
          }

          updateGroupChat(group, current => {
            const stranded = { ...(current.stranded || {}) }
            delete stranded[memberKey]

            return { ...current, stranded }
          })
        }

        await harvestStrandedGroupReply(group, member, fence)

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
        (member: GroupMember) =>
          !deliveryFailed.has(groupMemberKey(member)) &&
          !Object.prototype.hasOwnProperty.call(strandedNow, groupMemberKey(member))
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
    const externalIds = (Array.isArray(($groupChats.get()[group] || {}).log) ? $groupChats.get()[group].log : [])
      .filter(entry => entry?.external && groupThreadOf(entry) === thread && entry?.id)
      .map(entry => String(entry.id))

    if (isCurrent()) {
      const pendingCommands = new Set(
        Object.values($groupChats.get()[group]?.stranded || {})
          .map(marker => (typeof marker === 'object' ? marker?.classicTurn?.mailboxCommandId : undefined))
          .filter(Boolean)
      )

      recordGroupActivity(group, {
        kind: pendingCommands.has(fence?.commandId) ? 'failed' : exitKind,
        member: null,
        thread
      })
      updateGroupChat(group, (r: GroupChatRoom) => {
        r.running = false
        r.turn = null

        for (const id of externalIds) {
          if (exitKind !== 'failed' && !pendingCommands.has(id)) {
            r.desktopCommandSettled = settleDesktopCommand(group, r, id, 'send', {
              room_name: group,
              thread_id: thread
            })
          }
        }

        return r
      })

      // #89545: the loop's harvest pass only ran at the top of each round of
      // an ACTIVE loop — a member whose turn timed out after the final round
      // stayed stranded until the user's NEXT send. Poll for the late reply
      // in the background (bounded) so long work is late, never lost.
      // (window feature-detect: the engine also runs under node in tests.)
      const strandedLeft = Object.keys(($groupChats.get()[group] || {}).stranded || {})

      if (!fence && strandedLeft.length && typeof window !== 'undefined') {
        void harvestStrandedUntilSettled(group, members, thread)
      }
    }

    binding.dispose()
  }
}

/** Bounded background harvest for members whose replies outlived the turn
 *  loop. Polls every 5s for up to 5 minutes; stops early when nothing is
 *  stranded, a new loop takes the room over (it harvests on its own), or the
 *  room record disappears (disband). */
async function harvestStrandedUntilSettled(group: string, members: GroupMember[], thread: string) {
  const binding = followGroupChat(group, name => {
    group = name
  })

  try {
    const HARVEST_INTERVAL_MS = 5000
    const HARVEST_MAX_TRIES = 60

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

/** Composer send with a durable hosted boundary. File staging and verified
 * outbox insertion hold the room-order lock; only then is the optimistic
 * message painted. Classic rooms retain the synchronous round engine. */
export async function sendToGroupChatDurably(
  group: string,
  members: GroupMember[],
  text: string,
  thread?: null | string,
  images?: Attachment[]
) {
  const room = $groupChats.get()[group]

  if (!groupChatHostedGateway(room)) {
    return sendToGroupChat(group, members, text, thread, images)
  }

  const trimmed = String(text || '').trim()
  const attached = Array.isArray(images) ? images.filter((image): image is Attachment => Boolean(image?.data)) : []

  if ((!trimmed && !attached.length) || !members.length || room.hostedStatus?.state === 'deleted') {
    return null
  }

  if (!groupChatContinuityReady(room)) {
    updateGroupChat(
      group,
      current => ({
        ...current,
        continuityIssue: current.continuityIssue || current.hostedStatus?.label || botsText().group.hostedSyncing
      }),
      { sync: false }
    )

    return null
  }

  const target = thread || mintGroupThreadId()
  const commandId = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2)}`
  const roomId = String(room.roomId || '')
  const generation = beginHostedRoomMutation(roomId)

  const message: GroupMessage = {
    at: Date.now(),
    from: { kind: 'user', name: 'You' },
    id: commandId,
    ...(attached.length ? { images: attached } : {}),
    text: trimmed,
    thread: target
  }

  await queueHostedGroupChat(group, message, target)

  if (!hostedRoomMutationIsCurrent(roomId, generation) || !$groupChats.get()[group]) {
    return target
  }

  $groupNeedsYou.set({
    ...$groupNeedsYou.get(),
    [group]: false
  })
  updateGroupChat(group, current => ({
    ...current,
    members: durableGroupChatMembers(members),
    running: true,
    hostedStatus: {
      state: 'queued',
      label: botsText().group.hostedQueued(hostedConnectionName(room))
    },
    continuityIssue: null
  }))
  const visibleAttachments = attached.map(({ uploadId: _uploadId, ...attachment }) => attachment)

  appendGroupChatEntry(group, message.from, trimmed, target, visibleAttachments, { entryId: commandId })
  recordGroupActivity(group, { kind: 'queued', member: 'You', thread: target })

  return target
}

/** User send into a group room. `thread` continues that thread (its reply
 *  box); omitted/null mints a NEW thread — the main composer's Slack shape.
 *  Appends, bumps the room epoch (supersedes any running loop at its next
 *  member boundary), and starts the turn drive for the target thread.
 *  Returns the thread id the message landed in. */
export function sendToGroupChat(
  group: string,
  members: GroupMember[],
  text: string,
  thread?: null | string,
  images?: Attachment[],
  options: SendGroupChatOptions = {}
): null | string {
  const trimmed = String(text || '').trim()
  const attached = Array.isArray(images) ? images.filter((img: Attachment) => img && img.data) : []
  const roomBeforeSend = $groupChats.get()[group]
  const hosted = groupChatHostedGateway(roomBeforeSend)
  const externalId = String(options.entryId || '').trim()
  const fence = options.commandFence

  if (
    !groupCommandFenceLive(fence) ||
    (fence && (fence.roomId !== desktopRoomIdentity(group, roomBeforeSend) || fence.commandId !== externalId))
  ) {
    return null
  }

  const userName =
    String(options.userName || 'You')
      .trim()
      .slice(0, 128) || 'You'

  if ((!trimmed && !attached.length) || !members.length) {
    return null
  }

  if (hosted && roomBeforeSend?.hostedStatus?.state === 'deleted') {
    return null
  }

  if (!groupChatContinuityReady(roomBeforeSend)) {
    updateGroupChat(
      group,
      current => ({
        ...current,
        continuityIssue: current.continuityIssue || current.hostedStatus?.label || botsText().group.hostedSyncing
      }),
      { sync: false }
    )

    return null
  }

  if (hosted) {
    return null
  }

  // Runtime callers verify the durable snapshot before entering this function.
  // A retained settlement must never mutate the log or restart a trimmed turn.
  if (externalId && Object.hasOwn(roomBeforeSend?.desktopCommandSettled || {}, externalId)) {
    return desktopCommandResult(group, roomBeforeSend, externalId, 'send')?.thread_id || null
  }

  const target = thread || mintGroupThreadId()

  if (externalId) {
    const existing = (Array.isArray(roomBeforeSend?.log) ? roomBeforeSend.log : []).find(
      entry => entry?.id === externalId
    )

    if (existing) {
      if (!existing.external || existing.text !== trimmed || existing.from?.name !== userName) {
        return null
      }

      const existingThread = existing.thread || 'legacy'

      if (roomBeforeSend?.desktopCommandSettled?.[externalId] || roomBeforeSend?.running) {
        return existingThread
      }

      updateGroupChat(group, current => ({
        ...current,
        members: durableGroupChatMembers(members),
        epoch: (current.epoch || 0) + 1,
        running: true
      }))
      recordGroupActivity(group, { kind: 'queued', member: userName, thread: existingThread })
      bindGroupCommandFence(fence, existingThread, $groupChats.get()[group].epoch || 0)
      const binding = followGroupChat(group, name => {
        group = name
      })
      void runGroupChatRounds(group, members, existingThread, fence)
        .catch(() => {
          if (binding.isLive()) {
            updateGroupChat(group, current => ({ ...current, running: false }))
          }
        })
        .finally(binding.dispose)

      return existingThread
    }
  }

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
      name: userName
    },
    trimmed,
    target,
    attached,
    { entryId: externalId, external: Boolean(externalId) }
  )

  if (!sent) {
    return null
  }

  const wasRunning = ($groupChats.get()[group] || {}).running === true
  updateGroupChat(group, (room: GroupChatRoom) => {
    room.epoch = (room.epoch || 0) + 1
    room.running = true
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
    member: userName,
    thread: target
  })
  bindGroupCommandFence(fence, target, $groupChats.get()[group].epoch || 0)

  const binding = followGroupChat(group, name => {
    group = name
  })

  const drive = () => {
    if (!binding.isLive()) {
      binding.dispose()

      return
    }

    void runGroupChatRounds(group, members, target, fence)
      .catch(() => {
        if (binding.isLive()) {
          updateGroupChat(group, (r: GroupChatRoom) => {
            r.running = false

            return r
          })
        }
      })
      .finally(binding.dispose)
  }

  if (!wasRunning) {
    drive()
  } else {
    // Preserve the existing newer-send handoff delay, without pinning its name.
    setTimeout(drive, 250)
  }

  return target
}
