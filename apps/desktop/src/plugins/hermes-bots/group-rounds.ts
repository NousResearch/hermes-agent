/**
 * Room-level coordination: who speaks, in what order, for how long — the
 * @mention parse, the round-robin driver, the #93129 member holds, the stop
 * path, and the user send that starts it all.
 */

import { botFriendlyNames, botHandle, clearBotAttention, mentionNameForms, noteBotAttention } from './data'
import { recordGroupActivity } from './group-activity'
import {
  $groupChats,
  $groupNeedsYou,
  appendGroupChatEntry,
  GROUP_CHAT_HISTORY_LIMIT,
  GROUP_CHAT_MAX_CONTINUATIONS,
  GROUP_CHAT_MAX_MESSAGES,
  GROUP_CHAT_MAX_ROUNDS,
  groupSpeakerLabel,
  groupThreadOf,
  mintGroupThreadId,
  shouldCommitMemberTurn,
  updateGroupChat
} from './group-chat'
import type { GroupChatRoom, GroupHoldStamp } from './group-chat'
import { durableGroupChatMembers, groupMemberKey } from './group-membership'
import { harvestStrandedGroupReply, isGroupPassText, runGroupChatMemberTurn } from './group-turns'
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

/** Room-log line as a member sees it: `Name (user): …` / `Name: …` /
 *  `Name (you): …`. */
export function formatGroupChatLine(entry: GroupMessage, viewerName: string) {
  // Attachments are staged into each member's session as real payloads; the
  // transcript line names them so the delta text and the bytes line up.
  const attached =
    Array.isArray(entry.images) && entry.images.length
      ? ` ${entry.images
          .map(img => {
            const label = img.kind === 'pdf' ? 'attached PDF' : img.kind === 'file' ? 'attached file' : 'attached image'

            return `[${label}: ${img.name || 'image'}]`
          })
          .join(' ')}`
      : ''

  if (entry.from.kind === 'user') {
    return `${entry.from.name || 'User'} (user): ${entry.text}${attached}`
  }

  const suffix = entry.from.name === viewerName ? ' (you)' : ''
  // Cross-connection speakers carry their device so same-named agents on
  // two machines stay tellable apart in every member's transcript.
  const source = entry.from.source ? ` [${entry.from.source}]` : ''

  return `${groupSpeakerLabel(entry.from.name)}${suffix}${source}: ${entry.text}${attached}`
}

interface GroupChatTurnPromptInput {
  deltaLines: string[]
  groupName: string
  members: GroupMember[]
  viewer: GroupMember
}

/** The full per-turn payload for one member: participation rules + the room
 *  delta. Rules travel in the turn payload (not SOUL) so every existing bot
 *  can join a group chat without a profile migration. */
export function buildGroupChatTurnPrompt({ groupName, members, viewer, deltaLines }: GroupChatTurnPromptInput) {
  const viewerKey = groupMemberKey(viewer)
  const peers = members.filter(m => groupMemberKey(m) !== viewerKey)

  const peerNames = peers
    .map(m => {
      const handle = m.title ? `${m.title} (@${botHandle(m.name, m)})` : `@${botHandle(m.name, m)}`

      return m.remoteSource ? `${handle} [on ${m.connectionLabel || m.connectionId}]` : handle
    })
    .join(', ')

  return [
    `[Group chat: "${groupName}"] You are @${botHandle(viewer.name, viewer)}, one participant in a group chat with ${peerNames || 'no one else yet'} and the user.`,
    '',
    'New messages in the room since your last turn (oldest first):',
    ...deltaLines.map(line => `  ${line}`),
    '',
    'Rules for this room:',
    '- Reply with ONE conversational message ONLY if you have something new worth adding: build on what was just said, claim or hand off work, answer a question aimed at you, or report a real result. Keep chatter short (1-3 sentences) — but when you are delivering a result, an answer the user asked for, or substantive work, give it at full quality and length; never thin out real content to fit the room.',
    '- If you have nothing new to add, reply with exactly "(pass)". Passing is good — it lets the conversation settle.',
    '- Mention a teammate as @name to pull them in; mention @user only for a judgment call or a result the user needs. Do not repeat points already made.',
    '- Never reveal content from your private 1:1 chats. Your reply text goes to the room verbatim — no preamble, no meta-commentary.'
  ].join('\n')
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

/** #93129: a held member's skip must consume its delta exactly once —
 *  advance the watermark past the current log so the same entries never
 *  re-trigger the skip. Null = nothing to consume (no write, no spin). */
export function heldMemberWatermarkAdvance(seen: number | undefined, logLength: number): null | number {
  return logLength > (seen || 0) ? logLength : null
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

/** Drive one bounded round-robin turn for ONE THREAD. Serial — one member at
 *  a time. A newer user send bumps the room epoch; this loop notices at the
 *  next member boundary, bails, and the newest send's own loop takes over.
 *  Watermarks are per thread+member (`${thread}::${memberKey}`), so parallel
 *  topics never eat each other's deltas. */
export async function runGroupChatRounds(group: string, members: GroupMember[], thread: string) {
  const startEpoch = ($groupChats.get()[group] || {}).epoch || 0
  const isCurrent = () => (($groupChats.get()[group] || {}).epoch || 0) === startEpoch
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

        const room = $groupChats.get()[group] || {
          log: [],
          watermarks: {}
        }

        const memberKey = groupMemberKey(member)
        const markKey = `${thread}::${memberKey}`
        const seen = room.watermarks[markKey] || 0
        // Delta: NEW room entries, narrowed to this thread — the member's
        // turn sees only the conversation it's part of.
        const delta = room.log.slice(seen).filter((e: GroupMessage) => groupThreadOf(e) === thread)

        if (!delta.length) {
          continue
        }

        // #93129: a member the user told to stop is HELD — no turn until an
        // explicit release (resume / @all resume / a direct non-stop
        // mention). Consume the delta exactly once (watermark past the
        // current log) so the same entries never re-trigger this skip, and
        // surface WHY the bot is silent in the activity feed the first time.
        const heldEntry = (room.holds || {})[memberKey]

        if (heldEntry) {
          const advance = heldMemberWatermarkAdvance(seen, room.log.length)
          updateGroupChat(group, (r: GroupChatRoom) => {
            if (advance !== null) {
              r.watermarks[markKey] = advance
            }

            if (r.holds?.[memberKey] && !r.holds[memberKey].noted) {
              r.holds = {
                ...r.holds,
                [memberKey]: {
                  ...r.holds[memberKey],
                  noted: true
                }
              }
            }

            return r
          })

          if (!heldEntry.noted) {
            recordGroupActivity(group, {
              kind: 'held',
              member: member.name,
              thread
            })
          }

          continue
        }

        const prompt = buildGroupChatTurnPrompt({
          groupName: group,
          members,
          viewer: member,
          deltaLines: delta
            .slice(-GROUP_CHAT_HISTORY_LIMIT)
            .map((e: GroupMessage) => formatGroupChatLine(e, member.name))
        })

        // Images riding this delta (user attachments — member entries don't
        // carry images today, but flatMap keeps this future-proof) get staged
        // into the member's session so the model sees the pixels, not just
        // the transcript's [attached image: …] marker.
        const deltaImages = delta.flatMap((e: GroupMessage) => (Array.isArray(e.images) ? e.images : []))

        // Surface WHO is on turn (runtime-only, like running/epoch) so the
        // room shows "Radar is thinking…" instead of a generic working line —
        // long model turns otherwise read as the room being stuck.
        updateGroupChat(group, (r: GroupChatRoom) => {
          r.turn = member.name

          return r
        })
        let reply: null | string = null

        try {
          reply = await runGroupChatMemberTurn(group, member, prompt, thread, deltaImages)

          // Needs-attention hook (#93091 item 3): a turn that produced a real
          // reply (or an explicit pass) is a good turn — clear the badge.
          // A timed-out turn also returns null but never threw; leaving any
          // prior badge in place there is the conservative choice.
          if (reply !== null) {
            clearBotAttention(groupMemberKey(member))
          }
        } catch (error: any) {
          const reason = String(error?.data?.reason || '').trim()
          recordGroupActivity(group, {
            kind: 'failed',
            member: member.name,
            thread,
            ...(reason
              ? {
                  reason
                }
              : {})
          })
          noteBotAttention(groupMemberKey(member), reason || error?.message || error)
          reply = null // a failed turn is a pass, never a room error
        }

        // #93127: the turn may have finished AFTER a newer user send bumped
        // the room epoch. That newer send's loop re-drives this member with
        // the full delta, so committing this stale result (watermark advance
        // + append) would double-deliver the same reply. Drop it here —
        // BEFORE the watermark advance and BEFORE the append. Only a newer
        // USER entry in THIS thread makes the re-drive premise true: a
        // cross-thread send bumps the epoch too, but its loop filters this
        // thread out and would never regenerate the finished reply. The
        // during-turn tail is anchored by entry id, not index — the history
        // trim drops entries from the FRONT, so an index slice could
        // overshoot after a mid-turn trim and silently commit a stale turn.
        const roomNow = $groupChats.get()[group] || {
          log: []
        }

        const epochNow = roomNow.epoch || 0
        const anchorId = room.log.length ? room.log[room.log.length - 1].id : null
        const anchorIdx = anchorId === null ? -1 : roomNow.log.findIndex((e: GroupMessage) => e.id === anchorId)
        // Anchor trimmed away ⇒ every pre-turn entry was dropped, so every
        // surviving entry is newer — scanning the whole log stays exact.
        const turnTail = anchorIdx >= 0 ? roomNow.log.slice(anchorIdx + 1) : roomNow.log

        const newerUserEntryInThread = turnTail.some(
          (e: GroupMessage) => e.from?.kind === 'user' && groupThreadOf(e) === thread
        )

        if (!shouldCommitMemberTurn(startEpoch, epochNow, newerUserEntryInThread)) {
          recordGroupActivity(group, {
            kind: 'cancelled',
            member: member.name,
            thread
          })

          return
        }

        // The member has now seen everything up to the pre-reply log length.
        updateGroupChat(group, (r: GroupChatRoom) => {
          r.watermarks[markKey] = r.log.length

          return r
        })

        if (reply !== null && !isGroupPassText(reply)) {
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
            thread
          )
          // Its own message counts as seen too.
          updateGroupChat(group, (r: GroupChatRoom) => {
            r.watermarks[markKey] = r.log.length

            return r
          })
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

        if (pendingKeys.length && continuations <= GROUP_CHAT_MAX_CONTINUATIONS) {
          const citedMembers = members.filter((member: GroupMember) => pendingKeys.includes(groupMemberKey(member)))

          if (citedMembers.length && posted < GROUP_CHAT_MAX_MESSAGES) {
            const strandedNow = ($groupChats.get()[group] || {}).stranded || {}

            const continuationResponders = citedMembers.filter(
              (member: GroupMember) => !Object.prototype.hasOwnProperty.call(strandedNow, groupMemberKey(member))
            )

            for (const member of continuationResponders) {
              if (!isCurrent() || posted >= GROUP_CHAT_MAX_MESSAGES || continuations > GROUP_CHAT_MAX_CONTINUATIONS) {
                break
              }

              const room = $groupChats.get()[group] || {
                log: [],
                watermarks: {}
              }

              const memberKey = groupMemberKey(member)
              const markKey = `${thread}::${memberKey}`
              const seen = room.watermarks[markKey] || 0
              const delta = room.log.slice(seen).filter((e: GroupMessage) => groupThreadOf(e) === thread)

              // A cited member always has delta here (the citing reply IS in
              // its tail); skip defensively anyway so an empty prompt never
              // fires.
              if (!delta.length) {
                continue
              }

              const heldEntry = (room.holds || {})[memberKey]

              if (heldEntry) {
                continue // holds still apply to continuation turns (#93129)
              }

              const prompt = buildGroupChatTurnPrompt({
                groupName: group,
                members,
                viewer: member,
                // The continuation prompt centers on what the member missed:
                // everything since its watermark, which includes the reply
                // that cites it.
                deltaLines: delta
                  .slice(-GROUP_CHAT_HISTORY_LIMIT)
                  .map((e: GroupMessage) => formatGroupChatLine(e, member.name))
              })

              updateGroupChat(group, (r: GroupChatRoom) => {
                r.turn = member.name

                return r
              })
              let continuationReply: null | string = null

              try {
                continuationReply = await runGroupChatMemberTurn(group, member, prompt, thread)

                if (continuationReply !== null) {
                  clearBotAttention(memberKey)
                }
              } catch (error: any) {
                recordGroupActivity(group, {
                  kind: 'failed',
                  member: member.name,
                  thread
                })
                noteBotAttention(memberKey, error?.message || error)
                continuationReply = null
              }

              if (!isCurrent()) {
                return
              }

              updateGroupChat(group, (r: GroupChatRoom) => {
                r.watermarks[markKey] = r.log.length

                return r
              })

              if (continuationReply !== null && !isGroupPassText(continuationReply)) {
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
                  continuationReply,
                  thread
                )
                updateGroupChat(group, (r: GroupChatRoom) => {
                  r.watermarks[markKey] = r.log.length

                  return r
                })
                posted += 1

                // The continuation's own reply may cite someone else — fall
                // through to the normal loop so the next round handles it via
                // the same responder machinery. Reaching here means the loop
                // continues rather than settling; the outer for-loop's next
                // iteration re-evaluates everything.
                spokeThisRound += 1
              }
            }
          }
        }

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
  }
}

/** Bounded background harvest for members whose replies outlived the turn
 *  loop. Polls every 5s for up to 5 minutes; stops early when nothing is
 *  stranded, a new loop takes the room over (it harvests on its own), or the
 *  room record disappears (disband). */
async function harvestStrandedUntilSettled(group: string, members: GroupMember[], thread: string) {
  const HARVEST_INTERVAL_MS = 5000
  const HARVEST_MAX_TRIES = 60

  for (let attempt = 0; attempt < HARVEST_MAX_TRIES; attempt++) {
    await new Promise(resolve => window.setTimeout(resolve, HARVEST_INTERVAL_MS))
    const room = $groupChats.get()[group]

    if (!room || room.running) {
      return
    }

    const stranded = room.stranded || {}

    if (!Object.keys(stranded).length) {
      return
    }

    for (const member of members) {
      if (Object.prototype.hasOwnProperty.call(stranded, groupMemberKey(member))) {
        try {
          await harvestStrandedGroupReply(group, member)
        } catch {
          // Best-effort: the next tick retries; the bound stops runaways.
        }
      }
    }
  }

  recordGroupActivity(group, {
    kind: 'failed',
    member: null,
    thread
  })
}

/** User send into a group room. `thread` continues that thread (its reply
 *  box); omitted/null mints a NEW thread — the main composer's Slack shape.
 *  Appends, bumps the room epoch (supersedes any running loop at its next
 *  member boundary), and starts the turn drive for the target thread.
 *  Returns the thread id the message landed in. */
/** Options for a send made ON THE USER'S BEHALF by a relay (`hermes group
 *  send`): `via` is recorded on the user entry as provenance; the entry is
 *  otherwise indistinguishable from one typed in the composer. */
export interface SendToGroupChatOptions {
  via?: string
}

export function sendToGroupChat(
  group: string,
  members: GroupMember[],
  text: string,
  thread?: null | string,
  images?: Attachment[],
  opts?: SendToGroupChatOptions
): null | string {
  const trimmed = String(text || '').trim()
  const attached = Array.isArray(images) ? images.filter((img: Attachment) => img && img.data) : []

  if ((!trimmed && !attached.length) || !members.length) {
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

  const via = String(opts?.via || '').trim()

  const sent = appendGroupChatEntry(
    group,
    {
      kind: 'user',
      name: 'You',
      ...(via ? { via } : {})
    },
    trimmed,
    target,
    attached
  )

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
    member: 'You',
    thread: target
  })

  if (!wasRunning) {
    void runGroupChatRounds(group, members, target).catch(() => {
      updateGroupChat(group, (r: GroupChatRoom) => {
        r.running = false

        return r
      })
    })
  } else {
    // A loop is live; it bails at its next boundary. Chain the fresh loop
    // after a short settle so exactly one drive owns the room.
    setTimeout(() => {
      void runGroupChatRounds(group, members, target).catch(() => {
        updateGroupChat(group, (r: GroupChatRoom) => {
          r.running = false

          return r
        })
      })
    }, 250)
  }

  return target
}
