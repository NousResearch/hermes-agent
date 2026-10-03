import type { GroupChatSend, GroupChatsSnapshot } from '@hermes/plugin-sdk'

// Built-in composition seam; external plugins cannot register an engine.
// eslint-disable-next-line no-restricted-imports
import type { GroupChatsProvider } from '../../sdk/group-chats'

import { $botMeta, $lastRoster } from './data'
import { $groupActivity, currentGroupActivity } from './group-activity'
import { $groupChats, $groupClarify, $groupNeedsYou } from './group-chat'
import { groupChatMemberBots } from './group-membership'
import { sendToGroupChat } from './group-rounds'
import { botRosterMeta } from './routing'

const text = (value: unknown) => (typeof value === 'string' ? value : '')
const timestamp = (value: unknown) => (typeof value === 'number' && Number.isFinite(value) ? value : 0)

function freeze<T>(value: T): T {
  if (value && typeof value === 'object') {
    for (const child of Object.values(value)) {
      freeze(child)
    }

    Object.freeze(value)
  }

  return value
}

/** Adapter only: orchestration and routing remain in the Bot Mode singleton. */
export function createGroupChatBridge() {
  let ready = false
  let disposed = false
  let snapshot: GroupChatsSnapshot | undefined
  const submissions = new Map<string, { fingerprint: string; threadId: string }>()
  const listeners = new Set<() => void>()

  const changed = () => {
    snapshot = undefined

    for (const listener of listeners) {
      listener()
    }
  }

  const unbind = [$groupChats, $groupClarify, $groupNeedsYou, $groupActivity, $lastRoster, $botMeta].map(store =>
    store.listen(changed)
  )

  const provider: GroupChatsProvider & { setReady(): void; dispose(): void } = {
    status: () => (disposed ? ('unavailable' as const) : ready ? ('ready' as const) : ('loading' as const)),
    setReady() {
      if (!disposed) {
        ready = true
        changed()
      }
    },
    dispose() {
      if (disposed) {
        return
      }

      disposed = true
      unbind.forEach(stop => stop())
      changed()
      listeners.clear()
    },
    getSnapshot(): GroupChatsSnapshot {
      if (snapshot) {
        return snapshot
      }

      snapshot = freeze({
        rooms:
          !ready || disposed
            ? []
            : Object.entries($groupChats.get())
                .filter(([, room]) => !room.tombstone)
                .map(([name, room]) => ({
                  roomId: text(room.roomId) || null,
                  name,
                  running: Boolean(room.running),
                  needsYou: Boolean($groupNeedsYou.get()[name]),
                  log: room.log.map(entry => ({
                    at: timestamp(entry.at),
                    text: text(entry.text),
                    from: {
                      kind: entry.from?.kind === 'member' ? ('member' as const) : ('user' as const),
                      name: text(entry.from?.name),
                      ...(text(entry.from?.source) ? { source: text(entry.from?.source) } : {}),
                      ...(text(entry.from?.gateway) ? { gateway: text(entry.from?.gateway) } : {})
                    },
                    ...(text(entry.id) ? { id: text(entry.id) } : {}),
                    ...(text(entry.thread) ? { thread: text(entry.thread) } : {}),
                    ...(entry.truncated ? { truncated: true } : {})
                  })),
                  members: groupChatMemberBots(name, $lastRoster.get(), $botMeta.get()).map(member => ({
                    name: text(member.name),
                    title: text(botRosterMeta(member, $botMeta.get())?.title) || text(member.title),
                    ...(text(member.connectionId) ? { connectionId: text(member.connectionId) } : {}),
                    ...(text(member.connectionLabel) ? { connectionLabel: text(member.connectionLabel) } : {}),
                    ...(text(member.installId) ? { installId: text(member.installId) } : {})
                  })),
                  activity: currentGroupActivity(name).map(event => ({
                    at: timestamp(event.at),
                    kind: text(event.kind),
                    member: text(event.member),
                    thread: text(event.thread),
                    reason: text(event.reason),
                    preview: text(event.preview)
                  })),
                  requests: Object.values($groupClarify.get())
                    .filter(request => request.group === name)
                    .map(request => ({
                      at: timestamp(request.at),
                      kind: request.kind === 'approval' ? ('approval' as const) : ('clarify' as const),
                      member: text(request.member),
                      thread: text(request.thread),
                      question: text(request.question),
                      choices: (Array.isArray(request.choices) ? request.choices : []).map(text),
                      multiSelect: Boolean(request.multiSelect),
                      command: text(request.command)
                    }))
                }))
      })

      return snapshot
    },
    subscribe(listener: () => void) {
      listeners.add(listener)

      return () => {
        listeners.delete(listener)
      }
    },
    send(input: GroupChatSend) {
      const status = provider.status()

      if (status !== 'ready') {
        return { accepted: false as const, error: status }
      }

      if (
        !input ||
        typeof input.roomId !== 'string' ||
        !input.roomId ||
        typeof input.text !== 'string' ||
        !input.text.trim() ||
        (input.threadId !== undefined && (typeof input.threadId !== 'string' || !input.threadId))
      ) {
        return { accepted: false as const, error: 'invalid-input' }
      }

      const entries = Object.entries($groupChats.get()).filter(
        ([, room]) => !room.tombstone && room.roomId === input.roomId
      )

      if (entries.length !== 1) {
        return { accepted: false as const, error: 'invalid-room' }
      }

      const [name, room] = entries[0]

      if (input.threadId !== undefined && !room.log.some(entry => entry.thread === input.threadId)) {
        return { accepted: false as const, error: 'invalid-thread' }
      }

      if (
        input.submissionKey !== undefined &&
        (typeof input.submissionKey !== 'string' || !input.submissionKey || input.submissionKey.length > 256)
      ) {
        return { accepted: false, error: 'invalid-input' }
      }

      const fingerprint = JSON.stringify([input.roomId, input.text, input.threadId ?? null])
      const previous = input.submissionKey ? submissions.get(input.submissionKey) : undefined

      if (previous) {
        return previous.fingerprint === fingerprint
          ? { accepted: true, threadId: previous.threadId }
          : { accepted: false, error: 'submission-conflict' }
      }

      const members = groupChatMemberBots(name, $lastRoster.get(), $botMeta.get()).map(member => ({
        ...member,
        title: botRosterMeta(member, $botMeta.get())?.title || member.title || ''
      }))

      const threadId = sendToGroupChat(name, members, input.text, input.threadId)

      if (threadId && input.submissionKey) {
        submissions.set(input.submissionKey, { fingerprint, threadId })

        if (submissions.size > 256) {
          submissions.delete(submissions.keys().next().value!)
        }
      }

      return threadId ? { accepted: true as const, threadId } : { accepted: false as const, error: 'rejected' }
    }
  } satisfies GroupChatsProvider & { setReady(): void; dispose(): void }

  return provider
}
