import { ThreadPrimitive, useAuiState } from '@assistant-ui/react'
import { type ComponentProps, createContext, memo, useMemo } from 'react'

import { contentHasVisibleText, messageContentText, PROCESS_NOTIFICATION_RE } from './content'

interface GroupMessage {
  role: string
  content: unknown
  metadata?: { custom?: Record<string, unknown> }
}

/** Background deliveries continue the response without becoming human prompts. */
export function responseMessageRole(message: GroupMessage): string {
  const custom = message.metadata?.custom

  const background =
    message.role === 'system'
      ? Boolean(custom?.asyncResult || custom?.asyncResultKind)
      : message.role === 'user' && PROCESS_NOTIFICATION_RE.test(messageContentText(message.content))

  return background ? 'background' : message.role
}

export const ResponseMessageIds = createContext<readonly string[]>([])

interface ResponseMessagesProps {
  components: ComponentProps<typeof ThreadPrimitive.MessageByIndex>['components']
  /** The turn's messages, by id. Ids, not positions: older history loading in front of a turn
   *  moves every position after it, and a row addressed by position was rebuilt on each load. */
  ids: readonly string[]
}

interface ResponseRow {
  id: string
  role: string
  hasText: boolean
}

interface ResponseSection {
  key: string
  ids: string[]
  assistantIds: string[]
  response: boolean
}

type ThreadMessages = readonly (GroupMessage & { id: string })[]

const messagesById = new WeakMap<ThreadMessages, Map<string, ThreadMessages[number]>>()

/** One id index per message list, shared by every row that reads that list. */
function messageById(messages: ThreadMessages, id: string) {
  let index = messagesById.get(messages)

  if (!index) {
    index = new Map(messages.map(message => [message.id, message]))
    messagesById.set(messages, index)
  }

  return index.get(id)
}

const sameIds = (a: readonly string[], b: readonly string[]) =>
  a === b || (a.length === b.length && a.every((id, index) => id === b[index]))

/**
 * Keep message runtimes intact; only their visual container and footer are shared.
 *
 * memo() on the ids themselves: the list rebuilds its groups whenever the transcript changes
 * shape, and a turn whose own messages did not change must not re-render for it. With rows
 * addressed by position, every older page loading in front remounted each loaded reply (their keys
 * were positions), a freeze that grew with the transcript: 1s at 240 messages, 4s at 1,200.
 */
export const ResponseMessages = memo(
  function ResponseMessages({ components, ids }: ResponseMessagesProps) {
    const signature = useAuiState(s =>
      JSON.stringify(
        ids.flatMap(id => {
          const message = messageById(s.thread.messages, id)

          // A history replacement can notify this row before its parent updates ids.
          return message
            ? [{ id, role: responseMessageRole(message), hasText: contentHasVisibleText(message.content) }]
            : []
        })
      )
    )

    const sections = useMemo(() => {
      const rows = JSON.parse(signature) as ResponseRow[]
      const result: ResponseSection[] = []

      for (const row of rows) {
        const response = row.role === 'assistant' || row.role === 'background'
        const previous = result.at(-1)

        const section: ResponseSection =
          response && previous?.response ? previous : { key: row.id, ids: [], assistantIds: [], response }

        if (section !== previous) {
          result.push(section)
        }

        section.ids.push(row.id)

        if (row.role === 'assistant' && row.hasText) {
          section.assistantIds.push(row.id)
        }
      }

      return result
    }, [signature])

    return sections.map(section =>
      section.response ? (
        <ResponseMessageIds.Provider key={section.key} value={section.assistantIds}>
          <div className="group flex min-w-0 flex-col gap-(--scaffold-block-gap)" data-slot="aui_response-group">
            {section.ids.map(id => (
              <ThreadPrimitive.Unstable_MessageById components={components} key={id} messageId={id} />
            ))}
          </div>
        </ResponseMessageIds.Provider>
      ) : (
        <ThreadPrimitive.Unstable_MessageById components={components} key={section.key} messageId={section.ids[0]!} />
      )
    )
  },
  (previous, next) => previous.components === next.components && sameIds(previous.ids, next.ids)
)
