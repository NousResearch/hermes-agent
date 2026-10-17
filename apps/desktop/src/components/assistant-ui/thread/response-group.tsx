import { ThreadPrimitive, useAuiState } from '@assistant-ui/react'
import { type ComponentProps, createContext, useMemo } from 'react'

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
  indices: readonly number[]
}

interface ResponseRow {
  index: number
  id: string
  role: string
  hasText: boolean
}

interface ResponseSection {
  key: string
  indices: number[]
  ids: string[]
  assistantIds: string[]
  response: boolean
}

/** Keep message runtimes intact; only their visual container and footer are shared. */
export function ResponseMessages({ components, indices }: ResponseMessagesProps) {
  const signature = useAuiState(s =>
    JSON.stringify(
      indices.flatMap(index => {
        const message = s.thread.messages[index]

        // A history replacement can notify this row before its parent updates indices.
        return message
          ? [
              {
                index,
                id: message.id,
                role: responseMessageRole(message),
                hasText: contentHasVisibleText(message.content)
              }
            ]
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
        response && previous?.response ? previous : { key: row.id, indices: [], ids: [], assistantIds: [], response }

      if (section !== previous) {
        result.push(section)
      }

      section.indices.push(row.index)
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
          {/* Key by message id, not array index: a store replacement (session switch,
              history backfill, compaction) reuses indices across different logical
              messages, and React would keep the old row instance — and its cached
              parts — alive on the new message (#121096). */}
          {section.indices.map((index, i) => (
            <ThreadPrimitive.MessageByIndex components={components} index={index} key={section.ids[i] ?? index} />
          ))}
        </div>
      </ResponseMessageIds.Provider>
    ) : (
      <ThreadPrimitive.MessageByIndex components={components} index={section.indices[0]!} key={section.key} />
    )
  )
}
