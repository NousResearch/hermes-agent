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

// PERF: the grouping selector below runs on EVERY store notify (one per
// streamed chunk) for every mounted ResponseMessages, and it used to re-derive
// each row's role and hasText from scratch there — walking every part, and for
// user rows concatenating the ENTIRE message text to run the process-notice
// regex. At mounted-row breadth N that made the per-token cost scale with N
// (and with transcript text size) even though nothing visible had changed.
// The runtime publishes a fresh messages ARRAY per notify but keeps settled
// message objects referentially identical — only the streaming tail gets a new
// object per delta (verified through MessageRepository/addOrUpdateMessage and
// the WeakMap normalization cache in runtime-repository.ts) — so keying the
// derived metadata on the message object re-derives it once per object
// VERSION, not once per notify. The selector itself is left with array
// lookups, WeakMap hits, and the stringify of tiny primitive rows.
const rowMetaCache = new WeakMap<object, { hasText: boolean; role: string }>()

const rowMeta = (message: GroupMessage): { hasText: boolean; role: string } => {
  let meta = rowMetaCache.get(message)

  if (!meta) {
    meta = { hasText: contentHasVisibleText(message.content), role: responseMessageRole(message) }

    rowMetaCache.set(message, meta)
  }

  return meta
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
        if (!message) {
          return []
        }

        const meta = rowMeta(message)

        return [
          {
            index,
            id: message.id,
            role: meta.role,
            hasText: meta.hasText
          }
        ]
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
        response && previous?.response ? previous : { key: row.id, indices: [], assistantIds: [], response }

      if (section !== previous) {
        result.push(section)
      }

      section.indices.push(row.index)

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
          {section.indices.map(index => (
            <ThreadPrimitive.MessageByIndex components={components} index={index} key={index} />
          ))}
        </div>
      </ResponseMessageIds.Provider>
    ) : (
      <ThreadPrimitive.MessageByIndex components={components} index={section.indices[0]!} key={section.key} />
    )
  )
}
