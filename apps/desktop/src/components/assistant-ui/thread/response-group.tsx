import { ThreadPrimitive, useAuiState } from '@assistant-ui/react'
import { type ComponentProps, createContext, useMemo } from 'react'

import { contentHasVisibleText, messageContentText, PROCESS_NOTIFICATION_RE } from './content'

interface GroupMessage {
  role: string
  content: unknown
  metadata?: { custom?: Record<string, unknown> }
}

/**
 * Whether a USER message is a background delivery, keyed by its content array.
 *
 * This is the only part of `responseMessageRole` that reads a message's text, and
 * the transcript-wide structural signature calls it for every message on every
 * store notification. Extracting the whole text to test one prefix meant each
 * streamed delta re-copied and re-trimmed every settled prompt in the thread, so
 * per-frame cost grew with the transcript's characters, not with the live tail
 * (#126486). The sibling `contentHasVisibleText` in `./content` already exists to
 * avoid exactly this concatenation.
 *
 * A content array is the right key because assistant-ui publishes a NEW array
 * when a message changes and keeps the settled one — the same invariant
 * `messagePaintWeight`'s WeakMap is built on. A same-length replacement, a
 * rewrite, or a rewind all arrive as a new array and miss the cache. A `string`
 * content is not weak-referenceable and is left uncached.
 */
const backgroundDeliveryByContent = new WeakMap<object, boolean>()

function isBackgroundDelivery(message: GroupMessage): boolean {
  const content = message.content

  if (!Array.isArray(content)) {
    return PROCESS_NOTIFICATION_RE.test(messageContentText(content))
  }

  const cached = backgroundDeliveryByContent.get(content)

  if (cached !== undefined) {
    return cached
  }

  const background = PROCESS_NOTIFICATION_RE.test(messageContentText(content))

  backgroundDeliveryByContent.set(content, background)

  return background
}

/** Background deliveries continue the response without becoming human prompts. */
export function responseMessageRole(message: GroupMessage): string {
  const custom = message.metadata?.custom

  const background =
    message.role === 'system'
      ? Boolean(custom?.asyncResult || custom?.asyncResultKind)
      : message.role === 'user' && isBackgroundDelivery(message)

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
