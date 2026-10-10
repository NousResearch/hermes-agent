import { MessagePrimitive, useAuiState } from '@assistant-ui/react'

/** Rebuild index-owned child scopes only when the parts topology changes.
 * Text deltas never enter this key or the transcript's grouping signature. */
export const StructuralMessageParts = (props: MessagePrimitive.Parts.Props) => {
  const shape = useAuiState(s => JSON.stringify(s.message.parts.map(part =>
    part.type === 'tool-call' ? [part.type, part.toolCallId] : [part.type]
  )))
  return <MessagePrimitive.Parts key={shape} {...props} />
}
