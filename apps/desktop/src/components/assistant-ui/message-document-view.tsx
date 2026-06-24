import { Fragment, type FC } from 'react'

import { DIRECTIVE_CHIP_CLASS, directiveIconSvg } from '@/components/assistant-ui/directive-text'
import { type MessageDocument, isAttachmentToken, isTextToken } from '@/lib/message-document'

interface MessageDocumentViewProps {
  document: MessageDocument
  className?: string
}

/**
 * Inline prose + attachment chips from a MessageDocument. Image tokens render as
 * chips (not zoomable thumbnails) so we never nest interactive buttons inside
 * the edit-message bubble wrapper in thread.tsx.
 */
export const MessageDocumentView: FC<MessageDocumentViewProps> = ({ className, document }) => (
  <span className={className} data-slot="aui_message-document-view">
    {document.map((token, index) => {
      if (isTextToken(token)) {
        return (
          <span className="whitespace-pre-line" key={`text-${index}`}>
            {token.value}
          </span>
        )
      }

      if (isAttachmentToken(token)) {
        return (
          <span
            className={DIRECTIVE_CHIP_CLASS}
            data-directive-id={token.path}
            data-directive-type={token.kind}
            data-ref-kind={token.kind}
            data-ref-text={`@${token.kind}:${token.path}`}
            data-slot="aui_directive-chip"
            key={`attachment-${token.id}`}
            title={token.path}
          >
            <span
              className="inline-flex shrink-0 opacity-80 [&>svg]:size-3"
              dangerouslySetInnerHTML={{ __html: directiveIconSvg(token.kind) }}
            />
            <span className="truncate">{token.displayName}</span>
          </span>
        )
      }

      return <Fragment key={`unknown-${index}`} />
    })}
  </span>
)
