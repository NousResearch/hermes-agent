import { MessageTextContent } from '@hermes/plugin-sdk'

import { decorateFilePaths } from './filepath-links'

interface TaskMarkdownProps {
  linkPaths?: boolean
  text: string
}

// Kanban text has no session to own MEDIA resolution; retain the shared renderer.
export function TaskMarkdown({ linkPaths = false, text }: TaskMarkdownProps) {
  return (
    <div className="min-w-0 [&_.aui-md>:first-child]:mt-0 [&_.aui-md>:last-child]:mb-0" data-selectable-text="true">
      <MessageTextContent decorateText={linkPaths ? decorateFilePaths : undefined} media={false} text={text} />
    </div>
  )
}
