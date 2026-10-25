import { MarkdownTextContent } from '@/components/assistant-ui/markdown-text'

/**
 * A static view of the composer's current draft. The shared transcript
 * renderer keeps Markdown semantics consistent with sent messages, while
 * `previewOnly` and `disableArtifacts` ensure an unsent draft cannot trigger
 * links, directives, or artifact registration.
 */
export function ComposerMarkdownPreview({ text }: { text: string }) {
  return (
    <div
      className="min-h-[1.625rem] min-h-(--composer-input-min-height) max-h-(--composer-input-max-height) overflow-y-auto py-1"
      data-slot="composer-markdown-preview"
    >
      <MarkdownTextContent
        containerClassName="text-[length:inherit] leading-normal"
        disableArtifacts
        isRunning={false}
        previewOnly
        text={text}
      />
    </div>
  )
}
