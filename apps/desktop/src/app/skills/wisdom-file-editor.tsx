import { useState } from 'react'

import { MarkdownPreview } from '@/app/chat/right-rail/preview-file'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import type { WisdomDraftReview } from '@/hermes'
import { useI18n } from '@/i18n'

import { WisdomManifestEditor } from './wisdom-manifest-editor'

export function WisdomFileEditor({
  file,
  value,
  disabled = false,
  reviewSource = 'server',
  onChange
}: {
  file: WisdomDraftReview['files'][number]
  value: string
  disabled?: boolean
  reviewSource?: 'local' | 'server'
  onChange: (value: string) => void
}) {
  const { t } = useI18n()
  const copy = t.skills.collective
  const [mode, setMode] = useState<'preview' | 'source'>('source')
  const reviewLabel = reviewSource === 'local' ? copy.localDraft : copy.serverReviewed

  if (file.path === 'skill.manifest.json') {
    return (
      <details className="border-t border-(--ui-stroke-tertiary) py-3" open>
        <summary className="cursor-pointer break-all font-mono text-[0.68rem]">
          {file.path} · {reviewLabel} · {file.hash}
        </summary>
        <WisdomManifestEditor disabled={disabled} onChange={onChange} value={value} />
      </details>
    )
  }

  const supportsPreview = file.path.toLocaleLowerCase().endsWith('.md')

  return (
    <details className="border-t border-(--ui-stroke-tertiary) py-3" open>
      <summary className="cursor-pointer break-all font-mono text-[0.68rem]">
        {file.path} · {reviewLabel} · {file.hash}
      </summary>
      {supportsPreview && (
        <div aria-label={`${file.path} editor mode`} className="mt-3 flex gap-2">
          <Button onClick={() => setMode('source')} size="xs" variant={mode === 'source' ? 'secondary' : 'text'}>
            {copy.source}
          </Button>
          <Button onClick={() => setMode('preview')} size="xs" variant={mode === 'preview' ? 'secondary' : 'text'}>
            {copy.preview}
          </Button>
        </div>
      )}
      {supportsPreview && mode === 'preview' ? (
        <div className="mt-3 max-h-[32rem] min-h-48 overflow-auto bg-(--ui-bg-quaternary)">
          <MarkdownPreview text={value} />
        </div>
      ) : (
        <Textarea
          aria-label={`Edit ${file.path}`}
          className="mt-3 max-h-[32rem] min-h-64 w-full resize-y font-mono text-[0.68rem] leading-relaxed"
          disabled={disabled}
          onChange={event => onChange(event.target.value)}
          spellCheck={false}
          value={value}
        />
      )}
    </details>
  )
}
