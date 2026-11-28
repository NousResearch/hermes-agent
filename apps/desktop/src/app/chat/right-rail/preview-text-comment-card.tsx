import { useId, useRef, useState } from 'react'

import { useVoiceRecorder } from '@/app/chat/composer/hooks/use-voice-recorder'
import { ComposerScopeProvider, MAIN_COMPOSER_SCOPE } from '@/app/chat/composer/scope'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Popover, PopoverAnchor, PopoverContent } from '@/components/ui/popover'
import { Textarea } from '@/components/ui/textarea'
import { Tip } from '@/components/ui/tooltip'
import { type ResolvedOwner, transcribeAudio } from '@/hermes'
import { useI18n } from '@/i18n'
import { blobToDataUrl } from '@/lib/base64'
import { isSubmitEnter } from '@/lib/ime'
import { transcribeAudioClientDirect } from '@/lib/voice-client-direct'

interface PreviewTextCommentCardProps {
  onCancel: () => void
  /** True only when the pinned chat accepted the edit. The parent closes on success. */
  onSubmit: (note: string) => boolean
  owner?: Pick<ResolvedOwner, 'connectionId' | 'profile'>
  paneHeight: number
  paneWidth: number
  rect: { height: number; width: number; x: number; y: number }
  selectedText: string
}

async function transcribeComment(audio: Blob, owner?: ResolvedOwner): Promise<string> {
  const direct = await transcribeAudioClientDirect(audio, owner)

  if (direct !== null) {
    return direct
  }

  return (await transcribeAudio(await blobToDataUrl(audio), audio.type, owner)).transcript
}

export function PreviewTextCommentCard(props: PreviewTextCommentCardProps) {
  return (
    <ComposerScopeProvider value={{ ...MAIN_COMPOSER_SCOPE, ...props.owner }}>
      <TextCommentCard {...props} />
    </ComposerScopeProvider>
  )
}

function TextCommentCard({
  onCancel,
  onSubmit,
  owner,
  paneHeight,
  paneWidth,
  rect,
  selectedText
}: PreviewTextCommentCardProps) {
  const { t } = useI18n()
  const copy = t.preview.web
  const field = useRef<HTMLTextAreaElement>(null)
  const [anchor, setAnchor] = useState<HTMLSpanElement | null>(null)
  const [note, setNote] = useState('')
  const [unavailable, setUnavailable] = useState(false)
  const selectionId = useId()

  const { dictate, voiceStatus, voiceActivityState } = useVoiceRecorder({
    focusInput: () => field.current?.focus(),
    maxRecordingSeconds: 120,
    onTranscribeAudio: owner ? transcribeComment : undefined,
    onTranscript: text => setNote(previous => (previous ? `${previous} ${text}` : text))
  })

  const voiceLabel =
    voiceStatus === 'recording'
      ? t.composer.stopDictation
      : voiceStatus === 'transcribing'
        ? t.composer.transcribingDictation
        : t.composer.voiceDictation

  const canSubmit = Boolean(note.trim()) && voiceStatus === 'idle'

  const submit = () => {
    if (canSubmit) {
      setUnavailable(!onSubmit(note.trim()))
    }
  }

  return (
    <Popover
      onOpenChange={open => {
        if (!open) {
          onCancel()
        }
      }}
      open
    >
      <PopoverAnchor asChild>
        <span
          aria-hidden="true"
          className="pointer-events-none absolute"
          ref={setAnchor}
          style={{ height: rect.height, left: rect.x, top: rect.y, width: rect.width }}
        />
      </PopoverAnchor>
      <PopoverContent
        align="start"
        aria-label={copy.textComment}
        className="overflow-auto"
        collisionBoundary={anchor?.parentElement}
        onCloseAutoFocus={event => event.preventDefault()}
        onEscapeKeyDown={event => {
          event.preventDefault()
          event.stopPropagation()
          onCancel()
        }}
        onOpenAutoFocus={event => {
          event.preventDefault()
          field.current?.focus()
        }}
        style={{ maxHeight: Math.max(0, paneHeight - 16), maxWidth: Math.max(0, paneWidth - 16) }}
      >
        <form
          className="flex flex-col gap-2"
          onSubmit={event => {
            event.preventDefault()
            submit()
          }}
        >
          <blockquote className="line-clamp-2 break-words text-xs text-muted-foreground" id={selectionId}>
            {selectedText}
          </blockquote>
          <Textarea
            aria-describedby={selectionId}
            aria-label={copy.textCommentPlaceholder}
            onChange={event => {
              setNote(event.target.value)
              setUnavailable(false)
            }}
            onKeyDown={event => {
              if (isSubmitEnter(event) && (event.metaKey || event.ctrlKey)) {
                event.preventDefault()
                event.stopPropagation()
                submit()
              }
            }}
            placeholder={copy.textCommentPlaceholder}
            ref={field}
            rows={3}
            size="sm"
            spellCheck
            value={note}
          />
          {voiceStatus !== 'idle' && (
            <p aria-live="polite" className="text-xs text-muted-foreground" role="status">
              {voiceActivityState.partial || voiceLabel}
            </p>
          )}
          {unavailable && (
            <p className="text-xs text-destructive" role="alert">
              {copy.textCommentUnavailable}
            </p>
          )}
          <div className="flex flex-wrap items-center justify-end gap-1">
            {owner && (
              <Tip label={voiceLabel}>
                <Button
                  aria-label={voiceLabel}
                  aria-pressed={voiceStatus === 'recording'}
                  disabled={voiceStatus === 'transcribing'}
                  onClick={dictate}
                  size="icon-sm"
                  type="button"
                  variant={voiceStatus === 'recording' ? 'secondary' : 'ghost'}
                >
                  <Codicon name={voiceStatus === 'recording' ? 'primitive-square' : 'mic'} />
                </Button>
              </Tip>
            )}
            <Button onClick={onCancel} size="sm" type="button" variant="text">
              {t.common.cancel}
            </Button>
            <Button disabled={!canSubmit} size="sm" type="submit">
              {copy.sendTextComment}
            </Button>
          </div>
        </form>
      </PopoverContent>
    </Popover>
  )
}
