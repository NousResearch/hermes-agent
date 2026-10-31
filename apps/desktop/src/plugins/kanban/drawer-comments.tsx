/**
 * The card's comment thread pieces: the composer (text plus pasted, dropped or
 * picked images, each uploaded as a task attachment and referenced by its
 * stored path) and the image renderer that paints those references inline.
 */

import { Button, cn, Codicon, host, isSubmitEnter, Textarea, Tip } from '@hermes/plugin-sdk'
import {
  type ComponentProps,
  type ComponentType,
  createContext,
  type ReactNode,
  useCallback,
  useContext,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState
} from 'react'

import type { KanbanAttachment } from './types'
import { errText, useKanban } from './ui'

export type ReadAttachment = (path: string) => Promise<string>
export type UploadImage = (file: File, filename: string) => Promise<KanbanAttachment | null>

// The thread reads an image back as a data URL, which the backend caps at
// `_FS_DATA_URL_MAX_BYTES` (hermes_cli/web_server.py); a bigger upload would
// land but never paint.
export const MAX_COMMENT_IMAGE_MB = 16

const isImage = (file: File) => file.type.startsWith('image/')

const imageFiles = (files: FileList | null | undefined) => Array.from(files ?? []).filter(isImage)

/** Clipboard images arrive as a generic `image.png`; name them by paste time
 *  so the attachments list (and the worker reading it) can tell them apart. */
export function pastedImageName(file: File, now = new Date(), index?: number): string {
  const stamp = now.toISOString().replace(/\D/g, '').slice(0, 14)
  const ext = file.name.match(/\.[a-z0-9]+$/i)?.[0] ?? `.${file.type.split('/')[1]?.split('+')[0] || 'png'}`

  return `pasted-${stamp}${index === undefined ? '' : `-${index + 1}`}${ext}`
}

// Collision-resolved names carry spaces and parens (`shot (1).png`), which end
// a bare link destination; the `<…>` form takes them. Backslashes (Windows
// paths) are escapes in Markdown, so they are doubled to survive the parse.
export function imageMarkdown(filename: string, path: string): string {
  const alt = filename.replace(/[[\]\\]/g, '\\$&')
  const escaped = path.replace(/\\/g, '\\\\')

  return `![${alt}](${/[\s()<>]/.test(path) ? `<${escaped.replace(/[<>]/g, '\\$&')}>` : escaped})`
}

// Mirrors the review pane's commit-message field: one row tall to start
// (button-height), CSS field-sizing grows it with content, button hugs the
// bottom edge as it grows.
//
// On a RUNNING task the worker polls its comment thread and folds new notes
// into the live turn (OUT-OF-BAND steer), so a plain note reaches the agent
// mid-run within a few seconds — no block/unblock dance. `onRequeue` is the
// heavier option: post the note AND reclaim so the task restarts from scratch
// with the note in context (use when the current run has gone off the rails).
//
// `onUploadImage` (absent on backends without attachments) turns a pasted,
// dropped or picked image into a task attachment and inserts its Markdown at
// the caret; posting waits until every upload has landed. The draft is one
// card's: the drawer keys this by task id.
export function CommentComposer({
  onRequeue,
  onSubmit,
  onUploadImage,
  pending,
  running
}: {
  onRequeue?: (body: string) => void
  onSubmit: (body: string) => void
  onUploadImage?: UploadImage
  pending: boolean
  running?: boolean
}) {
  const k = useKanban()
  const [body, setBody] = useState('')
  const [uploads, setUploads] = useState(0)
  const fieldRef = useRef<HTMLTextAreaElement>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  // Where the next insert goes while earlier ones are still uncommitted, so
  // uploads landing together queue in order instead of sharing one caret.
  const caretRef = useRef<null | number>(null)

  useLayoutEffect(() => {
    if (caretRef.current !== null) {
      fieldRef.current?.setSelectionRange(caretRef.current, caretRef.current)
      caretRef.current = null
    }
  }, [body])

  const insert = (snippet: string) => {
    const field = fieldRef.current
    const start = caretRef.current ?? field?.selectionStart ?? null
    const end = caretRef.current ?? field?.selectionEnd ?? start

    caretRef.current = (start ?? body.length) + snippet.length
    setBody(prev => {
      const at = Math.min(start ?? prev.length, prev.length)

      return prev.slice(0, at) + snippet + prev.slice(Math.max(at, Math.min(end ?? at, prev.length)))
    })
  }

  const upload = async (file: File, filename = file.name) => {
    if (!onUploadImage) {
      return
    }

    if (file.size > MAX_COMMENT_IMAGE_MB * 1024 * 1024) {
      host.notify({ kind: 'error', message: k.imageTooLarge(filename, MAX_COMMENT_IMAGE_MB) })

      return
    }

    setUploads(n => n + 1)

    try {
      const image = await onUploadImage(file, filename)

      if (image?.stored_path) {
        insert(imageMarkdown(image.filename, image.stored_path))
      } else {
        host.notify({ kind: 'error', message: k.imageUploadFailed(filename) })
      }
    } catch (err) {
      host.notify({ kind: 'error', message: errText(err) })
    } finally {
      setUploads(n => n - 1)
    }
  }

  const post = (send: ((body: string) => void) | undefined) => {
    const trimmed = body.trim()

    if (trimmed && !pending && !uploads && send) {
      send(trimmed)
      setBody('')
    }
  }

  const empty = !body.trim() || pending || uploads > 0
  const sendLabel = running ? k.send : k.comment

  // Sized like the new-project idea field (default control padding, inset icon
  // actions); a labelled text button floating inside the textarea read as part
  // of the input.
  return (
    <div className="flex flex-col gap-1.5">
      <div
        className="relative"
        onDragOver={event => {
          if (onUploadImage && event.dataTransfer.types.includes('Files')) {
            event.preventDefault()
          }
        }}
        onDrop={event => {
          const files = onUploadImage ? Array.from(event.dataTransfer.files ?? []) : []

          if (!files.length) {
            return
          }

          event.preventDefault()
          event.stopPropagation()
          const skipped = files.filter(file => !isImage(file))

          if (skipped.length) {
            host.notify({ kind: 'warning', message: k.notAnImage(skipped.map(file => file.name).join(', ')) })
          }

          files.filter(isImage).forEach(file => void upload(file))
        }}
      >
        <Textarea
          className={cn('field-sizing-content max-h-40 resize-none text-[0.8125rem]', onUploadImage ? 'pr-16' : 'pr-9')}
          onChange={event => setBody(event.target.value)}
          onKeyDown={event => {
            if (isSubmitEnter(event) && !event.shiftKey) {
              event.preventDefault()
              post(onSubmit)
            }
          }}
          onPaste={event => {
            // A clipboard with text (a copied spreadsheet cell, a rich-text
            // selection) often carries a rendered image too; the text wins.
            const files =
              onUploadImage && !event.clipboardData.getData('text/plain') ? imageFiles(event.clipboardData.files) : []

            if (files.length) {
              event.preventDefault()
              const now = new Date()
              files.forEach(
                (file, i) => void upload(file, pastedImageName(file, now, files.length > 1 ? i : undefined))
              )
            }
          }}
          placeholder={running ? k.messageWorker : k.addComment}
          ref={fieldRef}
          value={body}
        />
        <div className="absolute top-1 right-1 flex items-center gap-0.5">
          {onUploadImage && (
            <>
              <input
                accept="image/*"
                hidden
                multiple
                onChange={event => {
                  imageFiles(event.target.files).forEach(file => void upload(file))
                  event.target.value = ''
                }}
                ref={fileRef}
                type="file"
              />
              <Tip label={k.attachImage}>
                <Button
                  aria-label={k.attachImage}
                  className="text-muted-foreground/80 hover:text-foreground"
                  onClick={() => fileRef.current?.click()}
                  size="icon-xs"
                  type="button"
                  variant="ghost"
                >
                  <Codicon name="file-media" size="0.85rem" />
                </Button>
              </Tip>
            </>
          )}
          <Tip label={sendLabel}>
            <Button
              aria-label={sendLabel}
              className="text-muted-foreground/80 hover:text-foreground"
              disabled={empty}
              onClick={() => post(onSubmit)}
              size="icon-xs"
              type="button"
              variant="ghost"
            >
              <Codicon name="arrow-up" size="0.85rem" />
            </Button>
          </Tip>
        </div>
      </div>
      <div aria-live="polite" role="status">
        {uploads > 0 && (
          <span className="flex items-center gap-1 text-[0.625rem] leading-tight text-(--ui-text-quaternary)">
            <Codicon name="sync" size="0.65rem" spinning />
            {k.uploadingImage}
          </span>
        )}
      </div>
      {running && onRequeue && (
        <div className="flex items-center justify-between gap-2">
          <span className="text-[0.625rem] leading-tight text-(--ui-text-quaternary)">{k.deliveredLive}</span>
          <Button className="shrink-0" disabled={empty} onClick={() => post(onRequeue)} size="xs" variant="outline">
            <Codicon name="debug-restart" size="0.7rem" />
            {k.requeueWithNote}
          </Button>
        </div>
      )}
    </div>
  )
}

type ImageComponent = ComponentType<ComponentProps<'img'>>

/** A task attachment painted inline. The path lives on the gateway that
 *  returned the task, so its bytes are read through that backend as a data
 *  URL; the renderer's default image (`Image`) then supplies the bounded frame
 *  and click-to-zoom. A failed read says so in a line: the raw path means
 *  nothing to the active gateway. */
function AttachmentImage({
  alt,
  Image,
  path,
  read
}: {
  alt?: string
  Image: ImageComponent
  path: string
  read: ReadAttachment
}) {
  const k = useKanban()
  const [url, setUrl] = useState('')
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    let live = true

    setUrl('')
    setFailed(false)
    read(path)
      .then(dataUrl => live && setUrl(dataUrl))
      .catch(() => live && setFailed(true))

    return () => {
      live = false
    }
  }, [path, read])

  if (failed) {
    return (
      <span className="text-[0.75rem] text-(--ui-text-quaternary)">
        {k.imageLoadFailed(alt || path.split(/[\\/]/).pop() || path)}
      </span>
    )
  }

  return url ? <Image alt={alt} src={url} /> : null
}

const decoded = (src: string) => {
  try {
    return decodeURI(src)
  } catch {
    return src
  }
}

interface AttachmentReader {
  known: ReadonlySet<string>
  read: ReadAttachment
}

const AttachmentReaderContext = createContext<AttachmentReader | null>(null)

/** Scopes `CommentImage` to one task's attachments. Keyed on the set of paths
 *  (not the detail object) so a refetch doesn't re-render every image; a new
 *  attachment re-renders them without remounting, so none re-reads its bytes. */
export function AttachmentImages({
  attachments,
  children,
  readAttachment
}: {
  attachments: KanbanAttachment[] | null | undefined
  children: ReactNode
  readAttachment: ReadAttachment | undefined
}) {
  const readRef = useRef(readAttachment)
  readRef.current = readAttachment
  const read = useCallback<ReadAttachment>(path => readRef.current!(path), [])

  const paths = (attachments ?? [])
    .map(attachment => attachment.stored_path?.trim())
    .filter(Boolean)
    .join('\n')

  const hasReader = !!readAttachment

  const reader = useMemo(
    () => (paths && hasReader ? { known: new Set(paths.split('\n')), read } : null),
    [hasReader, paths, read]
  )

  return <AttachmentReaderContext.Provider value={reader}>{children}</AttachmentReaderContext.Provider>
}

/** The comment thread's `img` component (one stable type): a source that is
 *  one of the enclosing `AttachmentImages`' paths renders through
 *  `AttachmentImage`; anything else keeps the renderer's default. */
export function CommentImage({ fallback: Fallback, ...props }: ComponentProps<'img'> & { fallback: ImageComponent }) {
  const reader = useContext(AttachmentReaderContext)
  const src = typeof props.src === 'string' ? props.src : ''
  const path = !reader ? null : reader.known.has(src) ? src : reader.known.has(decoded(src)) ? decoded(src) : null

  return path && reader ? (
    <AttachmentImage alt={props.alt} Image={Fallback} path={path} read={reader.read} />
  ) : (
    <Fallback {...props} />
  )
}
