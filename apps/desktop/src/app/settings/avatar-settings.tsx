import { useStore } from '@nanostores/react'
import { type ChangeEvent, type DragEvent, useCallback, useRef, useState } from 'react'

import { AssistantAvatar } from '@/components/assistant-ui/thread/assistant-avatar'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { ImageIcon, LinkIcon, Loader2, Trash2, Upload } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $avatarDataUrl, resetAvatar, setAvatar } from '@/store/avatar'

import { SectionHeading } from './primitives'

const COPY_FALLBACK = {
  heading: 'Avatar',
  description: 'Set a profile picture for the AI agent that appears next to responses in the chat.',
  upload: 'Choose Image',
  change: 'Change Image',
  remove: 'Remove Avatar',
  saving: 'Saving…',
  removeConfirm: 'Remove avatar picture?',
  removeDescription: 'The avatar will be removed from chat displays. This can be undone by uploading a new image.',
  unsupported: 'Unsupported image format. Please use PNG, JPEG, WebP, or GIF.',
  tooLarge: 'Image is too large. Please choose one under 5 MB.',
  saveFailed: 'Failed to save avatar.',
  urlLabel: 'Import from URL',
  urlPlaceholder: 'https://example.com/avatar.png',
  urlImport: 'Import',
  urlInvalid: 'Please enter a valid image URL.',
  dragHere: 'Drop an image here',
  or: 'or'
}

/** Convert a File or image URL to a 256×256 center-cropped PNG data URL. */
async function imageToPngDataUrl(source: File | string, maxSize = 256): Promise<string> {
  let rawDataUrl: string

  if (typeof source === 'string') {
    rawDataUrl = source
  } else {
    rawDataUrl = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve(reader.result as string)
      reader.onerror = () => reject(new Error('Failed to read file'))
      reader.readAsDataURL(source)
    })
  }

  // Decode and crop-square
  const img = await new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image()
    image.onload = () => resolve(image)
    image.onerror = () => reject(new Error('Failed to decode image'))
    image.src = rawDataUrl
  })

  const size = Math.min(img.width, img.height, maxSize)
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')!

  const sx = (img.width - size) / 2
  const sy = (img.height - size) / 2
  ctx.drawImage(img, sx, sy, size, size, 0, 0, size, size)

  return canvas.toDataURL('image/png')
}

export function AvatarSettings() {
  const { t } = useI18n()
  const copy = (t as any).settings?.appearance?.avatar ?? COPY_FALLBACK
  const avatarDataUrl = useStore($avatarDataUrl)
  const [saving, setSaving] = useState(false)
  const [confirmRemove, setConfirmRemove] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const [urlInput, setUrlInput] = useState('')
  const [urlImporting, setUrlImporting] = useState(false)
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  // ── File upload ────────────────────────────────────────────────────

  const processFile = useCallback(async (file: File) => {
    if (!file.type.startsWith('image/')) {
      alert(copy.unsupported)
      return
    }

    if (file.size > 5 * 1024 * 1024) {
      alert(copy.tooLarge)
      return
    }

    setSaving(true)

    try {
      const pngDataUrl = await imageToPngDataUrl(file)
      await setAvatar(pngDataUrl)
      triggerHaptic('success')
    } catch (error) {
      console.error('[avatar] Failed to save:', error)
      alert(copy.saveFailed)
    } finally {
      setSaving(false)

      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
    }
  }, [copy])

  const handleFileSelect = useCallback(async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]

    if (!file) return
    await processFile(file)
  }, [processFile])

  // ── Drag-and-drop ──────────────────────────────────────────────────

  const handleDragOver = useCallback((e: DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e: DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragOver(false)
  }, [])

  const handleDrop = useCallback(async (e: DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragOver(false)

    const file = e.dataTransfer.files?.[0]

    if (file) {
      await processFile(file)
    }
  }, [processFile])

  // ── URL import ─────────────────────────────────────────────────────

  const handleUrlImport = useCallback(async () => {
    const url = urlInput.trim()

    if (!url || (!url.startsWith('http://') && !url.startsWith('https://'))) {
      alert(copy.urlInvalid)
      return
    }

    setUrlImporting(true)

    try {
      // Use the existing saveImageFromUrl IPC to download to disk,
      // then read it back. Simpler: fetch in the renderer.
      const response = await fetch(url)

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }

      const blob = await response.blob()

      if (!blob.type.startsWith('image/')) {
        alert(copy.unsupported)
        return
      }

      if (blob.size > 5 * 1024 * 1024) {
        alert(copy.tooLarge)
        return
      }

      // Convert blob to data URL then to cropped PNG
      const dataUrl = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader()
        reader.onload = () => resolve(reader.result as string)
        reader.onerror = () => reject(new Error('Failed to read blob'))
        reader.readAsDataURL(blob)
      })

      const pngDataUrl = await imageToPngDataUrl(dataUrl)
      await setAvatar(pngDataUrl)
      setUrlInput('')
      triggerHaptic('success')
    } catch (error) {
      console.error('[avatar] URL import failed:', error)
      alert(copy.saveFailed)
    } finally {
      setUrlImporting(false)
    }
  }, [copy, urlInput])

  // ── Remove ─────────────────────────────────────────────────────────

  const handleRemove = useCallback(async () => {
    setConfirmRemove(false)
    setSaving(true)

    try {
      await resetAvatar()
      triggerHaptic('crisp')
    } catch (error) {
      console.error('[avatar] Failed to reset:', error)
    } finally {
      setSaving(false)
    }
  }, [])

  return (
    <div className="flex flex-col gap-6">
      <SectionHeading description={copy.description} title={copy.heading} />

      {/* ── Current avatar + upload buttons ────────────────────────── */}
      <div className="flex items-center gap-6">
        <div
          className={cn(
            'flex size-20 shrink-0 items-center justify-center overflow-hidden rounded-full border-2 border-(--ui-stroke-tertiary)',
            !avatarDataUrl && 'bg-(--ui-surface-secondary)'
          )}
        >
          {avatarDataUrl ? (
            <img
              alt="AI agent avatar"
              className="size-full object-cover"
              src={avatarDataUrl}
            />
          ) : (
            <ImageIcon className="size-8 text-(--ui-text-quaternary)" />
          )}
        </div>

        <div className="flex flex-col gap-2">
          <input
            accept="image/png,image/jpeg,image/webp,image/gif"
            className="hidden"
            onChange={handleFileSelect}
            ref={fileInputRef}
            type="file"
          />

          <Button
            disabled={saving}
            onClick={() => fileInputRef.current?.click()}
            variant="outline"
          >
            {saving ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              <Upload className="size-4" />
            )}
            {avatarDataUrl ? copy.change : copy.upload}
          </Button>

          {avatarDataUrl && (
            <Button
              disabled={saving}
              onClick={() => setConfirmRemove(true)}
              variant="ghost"
            >
              <Trash2 className="size-4" />
              {copy.remove}
            </Button>
          )}
        </div>
      </div>

      {/* ── Drag-and-drop zone ─────────────────────────────────────── */}
      <div
        className={cn(
          'flex flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed p-6 text-center transition-colors',
          dragOver
            ? 'border-(--dt-accent) bg-(--dt-accent)/5'
            : 'border-(--ui-stroke-tertiary) bg-(--ui-surface-secondary)/50'
        )}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        role="button"
      >
        <Upload className="size-6 text-(--ui-text-tertiary)" />
        <p className="text-xs text-(--ui-text-secondary)">
          {dragOver ? '\u2B07 Drop it here' : copy.dragHere}
        </p>
      </div>

      {/* ── URL import ─────────────────────────────────────────────── */}
      <div className="flex flex-col gap-2">
        <label className="text-xs font-medium text-(--ui-text-secondary)">
          {copy.urlLabel}
        </label>
        <div className="flex gap-2">
          <Input
            className="flex-1"
            disabled={urlImporting}
            onChange={e => setUrlInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleUrlImport()}
            placeholder={copy.urlPlaceholder}
            type="url"
            value={urlInput}
          />
          <Button
            disabled={urlImporting || !urlInput.trim()}
            onClick={handleUrlImport}
            variant="outline"
          >
            {urlImporting ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              <LinkIcon className="size-4" />
            )}
            {copy.urlImport}
          </Button>
        </div>
      </div>

      {/* ── Inline chat preview ────────────────────────────────────── */}
      <div className="rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-surface-secondary) p-4">
        <div className="mb-2 text-xs font-medium text-(--ui-text-secondary)">
          Preview
        </div>
        <div className="flex items-start gap-2.5">
          <AssistantAvatar />
          <div className="rounded-xl border border-(--ui-stroke-tertiary) bg-(--dt-user-bubble) px-3 py-2 text-sm text-foreground">
            Hello! How can I help you today?
          </div>
        </div>
      </div>

      {/* ── Remove confirmation dialog ─────────────────────────────── */}
      {confirmRemove && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
          onClick={() => setConfirmRemove(false)}
        >
          <div
            className="mx-4 max-w-sm rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-surface-primary) p-6 shadow-lg"
            onClick={e => e.stopPropagation()}
          >
            <h3 className="mb-2 text-sm font-semibold">
              {copy.removeConfirm}
            </h3>
            <p className="mb-4 text-xs text-(--ui-text-secondary)">
              {copy.removeDescription}
            </p>
            <div className="flex justify-end gap-2">
              <Button
                onClick={() => setConfirmRemove(false)}
                variant="ghost"
              >
                Cancel
              </Button>
              <Button
                onClick={handleRemove}
                variant="destructive"
              >
                {copy.remove}
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
