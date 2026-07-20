import { useStore } from '@nanostores/react'
import { type FC, useCallback, useRef } from 'react'

import { $avatarImages, $avatarNames, DEFAULT_NAMES, openAvatarEditor, type AvatarRole } from '@/store/avatar'
import { cn } from '@/lib/utils'

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const AVATAR_SIZE = 28 // px — matches the design system's control size

const FALLBACK_COLORS: Record<AvatarRole, { bg: string; fg: string }> = {
  assistant: { bg: '#00796B', fg: '#FFFFFF' },
  user: { bg: '#5C6BC0', fg: '#FFFFFF' }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function initials(name: string): string {
  return name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map(w => w[0].toUpperCase())
    .join('')
}

function acceptImageFiles(event: React.ChangeEvent<HTMLInputElement>, onDataUrl: (url: string) => void) {
  const file = event.target.files?.[0]

  if (!file) {
    return
  }

  const reader = new FileReader()

  reader.onload = () => {
    if (typeof reader.result === 'string') {
      onDataUrl(reader.result)
    }
  }

  reader.readAsDataURL(file)
  // allow re-upload of the same file
  event.target.value = ''
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface MessageAvatarProps {
  role: AvatarRole
  /** When true, clicking opens the native file picker (used in editor dialog). */
  editable?: boolean
  /** When true, clicking opens the avatar editor dialog. */
  clickToEdit?: boolean
  onImageChange?: (role: AvatarRole, dataUrl: string) => void
}

export const MessageAvatar: FC<MessageAvatarProps> = ({ clickToEdit = false, editable = false, onImageChange, role }) => {
  const images = useStore($avatarImages)
  const avatarNames = useStore($avatarNames)
  const name = avatarNames[role] || DEFAULT_NAMES[role]
  const imageSrc = images[role]
  const colors = FALLBACK_COLORS[role]
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFileChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      acceptImageFiles(event, url => onImageChange?.(role, url))
    },
    [onImageChange, role]
  )

  const avatar = imageSrc ? (
    <img
      alt={name}
      className="size-full rounded-full object-cover"
      data-slot="message-avatar-image"
      src={imageSrc}
    />
  ) : (
    <span
      className="flex size-full select-none items-center justify-center rounded-full text-[0.6875rem] font-semibold leading-none"
      data-slot="message-avatar-fallback"
      style={{ backgroundColor: colors.bg, color: colors.fg }}
    >
      {initials(name)}
    </span>
  )

  if (!editable) {
    if (clickToEdit) {
      return (
        <button
          aria-label={`${name} — click to edit avatar`}
          className={cn('message-avatar shrink-0 cursor-pointer transition-opacity hover:opacity-85', !imageSrc && 'avatar-fallback-ring')}
          data-slot="message-avatar"
          onClick={() => openAvatarEditor()}
          style={{ width: AVATAR_SIZE, height: AVATAR_SIZE }}
          title="Click to edit avatar"
          type="button"
        >
          {avatar}
        </button>
      )
    }

    return (
      <div
        aria-label={name}
        className={cn('message-avatar shrink-0', !imageSrc && 'avatar-fallback-ring')}
        data-slot="message-avatar"
        role="img"
        style={{ width: AVATAR_SIZE, height: AVATAR_SIZE }}
      >
        {avatar}
      </div>
    )
  }

  return (
    <button
      aria-label={`${name} — click to change avatar`}
      className="message-avatar message-avatar-editable shrink-0 cursor-pointer transition-opacity hover:opacity-80"
      data-slot="message-avatar"
      onClick={() => inputRef.current?.click()}
      style={{ width: AVATAR_SIZE, height: AVATAR_SIZE }}
      title="Click to change avatar"
      type="button"
    >
      {avatar}
      <input
        accept="image/png,image/jpeg,image/webp,image/gif,image/svg+xml"
        aria-hidden="true"
        className="hidden"
        onChange={handleFileChange}
        ref={inputRef}
        type="file"
      />
    </button>
  )
}
