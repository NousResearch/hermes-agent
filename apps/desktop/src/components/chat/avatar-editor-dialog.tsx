import { useStore } from '@nanostores/react'
import { type FC, useState } from 'react'

import { MessageAvatar } from '@/components/chat/message-avatar'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import {
  $avatarImages,
  $avatarNames,
  type AvatarRole,
  DEFAULT_NAMES,
  getAvatarName,
  setAvatarImage,
  setAvatarName
} from '@/store/avatar'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface AvatarEditorDialogProps {
  open: boolean
  onClose: () => void
}

// ---------------------------------------------------------------------------
// Single participant row
// ---------------------------------------------------------------------------

interface ParticipantRowProps {
  role: AvatarRole
  label: string
}

const ParticipantRow: FC<ParticipantRowProps> = ({ label, role }) => {
  const names = useStore($avatarNames)
  const images = useStore($avatarImages)
  const [localName, setLocalName] = useState(names[role])
  const hasImage = Boolean(images[role])

  return (
    <div className="flex items-center gap-3">
      <MessageAvatar
        editable
        onImageChange={setAvatarImage}
        role={role}
      />
      <div className="min-w-0 flex-1 space-y-1">
        <label className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
          {label}
        </label>
        <div className="flex items-center gap-2">
          <Input
            className="h-8 text-[length:var(--conversation-text-font-size)]"
            onChange={event => setLocalName(event.target.value)}
            onKeyDown={event => {
              if (event.key === 'Enter') {
                setAvatarName(role, localName)
              }
            }}
            onBlur={() => {
              const trimmed = localName.trim()

              if (trimmed && trimmed !== names[role]) {
                setAvatarName(role, trimmed)
              } else if (!trimmed) {
                setLocalName(names[role] || DEFAULT_NAMES[role])
              }
            }}
            placeholder={DEFAULT_NAMES[role]}
            value={localName}
          />
          {hasImage && (
            <Button
              aria-label="Remove avatar image"
              onClick={() => setAvatarImage(role, '')}
              size="icon-sm"
              variant="ghost"
            >
              ✕
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Dialog
// ---------------------------------------------------------------------------

export const AvatarEditorDialog: FC<AvatarEditorDialogProps> = ({ onClose, open }) => (
  <Dialog onOpenChange={onClose} open={open}>
    <DialogContent className="max-w-sm">
      <DialogHeader>
        <DialogTitle>Edit chat avatars</DialogTitle>
        <DialogDescription>Customise the display name and avatar for each participant.</DialogDescription>
      </DialogHeader>

      <div className="space-y-4 py-2">
        <ParticipantRow label="You (user)" role="user" />
        <ParticipantRow label="Hermes (assistant)" role="assistant" />
      </div>

      <DialogFooter>
        <Button onClick={onClose} variant="secondary">
          Done
        </Button>
      </DialogFooter>
    </DialogContent>
  </Dialog>
)
