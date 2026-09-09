import type { BotMeta } from './types'

/** What Edit Profile shows for a bot's look, as opened and as submitted. */
export interface EditProfileLook {
  color: null | string
  image: null | string
  shape: BotMeta['shape']
  title: string
  circle: string
}

/** The fields the user actually changed since the dialog opened, as a
 *  saveBotMeta patch; null when nothing changed. Untouched fields stay out of
 *  the patch: a form value seeded minutes ago is not an edit, and sending it
 *  would revert whatever another Desktop or the backend wrote since (a rename
 *  undone by a color change). */
export function editedLook(opened: EditProfileLook, current: EditProfileLook): BotMeta | null {
  const patch: BotMeta = {}

  if (current.shape !== opened.shape) {
    patch.shape = current.shape
  }

  if (current.color !== opened.color) {
    patch.color = current.color ?? undefined
  }

  if (current.image !== opened.image) {
    patch.image = current.image
    patch.imageKind = current.image ? 'photo' : 'shape'
  }

  if (Object.keys(patch).length > 0) {
    patch.custom = true
  }

  if (current.title.trim() !== opened.title.trim()) {
    patch.title = current.title.trim()
  }

  // Circle is mesh membership, not look: it never marks the look custom.
  const circle = normalizeCircle(current.circle)

  if (circle !== normalizeCircle(opened.circle)) {
    patch.circle = circle
  }

  return Object.keys(patch).length > 0 ? patch : null
}

/** Same rule as the gateway's circle reader: trimmed, case-insensitive, capped at 64. */
function normalizeCircle(value: string): string {
  return value.trim().toLowerCase().slice(0, 64)
}
