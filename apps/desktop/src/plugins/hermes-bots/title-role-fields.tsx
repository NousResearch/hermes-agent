/**
 * The Title and Role inputs New agent and Edit profile share. Title is the
 * bot's name; Role is the short job label shown beside it (`bot-role.ts`).
 */

import { Input } from '@hermes/plugin-sdk'

import { BOT_ROLE_MAX } from './bot-role'
import { labeled } from './dialog-parts'
import { useBots } from './i18n'

interface TitleRoleFieldsProps {
  onRole: (value: string) => void
  onTitle: (value: string) => void
  /** The Title input's placeholder. */
  placeholder: string
  role: string
  title: string
}

export function TitleRoleFields({ onRole, onTitle, placeholder, role, title }: TitleRoleFieldsProps) {
  const b = useBots()

  return (
    <>
      {labeled(
        b.editor.title,
        <Input onChange={event => onTitle(event.target.value)} placeholder={placeholder} value={title} />
      )}
      {labeled(
        b.editor.role,
        <Input
          maxLength={BOT_ROLE_MAX}
          onChange={event => onRole(event.target.value)}
          placeholder={b.editor.rolePlaceholder}
          value={role}
        />
      )}
    </>
  )
}
