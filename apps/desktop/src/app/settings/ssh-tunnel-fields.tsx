import type { ReactElement } from 'react'

import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'

import { ListRow, ToggleRow } from './primitives'

export interface SshTunnelDraft {
  host: string
}

/**
 * The draft's SSH target as the save and sign-in payloads carry it. An enabled
 * toggle always sends its host, so an empty one is rejected rather than read
 * as "no tunnel"; `clear` sends `null` to turn a saved tunnel off.
 */
export function sshTunnelPayload(
  kind: string,
  sshTunnel: SshTunnelDraft | undefined,
  clear: boolean
): { sshTunnel?: SshTunnelDraft | null } {
  if (kind !== 'remote') {
    return {}
  }

  return sshTunnel ? { sshTunnel: { host: sshTunnel.host.trim() } } : clear ? { sshTunnel: null } : {}
}

/** Whether the saved entry has a tunnel the save must explicitly turn off. */
export function hadSshTunnel(
  registry: { connections: { id: string; sshTunnel?: SshTunnelDraft }[] } | null,
  id: null | string
): boolean {
  return Boolean(registry?.connections.find(c => c.id === id)?.sshTunnel)
}

/** Remote gateway editor rows: reach a loopback gateway URL through an SSH host. */
export function SshTunnelFields<T extends { kind: string; sshTunnel?: SshTunnelDraft }>({
  disabled,
  editor,
  onChange
}: {
  disabled: boolean
  editor: T
  onChange: (editor: T) => void
}): null | ReactElement {
  const { t } = useI18n()
  const s = t.settings.connections.sshTunnel

  if (editor.kind !== 'remote') {
    return null
  }

  return (
    <>
      <ToggleRow
        checked={Boolean(editor.sshTunnel)}
        description={s.description}
        disabled={disabled}
        label={s.title}
        onChange={enabled => onChange({ ...editor, sshTunnel: enabled ? { host: '' } : undefined })}
      />
      {editor.sshTunnel && (
        <ListRow
          action={
            <Input
              aria-label={s.host}
              disabled={disabled}
              onChange={e => onChange({ ...editor, sshTunnel: { host: e.target.value } })}
              placeholder={s.hostPlaceholder}
              value={editor.sshTunnel.host}
            />
          }
          title={s.host}
        />
      )}
    </>
  )
}
