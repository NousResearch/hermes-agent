import type { ReactElement } from 'react'

import { type SshTunnelDraft, SshTunnelFields } from '@/app/settings/ssh-tunnel-fields'

import type { RemoteSetup } from './use-remote-setup'

/** Optional SSH transport rows shared by Settings, boot recovery and first run. */
export function RemoteSetupTunnelFields({
  disabled,
  hidden,
  setup,
  urlDisabled
}: {
  disabled: boolean
  hidden: boolean
  setup: RemoteSetup
  urlDisabled: boolean
}): null | ReactElement {
  if (hidden) {
    return null
  }

  return (
    <SshTunnelFields
      disabled={disabled || urlDisabled}
      editor={{ kind: 'remote', sshTunnel: setup.credentials.sshTunnel ?? undefined }}
      onChange={(editor: { kind: string; sshTunnel?: SshTunnelDraft }) => setup.setSshTunnel(editor.sshTunnel ?? null)}
    />
  )
}
