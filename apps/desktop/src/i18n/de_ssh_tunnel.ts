import type { SshTunnelCopy } from './types_ssh_tunnel'

export const deSshTunnel: SshTunnelCopy = {
  title: 'SSH-Tunnel verwenden',
  description:
    'Dieses Gateway über SSH erreichen. Die URL ist das Gateway auf dem Loopback des SSH-Hosts (z. B. http://127.0.0.1:9119). Hermes startet oder stoppt es nie.',
  host: 'SSH-Host',
  hostPlaceholder: 'user@host:22'
}
