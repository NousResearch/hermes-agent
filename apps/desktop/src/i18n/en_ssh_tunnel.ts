import type { SshTunnelCopy } from './types_ssh_tunnel'

export const enSshTunnel: SshTunnelCopy = {
  title: 'Use SSH tunnel',
  description:
    'Reach this gateway through SSH. The URL is the gateway on the SSH host’s loopback (e.g. http://127.0.0.1:9119). Hermes never starts or stops it.',
  host: 'SSH host',
  hostPlaceholder: 'user@host:22'
}
