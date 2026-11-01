import type { SshTunnelCopy } from './types_ssh_tunnel'

export const esSshTunnel: SshTunnelCopy = {
  title: 'Usar túnel SSH',
  description:
    'Accede a este gateway mediante SSH. La URL es la del gateway en el loopback del host SSH (p. ej. http://127.0.0.1:9119). Hermes nunca lo inicia ni lo detiene.',
  host: 'Host SSH',
  hostPlaceholder: 'user@host:22'
}
