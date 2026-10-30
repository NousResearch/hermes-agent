import type { SshTunnelCopy } from './types_ssh_tunnel'

export const frSshTunnel: SshTunnelCopy = {
  title: 'Utiliser un tunnel SSH',
  description:
    'Joindre cette gateway via SSH. L’URL est celle de la gateway sur le loopback de l’hôte SSH (p. ex. http://127.0.0.1:9119). Hermes ne la démarre ni ne l’arrête jamais.',
  host: 'Hôte SSH',
  hostPlaceholder: 'user@host:22'
}
