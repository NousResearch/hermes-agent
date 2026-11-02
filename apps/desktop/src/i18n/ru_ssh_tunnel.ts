import type { SshTunnelCopy } from './types_ssh_tunnel'

export const ruSshTunnel: SshTunnelCopy = {
  title: 'Использовать SSH-туннель',
  description:
    'Подключаться к этому шлюзу через SSH. URL — адрес шлюза на loopback SSH-хоста (например, http://127.0.0.1:9119). Hermes никогда не запускает и не останавливает его.',
  host: 'SSH-хост',
  hostPlaceholder: 'user@host:22'
}
