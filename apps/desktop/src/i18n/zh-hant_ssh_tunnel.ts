import type { SshTunnelCopy } from './types_ssh_tunnel'

export const zhHantSshTunnel: SshTunnelCopy = {
  title: '使用 SSH 通道',
  description:
    '透過 SSH 存取此閘道。URL 是該閘道在 SSH 主機回送位址上的位址（如 http://127.0.0.1:9119）。Hermes 從不啟動或停止它。',
  host: 'SSH 主機',
  hostPlaceholder: 'user@host:22'
}
