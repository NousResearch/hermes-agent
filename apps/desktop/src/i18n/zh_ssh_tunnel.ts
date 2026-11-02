import type { SshTunnelCopy } from './types_ssh_tunnel'

export const zhSshTunnel: SshTunnelCopy = {
  title: '使用 SSH 隧道',
  description:
    '通过 SSH 访问此网关。URL 是该网关在 SSH 主机回环地址上的地址（如 http://127.0.0.1:9119）。Hermes 从不启动或停止它。',
  host: 'SSH 主机',
  hostPlaceholder: 'user@host:22'
}
