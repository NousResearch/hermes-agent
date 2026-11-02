import type { SshTunnelCopy } from './types_ssh_tunnel'

export const jaSshTunnel: SshTunnelCopy = {
  title: 'SSH トンネルを使う',
  description:
    'SSH 経由でこのゲートウェイに接続します。URL は SSH ホストのループバック上のゲートウェイです（例: http://127.0.0.1:9119）。Hermes が起動・停止することはありません。',
  host: 'SSH ホスト',
  hostPlaceholder: 'user@host:22'
}
