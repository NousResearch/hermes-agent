import type { SshTunnelCopy } from './types_ssh_tunnel'

export const arSshTunnel: SshTunnelCopy = {
  title: 'استخدام نفق SSH',
  description:
    'الوصول إلى هذه البوابة عبر SSH. عنوان URL هو البوابة على واجهة loopback لمضيف SSH (مثل http://127.0.0.1:9119). لا يقوم Hermes بتشغيلها أو إيقافها أبدًا.',
  host: 'مضيف SSH',
  hostPlaceholder: 'user@host:22'
}
