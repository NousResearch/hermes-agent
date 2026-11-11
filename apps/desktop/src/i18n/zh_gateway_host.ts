import type { GatewayHostTranslations } from './types_gateway_host'

// Settings → Gateways copy where the connection cannot change, spread into zh.ts.
export const zhGatewayHost: GatewayHostTranslations = {
  unavailableTitle: '机器设置不可用',
  unavailableDesc: '桌面 IPC 桥未暴露机器设置。',
  webappHostTitle: 'Hermes 主机',
  webappHostDesc:
    'Webapp 始终运行在提供它的 Hermes 主机上。下方已保存的连接是这台机器的列表，与 Hermes Desktop 应用共享；其他机器通过此主机访问。',
  webappOauthPaste: {
    title: '完成登录',
    description: '如果登录标签页停在无法打开的地址（此浏览器不在连接的机器上），请将该地址粘贴到这里。',
    placeholder: 'http://127.0.0.1:…/webapp/gateway-connections/oauth/callback?…',
    finish: '完成',
    failed: '无法完成登录'
  },
  webAccess: {
    title: '网页访问',
    intro: '用手机或其他电脑，以你的 Nous 账户登录并打开这台电脑上的 Hermes。登录的人可以在这台电脑上运行命令。',
    failed: '无法更改网页访问',
    running: '运行中',
    stopped: '已停止',
    starting: '正在启动',
    stopping: '正在停止',
    restartNeeded: '需要重启',
    unavailable: '不可用',
    notRunningDetail: '启动后即可从其他设备打开 Hermes。',
    needsNous: '请先在“提供商”中登录你的 Nous 账户。',
    runningLan: '在同一网络中的设备上打开此地址。',
    runningPublic: '通过你的网址提供服务。',
    runningLocal: '只有这台电脑可以打开。要从其他设备打开，请先停止，再重新启动。',
    pendingOn: '已保存。Hermes 重启后将提供网页应用。',
    pendingOff: '已关闭，但 Hermes 在重启前仍会继续提供网页应用。',
    unavailableDetail: reason => `Hermes 无法提供网页应用：${reason}`,
    restartBusy: '已保存。有对话正在进行，因此 Hermes 没有重启。对话结束后请选择“重启 Hermes”。',
    restartFailed: '已保存，但 Hermes 无法重启。请选择“重启 Hermes”重试。',
    restartFailedBecause: reason => `Hermes 无法重启：${reason}`,
    startTimedOut: 'Hermes 未能在 90 秒内提供网页应用。',
    stopTimedOut: 'Hermes 未能在 90 秒内停止提供网页应用。',
    startStages: {
      preparing: '正在准备网页应用。构建可能需要几分钟。',
      building: '正在构建网页应用，可能需要几分钟。',
      registering: '正在为此地址设置 Nous 登录。',
      restarting: '正在重启 Hermes 以提供网页应用。'
    },
    startElapsed: time => `已用时 ${time}`,
    start: '启动',
    stop: '停止',
    restartGateway: '重启 Hermes',
    modeLan: '当前网络',
    modeLanDetail: url => `你的 Wi-Fi 中的设备可打开 ${url}。仅在你信任的网络中使用。`,
    lanUnencrypted: '连接未加密：同一网络中的其他人可能读取你发送和接收的内容。',
    noLan: '这台电脑没有其他设备可以访问的网络地址。',
    modePublic: '我自己的网址',
    modePublicDetail: port => `你已有的 HTTPS 地址（例如 Tailscale Serve 或隧道），并转发到 http://127.0.0.1:${port}。`
  }
}
