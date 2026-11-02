import type { GatewayHostTranslations } from './types_gateway_host'

// Settings → Gateways copy where the connection cannot change, spread into zh.ts.
export const zhGatewayHost: GatewayHostTranslations = {
  unavailableTitle: '网关设置不可用',
  unavailableDesc: '桌面 IPC 桥未暴露网关设置。',
  webappHostTitle: 'Hermes 主机',
  webappHostDesc:
    'Webapp 始终使用提供它的 Hermes 主机。如需切换网关、登录 Hermes Cloud 或管理已保存的连接，请使用 Hermes Desktop 应用。',
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
    pendingOn: '已保存。网关重启后将提供网页应用。',
    pendingOff: '已关闭，但网关在重启前仍会继续提供网页应用。',
    unavailableDetail: reason => `网关无法提供网页应用：${reason}`,
    restartBusy: '已保存。有对话正在进行，因此网关没有重启。对话结束后请选择“重启网关”。',
    restartFailed: '已保存，但网关无法重启。请选择“重启网关”重试。',
    restartFailedBecause: reason => `网关无法重启：${reason}`,
    startTimedOut: '网关未能在 90 秒内提供网页应用。',
    stopTimedOut: '网关未能在 90 秒内停止提供网页应用。',
    start: '启动',
    stop: '停止',
    restartGateway: '重启网关',
    modeLan: '当前网络',
    modeLanDetail: url => `你的 Wi-Fi 中的设备可打开 ${url}。仅在你信任的网络中使用。`,
    lanUnencrypted: '连接未加密：同一网络中的其他人可能读取你发送和接收的内容。',
    noLan: '这台电脑没有其他设备可以访问的网络地址。',
    modePublic: '我自己的网址',
    modePublicDetail: port => `你已有的 HTTPS 地址（例如 Tailscale Serve 或隧道），并转发到 http://127.0.0.1:${port}。`
  }
}
