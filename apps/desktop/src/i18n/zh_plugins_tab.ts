import type { PluginsTabCopy } from './types_plugins_tab'

export const zhPluginsTab: Omit<PluginsTabCopy, 'serverStates'> = {
  desktopHalfPending: '复制中…',
  desktopHalfPendingTip: '此包附带的桌面部分尚未复制到应用中。请重新扫描或重启应用。',
  desktopHalfRemote: '不可用（远程后端）',
  desktopHalfRemoteTip:
    '此包的桌面部分位于远程后端的磁盘上，本应用无法读取。要在此使用，请通过“从 Git 安装”输入该包的仓库地址并勾选桌面目标，即可将桌面部分克隆到本机。',
  fleet: {
    updateEverywhere: (count: number) => `全部更新 · ${count}`,
    updateHereOnly: (profile: string) => `仅 ${profile}`,
    updateAll: (count: number) => `全部更新 · ${count}`,
    updateAllTip: (installs: number, plugins: number) => `${plugins} 个插件在你网关上的 ${installs} 个配置中已过时`,
    titleOne: (name: string) => `在所有位置更新 ${name}`,
    titleAll: '在所有位置更新插件',
    summary: (installs: number, gateways: number) => `${gateways} 个网关上的 ${installs} 个配置运行旧版本。`,
    summaryDone: (installs: number) => `已更新 ${installs} 个配置。重启这些网关以加载新版本。`,
    summaryFailed: (ok: number, failed: number) => `${ok} 个已更新，${failed} 个失败。悬停失败项查看原因。`,
    summaryDeclined: (ok: number, declined: number) => `${ok} 个已更新。${declined} 个未应用，因为新权限被拒绝。`,
    thisProfile: '· 当前配置',
    failed: '失败',
    declined: '未应用',
    skippedOffline: '离线 · 已跳过',
    skippedSignIn: '未登录 · 已跳过',
    skippedIdle: '未连接 · 已跳过',
    skippedUnreadable: '无法检查',
    confirm: (installs: number) => `更新 ${installs} 个`,
    updating: '正在更新…',
    done: '完成',
    consentBody: (names: string, installs: number) =>
      `${names} 的新目录版本增加了已安装版本没有的功能。批准后将应用到 ${installs} 个配置：`,
    finished: (installs: number) => `已更新 ${installs} 个配置。重启这些网关后生效。`
  }
}
