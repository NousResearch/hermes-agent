import type { PluginUpdateAllTranslations } from './types_plugin_update'

/** Plugins tab: "Update all" and the auto-update switch. */
export const zhPluginUpdateAll: PluginUpdateAllTranslations = {
  updateAll: '全部更新',
  updatingAll: '正在更新…',
  updateAllTip: '检查所有已安装的插件，并应用无需审核的更新',
  updateAllDone: (updated: number, review: number, failed: number) =>
    `已更新 ${updated} 个` + (review ? `，${review} 个需要审核` : '') + (failed ? `，${failed} 个失败` : ''),
  updateAllNone: '所有插件均已是最新',
  updateAllFailed: '无法更新插件',
  autoUpdate: '自动更新',
  autoUpdateTip: (hours: number) => `网关检查时（每 ${hours} 小时）自动应用插件更新。更新后的代码在新会话中加载。`,
  autoUpdateFailed: '无法更改插件自动更新'
}
