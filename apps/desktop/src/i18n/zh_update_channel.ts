export const zhUpdateChannel = {
  title: '更新渠道',
  description: '稳定版只安装已发布的版本。每个提交会随 main 分支的变更持续更新。',
  stable: '稳定版发布',
  main: '每个提交',
  stableConfirmTitle: '切换到稳定版发布？',
  stableConfirmBody:
    'Hermes 将切换到最新的已发布版本，它可能比你当前运行的代码更旧。较新代码写入的数据可能无法正确读取，请先备份。',
  stableConfirm: '切换到稳定版',
  failed: '无法更改更新渠道',
  detailsLabel: '渠道',
  branch: (b: string) => `分支：${b}`,
  change: '更改',
  latestRelease: (v: string) => `你正在使用最新的稳定版发布（${v}）。`,
  rollbackTitle: (v: string) => `回退到稳定版 ${v}`,
  rollbackBody: (v: string) => `当前构建比稳定版 ${v} 更新。安装后 Hermes 将回退到该版本。`,
  heldBody: (v: string) =>
    `此版本比稳定版 ${v} 更新，且你的会话已在使用 ${v} 无法打开的网关运行时。在更新的稳定版发布前，Hermes 将保持当前版本。`,
  rollbackAction: (v: string) => `回退到 ${v}`,
  allChanges: '查看全部更改'
}
