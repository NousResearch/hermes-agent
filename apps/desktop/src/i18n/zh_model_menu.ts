export const zhModelMenu = {
  search: '搜索模型',
  noModels: '未找到模型',
  editModels: '编辑模型…',
  followDefault: '使用设置中的默认模型',
  refreshModels: '刷新模型',
  favorites: '收藏',
  addFavorite: '添加到收藏',
  removeFavorite: '从收藏中移除',
  favoriteShortcut: '⇧ 单击',
  fast: '快速',
  free: '免费',
  cacheRead: '缓存读取',
  priceTitle: (input: string, output: string, cache: string) =>
    `输入 ${input}/Mtok · 输出 ${output}/Mtok` + (cache ? ` · 缓存读取 ${cache}/Mtok` : ''),
  catalogPrice: 'models.dev 目录中的标价；该提供商未提供自己的价格',
  contextTitle: (context: string) => `${context} token 上下文窗口`,
  vision: '支持图像输入',
  maxOutputLabel: (tokens: string) => `输出 ${tokens}`,
  maxOutputTitle: (tokens: string) => `回复最多 ${tokens} token`,
  perThousandTitle: (input: string, output: string) => `每 1K token：输入 ${input} · 输出 ${output}`,
  tools: '支持工具调用',
  localSetup: {
    title: '本地运行 · 免费、私密',
    text: (model: string, size: string) => `${model} 适合这台电脑 · 下载 ${size}`,
    action: '设置'
  },
  limited: '已限额',
  limitedUntil: (time: string) => `限额至 ${time}`,
  limitedTip: (provider: string, time: null | string) =>
    time
      ? `${provider} 已达到用量上限，将于 ${time} 重置；现在就可以先选好之后要用的模型。`
      : `${provider} 已达到用量上限；现在就可以先选好重置后要用的模型。`,
  modelResets: (time: string) => `${time} 恢复`,
  modelLimitedTip: (time: string) => `该模型已达到自身上限，将于 ${time} 恢复。这里的其他模型仍可使用。`,
  usageLeft: (percent: number, time: null | string) => (time ? `剩余 ${percent}% · ${time} 重置` : `剩余 ${percent}%`),
  poolAccounts: (count: number) => `${count} 个账户`,
  poolLimited: (limited: number, total: number) => `${limited}/${total} 个账户已限额`,
  poolAccount: (number: number) => `账户 ${number}`,
  poolUnknown: '用量暂不可用',
  poolUnavailable: '请重新登录',
  usageTip: (provider: string) => `${provider} 即将达到用量上限。`,
  usageWindow: (label: string, percent: number, time: null | string) =>
    time ? `${label}：剩余 ${percent}%，${time} 重置` : `${label}：剩余 ${percent}%`
}
