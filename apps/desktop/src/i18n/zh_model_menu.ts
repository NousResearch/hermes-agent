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
  limited: '已限额',
  limitedUntil: (time: string) => `限额至 ${time}`,
  limitedTip: (provider: string, time: null | string) =>
    time
      ? `${provider} 已达到用量上限，将于 ${time} 重置；现在就可以先选好之后要用的模型。`
      : `${provider} 已达到用量上限；现在就可以先选好重置后要用的模型。`,
  modelResets: (time: string) => `${time} 恢复`,
  modelLimitedTip: (time: string) => `该模型已达到自身上限，将于 ${time} 恢复。这里的其他模型仍可使用。`
}
