'use strict'

const FOLDER_SPECS = [
  { id: 'desktop', label: '桌面', pathKey: 'desktop', aliases: ['desktop', '桌面'] },
  { id: 'documents', label: '文档', pathKey: 'documents', aliases: ['documents', 'document', 'docs', '文档'] },
  { id: 'downloads', label: '下载', pathKey: 'downloads', aliases: ['downloads', 'download', '下载'] },
  { id: 'home', label: '主页', pathKey: 'home', aliases: ['home', '主页', '主目录'] },
  { id: 'music', label: '音乐', pathKey: 'music', aliases: ['music', '音乐'] },
  { id: 'pictures', label: '图片', pathKey: 'pictures', aliases: ['pictures', 'picture', 'photos', '图片', '照片'] },
  { id: 'videos', label: '视频', pathKey: 'videos', aliases: ['videos', 'video', '视频'] }
]

function normalizeDesktopFolderTarget(rawTarget) {
  const normalized = String(rawTarget || '')
    .trim()
    .toLowerCase()

  if (!normalized) {
    return null
  }

  return FOLDER_SPECS.find(spec => spec.aliases.includes(normalized)) || null
}

module.exports = {
  FOLDER_SPECS,
  normalizeDesktopFolderTarget
}
