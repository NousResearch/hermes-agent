'use strict'

const APP_SPECS = [
  {
    id: 'notepad',
    label: '记事本',
    target: 'notepad.exe',
    aliases: ['notepad', '记事本', '文本编辑器']
  },
  {
    id: 'explorer',
    label: '资源管理器',
    target: 'explorer.exe',
    aliases: ['explorer', '文件管理器', '资源管理器', '我的电脑']
  },
  {
    id: 'calculator',
    label: '计算器',
    target: 'calc.exe',
    aliases: ['calc', 'calculator', '计算器']
  },
  {
    id: 'chrome',
    label: 'Chrome',
    target: 'chrome.exe',
    aliases: ['chrome', 'google chrome', '谷歌', '谷歌浏览器', 'chrome浏览器']
  },
  {
    id: 'edge',
    label: 'Edge',
    target: 'msedge.exe',
    aliases: ['edge', 'microsoft edge', '微软浏览器', 'edge浏览器']
  }
]

function normalizeDesktopAppTarget(rawTarget) {
  const normalized = String(rawTarget || '')
    .trim()
    .toLowerCase()

  if (!normalized) {
    return null
  }

  return APP_SPECS.find(spec => spec.aliases.includes(normalized)) || null
}

module.exports = {
  APP_SPECS,
  normalizeDesktopAppTarget
}
