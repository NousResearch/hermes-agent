import type { TranslationOverrides } from './define-locale'

export const zhChrome = {
  fileMenu: {
    openWithApp: '用默认应用打开',
    revealFinder: '在访达中显示',
    revealExplorer: '在文件资源管理器中显示',
    revealFileManager: '打开所在文件夹',
    revealInSidebar: '在文件树中显示',
    copyPath: '复制路径',
    copyRelativePath: '复制相对路径',
    download: '下载',
    downloadSaved: '已保存',
    downloadFailed: '下载失败',
    rename: '重命名…',
    delete: '删除',
    renameTitle: '重命名',
    renameLabel: '新名称',
    deleteTitle: (name: string) => `删除 ${name}？`,
    deleteBody: '将移至废纸篓，你可以从那里恢复。',
    pathCopied: '已复制路径',
    revealMissing: '该文件夹不在这台电脑上'
  }
} satisfies TranslationOverrides
