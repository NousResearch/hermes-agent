// The repo-scan field copy (labels + descriptions), composed by zh.ts.
import type { FieldCopyTree } from '@/app/settings/field-copy'

export const zhRepoScanLabels: FieldCopyTree = {
  desktop: {
    repoScanEnabled: '自动发现代码仓库',
    repoScanNested: '同时发现嵌套仓库',
    repoScanRoots: '代码仓库扫描根目录',
    repoScanExcludePaths: '排除的代码仓库路径'
  }
}

export const zhRepoScanDescriptions: FieldCopyTree = {
  desktop: {
    repoScanEnabled: '扫描本地文件夹，并在“项目”中显示 Git 代码仓库。',
    repoScanNested: '将已有仓库内部的仓库显示为子项目。',
    repoScanRoots: '要扫描的其他文件夹。留空则扫描工作目录。',
    repoScanExcludePaths: '发现代码仓库时跳过这些文件夹及其子目录。'
  }
}
