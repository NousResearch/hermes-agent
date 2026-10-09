// The repo-scan field copy (labels + descriptions), composed by ja.ts.
import type { FieldCopyTree } from '@/app/settings/field-copy'

export const jaRepoScanLabels: FieldCopyTree = {
  desktop: {
    repoScanEnabled: 'リポジトリの自動検出',
    repoScanNested: 'ネストされたリポジトリも検出',
    repoScanRoots: 'リポジトリの検索ルート',
    repoScanExcludePaths: '除外するリポジトリパス'
  }
}

export const jaRepoScanDescriptions: FieldCopyTree = {
  desktop: {
    repoScanEnabled: 'ローカルフォルダを検索して Git リポジトリをプロジェクトに表示します。',
    repoScanNested: 'すでにあるリポジトリ内のリポジトリをサブプロジェクトとして表示します。',
    repoScanRoots: '追加で検索するフォルダー。空の場合は作業ディレクトリを検索します。',
    repoScanExcludePaths: 'リポジトリ検出時に除外するフォルダとその配下です。'
  }
}
