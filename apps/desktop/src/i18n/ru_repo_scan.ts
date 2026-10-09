// The repo-scan field copy (labels + descriptions), composed by ru.ts.
import type { FieldCopyTree } from '@/app/settings/field-copy'

export const ruRepoScanLabels: FieldCopyTree = {
  desktop: {
    repoScanEnabled: 'Автоматическое обнаружение репозиториев',
    repoScanNested: 'Также находить вложенные репозитории',
    repoScanRoots: 'Корни обнаружения репозиториев',
    repoScanExcludePaths: 'Исключаемые пути репозиториев'
  }
}

export const ruRepoScanDescriptions: FieldCopyTree = {
  desktop: {
    repoScanEnabled: 'Сканировать локальные папки на Git-репозитории, чтобы показывать их в Проектах.',
    repoScanNested: 'Показывать репозитории внутри уже добавленного как подпроекты.',
    repoScanRoots: 'Дополнительные папки для сканирования. Пусто — сканируется Рабочий каталог.',
    repoScanExcludePaths: 'Папки и их вложенные, которые нужно пропускать при обнаружении репозиториев.'
  }
}
