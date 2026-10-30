import { useStore } from '@nanostores/react'
import { useMemo } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { useI18n } from '@/i18n'

import { $folderSortPrefs, folderSortPreference, setFolderSortPreference } from './prefs'
import {
  FILE_SORT_KEYS,
  type FileSortKey,
  fileSortKey,
  type FileSortPreference,
  parseFileSortKey
} from './sort'

interface FilesSortMenuProps {
  /** Folder the control edits (workspace root for the Files header). */
  folderPath: string
}

export function FilesSortMenu({ folderPath }: FilesSortMenuProps) {
  const { t } = useI18n()
  const copy = t.rightSidebar.filesSort
  // Subscribe so the trigger label updates when prefs change.
  useStore($folderSortPrefs)

  const current = folderSortPreference(folderPath)
  const currentKey = fileSortKey(current)

  const labels = useMemo<Record<FileSortKey, string>>(
    () => ({
      'name:asc': copy.nameAsc,
      'name:desc': copy.nameDesc,
      'mtime:desc': copy.mtimeDesc,
      'mtime:asc': copy.mtimeAsc,
      'size:desc': copy.sizeDesc,
      'size:asc': copy.sizeAsc
    }),
    [copy]
  )

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          aria-label={copy.ariaLabel(labels[currentKey])}
          className="size-5 text-sidebar-foreground/70 hover:bg-sidebar-accent! hover:text-sidebar-accent-foreground!"
          size="icon-xs"
          title={copy.ariaLabel(labels[currentKey])}
          type="button"
          variant="ghost"
        >
          <Codicon name="filter" size="0.8125rem" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-56 p-1">
        <DropdownMenuLabel>{copy.title}</DropdownMenuLabel>
        <DropdownMenuSeparator />
        <DropdownMenuRadioGroup
          onValueChange={value => {
            const pref = parseFileSortKey(value) as FileSortPreference | null

            if (pref) {
              setFolderSortPreference(folderPath, pref)
            }
          }}
          value={currentKey}
        >
          {FILE_SORT_KEYS.map(key => (
            <DropdownMenuRadioItem key={key} value={key}>
              {labels[key]}
            </DropdownMenuRadioItem>
          ))}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
