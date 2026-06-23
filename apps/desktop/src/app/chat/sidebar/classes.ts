import { cn } from '@/lib/utils'

export function sidebarSessionsSectionRootClassName(rootClassName: string | undefined, open: boolean): string {
  return cn(rootClassName, !open && 'min-h-0 flex-none shrink-0 overflow-visible')
}
