import type { ReactNode } from 'react'

export function RightSidebarSectionHeader({ children }: { children: ReactNode }) {
  return <div className="flex h-7 shrink-0 items-center px-2.5">{children}</div>
}
