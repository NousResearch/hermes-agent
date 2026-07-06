import { asRecords, asText, InvalidPlanBlock, MetaBadge, parseJsonRecord, PlanBlockShell } from './plan-block-utils'
import type { RichFenceProps } from './types'

const CHANGE_TONE: Record<string, string> = {
  added: 'text-emerald-600 dark:text-emerald-300',
  deleted: 'text-rose-600 dark:text-rose-300',
  modified: 'text-blue-600 dark:text-blue-300',
  renamed: 'text-violet-600 dark:text-violet-300'
}

function pathDepth(path: string): number {
  return Math.max(0, path.split('/').length - 1)
}

export default function FileTreeRenderer({ code }: RichFenceProps) {
  const tree = parseJsonRecord(code)

  if (!tree) {
    return <InvalidPlanBlock code={code} kind="file-tree" />
  }

  const title = asText(tree.title, 'File tree')
  const entries = asRecords(tree.entries ?? tree.files ?? tree.paths)

  if (entries.length === 0) {
    return <InvalidPlanBlock code={code} kind="file-tree" />
  }

  return (
    <PlanBlockShell title={title}>
      <div className="grid gap-1 font-mono text-xs">
        {entries.map((entry, index) => {
          const path = asText(entry.path ?? entry.file, `file-${index}`)
          const change = asText(entry.change ?? entry.status, 'changed')
          const note = asText(entry.note ?? entry.description)
          const depth = Math.min(pathDepth(path), 6)

          return (
            <div className="rounded-md px-2 py-1.5 hover:bg-muted/35" key={`${path}-${index}`}>
              <div className="flex min-w-0 items-center gap-2" style={{ paddingLeft: `${depth * 0.75}rem` }}>
                <span className="shrink-0 text-muted-foreground">└</span>
                <span className="min-w-0 flex-1 truncate">{path}</span>
                <MetaBadge className={CHANGE_TONE[change] ?? undefined}>{change}</MetaBadge>
              </div>
              {note ? <div className="mt-1 pl-6 text-[0.68rem] text-muted-foreground">{note}</div> : null}
            </div>
          )
        })}
      </div>
    </PlanBlockShell>
  )
}
