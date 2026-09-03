export type DrawerTab = 'execution' | 'overview' | 'timeline'

type DrawerTabStatsInput = {
  comments: number
  events: number
  hasLog: boolean
  running: boolean
  runs: number
}

export function drawerTabStats({ comments, events, hasLog, running, runs }: DrawerTabStatsInput) {
  return {
    // Older tasks can retain worker output after their run history has been pruned.
    // Give that output a discoverable Execution badge without claiming it is a run count.
    executionCount: runs || (hasLog ? 1 : 0),
    executionLive: running,
    timelineCount: comments + events
  }
}
