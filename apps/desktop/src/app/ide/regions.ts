// The IDE shell's region contract: one component per region, consumed by
// ide-shell.tsx. Leaves replace a region's internals without touching the
// frame — the shell only ever refers to these four names.

export { BrowserRegion } from './regions/browser'
export { ChatRegion } from './regions/chat'
export { EditorRegion } from './regions/editor'
export { ExplorerRegion } from './regions/explorer'
