import { Codecs, persistentAtom } from '@/lib/persisted'

// Agents panel open state — mirrors the Logs pane's summon-only pattern
// (see app/contrib/controller.tsx's $logsOpen/syncLogsPane). This turns
// AgentsView from a route-based overlay into a standing right-sidebar panel
// that can be toggled on/off without navigating away from the current chat.
//
// Persisted (like $reviewOpen in store/review.ts) — a fresh boot re-opens the
// panel exactly as the user left it. Default true: the panel shows itself on
// first run (a fresh profile with no stored preference yet) instead of
// requiring the user to discover the ⌘K toggle before ever seeing it.
const OPEN_KEY = 'hermes.desktop.agentsPanelOpen'

export const $agentsPanelOpen = persistentAtom(OPEN_KEY, true, Codecs.bool)

export const openAgentsPanel = (): void => $agentsPanelOpen.set(true)
export const closeAgentsPanel = (): void => $agentsPanelOpen.set(false)
export const toggleAgentsPanel = (): void => $agentsPanelOpen.set(!$agentsPanelOpen.get())
