import { findGroupOfPane } from '@/components/pane-shell/tree/model'
import { $layoutTree, removeTreePane } from '@/components/pane-shell/tree/store'
import { $botsPaneHosted } from '@/plugins/hermes-bots/bots-pane-host'
import { modeLayout } from '@/store/interface-mode'
import { $botsInAgentsPage } from '@/store/shell-policy'

/** The Bots plugin's sidebar pane, as the registry namespaces it. */
export const BOTS_PANE_ID = 'hermes-bots:pane'

/** The redesign's layout never holds the Bots pane. A classic→redesign flip loads the redesign's
 *  layout while the pane may still be registered (adoption takes it in), and the pane unregisters
 *  a moment later; the order of the two is not ours to rely on, so both ends prune. Only the
 *  redesign's own scope is touched: the classic layout keeps its Bots tab for a rollback. */
function pruneUnhostedBotsPane(): void {
  const tree = $layoutTree.get()

  if ($botsInAgentsPage.get() && modeLayout.scope.shell === 'portal' && tree && findGroupOfPane(tree, BOTS_PANE_ID)) {
    removeTreePane(BOTS_PANE_ID)
  }
}

/** The redesign's Agents page hosts Bots, so the Bots plugin's sidebar pane is classic-only. The
 *  app owns that decision; the plugin only follows it, so no rollout flag reaches the plugin SDK. */
export function bindBotsPaneToShell(): () => void {
  const stopHosting = $botsInAgentsPage.subscribe(inAgentsPage => {
    $botsPaneHosted.set(!inAgentsPage)
    pruneUnhostedBotsPane()
  })

  const stopRestore = modeLayout.onRestore(pruneUnhostedBotsPane)

  return () => {
    stopHosting()
    stopRestore()
  }
}
