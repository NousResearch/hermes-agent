import { isPaneVisible, revealTreePane, setPaneCollapsed } from '@/components/pane-shell/tree/store'
import { registry } from '@/contrib/registry'

/**
 * `host.togglePane`: flip a plugin's own contributed pane from an explicit user
 * action (its keybind, its palette row). Like ⌘G and ⌃`, it asks the tree what
 * is on screen instead of keeping a flag that drifts.
 *
 * On screen, the pane folds away the way its zone's Minimize does. In a zone
 * shared with the conversation, the conversation takes the tab back, because
 * that zone cannot minimize. It never goes through Close: a closer-less pane
 * is dismissed by Close, and a single-pane plugin is switched off with it, so
 * "hide" would mean "disable until re-enabled".
 *
 * Panes the app owns are left alone, so a plugin cannot fold the conversation
 * or the app's chrome.
 */
export function togglePluginPane(paneId: string): void {
  const id = (paneId ?? '').trim()

  if (!id || !registry.getArea('panes').some(c => c.id === id && c.source?.startsWith('plugin:'))) {
    return
  }

  if (isPaneVisible(id)) {
    setPaneCollapsed(id, true)
  } else {
    revealTreePane(id)
  }
}
