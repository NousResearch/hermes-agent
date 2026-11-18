import { beforeEach, expect, it, vi } from 'vitest'

beforeEach(() => {
  window.localStorage.clear()
  vi.resetModules()
})

// Boots in the classic shell with the real tree store, registry and shell switch; the Bots pane
// follows `$botsPaneHosted` exactly as the plugin's register() does.
async function bootClassicWithBots() {
  window.localStorage.setItem('hermes.desktop.agentShell.v1', 'false')
  const { BOTS_PANE_ID, bindBotsPaneToShell } = await import('./bots-pane-shell')
  const { findGroupOfPane, group, split } = await import('@/components/pane-shell/tree/model')
  const tree = await import('@/components/pane-shell/tree/store')
  const { registry } = await import('@/contrib/registry')
  const { $botsPaneHosted } = await import('@/plugins/hermes-bots/bots-pane-host')
  const { $shellChoice } = await import('@/store/shell-variant')

  registry.registerMany([
    { area: 'panes', data: { placement: 'main' }, id: 'workspace', render: () => null, title: 'Chat' },
    { area: 'panes', data: { placement: 'left' }, id: 'sessions', render: () => null, title: 'Sessions' }
  ])
  tree.declareDefaultTree(split('row', [group(['sessions'], { id: 'g-side' }), group(['workspace'], { id: 'g-main' })]))
  tree.watchContributedPanes()

  let disposeBots: (() => void) | undefined

  $botsPaneHosted.subscribe(hosted => {
    if (!hosted) {
      disposeBots?.()
      disposeBots = undefined
    } else if (!disposeBots) {
      disposeBots = registry.register({
        area: 'panes',
        data: { dock: { enforce: true, pane: 'sessions', pos: 'center' }, placement: 'left' },
        id: BOTS_PANE_ID,
        render: () => null,
        title: 'Bots'
      })
    }
  })
  bindBotsPaneToShell()

  const zoneOfBots = () => findGroupOfPane(tree.$layoutTree.get()!, BOTS_PANE_ID)?.panes

  return { $shellChoice, zoneOfBots }
}

it('keeps the unregistered Bots pane out of the redesign layout and hands classic its tab back', async () => {
  const { $shellChoice, zoneOfBots } = await bootClassicWithBots()

  expect(zoneOfBots()).toEqual(['sessions', 'hermes-bots:pane'])

  $shellChoice.set(true)
  expect(zoneOfBots()).toBeUndefined()

  $shellChoice.set(false)
  expect(zoneOfBots()).toEqual(['sessions', 'hermes-bots:pane'])
})
