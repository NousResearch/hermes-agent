/**
 * Runtime-loaded example — this file is NOT bundled as a module: it ships as
 * raw text (`?raw`) and goes through the real runtime pipeline (specifier
 * rewrite -> SDK/react shim blobs -> blob import -> register). Plain ESM js
 * with `jsx()` calls — exactly what an agent (or a compiler) writes into
 * `~/.hermes/desktop-plugins/<name>/plugin.js`.
 *
 * It also shows the Settings ▸ Plugins contract: one settings page with two
 * sub-pages (`ctx.registerSettingsPage`), persisted with `ctx.storage`.
 */

import { atom, cn, host, ListRow, SegmentedControl, Tip, ToggleRow, useValue } from '@hermes/plugin-sdk'
import { jsx, jsxs } from 'react/jsx-runtime'

const DEFAULTS = { label: 'word', showChip: true }

const GATEWAY_STATUS = { closed: 'Disconnected', connecting: 'Connecting…', error: 'Error', idle: 'Idle', open: 'Connected' }

function RuntimeChip({ prefs }) {
  const gateway = useValue(host.state.gateway)
  const { label, showChip } = useValue(prefs)

  if (!showChip) {
    return null
  }

  return jsx(Tip, {
    label: `Loaded at RUNTIME through blob import + SDK injection (gateway: ${gateway})`,
    children: jsxs('span', {
      className: cn('inline-flex h-full items-center gap-1 px-1.5 text-[0.6875rem]', 'text-(--ui-text-tertiary)'),
      children: [
        jsx('span', { 'aria-hidden': true, children: '⚡' }),
        label === 'word' ? jsx('span', { children: 'runtime' }) : null
      ]
    })
  })
}

export default {
  id: 'hello-runtime',
  name: 'Hello Runtime',
  register(ctx) {
    const prefs = atom({ ...DEFAULTS, ...ctx.storage.get('prefs', {}) })

    const update = patch => {
      prefs.set({ ...prefs.get(), ...patch })
      ctx.storage.set('prefs', prefs.get())
    }

    ctx.register({
      id: 'chip',
      area: 'statusBar.right',
      order: 110,
      render: () => jsx(RuntimeChip, { prefs })
    })

    // Settings ▸ Plugins ▸ Hello Runtime (▸ Chip, ▸ About). Feature-detected
    // so the plugin still loads on hosts that predate the helper.
    ctx.registerSettingsPage?.({
      id: 'settings',
      title: 'Hello Runtime',
      icon: 'zap',
      render: function General() {
        const { showChip } = useValue(prefs)

        return jsx(ToggleRow, {
          checked: showChip,
          description: 'The ⚡ chip at the right end of the status bar.',
          label: 'Show status-bar chip',
          onChange: on => update({ showChip: on })
        })
      },
      children: [
        {
          id: 'chip',
          title: 'Chip',
          render: function Chip() {
            const { label } = useValue(prefs)

            return jsx(ListRow, {
              action: jsx(SegmentedControl, {
                onChange: value => update({ label: value }),
                options: [
                  { id: 'word', label: '⚡ runtime' },
                  { id: 'icon', label: '⚡' }
                ],
                value: label
              }),
              description: 'Show the word next to the ⚡, or the icon alone to save room.',
              title: 'Chip style'
            })
          }
        },
        {
          id: 'about',
          title: 'About',
          render: function About() {
            const gateway = useValue(host.state.gateway)

            return jsxs('div', {
              children: [
                jsx(ListRow, {
                  description: 'Plain ESM loaded through the runtime pipeline (blob import + SDK injection).',
                  title: 'Loaded at runtime'
                }),
                jsx(ListRow, {
                  action: jsx('span', {
                    className: 'text-[length:var(--conversation-caption-font-size)] text-(--ui-text-secondary)',
                    children: GATEWAY_STATUS[gateway] ?? String(gateway)
                  }),
                  description: 'The chip reads this live through host.state.gateway.',
                  title: 'Gateway connection'
                })
              ]
            })
          }
        }
      ]
    })
  }
}
