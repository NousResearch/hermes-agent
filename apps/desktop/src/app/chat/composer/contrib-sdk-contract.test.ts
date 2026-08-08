import type { HermesPlugin, PluginContribution } from '@hermes/plugin-sdk'
import { describe, expect, it } from 'vitest'

import { COMPOSER_AREAS } from './contrib'
import type { ComposerRenderContext } from './contrib'

const typedContribution: PluginContribution = {
  id: 'bold',
  area: COMPOSER_AREAS.actions,
  render: (bridge: ComposerRenderContext) => {
    bridge.insertText('**x**')

    return null
  }
}

const typedPlugin: HermesPlugin = {
  id: 'composer-types',
  register(ctx) {
    ctx.register({
      id: 'inline',
      area: COMPOSER_AREAS.actions,
      render: bridge => {
        bridge.insertText('__x__')

        return null
      }
    })
    ctx.registerMany([
      {
        id: 'batched',
        area: COMPOSER_AREAS.actions,
        render: bridge => {
          bridge.insertText('~~x~~')

          return null
        }
      }
    ])
  }
}

// @ts-expect-error Non-composer areas invoke render with no arguments.
const invalidNonComposerContribution: PluginContribution = {
  id: 'invalid-context',
  area: 'statusBar.right',
  render: (bridge: ComposerRenderContext) => {
    bridge.insertText('must not type-check')

    return null
  }
}

void typedPlugin
void invalidNonComposerContribution

describe('composer SDK render contract', () => {
  it('accepts typed composer callbacks at the public plugin seam', () => {
    expect(typedContribution.area).toBe(COMPOSER_AREAS.actions)
  })
})
