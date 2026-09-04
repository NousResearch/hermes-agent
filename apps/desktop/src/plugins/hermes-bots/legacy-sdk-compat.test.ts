/**
 * Bot Mode has to keep linking against an OLDER desktop SDK.
 *
 * `McpTab`, `ToolsetConfigPanel` and `SkillsView` are capability exports: the
 * shell that hosts the plugin may predate any of them. Every use site is
 * therefore guarded, and the plugin module graph must evaluate — and still
 * hand back a registrable plugin — when all three are missing. A bare
 * top-level use of one of them turns a missing export into a blank Bots pane
 * on an older build, which is exactly the failure this pins.
 */

import { describe, expect, it, vi } from 'vitest'

/** Names an older SDK is allowed not to export. */
const OPTIONAL_CAPABILITY_EXPORTS = new Set(['McpTab', 'SkillsView', 'ToolsetConfigPanel'])

vi.mock('@hermes/plugin-sdk', async () => {
  const { atom } = await import('nanostores')

  // Everything an older SDK DOES export answers as a callable stand-in, so
  // module-level `host.state.x.get()` / `foo()` evaluation succeeds without
  // pinning the real surface.
  const stub: unknown = new Proxy(function stubbed() {}, {
    apply: () => stub,
    get: (_target, key) => {
      if (key === Symbol.iterator) {
        return function* () {}
      }

      if (key === Symbol.toPrimitive) {
        return () => 0
      }

      // Never look thenable: an awaited stub would hang the import.
      if (key === 'then' || OPTIONAL_CAPABILITY_EXPORTS.has(String(key))) {
        return undefined
      }

      return stub
    }
  })

  return new Proxy({ atom } as Record<string, unknown>, {
    get: (target, key) => {
      if (typeof key === 'symbol' || key in target) {
        return target[key as string]
      }

      // The namespace itself is awaited by the loader: a callable `then`
      // would make it look thenable and never settle.
      return key === 'then' || OPTIONAL_CAPABILITY_EXPORTS.has(key) ? undefined : stub
    },
    // The loader validates namespace access against the mock, so the
    // capability names must READ as undefined rather than be absent —
    // absent would throw where an older bundled SDK simply gives undefined.
    has: () => true
  })
})

describe('an SDK without the optional capability exports', () => {
  it('still links Bot Mode into a registrable plugin', async () => {
    const plugin = (await import('./plugin')).default

    expect(plugin.id).toBe('hermes-bots')
    expect(typeof plugin.register).toBe('function')
  })

  it('leaves the SkillsView connection-routing capability off', async () => {
    // `skillsViewRoutesConnections` gates whether a source-scoped bot may open
    // the Capabilities tab at all — with no SkillsView it must read false, not
    // throw on the missing export.
    const { SkillsView, skillsViewRoutesConnections } = await import('./profile-config')

    expect(SkillsView).toBeUndefined()
    expect(skillsViewRoutesConnections).toBe(false)
  })
})
