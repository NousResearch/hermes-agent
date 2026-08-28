import { afterEach, describe, expect, it } from 'vitest'

import type { Contribution } from '@/contrib/types'
import { setTabStripDefault } from '@/store/tabstrip-prefs'

import { resolveTabStripVisible, type StripPane, tabStripVisibleForZone } from './strip-visibility'

const tile = (): StripPane => ({ collapsePane: false, placement: 'main' })
const workspace = (): StripPane => ({ collapsePane: false, placement: 'main', uncloseable: true })
const toolPanel = (): StripPane => ({ collapsePane: true, placement: 'bottom' })
const sideChrome = (): StripPane => ({ collapsePane: false, placement: 'right' })
const hideOnlyChrome = (): StripPane => ({ collapsePane: false, hideOnly: true, placement: 'left' })

describe('auto (no stored choice)', () => {
  it('gives a lone workspace no strip and a stack of two a strip', () => {
    expect(resolveTabStripVisible({ shown: [workspace()] })).toBe(false)
    expect(resolveTabStripVisible({ shown: [workspace(), sideChrome()] })).toBe(true)
  })

  it('leaves standing side chrome alone in its own zone', () => {
    expect(resolveTabStripVisible({ shown: [sideChrome()] })).toBe(false)
  })

  it('has nothing to draw for an empty zone', () => {
    expect(resolveTabStripVisible({ shown: [] })).toBe(false)
  })
})

describe('the stored choice', () => {
  it('overrides auto in both directions', () => {
    expect(resolveTabStripVisible({ mode: 'always', shown: [workspace()] })).toBe(true)
    expect(resolveTabStripVisible({ mode: 'never', shown: [workspace(), sideChrome()] })).toBe(false)
  })
})

// THE invariant the old boolean could not hold. `never` used to sit above the
// force-visible rule, so hiding a zone that held only a closeable tile left a
// surface with no tab, no ✕ and no menu — the "how do I get it back" reports.
describe('no dead zone', () => {
  it('keeps the strip for a closeable tile even when the zone says never', () => {
    expect(resolveTabStripVisible({ mode: 'never', shown: [tile()] })).toBe(true)
  })

  it('keeps the strip for a lone tool panel even when the zone says never', () => {
    expect(resolveTabStripVisible({ mode: 'never', shown: [toolPanel()] })).toBe(true)
  })

  it('keeps the strip for hide-only chrome even when the zone says never', () => {
    // Sessions / Bots: the Show/Hide rows and the chips themselves live on
    // the strip. Hiding it is the #91223 trap — nothing left to click.
    expect(resolveTabStripVisible({ mode: 'never', shown: [hideOnlyChrome()] })).toBe(true)
    expect(resolveTabStripVisible({ mode: 'never', shown: [hideOnlyChrome(), hideOnlyChrome()] })).toBe(true)
  })

  it('still hides a zone that cannot strand anything', () => {
    // The workspace is uncloseable, and a stack is reachable by tab cycling —
    // the invariant protects handles, it does not veto hiding as such.
    expect(resolveTabStripVisible({ mode: 'never', shown: [workspace()] })).toBe(false)
    expect(resolveTabStripVisible({ mode: 'never', shown: [toolPanel(), toolPanel()] })).toBe(false)
  })
})

// A full-page view is not a tab-able surface, and it lifts itself the moment
// the chat comes back — so it outranks even the stranding rule and, unlike
// `mode`, is never written to the tree.
describe('a full-page view', () => {
  it('suppresses the strip regardless of what the zone holds or says', () => {
    expect(resolveTabStripVisible({ headerVeto: true, mode: 'always', shown: [tile()] })).toBe(false)
    expect(resolveTabStripVisible({ headerVeto: true, shown: [workspace(), tile()] })).toBe(false)
  })
})

// The adapter both TreeGroup and the store call. Its job is to read the same
// chrome flags and fold in the app-wide default on both paths, so the strip on
// screen and the toggle command can never disagree.
describe('tabStripVisibleForZone', () => {
  const contributions: Record<string, Contribution> = {
    terminal: { area: 'panes', data: { placement: 'bottom' }, id: 'terminal', render: () => null, title: 'terminal' },
    'tile:a': { area: 'panes', data: { placement: 'main' }, id: 'tile:a', render: () => null, title: 'tile' },
    workspace: {
      area: 'panes',
      data: { placement: 'main', uncloseable: true },
      id: 'workspace',
      render: () => null,
      title: 'chat'
    }
  }

  const visible = (shown: string[], mode?: 'always' | 'never') =>
    tabStripVisibleForZone({
      active: shown[0],
      isCollapsePane: id => id === 'terminal',
      mode,
      paneFor: id => contributions[id],
      shown
    })

  afterEach(() => setTabStripDefault('auto'))

  it('reads placement, uncloseable and collapse off the contributions', () => {
    expect(visible(['workspace'])).toBe(false)
    expect(visible(['tile:a'], 'never')).toBe(true)
    expect(visible(['terminal'], 'never')).toBe(true)
  })

  it('falls back to the app default when the zone has no choice', () => {
    setTabStripDefault('always')
    expect(visible(['workspace'])).toBe(true)

    setTabStripDefault('never')
    expect(visible(['workspace', 'terminal'])).toBe(false)
  })

  it("lets a zone's own choice beat the app default", () => {
    setTabStripDefault('never')
    expect(visible(['workspace'], 'always')).toBe(true)

    setTabStripDefault('always')
    expect(visible(['workspace'], 'never')).toBe(false)
  })
})
