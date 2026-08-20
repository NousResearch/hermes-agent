import { expect, test } from './test'

import {
  type MockBackendFixture,
  setupMockBackend,
  waitForAppReady,
} from './fixtures'

/**
 * REGRESSION: dragging the seam between the workspace and the nested right
 * section (default tree's `spl-right`) snaps the section instead of previewing
 * the width it will commit.
 *
 * Root cause: `startSash` in tree-split.tsx resolves the fixed side's resize
 * target via `edgeFixedZone` — the INNER edge zone (grp-review) — so `b0px`
 * is the review zone's width (237) while the preview applied it as the
 * flex-basis of the whole SECTION wrapper (474). Mid-drag the section
 * collapsed to the review zone's clamped size (320px = 68% of 474 — the
 * "snap towards the right at 2/3"), then jumped to yet another width (557,
 * review@maxWidth 320 + files 237) on release.
 *
 * The fix previews with the seam partners' wrapper sizes captured at
 * pointerdown AND mirrors the commit on the resize target itself (the inner
 * zone), so the drag previews the same clamped width the release commits.
 */
let fixture: MockBackendFixture | null = null

test.beforeAll(async () => {
  fixture = await setupMockBackend()
  await waitForAppReady(fixture, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('the workspace|section sash previews the width it commits (no mid-drag snap)', async () => {
  const page = fixture!.page

  // Let the layout tree render.
  await page.locator('[data-tree-group]').first().waitFor({ state: 'visible', timeout: 30_000 })

  // The review/files groups start collapsed (their panes are closed at boot).
  await page.getByRole('button', { name: /right sidebar/i }).first().click().catch(() => {})
  await page.keyboard.press('Meta+g')
  await page.getByRole('button', { name: /right sidebar/i }).first().click().catch(() => {})

  // Wait for the actual precondition (both rail panes rendered) instead of
  // fixed delays.
  await page.waitForFunction(() => {
    const visible = (id: string) => {
      const r = document.querySelector<HTMLElement>(`[data-tree-group="${id}"]`)?.getBoundingClientRect()
      return Boolean(r && r.width > 0 && r.height > 0)
    }

    return visible('grp-files') && visible('grp-review')
  }, undefined, { timeout: 30_000 })

  const drag = await page.evaluate(async () => {
    const groups = [...document.querySelectorAll<HTMLElement>('[data-tree-group]')].map((el) => {
      const r = el.getBoundingClientRect()
      return { id: el.dataset.treeGroup ?? '', left: r.left, right: r.right, top: r.top, bottom: r.bottom }
    })

    const review = groups.find((g) => g.id === 'grp-review')
    const files = groups.find((g) => g.id === 'grp-files')

    if (!review || !files) {
      return { ok: false as const, reason: `review/files missing (groups: ${groups.map((g) => g.id).join(',')})` }
    }

    const separators = [...document.querySelectorAll<HTMLElement>('[role="separator"]')]
      .map((el) => {
        const r = el.getBoundingClientRect()
        return { el, left: r.left, right: r.right, top: r.top, bottom: r.bottom, w: r.width, h: r.height }
      })
      .filter((s) => s.w > 0 && s.h > 0)

    if (separators.length === 0) {
      return { ok: false as const, reason: 'no separators' }
    }

    // The sash on the section's left edge (workspace | spl-right seam).
    const target = separators
      .map((s) => ({ ...s, score: Math.abs((s.left + s.right) / 2 - review.left) }))
      .sort((a, b) => a.score - b.score)[0]

    const sectionWidthBefore = files.right - review.left

    const x = (target.left + target.right) / 2
    const y = (target.top + target.bottom) / 2
    const pointer = {
      bubbles: true,
      cancelable: true,
      pointerId: 77,
      pointerType: 'mouse',
      isPrimary: true,
      button: 0,
      buttons: 1,
    }

    target.el.dispatchEvent(new PointerEvent('pointerdown', { ...pointer, clientX: x, clientY: y }))

    let currentX = x
    for (let index = 0; index < 20; index += 1) {
      currentX -= 6 // drag LEFT ~120px: should WIDEN the section
      window.dispatchEvent(new PointerEvent('pointermove', { ...pointer, clientX: currentX, clientY: y }))
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()))
    }

    // Measure MID-DRAG (the preview state) — the "sudden snap" moment.
    await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()))
    const reviewMid = document.querySelector('[data-tree-group="grp-review"]')?.getBoundingClientRect()
    const filesMid = document.querySelector('[data-tree-group="grp-files"]')?.getBoundingClientRect()

    window.dispatchEvent(new PointerEvent('pointerup', { ...pointer, buttons: 0, clientX: currentX, clientY: y }))

    return {
      ok: true as const,
      sectionWidthBefore: Math.round(sectionWidthBefore),
      sectionWidthMid: reviewMid && filesMid ? Math.round(filesMid.right - reviewMid.left) : -1,
      draggedBy: Math.round(x - currentX),
    }
  })

  if (!drag.ok) {
    throw new Error(drag.reason)
  }

  // Wait for the release commit to settle: React's re-render rewrites the
  // wrappers' `flex` shorthand, which must clear the preview's inline
  // flex-basis (a leftover would be a stuck-style regression in the #72845
  // family), and the review zone must be at its committed clamped width.
  await page.waitForFunction(() => {
    const review = document.querySelector<HTMLElement>('[data-tree-group="grp-review"]')?.getBoundingClientRect()
    const separators = [...document.querySelectorAll<HTMLElement>('[role="separator"]')]
      .map((el) => ({ el, left: el.getBoundingClientRect().left, w: el.getBoundingClientRect().width }))
      .filter((s) => s.w > 0)
    const sash = separators.sort((a, b) => Math.abs(a.left - (review?.left ?? 0)) - Math.abs(b.left - (review?.left ?? 0)))[0]
    const sectionWrapper = sash?.el.nextElementSibling as HTMLElement | null

    return Boolean(
      review && review.width >= 310 && // review clamped at its max (~320)
        sectionWrapper &&
        !sectionWrapper.style.flexBasis && // preview basis cleared
        !sectionWrapper.querySelector('[style*="flex-basis" i]'),
    )
  }, undefined, { timeout: 5_000 })

  const after = await page.evaluate(() => {
    const review = document.querySelector('[data-tree-group="grp-review"]')?.getBoundingClientRect()
    const files = document.querySelector('[data-tree-group="grp-files"]')?.getBoundingClientRect()
    const separators = [...document.querySelectorAll<HTMLElement>('[role="separator"]')]
      .map((el) => ({ el, left: el.getBoundingClientRect().left, w: el.getBoundingClientRect().width }))
      .filter((s) => s.w > 0)
    const sash = separators.sort((a, b) => Math.abs(a.left - (review?.left ?? 0)) - Math.abs(b.left - (review?.left ?? 0)))[0]
    const sectionWrapper = sash?.el.nextElementSibling as HTMLElement | null

    return {
      sectionWidthAfter: review && files ? Math.round(files.right - review.left) : -1,
      wrapperFlexBasis: sectionWrapper?.style.flexBasis ?? 'missing',
    }
  })

  // The drag runs into the review zone's maxWidth (320px), so the section
  // legitimately stops at ~557px — but the PREVIEW must show exactly the
  // width the release commits. The regression snapped the section to 320
  // mid-drag (the inner zone's size) and jumped to 557 on release: a 237px
  // gap between preview and commit.
  expect(Math.abs(drag.sectionWidthMid - after.sectionWidthAfter)).toBeLessThanOrEqual(5)
  expect(after.sectionWidthAfter).toBeGreaterThanOrEqual(drag.sectionWidthBefore + 80)
  // No leftover inline flex-basis on the section wrapper after release.
  expect(after.wrapperFlexBasis).toBe('')
})
