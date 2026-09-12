/**
 * E2E fixture copied from the installed Follow-up plugin. Keep in sync.
 *
 * Follow-up — Codex-style click-to-run follow-up prompts for Hermes Desktop.
 *
 * Port of Codex++'s `codex-follow-up.js`, rebuilt on Hermes' own seams instead
 * of DOM scraping. Codex++ had to hide a raw ```json code block and inject a
 * panel after it via MutationObserver + viewport culling + an LRU parse cache;
 * Hermes exposes the transcript as a contribution area, so the model addresses
 * a real component directly and none of that machinery is needed.
 *
 * The model ends a reply with ONE paragraph:
 *
 *   ::followup{p1="Run the tests" p2="Open a PR" p3="Explain the tradeoff"}
 *
 * Rules the host enforces (see lib/transcript-directives.ts): the directive
 * must be the WHOLE paragraph, attrs are untrusted key="value" strings, the
 * brace body caps at 1024 chars, and an unclaimed directive degrades to plain
 * prose — so turning this plugin off breaks nothing.
 *
 * Click a row     → inserts the prompt into that chat's composer (you edit, then send).
 * Ctrl/Cmd+click  → sends it immediately.
 * "Insert all"    → appends every prompt as separate lines.
 *
 * Routing is surface-exact: the panel finds the chat surface it is rendered
 * inside (`[data-composer-target]`, which wraps transcript AND composer), so
 * with split tiles the prompt lands in the chat you clicked in, never another.
 */

import { cn, Codicon, haptic, Tip, usePluginI18n } from '@hermes/plugin-sdk'
import { useCallback, useEffect, useId, useRef, useState } from 'react'
import { jsx, jsxs } from 'react/jsx-runtime'

const ID = 'follow-up'

/** Hermes' internal composer bus (app/chat/composer/focus.ts). Not re-exported
 *  by the plugin SDK, so we speak its CustomEvent protocol directly. */
const INSERT_EVENT = 'hermes:composer-insert'
const SUBMIT_EVENT = 'hermes:composer-submit'

/** Inactive keep-alive tabs stay mounted under this attribute. */
const HIDDEN_PANE = '[data-pane-hidden]'

/** Core's rich editor node (app/chat/composer/rich-editor.ts). Carries the
 *  composer's disabled state, which decides whether the bus is even listening. */
const RICH_INPUT_SLOT = 'composer-rich-input'

const MAX_ITEMS = 5
const MAX_PROMPT_LEN = 400

/** How long a status line stays before clearing itself. Long enough to read,
 *  short enough that scrolling back to an old reply never shows a stale
 *  "Sent." from days ago. */
const STATUS_TTL_MS = 4000

/** Cut to at most `max` UTF-16 units without splitting a character.
 *
 *  `slice()` counts code units, so it happily cuts an emoji in half (leaving a
 *  lone surrogate that renders as a replacement box) or strips a Vietnamese
 *  tone mark off the letter it belongs to when the text arrives decomposed
 *  (NFD: "ạ" is two code units). Back off to the last boundary instead. */
function cutSafely(text, max) {
  if (text.length <= max) {
    return text
  }

  // Chromium and current Node expose Unicode grapheme segmentation. It keeps
  // complete not only surrogate pairs and NFD accents, but also joined emoji
  // such as 👨👩👧👦, skin-tone sequences and flags.
  if (typeof Intl?.Segmenter === 'function') {
    const graphemes = new Intl.Segmenter(undefined, { granularity: 'grapheme' })
    let end = 0

    for (const { index, segment } of graphemes.segment(text)) {
      const next = index + segment.length

      if (next > max) {
        break
      }

      end = next
    }

    return text.slice(0, end)
  }

  // Compatibility fallback for older runtimes: preserve the two boundaries
  // that most often corrupt user text (surrogate pairs and combining marks).
  let end = max

  if (end > 0 && /[\uD800-\uDBFF]/.test(text[end - 1])) {
    end -= 1
  }

  while (end > 0 && /\p{M}/u.test(text[end])) {
    end -= 1
  }

  return text.slice(0, end)
}

/** Prompts out of untrusted directive attrs: p1..p5, trimmed, capped, deduped.
 *
 *  Each item carries `truncated` so the row can say so: a prompt silently cut
 *  mid-word lands in the composer looking finished, and the user only finds
 *  out after sending it. */
function promptsFromAttrs(attrs) {
  const seen = new Set()
  const items = []

  for (let i = 1; i <= MAX_ITEMS; i += 1) {
    const raw = attrs?.[`p${i}`]

    if (typeof raw !== 'string') {
      continue
    }

    const full = raw.trim()
    const prompt = cutSafely(full, MAX_PROMPT_LEN)

    if (!prompt) {
      continue
    }

    // Dedupe on the full normalized value so distinct long prompts that share
    // the same preview remain separate actions.
    const key = full.normalize('NFC')

    if (seen.has(key)) {
      continue
    }

    seen.add(key)
    items.push({
      displayText: prompt,
      fullText: full,
      truncated: full.length > prompt.length
    })
  }

  return items
}

/** The chat surface this panel lives in — the element carrying both the
 *  composer routing key and the mounted-surface identity. Returns null when
 *  the surface is buried in an inactive tab (submit must fail closed). */
function surfaceFor(node) {
  const surface = node?.closest?.('[data-composer-target]')

  if (!surface || surface.closest(HIDDEN_PANE)) {
    return null
  }

  const target = surface.dataset.composerTarget
  const surfaceId = surface.dataset.composerSurfaceId

  return target ? { el: surface, surfaceId: surfaceId || null, target } : null
}

/**
 * True when that surface's composer is accepting input.
 *
 * This is NOT cosmetic. While the composer is disabled (gateway closed) core
 * tears down its insert subscription entirely (`use-composer-draft.ts` returns
 * early on `inputDisabled`) and gates submit on `!inputDisabled`
 * (`use-composer-submit.ts`). The event would be dispatched into the void, so
 * without this check the panel reports "Sent." for a prompt that never left.
 * Core marks the state on the editor node itself.
 */
function composerAccepting(surface) {
  const editor = surface?.querySelector?.(`[data-slot="${RICH_INPUT_SLOT}"]`)

  // No editor found: don't invent a failure — let the dispatch proceed.
  if (!editor) {
    return true
  }

  // Read the ATTRIBUTES React actually renders (`aria-disabled={... : undefined}`
  // and `contentEditable={!inputDisabled}`) rather than the `isContentEditable`
  // property — the property is a Chromium-only convenience that jsdom doesn't
  // implement, so testing it would be untestable and silently wrong off-browser.
  return editor.getAttribute('aria-disabled') !== 'true' && editor.getAttribute('contenteditable') !== 'false'
}

function insertPrompt(node, text) {
  const surface = surfaceFor(node)

  if (!surface || !composerAccepting(surface.el)) {
    return false
  }

  // Deferred to a macrotask, exactly like core's `requestComposerInsert`: the
  // synchronous click handler must finish first, or it steals focus back from
  // the composer effect. Submit, by contrast, dispatches NOW (see below).
  window.setTimeout(() => {
    window.dispatchEvent(new CustomEvent(INSERT_EVENT, { detail: { mode: 'block', target: surface.target, text } }))
  }, 0)

  return true
}

function submitPrompt(node, text) {
  const surface = surfaceFor(node)

  // Fail closed exactly like core: no visible surface identity, no broadcast.
  if (!surface?.surfaceId || !composerAccepting(surface.el)) {
    return false
  }

  // Synchronous on purpose. Core's `dispatchNow` comment: deferring a submit
  // lets a tab reveal switch the visible pane before subscribers run, so the
  // task gets dropped or claimed by the wrong composer.
  window.dispatchEvent(
    new CustomEvent(SUBMIT_EVENT, {
      detail: { surfaceId: surface.surfaceId, target: surface.target, text }
    })
  )

  return true
}

function FollowUpPanel({ attrs, streaming }) {
  const t = usePluginI18n(ID)
  const rootRef = useRef(null)
  const titleId = useId()
  const actionDescriptionId = useId()
  const [status, setStatus] = useState('')
  const statusTimer = useRef(0)

  const items = promptsFromAttrs(attrs)

  // Announce through a keyed live region. Re-running the SAME action must
  // re-announce, but an identical string in an unchanged region is silent to
  // screen readers — so bump a nonce and key the node on it.
  const announce = useCallback(message => {
    setStatus(prev => ({ message, nonce: (prev?.nonce ?? 0) + 1 }))

    window.clearTimeout(statusTimer.current)
    statusTimer.current = window.setTimeout(() => setStatus(''), STATUS_TTL_MS)
  }, [])

  const run = useCallback(
    (text, send) => {
      const node = rootRef.current
      const ok = send ? submitPrompt(node, text) : insertPrompt(node, text)

      if (ok) {
        haptic('tap')
      }
      announce(ok ? (send ? t('sent') : t('inserted')) : t('noComposer'))
    },
    [announce, t]
  )

  // Alt+1..5 runs the Nth prompt; add Ctrl/Cmd to send it outright. Bound on
  // the panel's own root, not the window: several replies can carry a panel at
  // once, and a window-level shortcut would fire every one of them. Focus must
  // be inside this panel, which Tab reaches natively.
  const onKeyDown = useCallback(
    event => {
      if (!event.altKey || event.isComposing || event.getModifierState?.('AltGraph')) {
        return
      }

      const codeMatch = /^Digit([1-5])$/.exec(event.code)
      const keyMatch = /^[1-5]$/.exec(event.key)
      const index = Number(codeMatch?.[1] ?? keyMatch?.[0]) - 1

      if (!Number.isInteger(index) || index < 0 || index >= items.length) {
        return
      }

      event.preventDefault()
      const item = items[index]
      run(item.fullText, item.truncated ? false : event.ctrlKey || event.metaKey)
    },
    [items, run]
  )

  // Clear a pending timer if the panel unmounts mid-countdown.
  useEffect(() => () => window.clearTimeout(statusTimer.current), [])

  // A directive mid-stream is still being typed — render nothing rather than
  // flashing a half-parsed panel.
  if (streaming) {
    return null
  }

  // A panel that renders nothing is indistinguishable from a directive that
  // was never emitted: the user sees an answer with no follow-ups and cannot
  // tell the difference. Mirror the core drop badge instead of vanishing, so
  // a bad `::followup` is visible rather than silent.
  if (!items.length) {
    return jsx(Tip, {
      label: t('emptyDirectiveTip'),
      side: 'top',
      children: jsxs('span', {
        className: cn(
          'inline-flex select-none items-center gap-1.5 rounded-md border border-dashed',
          'border-border/60 px-2 py-0.5 text-xs text-muted-foreground'
        ),
        'data-testid': 'directive-drop-badge',
        role: 'note',
        'aria-label': `${t('emptyDirective')}. ${t('emptyDirectiveTip')}`,
        title: t('emptyDirectiveTip'),
        children: [jsx(Codicon, { name: 'debug-disconnect', size: '0.85em' }), t('emptyDirective')]
      })
    })
  }

  return jsxs('section', {
    ref: rootRef,
    onKeyDown,
    role: 'group',
    'aria-labelledby': titleId,
    'data-testid': 'follow-up-panel',
    className: cn(
      'my-2 flex w-full flex-col gap-1.5 rounded-md border p-2',
      'border-(--ui-stroke-secondary) transition-colors duration-150 ease-out',
      'hover:border-(--ui-stroke-primary)',
      'motion-reduce:transition-none'
    ),
    children: [
      jsxs('div', {
        className: 'flex min-w-0 flex-wrap items-center gap-1.5 text-xs text-(--ui-text-tertiary)',
        children: [
          jsx(Codicon, { name: 'debug-step-over' }),
          jsx('span', { id: titleId, className: 'font-medium text-(--ui-text-secondary)', children: t('title') }),
          jsx('span', {
            className: 'text-[0.6875rem] tabular-nums text-(--ui-text-quaternary)',
            'aria-hidden': true,
            children: items.length
          }),
          items.length > 1 &&
            jsx(Tip, {
              label: t('insertAllTip'),
              children: jsxs('button', {
                type: 'button',
                className: cn(
                  'ml-auto inline-flex min-h-8 items-center gap-1 rounded px-2 py-1 transition-colors',
                  'text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground',
                  'focus-visible:outline-none focus-visible:ring-2 ring-(--ui-accent)',
                  'motion-reduce:transition-none'
                ),
                onClick: () => run(items.map(item => item.fullText).join('\n'), false),
                children: [jsx(Codicon, { name: 'insert', size: '0.85em' }), t('insertAll')]
              })
            })
        ]
      }),

      jsx('span', {
        id: actionDescriptionId,
        className: 'sr-only',
        children: t('actionDescription')
      }),
      jsx('div', {
        role: 'list',
        className: 'flex flex-col gap-0.5',
        children: items.map((item, index) =>
          jsx(
            Tip,
            {
              label: item.truncated ? t('rowTipTruncated') : t('rowTip'),
              children: jsxs('button', {
                type: 'button',
                role: 'listitem',
                'aria-describedby': actionDescriptionId,
                // Alt+N is invisible to a screen reader unless the control
                // says so; the number badge alone is decoration.
                'aria-keyshortcuts': index < 9 ? `Alt+${index + 1}` : undefined,
                className: cn(
                  'group flex w-full min-h-10 items-start gap-2 rounded px-2 py-2 text-left text-sm',
                  'leading-snug transition-colors duration-150 ease-out',
                  'hover:bg-(--chrome-action-hover)',
                  'focus-visible:outline-none focus-visible:ring-2 ring-(--ui-accent)',
                  'motion-reduce:transition-none'
                ),
                onClick: event => run(item.fullText, item.truncated ? false : event.ctrlKey || event.metaKey),
                children: [
                  jsx('span', {
                    className: cn(
                      'mt-px inline-flex h-4 w-4 shrink-0 items-center justify-center rounded-sm',
                      'text-[0.625rem] font-semibold tabular-nums',
                      // The index is wayfinding, not content: quiet until the
                      // row is the one under the cursor.
                      'text-(--ui-text-quaternary) transition-colors duration-150',
                      'group-hover:text-foreground',
                      'motion-reduce:transition-none'
                    ),
                    children: String(index + 1)
                  }),
                  jsxs('span', {
                    className: 'min-w-0 flex-1 break-words',
                    children: [
                      item.displayText,
                      item.truncated &&
                        jsx('span', {
                          className: cn(
                            'ml-1.5 rounded px-1 align-middle text-[0.625rem]',
                            'bg-(--chrome-action-hover) text-(--ui-text-quaternary)'
                          ),
                          // The ellipsis is decorative; the label carries it.
                          title: t('truncatedTip'),
                          children: t('truncated')
                        })
                    ]
                  }),
                  // Affordance revealed on hover: says what a click will do
                  // without adding permanent clutter to every row.
                  jsx(Codicon, {
                    name: 'arrow-right',
                    size: '0.85em',
                    className: cn(
                      'mt-0.5 shrink-0 opacity-40 transition-all duration-150',
                      'text-(--ui-accent)',
                      'group-hover:translate-x-0.5 group-hover:opacity-100',
                      'motion-reduce:transition-none'
                    )
                  }),
                  jsx('span', {
                    className: cn(
                      'shrink-0 text-xs text-(--ui-text-quaternary) opacity-0 transition-opacity',
                      'group-hover:opacity-100 motion-reduce:transition-none'
                    ),
                    'aria-hidden': true,
                    children: t('insertAction')
                  })
                ]
              })
            },
            item.fullText
          )
        )
      }),

      // Always mounted so assistive tech has a region to observe; keyed by a
      // nonce so repeating the same action re-announces it.
      jsx('div', {
        className: 'px-2 text-[0.6875rem] text-(--ui-text-quaternary)',
        role: 'status',
        'aria-live': 'polite',
        children: status ? jsx('span', { children: status.message }, String(status.nonce)) : null
      })
    ]
  })
}

export default {
  id: ID,
  name: 'Follow-up',
  // This plugin was explicitly installed for the user's Codex-style workflow.
  // Keep it active when no persisted choice exists; Settings → Plugins can
  // still override this with an explicit false decision.
  defaultEnabled: true,
  register(ctx) {
    ctx.i18n.register({
      en: {
        title: 'Follow-up',
        rowTip: 'Click to insert · Ctrl/Cmd+click to send · Alt+N',
        rowTipTruncated: 'Shortened to fit · click to insert the full prompt for review',
        actionDescription:
          'Click to insert. Ctrl or Command plus click sends immediately. Shortened prompts are always inserted in full for review.',
        truncatedDescription: 'This prompt is shortened in the preview and will be inserted in full for review.',
        insertAction: 'Insert',
        truncated: 'shortened',
        truncatedTip: 'This prompt was longer than 400 characters and was cut short.',
        insertAll: 'Insert all',
        insertAllTip: 'Append every prompt to the composer',
        inserted: 'Inserted into the composer.',
        sent: 'Sent.',
        noComposer: 'No visible composer in this chat.',
        emptyDirective: 'Follow-ups skipped: no prompts given',
        emptyDirectiveTip: 'The follow-up directive carried no usable prompt (expected p1="…").'
      },
      vi: {
        title: 'Gợi ý tiếp theo',
        rowTip: 'Nhấn để chèn · Ctrl/Cmd+nhấn để gửi ngay · Alt+N',
        rowTipTruncated: 'Đã rút gọn · nhấn để chèn toàn bộ nội dung để xem lại',
        actionDescription:
          'Nhấn để chèn. Ctrl hoặc Command và nhấn để gửi ngay. Gợi ý đã rút gọn luôn được chèn đầy đủ để xem lại.',
        truncatedDescription: 'Gợi ý này đã được rút gọn trong phần xem trước và sẽ được chèn đầy đủ để xem lại.',
        insertAction: 'Chèn',
        truncated: 'đã rút gọn',
        truncatedTip: 'Gợi ý này dài hơn 400 ký tự nên đã bị cắt bớt.',
        insertAll: 'Chèn tất cả',
        insertAllTip: 'Thêm mọi gợi ý vào khung soạn',
        inserted: 'Đã chèn vào khung soạn.',
        sent: 'Đã gửi.',
        noComposer: 'Không thấy khung soạn đang hiển thị.',
        emptyDirective: 'Đã bỏ qua gợi ý: không có nội dung',
        emptyDirectiveTip: 'Directive gợi ý không kèm nội dung dùng được (cần p1="…").'
      }
    })

    ctx.register({
      id: 'directive',
      area: 'transcript.directives',
      data: {
        name: 'followup',
        render: ({ attrs, streaming }) => jsx(FollowUpPanel, { attrs, streaming })
      }
    })
  }
}
