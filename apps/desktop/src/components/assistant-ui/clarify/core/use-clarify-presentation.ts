// Report conservative software visibility for a live clarification card.
// A hidden or touched card can still be answered, but cannot earn presentation again.
import { type RefObject, useCallback, useEffect, useRef } from 'react'

import { isElementInHiddenPane, usePaneLifecycle, usePaneVisible } from '@/components/pane-shell/pane-visibility'
import type { ClarifyRequest } from '@/store/clarify'
import { clarifyPresentationReporter } from '@/store/server-requests'

function visibleBox(element: HTMLElement): boolean {
  const style = getComputedStyle(element)
  const rect = element.getBoundingClientRect()

  return element.isConnected && typeof element.checkVisibility === 'function'
    && element.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true })
    && style.visibility === 'visible' && style.display !== 'none' && style.opacity !== '0'
    && rect.width > 0 && rect.height > 0 && rect.top >= 0 && rect.left >= 0
    && rect.bottom <= window.innerHeight && rect.right <= window.innerWidth
}

function unobscured(element: HTMLElement): boolean {
  if (!visibleBox(element)) { return false }
  const rect = element.getBoundingClientRect()

  return [[0.1, 0.1], [0.9, 0.1], [0.5, 0.5], [0.1, 0.9], [0.9, 0.9]].every(([x, y]) => {
    const hit = document.elementFromPoint(rect.left + rect.width * x, rect.top + rect.height * y)

    // Disabled Confirm lets pointer events through; its own bar must still be the visible hit.
    const disabledSubmit = element instanceof HTMLButtonElement && element.disabled && element.type === 'submit'

    return hit !== null && (element.contains(hit) || (disabledSubmit
      && getComputedStyle(element).pointerEvents === 'none' && hit === element.parentElement))
  })
}

function answerableForm(form: HTMLFormElement): boolean {
  if (!visibleBox(form) || isElementInHiddenPane(form)) { return false }
  const controls = [...form.querySelectorAll<HTMLButtonElement | HTMLTextAreaElement>('button, textarea')]
  const questions = [...form.querySelectorAll<HTMLElement>('[data-clarify-batch-question] > div:first-child > span:first-child')]

  // Confirm is intentionally disabled until a real answer is staged; Skip and answer entry are not.
  return questions.length > 0 && controls.some(control => control.tagName === 'TEXTAREA')
    && controls.every(control => (control.type === 'submit' || !control.disabled) && unobscured(control))
    && questions.every(unobscured)
}

export function useClarifyPresentation(
  formRef: RefObject<HTMLFormElement | null>, request: ClarifyRequest | null, disabled: boolean, undelivered: boolean
): () => void {
  const requestId = request?.requestId
  const ready = !disabled && !undelivered
  const locked = Object.keys(request?.lockedAnswers ?? {}).length > 0
  const report = clarifyPresentationReporter(requestId)
  const visible = usePaneVisible()
  const lifecycle = usePaneLifecycle()
  const current = useRef({ requestId, report, touched: false, presented: false, invalidated: false })

  if (current.current.requestId !== requestId) {
    current.current = { requestId, report, touched: false, presented: false, invalidated: false }
  } else if (current.current.report !== report) {
    const prior = current.current
    current.current = { ...prior, report, presented: false,
      invalidated: prior.invalidated || prior.presented || prior.touched }
  }
  const card = current.current
  const interaction = useCallback(() => {
    card.touched = true
    card.report?.('interaction')
  }, [card])

  useEffect(() => {
    if (locked) { interaction() }
    const form = formRef.current
    if (!report || !form || typeof IntersectionObserver === 'undefined'
        || typeof document.elementFromPoint !== 'function') { return }
    let active = true
    let frame = 0
    const hide = () => {
      if (card.presented && !card.invalidated) {
        card.invalidated = true
        report('hidden')
      }
    }
    const check = () => {
      if (!active) { return }
      const shown = ready && visible && lifecycle === 'visible' && document.visibilityState === 'visible'
        && document.hasFocus() && answerableForm(form)
      if (!shown) { hide(); return }
      if (!card.touched && !card.invalidated && !card.presented) {
        card.presented = true
        report('presented')
      }
    }
    const schedule = () => {
      cancelAnimationFrame(frame)
      frame = requestAnimationFrame(() => { frame = requestAnimationFrame(check) })
    }
    const observer = new IntersectionObserver(schedule, { threshold: 1 })
    const mutations = new MutationObserver(schedule)
    observer.observe(form)
    mutations.observe(document.body, { attributes: true, childList: true, subtree: true })
    window.addEventListener('blur', hide)
    window.addEventListener('focus', schedule)
    window.addEventListener('scroll', check, true)
    window.addEventListener('resize', check)
    document.addEventListener('visibilitychange', check)
    document.addEventListener('transitionend', check, true)
    document.addEventListener('animationend', check, true)
    // Two frames allow a committed card a rendering opportunity before the final fresh check.
    frame = requestAnimationFrame(() => { frame = requestAnimationFrame(check) })

    return () => {
      active = false
      hide()
      cancelAnimationFrame(frame)
      observer.disconnect()
      mutations.disconnect()
      window.removeEventListener('blur', hide)
      window.removeEventListener('focus', schedule)
      window.removeEventListener('scroll', check, true)
      window.removeEventListener('resize', check)
      document.removeEventListener('visibilitychange', check)
      document.removeEventListener('transitionend', check, true)
      document.removeEventListener('animationend', check, true)
    }
  }, [card, formRef, interaction, lifecycle, locked, ready, report, visible])

  return interaction
}
