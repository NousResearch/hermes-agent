import type { ReactElement, ReactNode } from 'react'
import { Children, cloneElement, isValidElement } from 'react'

import { modelSearchText } from '@/lib/model-search-text'
import { foldIncludes } from '@/lib/text'

type SelectNodeProps = {
  children?: ReactNode
  'data-slot'?: string
  textValue?: string
  value?: unknown
}

function asElement(node: ReactNode): ReactElement<SelectNodeProps> | null {
  return isValidElement<SelectNodeProps>(node) ? node : null
}

/** `<SelectItem>` always carries a string `value`. Groups and labels do not. */
export function isSelectItemElement(node: ReactNode): node is ReactElement<SelectNodeProps> {
  const element = asElement(node)

  return Boolean(element && typeof element.props.value === 'string')
}

function childText(node: ReactNode): string {
  return Children.toArray(node)
    .map(child => {
      if (typeof child === 'string' || typeof child === 'number') {
        return String(child)
      }

      const element = asElement(child)

      return element ? childText(element.props.children) : ''
    })
    .join(' ')
}

/** Haystack for one `<SelectItem>`: value, visible label, and model aliases. */
export function selectItemSearchText(element: ReactElement<SelectNodeProps>): string {
  const value = String(element.props.value ?? '')
  const label = element.props.textValue || childText(element.props.children)
  const aliases = value ? modelSearchText(value) : ''

  return [value, label, aliases].filter(Boolean).join(' ')
}

export function selectItemMatchesQuery(element: ReactElement<SelectNodeProps>, query: string): boolean {
  return foldIncludes(selectItemSearchText(element), query)
}

function hasMatchingItem(node: ReactNode, query: string): boolean {
  return Children.toArray(node).some(child => {
    const element = asElement(child)

    if (!element) {
      return false
    }

    if (isSelectItemElement(element)) {
      return selectItemMatchesQuery(element, query)
    }

    return hasMatchingItem(element.props.children, query)
  })
}

function hasItemDescendant(node: ReactNode): boolean {
  return Children.toArray(node).some(child => {
    const element = asElement(child)

    if (!element) {
      return false
    }

    return isSelectItemElement(element) || hasItemDescendant(element.props.children)
  })
}

/** Keep groups that still have a matching item; drop non-matching items. */
export function filterSelectChildren(children: ReactNode, query: string): ReactNode {
  const q = String(query || '').trim()

  if (!q) {
    return children
  }

  return Children.map(children, child => {
    const element = asElement(child)

    if (!element) {
      return child
    }

    if (isSelectItemElement(element)) {
      return selectItemMatchesQuery(element, q) ? element : null
    }

    const nested = element.props.children

    if (hasItemDescendant(nested)) {
      return hasMatchingItem(nested, q) ? cloneElement(element, undefined, filterSelectChildren(nested, q)) : null
    }

    return element
  })
}

export function countSelectItems(node: ReactNode): number {
  return Children.toArray(node).reduce<number>((total, child) => {
    const element = asElement(child)

    if (!element) {
      return total
    }

    if (isSelectItemElement(element)) {
      return total + 1
    }

    return total + countSelectItems(element.props.children)
  }, 0)
}
