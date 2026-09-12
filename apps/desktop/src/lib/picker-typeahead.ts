import { type Dispatch, type RefObject, type SetStateAction, useEffect } from 'react'

const NAV_KEYS = new Set(['ArrowDown', 'ArrowUp', 'Enter', 'Escape', 'Tab', 'Home', 'End', 'PageUp', 'PageDown'])

const PICKER_SEARCH =
  '[data-slot="select-search"] input, [data-slot="dropdown-menu-search"] input'

/** A model picker overlay is open — Settings type-to-search / composer
 *  type-to-focus must stand down even if Radix left focus on a row. */
export function isPickerSearchOpen(): boolean {
  return Boolean(
    document.querySelector(
      `${PICKER_SEARCH}, [data-slot="select-content"][data-state="open"], [data-slot="dropdown-menu-content"][data-state="open"]`
    )
  )
}

/** Printable / edit keys that should filter a picker, not jump the list. */
export function isPickerFilterKey(event: KeyboardEvent | { altKey: boolean; ctrlKey: boolean; key: string; metaKey: boolean }): boolean {
  if (event.metaKey || event.ctrlKey || event.altKey) {
    return false
  }

  if (NAV_KEYS.has(event.key)) {
    return false
  }

  return event.key === 'Backspace' || event.key === 'Delete' || event.key.length === 1
}

export function applyPickerFilterKey(previous: string, key: string): string {
  if (key === 'Backspace') {
    return previous.slice(0, -1)
  }

  if (key === 'Delete') {
    return ''
  }

  return `${previous}${key}`
}

/** True when the event already landed in a real text field (leave it alone). */
export function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) {
    return false
  }

  if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement) {
    return !target.readOnly && !target.disabled
  }

  return target.isContentEditable
}

/** While a searchable picker is mounted, printable keys filter it — even
 *  when Radix parked focus on a list row and Settings/chat type-to-search
 *  would otherwise steal the keystroke. */
export function usePickerFilterCapture(
  active: boolean,
  searchRef: RefObject<HTMLInputElement | null>,
  setQuery: Dispatch<SetStateAction<string>>
): void {
  useEffect(() => {
    if (!active) {
      return
    }

    const onKeyDown = (event: KeyboardEvent) => {
      if (!isPickerFilterKey(event)) {
        return
      }

      const input = searchRef.current

      if (!input?.isConnected) {
        return
      }

      if (event.target === input) {
        return
      }

      event.preventDefault()
      event.stopPropagation()
      setQuery(previous => applyPickerFilterKey(previous, event.key))
      input.focus()
    }

    window.addEventListener('keydown', onKeyDown, true)

    return () => window.removeEventListener('keydown', onKeyDown, true)
  }, [active, searchRef, setQuery])
}
