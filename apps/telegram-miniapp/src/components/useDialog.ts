import { useEffect, useRef } from "react";

const FOCUSABLE =
  'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])';

/**
 * Accessible modal-dialog wiring for a container ref: on open it moves focus
 * inside, traps Tab within the dialog, closes on Escape, and restores focus to
 * the element that had it before opening. `busy` freezes Escape/close while an
 * action is in flight (matches the backdrop being non-dismissable then).
 */
export function useDialog(open: boolean, onClose: () => void, { busy = false }: { busy?: boolean } = {}) {
  const ref = useRef<HTMLElement | null>(null);
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;
  const busyRef = useRef(busy);
  busyRef.current = busy;

  useEffect(() => {
    if (!open) return undefined;
    const node = ref.current;
    if (!node) return undefined;

    const previouslyFocused = document.activeElement as HTMLElement | null;

    // The selector already drops disabled controls; the dialogs contain no
    // hidden-but-enabled focusables, so we avoid layout-based visibility checks
    // (offsetParent), which are unavailable under jsdom and would break tests.
    const focusables = () => Array.from(node.querySelectorAll<HTMLElement>(FOCUSABLE));
    // Move focus inside: first focusable, else the dialog container itself.
    const first = focusables()[0];
    if (first) first.focus();
    else {
      node.setAttribute("tabindex", "-1");
      node.focus();
    }

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        if (busyRef.current) return;
        event.preventDefault();
        onCloseRef.current();
        return;
      }
      if (event.key !== "Tab") return;
      const items = focusables();
      if (items.length === 0) {
        event.preventDefault();
        return;
      }
      const firstItem = items[0];
      const lastItem = items[items.length - 1];
      const active = document.activeElement;
      if (event.shiftKey && (active === firstItem || active === node)) {
        event.preventDefault();
        lastItem.focus();
      } else if (!event.shiftKey && active === lastItem) {
        event.preventDefault();
        firstItem.focus();
      }
    };

    document.addEventListener("keydown", onKeyDown, true);
    return () => {
      document.removeEventListener("keydown", onKeyDown, true);
      // Restore focus to the trigger so keyboard/switch users are not stranded.
      if (previouslyFocused && typeof previouslyFocused.focus === "function") previouslyFocused.focus();
    };
  }, [open]);

  return ref;
}
