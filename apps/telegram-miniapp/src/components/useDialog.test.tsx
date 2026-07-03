import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/react";
import { useDialog } from "./useDialog";
import { type RefObject } from "react";

function Dialog({ open, onClose, busy = false }: { open: boolean; onClose: () => void; busy?: boolean }) {
  const ref = useDialog(open, onClose, { busy });
  return (
    <div>
      <button type="button" data-testid="trigger">
        trigger
      </button>
      {open ? (
        <section ref={ref as RefObject<HTMLElement>} role="dialog" aria-modal="true">
          <button type="button" data-testid="first">
            first
          </button>
          <button type="button" data-testid="last">
            last
          </button>
        </section>
      ) : null}
    </div>
  );
}

describe("useDialog", () => {
  it("moves focus inside when opened", async () => {
    const { rerender } = render(<Dialog open={false} onClose={() => {}} />);
    screen.getByTestId("trigger").focus();
    await act(async () => {
      rerender(<Dialog open onClose={() => {}} />);
    });
    expect(document.activeElement).toBe(screen.getByTestId("first"));
  });

  it("closes on Escape and restores focus to the trigger", async () => {
    const onClose = vi.fn();
    // Single tree: focus the trigger, open, Escape closes, focus returns to it.
    const { rerender } = render(<Dialog open={false} onClose={onClose} />);
    const trigger = screen.getByTestId("trigger");
    trigger.focus();
    await act(async () => {
      rerender(<Dialog open onClose={onClose} />);
    });
    await act(async () => {
      fireEvent.keyDown(document, { key: "Escape" });
    });
    expect(onClose).toHaveBeenCalledTimes(1);
    await act(async () => {
      rerender(<Dialog open={false} onClose={onClose} />);
    });
    expect(trigger).toHaveFocus();
  });

  it("ignores Escape while busy", async () => {
    const onClose = vi.fn();
    render(<Dialog open onClose={onClose} busy />);
    await act(async () => {
      fireEvent.keyDown(document, { key: "Escape" });
    });
    expect(onClose).not.toHaveBeenCalled();
  });

  it("wraps Tab focus forward within the dialog", async () => {
    render(<Dialog open onClose={() => {}} />);
    screen.getByTestId("last").focus();
    await act(async () => {
      fireEvent.keyDown(document, { key: "Tab" });
    });
    expect(document.activeElement).toBe(screen.getByTestId("first"));
  });

  it("wraps Shift+Tab focus backward within the dialog", async () => {
    render(<Dialog open onClose={() => {}} />);
    screen.getByTestId("first").focus();
    await act(async () => {
      fireEvent.keyDown(document, { key: "Tab", shiftKey: true });
    });
    expect(document.activeElement).toBe(screen.getByTestId("last"));
  });
});
