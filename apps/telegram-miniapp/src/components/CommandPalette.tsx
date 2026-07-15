import { type RefObject } from "react";
import { quickActions, type NavKey, type QuickAction } from "../mockData";
import { useDialog } from "./useDialog";

export function CommandPalette({ isOpen, onClose, onNavigate }: { isOpen: boolean; onClose: () => void; onNavigate: (tab: NavKey) => void }) {
  // Hook must run unconditionally; it no-ops while closed and traps focus /
  // closes on Escape / restores focus to the trigger while open.
  const dialogRef = useDialog(isOpen, onClose);
  if (!isOpen) return null;

  const routeMap: Partial<Record<QuickAction["id"], NavKey>> = {
    status: "status",
    sessions: "sessions",
    approvals: "approvals",
    logs: "logs",
  };

  return (
    <div className="sheet-backdrop" role="presentation" onClick={onClose}>
      <section
        ref={dialogRef as RefObject<HTMLElement>}
        className="command-sheet glass-card"
        aria-label="Палитра команд"
        role="dialog"
        aria-modal="true"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="sheet-handle" />
        <div className="section-heading compact">
          <div>
            <p className="mono-label">ПАЛИТРА</p>
            <h2>Быстрый переход</h2>
          </div>
          <button className="ghost-button" type="button" onClick={onClose}>Закрыть</button>
        </div>
        <div className="palette-list">
          {quickActions.map((action) => {
            const target = routeMap[action.id];
            return (
              <button
                className="palette-row tap"
                disabled={!target || action.risk === "critical"}
                key={action.id}
                type="button"
                onClick={() => {
                  if (!target) return;
                  onNavigate(target);
                  onClose();
                }}
              >
                <span>
                  <strong>{action.label}</strong>
                  <small>{action.description}</small>
                </span>
                <em>{action.hint}</em>
              </button>
            );
          })}
        </div>
      </section>
    </div>
  );
}
