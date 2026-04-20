import React, { useState } from "react";

interface Toast {
  id: number;
  tone: "ok" | "error" | "info";
  title: string;
  body?: string;
}

export interface ToastsCtrl {
  toasts: Toast[];
  push: (t: Omit<Toast, "id">) => void;
}

export function useToasts(): ToastsCtrl {
  const [toasts, setToasts] = useState<Toast[]>([]);
  return {
    toasts,
    push: (t) => {
      const id = Date.now() + Math.random();
      setToasts((arr) => [...arr, { ...t, id }]);
      setTimeout(() => setToasts((arr) => arr.filter((x) => x.id !== id)), 4000);
    },
  };
}

export function Toasts({ ctrl }: { ctrl: ToastsCtrl }) {
  return (
    <div className="fixed bottom-4 right-4 z-50 space-y-2 pointer-events-none">
      {ctrl.toasts.map((t) => (
        <div
          key={t.id}
          className={`pointer-events-auto card px-4 py-3 max-w-xs ${
            t.tone === "error" ? "border-l-4 border-l-rose-500" :
            t.tone === "ok" ? "border-l-4 border-l-emerald-500" :
            "border-l-4 border-l-sky-500"
          }`}
        >
          <div className="font-medium">{t.title}</div>
          {t.body && <div className="text-sm text-slate-600 mt-0.5">{t.body}</div>}
        </div>
      ))}
    </div>
  );
}
