"use client";

import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import { PROVIDER_SEEDS, type ModelProvider } from "../data/providers";

/* ---------------- themes ---------------- */
export type ThemeId = "obsidian" | "daylight" | "nord" | "ember" | "paper";
export const THEMES: { id: ThemeId; label: string; desc: string; swatch: string[] }[] = [
  { id: "obsidian", label: "Obsidian", desc: "Deep neutral dark · violet accent", swatch: ["#0c0e14", "#12151d", "#7c6bff", "#35d6c4"] },
  { id: "daylight", label: "Daylight", desc: "Clean light · high legibility", swatch: ["#ffffff", "#f4f5f8", "#5b4ce6", "#0f9f90"] },
  { id: "nord", label: "Nord", desc: "Arctic blue-grey · frost accent", swatch: ["#2a303b", "#313845", "#88c0d0", "#a3be8c"] },
  { id: "ember", label: "Ember", desc: "Warm charcoal · copper accent", swatch: ["#171210", "#1f1815", "#ff8a4c", "#4fd1b5"] },
  { id: "paper", label: "Paper", desc: "Warm light · ink on paper", swatch: ["#faf7f0", "#f4f0e6", "#8a5cf6", "#0f8f86"] },
];

/* ---------------- product mode ---------------- */
export type ProductMode = "code" | "agent";

/* ---------------- todos ---------------- */
export type Todo = { id: string; text: string; state: "todo" | "doing" | "done"; by?: "agent" | "you" };

type Ctx = {
  theme: ThemeId;
  setTheme: (t: ThemeId) => void;
  product: ProductMode;
  setProduct: (p: ProductMode) => void;
  todos: Todo[];
  setTodos: React.Dispatch<React.SetStateAction<Todo[]>>;
  todoOpen: boolean;
  setTodoOpen: (b: boolean) => void;
  toast: (msg: string, tone?: "iris" | "mint" | "amber" | "rose") => void;
  toasts: { id: number; msg: string; tone: string }[];
  termOpen: boolean;
  setTermOpen: (b: boolean) => void;
  termFeed: TermLine[];
  termPush: (lines: TermLine[]) => void;
  dockOpen: boolean;
  setDockOpen: (b: boolean) => void;
  settingsSection: string;
  setSettingsSection: (s: string) => void;
  providers: ModelProvider[];
  setProviders: React.Dispatch<React.SetStateAction<ModelProvider[]>>;
};
export type TermLine = { t: string; k?: "cmd" | "ok" | "err" | "warn" | "dim" | "info" };

const AppCtx = createContext<Ctx>(null!);
export const useApp = () => useContext(AppCtx);

const SEED_TODOS: Todo[] = [
  { id: "t1", text: "Add SessionChannel with resumable cursor", state: "done", by: "agent" },
  { id: "t2", text: "Capability handshake in attachEditor()", state: "done", by: "agent" },
  { id: "t3", text: "Route editor edits through DiffReviewer", state: "doing", by: "agent" },
  { id: "t4", text: "Two-editor integration test", state: "todo", by: "agent" },
  { id: "t5", text: "Review PR #482 before standup", state: "todo", by: "you" },
];

export function AppProvider({ children }: { children: React.ReactNode }) {
  const [theme, setThemeState] = useState<ThemeId>(
    () => (localStorage.getItem("aro.theme") as ThemeId) || "obsidian",
  );
  const [product, setProduct] = useState<ProductMode>("code");
  const [todos, setTodos] = useState<Todo[]>(SEED_TODOS);
  const [todoOpen, setTodoOpen] = useState(false);
  const [toasts, setToasts] = useState<{ id: number; msg: string; tone: string }[]>([]);

  const [termOpen, setTermOpen] = useState(true);
  const [dockOpen, setDockOpen] = useState(true);
  const [settingsSection, setSettingsSection] = useState("general");
  const [providers, setProviders] = useState<ModelProvider[]>(() => {
    try {
      const saved = JSON.parse(localStorage.getItem("aro.providers") || "null") as ModelProvider[] | null;
      if (!saved) return PROVIDER_SEEDS;
      return PROVIDER_SEEDS.map((p) => ({ ...p, ...(saved.find((x) => x.id === p.id) ?? {}) }));
    } catch { return PROVIDER_SEEDS; }
  });
  const [termFeed, setTermFeed] = useState<TermLine[]>([
    { t: "aro agent shell · commands run by the agent stream here", k: "dim" },
    { t: "$ pnpm --filter @aro/bridge test", k: "cmd" },
    { t: " ✓ session-channel.test.ts (14 tests) 284ms", k: "ok" },
    { t: " ✓ attach-editor.test.ts (9 tests) 121ms", k: "ok" },
    { t: " • reconnect.test.ts (12 tests) running…", k: "info" },
  ]);
  const termPush = useCallback((lines: TermLine[]) => setTermFeed((f) => [...f, ...lines].slice(-400)), []);
  const setTheme = useCallback((t: ThemeId) => {
    setThemeState(t);
    localStorage.setItem("aro.theme", t);
  }, []);
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);
  useEffect(() => {
    localStorage.setItem("aro.providers", JSON.stringify(providers.map(({ id, baseUrl, status, models }) => ({ id, baseUrl, status, models }))));
  }, [providers]);

  const toast = useCallback((msg: string, tone = "iris") => {
    const id = Date.now() + Math.random();
    setToasts((t) => [...t, { id, msg, tone }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 2600);
  }, []);

  return (
    <AppCtx.Provider value={{ theme, setTheme, product, setProduct, todos, setTodos, todoOpen, setTodoOpen, toast, toasts, termOpen, setTermOpen, termFeed, termPush, dockOpen, setDockOpen, settingsSection, setSettingsSection, providers, setProviders }}>
      {children}
    </AppCtx.Provider>
  );
}

/* ---------------- vertical resize (bottom panels) ---------------- */
export function useResizableY(initial: number, min: number, max: number) {
  const [size, setSize] = useState(initial);
  const [active, setActive] = useState(false);
  const onMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    const y0 = e.clientY, s0 = size;
    setActive(true);
    document.body.style.cursor = "row-resize";
    document.body.style.userSelect = "none";
    const move = (ev: MouseEvent) => setSize(Math.max(min, Math.min(max, s0 - (ev.clientY - y0))));
    const up = () => {
      setActive(false);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  }, [size, min, max]);
  return { size, setSize, active, handle: { onMouseDown, "data-active": active, className: "resizer-y" } as const };
}

/* ---------------- resizable panel hook ---------------- */
export function useResizable(initial: number, min: number, max: number, dir: "left" | "right" = "right") {
  const [size, setSize] = useState(initial);
  const [active, setActive] = useState(false);
  const start = useRef<{ x: number; s: number } | null>(null);

  const onMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      start.current = { x: e.clientX, s: size };
      setActive(true);
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
      const move = (ev: MouseEvent) => {
        if (!start.current) return;
        const d = (ev.clientX - start.current.x) * (dir === "right" ? 1 : -1);
        setSize(Math.max(min, Math.min(max, start.current.s + d)));
      };
      const up = () => {
        start.current = null;
        setActive(false);
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
        window.removeEventListener("mousemove", move);
        window.removeEventListener("mouseup", up);
      };
      window.addEventListener("mousemove", move);
      window.addEventListener("mouseup", up);
    },
    [size, min, max, dir],
  );

  return { size, setSize, active, handle: { onMouseDown, "data-active": active, className: "resizer" } as const };
}
