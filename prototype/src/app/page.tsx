"use client";

import { useSyncExternalStore } from "react";
import App from "@/workbench/App";
import { LogoMark } from "@/workbench/components/Icons";

const emptySubscribe = () => () => {};

/**
 * Aro Workbench — the attached coding-agent design system, now the main prototype.
 * Rendered client-only: the workbench persists sessions/theme/layout in
 * localStorage, so it mounts after hydration to avoid SSR mismatches.
 */
export default function Page() {
  const mounted = useSyncExternalStore(
    emptySubscribe,
    () => true,
    () => false,
  );

  if (!mounted) {
    return (
      <div className="flex h-dvh flex-col items-center justify-center gap-3 bg-void">
        <LogoMark size={42} />
        <span className="font-display text-[13px] font-semibold tracking-[0.22em] text-ink-3 uppercase">
          Aro
        </span>
        <span className="font-mono text-[10px] text-ink-4">attaching daemon…</span>
      </div>
    );
  }

  return <App />;
}
