"use client";

import { useEffect, useState, type ReactNode } from "react";
import { cn } from "../utils/cn";
import { useApp } from "../lib/app";
import { IconCheck, IconFolder, IconGlobe, IconX, IconZap } from "./Icons";
import { Button, IconButton } from "./ui";

/* ====================================================================
   ONBOARDING STRIP — first-run checklist (P-E)
   Addresses the #1 gap: install friction. The doctor already verified
   runtime + model; one folder pick (plus optional channels) is all that
   stands between a fresh install and the first real prompt.
   ==================================================================== */

type StepState = "done" | "todo" | "optional" | "checking";

function StepMark({ state }: { state: StepState }) {
  return (
    <span className="flex size-[18px] shrink-0 items-center justify-center">
      {state === "done" && <IconCheck size={13} className="text-mint" />}
      {state === "todo" && <i className="size-[10px] rounded-full border-[1.5px] border-amber" />}
      {state === "optional" && <i className="size-[10px] rounded-full border-[1.5px] border-line-strong" />}
      {state === "checking" && <i className="size-[8px] animate-breathe rounded-full bg-iris" />}
    </span>
  );
}

function Step({ state, label, sub, action }: { state: StepState; label: string; sub?: string; action?: ReactNode }) {
  return (
    <div className="flex items-center gap-2">
      <StepMark state={state} />
      <div className="min-w-0">
        <p className={cn("text-[12px] leading-[1.35] font-medium", state === "checking" ? "text-ink-2" : "text-ink")}>{label}</p>
        {sub && <p className="mt-0.5 font-mono text-[9.5px] leading-[1.35] text-ink-4">{sub}</p>}
      </div>
      {action}
    </div>
  );
}

function BreathingZap({ size }: { size?: number }) {
  return <IconZap size={size} className="animate-breathe" />;
}

export function OnboardingStrip({ onDismiss }: { onDismiss: () => void }) {
  const { toast } = useApp();
  const [phase, setPhase] = useState(0); // 0 idle · 1 runtime · 2 providers · 3 workspace · 4 done
  const [run, setRun] = useState(0);
  const [wsDone, setWsDone] = useState(false);
  const running = phase > 0 && phase < 4;

  /* Fake doctor sweep: 3 checks, 400ms apart, timers cleaned up on re-run/unmount. */
  useEffect(() => {
    if (run === 0) return;
    const t1 = setTimeout(() => setPhase(2), 400);
    const t2 = setTimeout(() => setPhase(3), 800);
    const t3 = setTimeout(() => {
      setPhase(4);
      toast("Doctor: all green · 1 optional step left", "mint");
    }, 1200);
    return () => { clearTimeout(t1); clearTimeout(t2); clearTimeout(t3); };
  }, [run, toast]);

  const startDoctor = () => {
    if (running) return;
    setPhase(1);
    setRun((r) => r + 1);
  };

  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2.5 rounded-[10px] border border-iris/25 bg-iris-tint/40 px-4 py-3">
      {/* Aro mark substitute + headline */}
      <div className="flex min-w-[200px] items-center gap-2.5">
        <span className="flex size-[26px] shrink-0 items-center justify-center rounded-[8px] bg-gradient-to-br from-[#6EE7B7] to-[#10B981] text-[#07231a] shadow-glow-iris"><IconZap size={13} /></span>
        <div className="min-w-0">
          <p className="text-[13px] leading-[1.3] font-semibold text-ink">Get Aro running</p>
          <p className="text-[11.5px] leading-[1.4] text-ink-3">{wsDone ? "Workspace set — everything is ready." : "Doctor found everything it needs — one step left."}</p>
        </div>
      </div>

      <span className="hidden h-5 w-px bg-iris/20 sm:block" />

      {/* ① Runtime */}
      <Step state={phase === 1 ? "checking" : "done"} label="Runtime" sub={phase === 1 ? "checking…" : "Python 3.12 · venv ready"} />

      {/* ② Model */}
      <Step state={phase === 2 ? "checking" : "done"} label="Model" sub={phase === 2 ? "checking…" : "aro-4-70b · free tier"} />

      {/* ③ Workspace */}
      <Step
        state={wsDone ? "done" : phase === 3 ? "checking" : "todo"}
        label="Workspace"
        sub={wsDone ? "~/code/bridge" : phase === 3 ? "checking…" : undefined}
        action={!wsDone && phase !== 3 ? (
          <Button variant="secondary" size="xs" icon={IconFolder} className="ml-1" onClick={() => { setWsDone(true); toast("Workspace set — ~/code/bridge", "mint"); }}>Pick folder</Button>
        ) : undefined}
      />

      {/* ④ Channels (optional) */}
      <Step
        state="optional"
        label="Channels"
        sub="optional"
        action={<Button variant="ghost" size="xs" icon={IconGlobe} className="ml-1" onClick={() => toast("Channels live in Automations → Channels", "iris")}>Connect</Button>}
      />

      {/* Right: re-run doctor + dismiss */}
      <div className="ml-auto flex items-center gap-1.5">
        <Button variant="ghost" size="xs" icon={running ? BreathingZap : IconZap} disabled={running} onClick={startDoctor}>{running ? "Checking…" : "Run doctor"}</Button>
        <IconButton icon={IconX} label="Dismiss onboarding" size={24} onClick={onDismiss} />
      </div>
    </div>
  );
}
