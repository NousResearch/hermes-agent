"use client";

import { useEffect, useRef } from "react";

export type OrbState = "idle" | "listening" | "thinking" | "speaking" | "completed" | "error";

/** State → two-stop palette. Kept theme-independent so the orb reads the same everywhere. */
const PALETTE: Record<OrbState, [string, string]> = {
  idle: ["#8f7dff", "#35d6c4"],
  listening: ["#ff5c7a", "#e879c8"],
  thinking: ["#f2b33d", "#8f7dff"],
  speaking: ["#35d6c4", "#5ba8ff"],
  completed: ["#46d68f", "#35d6c4"],
  error: ["#ff6b7a", "#f2b33d"],
};
/** How much the outline wobbles and how fast it moves, per state. */
const MOTION: Record<OrbState, { amp: number; speed: number }> = {
  idle: { amp: 0.05, speed: 0.55 },
  listening: { amp: 0.11, speed: 1.4 },
  thinking: { amp: 0.08, speed: 2.2 },
  speaking: { amp: 0.13, speed: 1.8 },
  completed: { amp: 0.06, speed: 0.9 },
  error: { amp: 0.09, speed: 2.6 },
};
export const ORB_LABEL: Record<OrbState, string> = {
  idle: "Idle", listening: "Listening", thinking: "Thinking", speaking: "Speaking", completed: "Done", error: "Needs attention",
};

const hexToRgb = (h: string): [number, number, number] => {
  const n = parseInt(h.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
};
const rgba = (c: number[], a: number) => `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${a})`;

/** Live microphone loudness (0–1) in a ref, so the orb can react without re-rendering. */
export function useMicLevel(active: boolean) {
  const level = useRef(0);
  useEffect(() => {
    if (!active || !navigator.mediaDevices?.getUserMedia) { level.current = 0; return; }
    let cancelled = false;
    let raf = 0;
    let stream: MediaStream | null = null;
    let ctx: AudioContext | null = null;
    navigator.mediaDevices.getUserMedia({ audio: true }).then((s) => {
      if (cancelled) { s.getTracks().forEach((t) => t.stop()); return; }
      stream = s;
      ctx = new AudioContext();
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 512;
      ctx.createMediaStreamSource(s).connect(analyser);
      const buf = new Uint8Array(analyser.fftSize);
      const loop = () => {
        analyser.getByteTimeDomainData(buf);
        let sum = 0;
        for (let i = 0; i < buf.length; i += 1) { const v = (buf[i] - 128) / 128; sum += v * v; }
        const rms = Math.min(1, Math.sqrt(sum / buf.length) * 4.5);
        level.current = level.current * 0.72 + rms * 0.28;
        raf = requestAnimationFrame(loop);
      };
      loop();
    }).catch(() => { level.current = 0; });
    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
      stream?.getTracks().forEach((t) => t.stop());
      ctx?.close().catch(() => undefined);
      level.current = 0;
    };
  }, [active]);
  return level;
}

export function Orb({ state, size = 44, levelRef, className }: { state: OrbState; size?: number; levelRef?: React.RefObject<number>; className?: string }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const stateRef = useRef(state);
  useEffect(() => { stateRef.current = state; }, [state]);

  useEffect(() => {
    const el = canvas.current;
    if (!el) return;
    const ctx = el.getContext("2d");
    if (!ctx) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const pad = size * 0.35; // room for glow
    const W = size + pad * 2;
    el.width = W * dpr; el.height = W * dpr;
    el.style.width = `${W}px`; el.style.height = `${W}px`;
    ctx.scale(dpr, dpr);

    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const c1 = hexToRgb(PALETTE[stateRef.current][0]);
    const c2 = hexToRgb(PALETTE[stateRef.current][1]);
    let amp = MOTION[stateRef.current].amp;
    let speed = MOTION[stateRef.current].speed;
    let t = Math.random() * 100;
    let last = performance.now();
    let ripple = 0;
    let prevState = stateRef.current;
    let raf = 0;
    const N = 72;

    const frame = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      const st = stateRef.current;
      if (st !== prevState) { if (st === "completed") ripple = 1; prevState = st; }

      // ease palette + motion toward the target state
      const [tc1, tc2] = PALETTE[st].map(hexToRgb);
      for (let i = 0; i < 3; i += 1) { c1[i] += (tc1[i] - c1[i]) * 0.08; c2[i] += (tc2[i] - c2[i]) * 0.08; }
      amp += (MOTION[st].amp - amp) * 0.06;
      speed += (MOTION[st].speed - speed) * 0.06;
      t += dt * speed * (reduce ? 0.15 : 1);

      let level = levelRef?.current ?? 0;
      if (st === "speaking") level = 0.35 + 0.3 * Math.abs(Math.sin(t * 4.2)) * Math.abs(Math.sin(t * 1.7));
      if (st === "thinking") level = 0.12 + 0.08 * Math.sin(t * 3);
      const wobble = amp + level * 0.18;

      const cx = W / 2, cy = W / 2, R = size * 0.42 * (1 + level * 0.12);
      ctx.clearRect(0, 0, W, W);

      // outer glow
      const glow = ctx.createRadialGradient(cx, cy, R * 0.6, cx, cy, R * 1.9);
      glow.addColorStop(0, rgba(c1, 0.28 + level * 0.25));
      glow.addColorStop(1, rgba(c2, 0));
      ctx.fillStyle = glow;
      ctx.beginPath(); ctx.arc(cx, cy, R * 1.9, 0, Math.PI * 2); ctx.fill();

      // blob outline from layered sines (cheap, smooth noise)
      const pts: [number, number][] = [];
      for (let i = 0; i < N; i += 1) {
        const a = (i / N) * Math.PI * 2;
        const n = Math.sin(a * 3 + t * 1.3) * 0.5 + Math.sin(a * 5 - t * 0.9) * 0.3 + Math.sin(a * 2 + t * 0.6) * 0.2;
        const r = R * (1 + n * wobble);
        pts.push([cx + Math.cos(a) * r, cy + Math.sin(a) * r]);
      }
      ctx.beginPath();
      for (let i = 0; i <= N; i += 1) {
        const p = pts[i % N], q = pts[(i + 1) % N];
        const mx = (p[0] + q[0]) / 2, my = (p[1] + q[1]) / 2;
        if (i === 0) ctx.moveTo(mx, my); else ctx.quadraticCurveTo(p[0], p[1], mx, my);
      }
      ctx.closePath();
      const fill = ctx.createRadialGradient(cx - R * 0.35, cy - R * 0.4, R * 0.1, cx, cy, R * 1.15);
      fill.addColorStop(0, rgba([255, 255, 255], 0.95));
      fill.addColorStop(0.18, rgba(c1, 1));
      fill.addColorStop(1, rgba(c2, 1));
      ctx.fillStyle = fill;
      ctx.fill();

      // inner swirl highlight
      ctx.save();
      ctx.clip();
      ctx.globalCompositeOperation = "soft-light";
      for (let k = 0; k < 2; k += 1) {
        const ang = t * (0.7 + k * 0.4) + k * 2;
        const hx = cx + Math.cos(ang) * R * 0.45, hy = cy + Math.sin(ang) * R * 0.45;
        const hg = ctx.createRadialGradient(hx, hy, 0, hx, hy, R * 0.8);
        hg.addColorStop(0, rgba(k ? c2 : c1, 0.7));
        hg.addColorStop(1, rgba(c1, 0));
        ctx.fillStyle = hg;
        ctx.fillRect(0, 0, W, W);
      }
      ctx.restore();

      // thinking: orbiting arc
      if (st === "thinking") {
        ctx.strokeStyle = rgba(c1, 0.9);
        ctx.lineWidth = Math.max(1.5, size * 0.04);
        ctx.lineCap = "round";
        ctx.beginPath();
        ctx.arc(cx, cy, R * 1.28, t * 2.4, t * 2.4 + Math.PI * 0.6);
        ctx.stroke();
      }
      // completed: one expanding ripple
      if (ripple > 0) {
        ctx.strokeStyle = rgba(c1, ripple * 0.8);
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(cx, cy, R * (1.05 + (1 - ripple) * 0.75), 0, Math.PI * 2);
        ctx.stroke();
        ripple = Math.max(0, ripple - dt * 1.1);
      }
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [size, levelRef]);

  return <canvas ref={canvas} aria-hidden className={className} style={{ margin: -size * 0.35 }} />;
}
