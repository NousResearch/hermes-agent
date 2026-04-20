// Canvas2D office scene.
//
// The world coordinate system is a fixed 1600×900 box. We render at the
// canvas's CSS pixel size and scale via a single transform so the layout
// is stable across window sizes. Each "sprite" is a digital employee; their
// resting positions are derived from the zone they should be in (driven by
// their `activity` field). Smoothing/easing is done with a stiff spring per
// axis (no library — a few lines of simulation per frame).
//
// Performance budget: 60 FPS for ≤200 sprites on a 2020-era laptop. The
// sprite rendering uses simple `arc()` + emoji text to avoid bitmap loading
// at first run. A future iteration can swap in PNG sprite sheets.

import React, { useEffect, useMemo, useRef, useState } from "react";
import { useStore } from "../state";
import type { Employee, Activity, Zone } from "../types";
import { ACTIVITY_TO_ZONE } from "../types";
import { t } from "../i18n";

interface Props {
  onPickEmployee: (id: string) => void;
}

const W = 1600;
const H = 900;

interface ZoneLayout {
  id: Zone;
  rect: { x: number; y: number; w: number; h: number };
  fill: string;
  label: string;
  emoji: string;
}

function zones(): ZoneLayout[] {
  // 4 quadrants, padded margins, ample paths.
  const M = 60;
  const halfW = (W - 3 * M) / 2;
  const halfH = (H - 3 * M) / 2;
  return [
    {
      id: "work",
      rect: { x: M, y: M, w: halfW, h: halfH },
      fill: "#bbf7d0",
      label: "work",
      emoji: "💻",
    },
    {
      id: "talk",
      rect: { x: 2 * M + halfW, y: M, w: halfW, h: halfH },
      fill: "#fbcfe8",
      label: "talk",
      emoji: "💬",
    },
    {
      id: "learn",
      rect: { x: M, y: 2 * M + halfH, w: halfW, h: halfH },
      fill: "#bae6fd",
      label: "learn",
      emoji: "📚",
    },
    {
      id: "rest",
      rect: { x: 2 * M + halfW, y: 2 * M + halfH, w: halfW, h: halfH },
      fill: "#fde68a",
      label: "rest",
      emoji: "☕",
    },
  ];
}

interface Sprite {
  id: string;
  emp: Employee;
  x: number;
  y: number;
  vx: number;
  vy: number;
  tx: number;
  ty: number;
  hue: number;
  emoji: string;
  zone: Zone;
}

// deterministic per-id position inside a zone
function targetInZone(empId: string, z: ZoneLayout, idx: number, n: number): { x: number; y: number } {
  // pack employees in the zone in a soft grid.
  const cols = Math.max(2, Math.ceil(Math.sqrt(n || 1)));
  const cell = Math.min(z.rect.w / (cols + 1), z.rect.h / (cols + 1));
  const ix = idx % cols;
  const iy = Math.floor(idx / cols);
  // jitter from id hash so revisits stay stable
  let h = 0;
  for (let i = 0; i < empId.length; i++) h = (h * 31 + empId.charCodeAt(i)) >>> 0;
  const jx = ((h & 0xff) / 255 - 0.5) * cell * 0.4;
  const jy = (((h >> 8) & 0xff) / 255 - 0.5) * cell * 0.4;
  return {
    x: z.rect.x + cell * (ix + 1) + jx,
    y: z.rect.y + cell * (iy + 1) + jy,
  };
}

function emojiForSprite(sprite_id: string): string {
  const map: Record<string, string> = {
    "robot-1": "🤖",
    "robot-2": "🤖",
    cat: "🐱",
    fox: "🦊",
    panda: "🐼",
    wizard: "🧙",
    scientist: "🔬",
    writer: "✍️",
    designer: "🎨",
    analyst: "📊",
    translator: "🌐",
    tutor: "🎓",
  };
  return map[sprite_id] ?? "🤖";
}

export function OfficeCanvas({ onPickEmployee }: Props) {
  const employees = useStore((s) => s.employees);
  const departments = useStore((s) => s.departments);
  const selectedDeptId = useStore((s) => s.selectedDeptId);
  const activityByEmp = useStore((s) => s.activityByEmp);

  const visible = useMemo(
    () => (selectedDeptId == null ? employees : employees.filter((e) => e.department_id === selectedDeptId)),
    [employees, selectedDeptId],
  );

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const spritesRef = useRef<Sprite[]>([]);
  const popRef = useRef<Map<string, { text: string; until: number }>>(new Map());
  const hoverRef = useRef<{ id: string; x: number; y: number } | null>(null);
  const rafRef = useRef<number | null>(null);

  const zoneLayout = useMemo(() => zones(), []);

  // Sync sprites with employees roster
  useEffect(() => {
    const next: Sprite[] = [];
    const byZone: Record<Zone, Employee[]> = { rest: [], learn: [], talk: [], work: [] };
    for (const e of visible) byZone[ACTIVITY_TO_ZONE[e.activity as Activity]].push(e);

    for (const z of zoneLayout) {
      const list = byZone[z.id];
      list.forEach((emp, idx) => {
        const tgt = targetInZone(emp.id, z, idx, list.length);
        const existing = spritesRef.current.find((s) => s.id === emp.id);
        if (existing) {
          existing.tx = tgt.x;
          existing.ty = tgt.y;
          existing.zone = z.id;
          existing.emp = emp;
          existing.hue = emp.avatar.hue;
          existing.emoji = emojiForSprite(emp.avatar.sprite_id);
          next.push(existing);
        } else {
          // new arrival: spawn at the center of "rest"
          const restZ = zoneLayout.find((zz) => zz.id === "rest")!;
          const sx = restZ.rect.x + restZ.rect.w / 2;
          const sy = restZ.rect.y + restZ.rect.h / 2;
          next.push({
            id: emp.id,
            emp,
            x: sx,
            y: sy,
            vx: 0,
            vy: 0,
            tx: tgt.x,
            ty: tgt.y,
            hue: emp.avatar.hue,
            emoji: emojiForSprite(emp.avatar.sprite_id),
            zone: z.id,
          });
        }
      });
    }
    spritesRef.current = next;
  }, [visible, zoneLayout]);

  // Capture the latest assistant/tool_call utterance per employee for speech bubbles
  useEffect(() => {
    const now = performance.now();
    Object.entries(activityByEmp).forEach(([empId, list]) => {
      const last = list[list.length - 1];
      if (!last) return;
      if (last.kind === "assistant" || last.kind === "tool_call" || last.kind === "clarify") {
        const text = (last.text || "").slice(0, 48);
        popRef.current.set(empId, { text, until: now + 5000 });
      }
    });
  }, [activityByEmp]);

  // Animation + render loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = Math.max(1, window.devicePixelRatio || 1);

    const resize = () => {
      const wrap = wrapperRef.current;
      if (!wrap) return;
      const rect = wrap.getBoundingClientRect();
      canvas.width = Math.round(rect.width * dpr);
      canvas.height = Math.round(rect.height * dpr);
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
    };
    resize();
    const ro = new ResizeObserver(resize);
    if (wrapperRef.current) ro.observe(wrapperRef.current);

    let last = performance.now();

    const render = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      step(dt);
      draw(ctx, canvas, dpr);
      rafRef.current = requestAnimationFrame(render);
    };
    rafRef.current = requestAnimationFrame(render);

    return () => {
      ro.disconnect();
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  function step(dt: number) {
    // Spring physics; stiff enough to settle within ~600ms.
    const k = 70;       // spring constant
    const c = 12;       // damping
    for (const s of spritesRef.current) {
      const ax = k * (s.tx - s.x) - c * s.vx;
      const ay = k * (s.ty - s.y) - c * s.vy;
      s.vx += ax * dt;
      s.vy += ay * dt;
      s.x += s.vx * dt;
      s.y += s.vy * dt;
    }
  }

  function draw(ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement, dpr: number) {
    const cssW = canvas.width / dpr;
    const cssH = canvas.height / dpr;
    const scale = Math.min(cssW / W, cssH / H);
    const offX = (cssW - W * scale) / 2;
    const offY = (cssH - H * scale) / 2;

    // Background
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const bg = ctx.createLinearGradient(0, 0, 0, cssH);
    bg.addColorStop(0, "#f0f9ff");
    bg.addColorStop(1, "#e0e7ff");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, cssW, cssH);

    ctx.setTransform(scale * dpr, 0, 0, scale * dpr, offX * dpr, offY * dpr);

    // Floor
    ctx.fillStyle = "#ffffff";
    ctx.strokeStyle = "#cbd5e1";
    ctx.lineWidth = 4;
    roundRect(ctx, 30, 30, W - 60, H - 60, 32);
    ctx.fill();
    ctx.stroke();

    // Zones
    for (const z of zoneLayout) {
      ctx.fillStyle = z.fill;
      roundRect(ctx, z.rect.x, z.rect.y, z.rect.w, z.rect.h, 24);
      ctx.fill();
      // label
      ctx.font = "600 28px Inter, system-ui";
      ctx.fillStyle = "#1f2937";
      ctx.textBaseline = "top";
      ctx.fillText(`${z.emoji} ${labelForZone(z.id)}`, z.rect.x + 18, z.rect.y + 14);
    }

    // Department color rings around employees in same dept (subtle).
    const deptColor = (id: string) => departments.find((d) => d.id === id)?.color ?? "#94a3b8";

    // Sprites
    for (const s of spritesRef.current) {
      // dept ring
      ctx.beginPath();
      ctx.arc(s.x, s.y, 32, 0, Math.PI * 2);
      ctx.fillStyle = deptColor(s.emp.department_id);
      ctx.globalAlpha = 0.18;
      ctx.fill();
      ctx.globalAlpha = 1;

      // body
      ctx.beginPath();
      ctx.arc(s.x, s.y, 24, 0, Math.PI * 2);
      ctx.fillStyle = `hsl(${s.hue} 80% 65%)`;
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#1e293b";
      ctx.stroke();

      // emoji face
      ctx.font = "30px system-ui";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = "#0f172a";
      ctx.fillText(s.emoji, s.x, s.y + 1);

      // name plate
      ctx.font = "12px Inter, system-ui";
      ctx.fillStyle = "rgba(15,23,42,0.85)";
      ctx.fillText(s.emp.name.slice(0, 14), s.x, s.y + 40);

      // speech bubble
      const pop = popRef.current.get(s.id);
      if (pop && pop.until > performance.now()) {
        drawBubble(ctx, s.x, s.y - 36, pop.text);
      }
    }

    // Hover tooltip (drawn in CSS pixel space, so reset transform)
    if (hoverRef.current) {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const sp = spritesRef.current.find((s) => s.id === hoverRef.current!.id);
      if (sp) {
        const px = offX + sp.x * scale;
        const py = offY + sp.y * scale;
        drawTooltip(ctx, px, py - 50, `${sp.emp.name} — ${sp.emp.role}`, sp.emp.activity);
      }
    }
  }

  // Pick events
  const onClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const id = pickAt(e);
    if (id) onPickEmployee(id);
  };
  const onMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const id = pickAt(e);
    if (id) {
      const r = (e.target as HTMLCanvasElement).getBoundingClientRect();
      hoverRef.current = { id, x: e.clientX - r.left, y: e.clientY - r.top };
    } else {
      hoverRef.current = null;
    }
  };

  function pickAt(e: React.MouseEvent<HTMLCanvasElement>): string | null {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const cssW = rect.width;
    const cssH = rect.height;
    const scale = Math.min(cssW / W, cssH / H);
    const offX = (cssW - W * scale) / 2;
    const offY = (cssH - H * scale) / 2;
    const wx = (e.clientX - rect.left - offX) / scale;
    const wy = (e.clientY - rect.top - offY) / scale;
    let best: { id: string; d: number } | null = null;
    for (const s of spritesRef.current) {
      const dx = s.x - wx;
      const dy = s.y - wy;
      const d = Math.hypot(dx, dy);
      if (d <= 28 && (!best || d < best.d)) best = { id: s.id, d };
    }
    return best?.id ?? null;
  }

  return (
    <div ref={wrapperRef} className="absolute inset-0">
      <canvas
        ref={canvasRef}
        className="absolute inset-0 cursor-pointer"
        role="img"
        aria-label={t("appTitle")}
        onClick={onClick}
        onMouseMove={onMove}
        onMouseLeave={() => (hoverRef.current = null)}
      />
    </div>
  );
}

function labelForZone(z: Zone): string {
  return t(z as any);
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function drawBubble(ctx: CanvasRenderingContext2D, x: number, y: number, text: string) {
  ctx.font = "13px Inter, system-ui";
  ctx.textAlign = "center";
  const padding = 8;
  const w = Math.min(220, ctx.measureText(text).width + padding * 2);
  const h = 24;
  ctx.fillStyle = "rgba(255,255,255,0.95)";
  ctx.strokeStyle = "rgba(15,23,42,0.2)";
  ctx.lineWidth = 1;
  roundRect(ctx, x - w / 2, y - h, w, h, 12);
  ctx.fill();
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x - 6, y);
  ctx.lineTo(x, y + 7);
  ctx.lineTo(x + 6, y);
  ctx.closePath();
  ctx.fillStyle = "rgba(255,255,255,0.95)";
  ctx.fill();
  ctx.fillStyle = "#0f172a";
  ctx.fillText(text, x, y - 7);
}

function drawTooltip(ctx: CanvasRenderingContext2D, x: number, y: number, line1: string, line2: string) {
  ctx.font = "12px Inter, system-ui";
  ctx.textAlign = "left";
  const w = Math.max(ctx.measureText(line1).width, ctx.measureText(line2).width) + 18;
  const h = 38;
  ctx.fillStyle = "rgba(15,23,42,0.92)";
  roundRect(ctx, x - w / 2, y - h, w, h, 8);
  ctx.fill();
  ctx.fillStyle = "#fafafa";
  ctx.fillText(line1, x - w / 2 + 9, y - h + 14);
  ctx.fillStyle = "#cbd5e1";
  ctx.fillText(line2, x - w / 2 + 9, y - h + 30);
}
