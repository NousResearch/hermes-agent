"use client";

import * as React from "react";
import { IsoFigure } from "../../kit/react/draw";
import { asksForStillness, useReadout } from "../../kit/react/live";
import { keyOf, rowsOf, type GlassShape, type RowShape } from "./drops";
import { fanOf, type FanPose } from "./fan";
import { HEIGHT, MERGE, MERGE_LEAST, MERGE_MOST, MERGE_STEP, ROWS, WIDTH, mergeAtX, readoutOf, roundMerge, spokenOf } from "./rig";

const FOLLOW = 150;
const GLIDE = 560;
const SETTLED = 0.004;
const TOUR_FIRST = 1800;
const TOUR_BACK = 4200;
const TOUR_TRAVEL = 1400;
const TOUR: { merge: number; hold: number }[] = [
  { merge: MERGE_MOST, hold: 2400 },
  { merge: MERGE_LEAST, hold: 2200 },
  { merge: MERGE, hold: 3800 },
];
const BLADES = ROWS + 1;

function Glass({ shape, joined }: { shape: GlassShape; joined: boolean }) {
  const quiet = joined ? "lo" : "faint";
  return (
    <g className="iso-solid ml-glass" data-joined={joined ? "" : undefined}>
      <path className="iso-fill" d={shape.fill} />
      {shape.shades.map((shade, index) => (
        <path key={index} className="iso-shade" data-shade={shade.shade} d={shade.d} />
      ))}
      <path className="iso-top" d={shape.top} />
      <path className="iso-line" data-tone={quiet} d={shape.crease} />
      {shape.bevel ? <path className="iso-line iso-bevel" data-tone={quiet} d={shape.bevel} /> : null}
      <path className="iso-line iso-edge" data-tone={joined ? "hi" : "mid"} d={shape.outline} />
      {shape.shine ? <path className="iso-line ml-shine" data-tone="mid" d={shape.shine} /> : null}
    </g>
  );
}

const Row = React.memo(function Row({ shape }: { shape: RowShape }) {
  return (
    <>
      <g>
        <g>
          {shape.glass.map((glass, index) => (
            <Glass key={index} shape={glass} joined={shape.joined} />
          ))}
        </g>
      </g>
      {shape.neck.length ? (
        <g>
          <g className="ml-neck">
            {shape.neck.map((glint, index) => (
              <path key={index} className="iso-line" data-tone="lit" strokeOpacity={glint.alpha < 1 ? glint.alpha : undefined} d={glint.d} />
            ))}
          </g>
        </g>
      ) : null}
    </>
  );
});

interface BladeNodes {
  group: SVGGElement | null;
  fill: SVGPathElement | null;
  shades: (SVGPathElement | null)[];
  top: SVGPathElement | null;
  crease: SVGPathElement | null;
  outline: SVGPathElement | null;
  line: SVGPathElement | null;
}

const emptyBlade = (): BladeNodes => ({ group: null, fill: null, shades: [null, null, null, null], top: null, crease: null, outline: null, line: null });

function apply(nodes: BladeNodes[], halo: SVGPathElement | null, pose: FanPose) {
  halo?.setAttribute("d", pose.halo);
  pose.blades.forEach((blade, index) => {
    const node = nodes[index];
    if (!node) return;
    node.fill?.setAttribute("d", blade.fill);
    blade.shades.forEach((d, shade) => node.shades[shade]?.setAttribute("d", d));
    node.top?.setAttribute("d", blade.top);
    node.crease?.setAttribute("d", blade.crease);
    node.outline?.setAttribute("d", blade.outline);
    node.line?.setAttribute("d", blade.line);
    if (index < ROWS) {
      if (pose.out[index]) node.group?.setAttribute("data-out", "");
      else node.group?.removeAttribute("data-out");
    }
  });
}

interface Props {
  label: string;
  back: React.ReactNode;
  deck: React.ReactNode;
  front: React.ReactNode;
}

export function LiquidLive({ label, back, deck, front }: Props) {
  const { anchor, say } = useReadout<HTMLDivElement>();
  const root = React.useRef<HTMLDivElement | null>(null);
  const blades = React.useRef<BladeNodes[]>(Array.from({ length: BLADES }, emptyBlade));
  const halo = React.useRef<SVGPathElement | null>(null);
  const rest = React.useMemo(() => fanOf(MERGE), []);
  const goal = React.useRef(MERGE);
  const lead = React.useRef(MERGE);
  const shown = React.useRef(MERGE);
  const ease = React.useRef(FOLLOW);
  const frame = React.useRef(0);
  const last = React.useRef(0);
  const [quant, setQuant] = React.useState(keyOf(MERGE) / 4);
  const [hovering, setHovering] = React.useState(false);
  const [focused, setFocused] = React.useState(false);
  const [visible, setVisible] = React.useState(false);
  const [idle, setIdle] = React.useState(true);
  const [poke, setPoke] = React.useState(0);

  const rows = React.useMemo(() => rowsOf(quant), [quant]);
  const value = roundMerge(quant);

  const draw = React.useCallback((merge: number) => {
    apply(blades.current, halo.current, fanOf(merge));
    setQuant(keyOf(merge) / 4);
  }, []);

  const stop = React.useCallback(() => {
    cancelAnimationFrame(frame.current);
    frame.current = 0;
  }, []);

  const run = React.useCallback(() => {
    if (frame.current) return;
    last.current = 0;
    const tick = (now: number) => {
      const dt = last.current ? Math.min(now - last.current, 64) : 16;
      last.current = now;
      const blend = 1 - Math.exp(-dt / ease.current);
      lead.current += (goal.current - lead.current) * blend;
      shown.current += (lead.current - shown.current) * blend;
      const done = Math.abs(goal.current - lead.current) < SETTLED && Math.abs(goal.current - shown.current) < SETTLED;
      if (done) {
        lead.current = goal.current;
        shown.current = goal.current;
      }
      draw(shown.current);
      frame.current = done ? 0 : requestAnimationFrame(tick);
    };
    frame.current = requestAnimationFrame(tick);
  }, [draw]);

  const aim = React.useCallback(
    (merge: number, pace = FOLLOW) => {
      const next = Math.min(MERGE_MOST, Math.max(MERGE_LEAST, merge));
      ease.current = pace;
      goal.current = next;
      if (asksForStillness()) {
        stop();
        lead.current = next;
        shown.current = next;
        draw(next);
        return;
      }
      run();
    },
    [draw, run, stop],
  );

  React.useEffect(() => stop, [stop]);

  React.useEffect(() => {
    say(readoutOf(value));
  }, [value, say]);

  React.useEffect(() => {
    const node = root.current;
    if (!node) return;
    const observer = new IntersectionObserver((entries) => setVisible(Boolean(entries.at(-1)?.isIntersecting)));
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  React.useEffect(() => {
    if (!visible) stop();
  }, [visible, stop]);

  const engaged = hovering || focused;

  React.useEffect(() => {
    if (engaged) {
      setIdle(false);
      return;
    }
    const timer = window.setTimeout(() => setIdle(true), TOUR_BACK);
    return () => window.clearTimeout(timer);
  }, [engaged, poke]);

  const touring = visible && idle && !engaged;

  React.useEffect(() => {
    if (!touring || asksForStillness()) return;
    let timer = 0;
    let step = 0;
    const next = () => {
      const stop = TOUR[step % TOUR.length]!;
      aim(stop.merge, GLIDE);
      step += 1;
      timer = window.setTimeout(next, stop.hold + TOUR_TRAVEL);
    };
    timer = window.setTimeout(next, TOUR_FIRST);
    return () => window.clearTimeout(timer);
  }, [touring, aim]);

  const mergeAt = (clientX: number) => {
    const svg = root.current?.querySelector("svg");
    const box = svg?.getBoundingClientRect();
    if (!box || !box.width) return null;
    return mergeAtX(((clientX - box.left) / box.width) * WIDTH);
  };

  const hold = () => {
    setIdle(false);
    setPoke((count) => count + 1);
  };

  const follow = (clientX: number) => {
    const merge = mergeAt(clientX);
    if (merge !== null) aim(merge, FOLLOW);
  };

  const stepBy = (by: number) => roundMerge(goal.current + by);

  const bindBlade = (index: number) => ({
    group: (node: SVGGElement | null) => {
      blades.current[index]!.group = node;
    },
    path: (key: "fill" | "top" | "crease" | "outline" | "line") => (node: SVGPathElement | null) => {
      blades.current[index]![key] = node;
    },
    shade: (shade: number) => (node: SVGPathElement | null) => {
      blades.current[index]!.shades[shade] = node;
    },
  });

  return (
    <div
      ref={(node) => {
        root.current = node;
        anchor.current = node;
      }}
      className="ml"
      role="slider"
      tabIndex={0}
      aria-label="Merge, in pixels"
      aria-valuemin={MERGE_LEAST}
      aria-valuemax={MERGE_MOST}
      aria-valuenow={value}
      aria-valuetext={spokenOf(value)}
      onPointerEnter={(event) => {
        if (event.pointerType === "mouse") setHovering(true);
      }}
      onPointerMove={(event) => {
        if (event.pointerType === "mouse" || event.currentTarget.hasPointerCapture(event.pointerId)) {
          hold();
          follow(event.clientX);
        }
      }}
      onPointerDown={(event) => {
        if (event.pointerType !== "mouse") event.currentTarget.setPointerCapture(event.pointerId);
        hold();
        follow(event.clientX);
      }}
      onPointerUp={(event) => {
        if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
      }}
      onPointerLeave={(event) => {
        if (event.pointerType !== "mouse") return;
        setHovering(false);
        aim(MERGE, GLIDE);
      }}
      onFocus={() => {
        setFocused(true);
        aim(roundMerge(shown.current), FOLLOW);
      }}
      onBlur={() => setFocused(false)}
      onKeyDown={(event) => {
        let next: number | null = null;
        const big = event.shiftKey ? 4 : 1;
        if (event.key === "ArrowLeft" || event.key === "ArrowDown") next = stepBy(-MERGE_STEP * big);
        else if (event.key === "ArrowRight" || event.key === "ArrowUp") next = stepBy(MERGE_STEP * big);
        else if (event.key === "PageDown") next = stepBy(-MERGE_STEP * 8);
        else if (event.key === "PageUp") next = stepBy(MERGE_STEP * 8);
        else if (event.key === "Home") next = MERGE_LEAST;
        else if (event.key === "End") next = MERGE_MOST;
        if (next === null) return;
        event.preventDefault();
        hold();
        aim(next, FOLLOW);
      }}
    >
      <IsoFigure width={WIDTH} height={HEIGHT} label={label}>
        {back}
        {rows.map((shape) => (
          <Row key={shape.row} shape={shape} />
        ))}
        {deck}
        <g>
          <g>
            <path ref={(node) => void (halo.current = node)} className="iso-halo ml-fan-halo" d={rest.halo} />
            {rest.blades.map((blade, index) => {
              const bind = bindBlade(index);
              const isLead = index === ROWS;
              return (
                <g
                  key={index}
                  ref={bind.group}
                  className="iso-solid ml-blade"
                  data-lead={isLead ? "" : undefined}
                  data-lit={isLead ? "" : undefined}
                  data-out={!isLead && rest.out[index] ? "" : undefined}
                >
                  <path ref={bind.path("fill")} className="iso-fill" d={blade.fill} />
                  {[0, 1, 2, 3].map((shade) => (
                    <path key={shade} ref={bind.shade(shade)} className="iso-shade" data-shade={shade} d={blade.shades[shade] ?? ""} />
                  ))}
                  <path ref={bind.path("top")} className="iso-top" d={blade.top} />
                  <path ref={bind.path("crease")} className="iso-line" data-tone="faint" d={blade.crease} />
                  <path ref={bind.path("outline")} className="iso-line iso-edge" data-tone={isLead ? "hi" : "mid"} d={blade.outline} />
                  <path ref={bind.path("line")} className="iso-line ml-blade-line" data-tone={isLead ? "lit" : "lo"} d={blade.line} />
                </g>
              );
            })}
          </g>
        </g>
        {front}
      </IsoFigure>
    </div>
  );
}
