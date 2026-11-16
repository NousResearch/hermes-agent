"use client";

import * as React from "react";
import { IsoFigure } from "../../kit/react/draw";
import { asksForStillness, useReadout } from "../../kit/react/live";
import { frameOf, type Frame } from "./pose";
import { HALF, REST, memberAt, one, poseOf, readoutOf, settled, sizeText, step, type Member, type Pose } from "./rig";
import { EX, FINGER_MOST, GRAB_MOST, HEIGHT, HOVER, PROBE_HOME, PROBE_MAX, PROBE_MIN, WIDTH, at } from "./view";

const FOLLOW = 0.09;
const DOWN = 0.07;
const UP = 0.16;
const NUDGE = 10;
const TOUR_FIRST = 1.4;
const TOUR_BACK = 3600;
const LOOP = 7.6;
const MOST_DT = 1 / 30;

interface Hand {
  held: boolean;
  grab: number;
  finger: number;
}

const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));
const ease = (t: number) => {
  const u = clamp(t, 0, 1);
  return u * u * (3 - 2 * u);
};

function tour(t: number): Hand {
  const at = t % LOOP;
  if (at < 0.6) return { held: false, grab: PROBE_HOME, finger: PROBE_HOME };
  if (at < 3.7) return { held: true, grab: PROBE_HOME, finger: PROBE_HOME - 80 * ease((at - 1.2) / 1.2) };
  return { held: false, grab: PROBE_HOME, finger: PROBE_HOME - 80 + 80 * ease((at - 4.4) / 1.2) };
}

interface Nodes {
  glass: SVGGElement | null;
  fill: SVGPathElement | null;
  shades: (SVGPathElement | null)[];
  top: SVGPathElement | null;
  crease: SVGPathElement | null;
  bevel: SVGPathElement | null;
  edge: SVGPathElement | null;
  rim: SVGPathElement | null;
  shine: SVGPathElement | null;
  coilsBack: SVGPathElement | null;
  coilsLeft: SVGPathElement | null;
  coilsRightBack: SVGPathElement | null;
  coilsRightFront: SVGPathElement | null;
  carriage: SVGGElement | null;
  plunger: SVGGElement | null;
  needle: SVGPathElement | null;
  pull: SVGPathElement | null;
  grab: SVGCircleElement | null;
  touch: SVGCircleElement | null;
  rider: SVGGElement | null;
  probe: SVGGElement | null;
  arm: SVGGElement | null;
  tip: SVGGElement | null;
  rest: SVGGElement | null;
}

function apply(nodes: Nodes, frame: Frame, held: boolean) {
  const { glass } = frame;
  nodes.fill?.setAttribute("d", glass.fill);
  glass.shades?.forEach((d, index) => nodes.shades[index]?.setAttribute("d", d));
  nodes.top?.setAttribute("d", glass.top);
  nodes.crease?.setAttribute("d", glass.crease);
  nodes.bevel?.setAttribute("d", glass.bevel ?? "");
  nodes.edge?.setAttribute("d", glass.outline);
  nodes.rim?.setAttribute("d", glass.rim);
  nodes.shine?.setAttribute("d", glass.shine);
  if (nodes.glass) {
    if (held) nodes.glass.dataset.lit = "";
    else delete nodes.glass.dataset.lit;
  }
  nodes.coilsBack?.setAttribute("d", frame.coilsBack);
  nodes.coilsLeft?.setAttribute("d", frame.coilsLeft);
  nodes.coilsRightBack?.setAttribute("d", frame.coilsRightBack);
  nodes.coilsRightFront?.setAttribute("d", frame.coilsRightFront);
  nodes.carriage?.setAttribute("transform", frame.carriage);
  nodes.plunger?.setAttribute("transform", frame.carriage);
  nodes.needle?.setAttribute("d", frame.needle);
  nodes.pull?.setAttribute("d", frame.pull);
  for (const [node, point] of [
    [nodes.grab, frame.grab],
    [nodes.touch, frame.finger],
  ] as const) {
    if (!node) continue;
    node.setAttribute("cx", `${point[0]}`);
    node.setAttribute("cy", `${point[1]}`);
    node.style.opacity = frame.pull ? "" : "0";
  }
  nodes.rider?.setAttribute("transform", frame.rider);
  nodes.probe?.setAttribute("transform", frame.probe);
  nodes.arm?.setAttribute("transform", frame.probe);
  nodes.tip?.setAttribute("transform", frame.tip);
}

function restShown(pose: Pose) {
  const off = Math.max(Math.abs(pose.mid - pose.half[0] + HALF[0]), Math.abs(pose.mid + pose.half[0] - HALF[0]), Math.abs(pose.half[1] - HALF[1]));
  return `${Math.round(clamp((off - 0.4) / 1.6, 0, 1) * 100) / 100}`;
}

function spokenOf(member: Member, pull: number, held: boolean) {
  const pose = poseOf(member);
  const size = sizeText(pose).replace(" × ", " by ");
  const state = held ? "Pressed" : member.swell.at === 0 && member.lean.at === 0 ? "At rest" : "Springing back";
  return `${state}: ${size} pixels, pulled ${one(Math.abs(pull))} px, leaning ${one(Math.abs(pose.lean))} px`;
}

const RESTING_TEXT = spokenOf(memberAt(), 0, false);

interface Props {
  label: string;
  base: React.ReactNode;
  rail: React.ReactNode;
  rider: React.ReactNode;
  block: React.ReactNode;
  frameBack: React.ReactNode;
  rods: React.ReactNode;
  carriage: React.ReactNode;
  plunger: React.ReactNode;
  rest: React.ReactNode;
  frameFront: React.ReactNode;
  dial: React.ReactNode;
  finger: React.ReactNode;
  arm: React.ReactNode;
}

export function StretchLive(props: Props) {
  const { anchor, say } = useReadout<HTMLDivElement>();
  const root = React.useRef<HTMLDivElement | null>(null);
  const nodes = React.useRef<Nodes>({
    glass: null,
    fill: null,
    shades: [null, null, null, null],
    top: null,
    crease: null,
    bevel: null,
    edge: null,
    rim: null,
    shine: null,
    coilsBack: null,
    coilsLeft: null,
    coilsRightBack: null,
    coilsRightFront: null,
    carriage: null,
    plunger: null,
    needle: null,
    pull: null,
    grab: null,
    touch: null,
    rider: null,
    probe: null,
    arm: null,
    tip: null,
    rest: null,
  });
  const member = React.useRef<Member>(memberAt());
  const hand = React.useRef<Hand>({ held: false, grab: PROBE_HOME, finger: PROBE_HOME });
  const local = React.useRef(PROBE_HOME);
  const finger = React.useRef(PROBE_HOME);
  const lift = React.useRef(1);
  const frame = React.useRef(0);
  const last = React.useRef(0);
  const clock = React.useRef(0);
  const script = React.useRef<((t: number) => Hand) | null>(null);
  const [pressing, setPressing] = React.useState(false);
  const user = React.useRef(false);
  const [hovering, setHovering] = React.useState(false);
  const [focused, setFocused] = React.useState(false);
  const [visible, setVisible] = React.useState(false);
  const [idle, setIdle] = React.useState(true);
  const [poke, setPoke] = React.useState(0);

  const rest = React.useMemo(() => frameOf(REST, 0, PROBE_HOME, PROBE_HOME, HOVER, PROBE_HOME), []);

  const grip = React.useCallback((next: Hand) => {
    const was = hand.current;
    if (next.held && !was.held) {
      const pose = poseOf(member.current);
      local.current = (next.grab - pose.mid) / (pose.half[0] / HALF[0]);
    }
    hand.current = next;
  }, []);

  const draw = React.useCallback(() => {
    const pose = poseOf(member.current);
    const now = hand.current;
    apply(nodes.current, frameOf(pose, member.current.lean.at, local.current, finger.current, HOVER * lift.current, PROBE_HOME, now.held), now.held);
    say(readoutOf(pose));
    const restNode = nodes.current.rest;
    const shownRest = restShown(pose);
    if (restNode && restNode.style.opacity !== shownRest) restNode.style.opacity = shownRest;
    const pulled = now.held ? finger.current - now.grab : 0;
    const node = root.current;
    if (node) {
      const value = `${Math.round(pulled)}`;
      if (node.getAttribute("aria-valuenow") !== value) node.setAttribute("aria-valuenow", value);
      const spoken = spokenOf(member.current, pulled, now.held);
      if (node.getAttribute("aria-valuetext") !== spoken) node.setAttribute("aria-valuetext", spoken);
    }
  }, [say]);

  const stop = React.useCallback(() => {
    cancelAnimationFrame(frame.current);
    frame.current = 0;
  }, []);

  const run = React.useCallback(() => {
    if (frame.current) return;
    last.current = 0;
    const tick = (now: number) => {
      const dt = last.current ? Math.min((now - last.current) / 1000, MOST_DT) : 1 / 60;
      last.current = now;
      const calm = asksForStillness();
      if (script.current) {
        clock.current += dt;
        grip(script.current(clock.current));
      }
      const want = hand.current;
      const follow = calm ? 1 : 1 - Math.exp(-dt / FOLLOW);
      finger.current += (want.finger - finger.current) * follow;
      if (Math.abs(want.finger - finger.current) < 0.02) finger.current = want.finger;
      const goal = want.held ? 0 : 1;
      const drop = calm ? 1 : 1 - Math.exp(-dt / (want.held ? DOWN : UP));
      lift.current += (goal - lift.current) * drop;
      if (Math.abs(goal - lift.current) < 0.002) lift.current = goal;
      const pulled = want.held ? finger.current - want.grab : 0;
      step(member.current, want.held, pulled, dt, calm);
      draw();
      const done = !script.current && settled(member.current, want.held, pulled) && finger.current === want.finger && lift.current === goal;
      frame.current = done ? 0 : requestAnimationFrame(tick);
    };
    frame.current = requestAnimationFrame(tick);
  }, [draw, grip]);

  const steer = React.useCallback(
    (next: Hand) => {
      script.current = null;
      grip(next);
      run();
    },
    [grip, run],
  );

  React.useEffect(() => {
    const node = root.current;
    if (!node) return;
    const observer = new IntersectionObserver((entries) => setVisible(Boolean(entries.at(-1)?.isIntersecting)));
    observer.observe(node);
    return () => {
      observer.disconnect();
      stop();
    };
  }, [stop]);

  const engaged = hovering || focused || pressing;

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
    if (!visible) {
      script.current = null;
      stop();
      return;
    }
    if (touring && !asksForStillness()) {
      const timer = window.setTimeout(
        () => {
          clock.current = 0;
          script.current = tour;
          run();
        },
        TOUR_FIRST * 1000,
      );
      return () => {
        window.clearTimeout(timer);
        script.current = null;
      };
    }
    script.current = null;
    if (user.current) {
      run();
      return;
    }
    steer({ held: false, grab: hand.current.grab, finger: hovering ? hand.current.finger : PROBE_HOME });
  }, [visible, touring, hovering, run, steer, stop]);

  const xAt = (clientX: number) => {
    const svg = root.current?.querySelector("svg");
    const box = svg?.getBoundingClientRect();
    if (!box || !box.width) return null;
    const sx = ((clientX - box.left) / box.width) * WIDTH;
    const origin = at([0, 0, 0]);
    return (sx - origin[0]) / EX[0];
  };

  const wake = () => {
    setIdle(false);
    setPoke((count) => count + 1);
  };

  const hoverTo = (x: number) => steer({ held: false, grab: hand.current.grab, finger: clamp(x, PROBE_MIN, PROBE_MAX) });

  const hold = (on: boolean) => {
    user.current = on;
    setPressing(on);
  };

  const pressAt = (x: number) => {
    hold(true);
    const grab = clamp(x, -GRAB_MOST, GRAB_MOST);
    steer({ held: true, grab, finger: grab });
  };

  const pullTo = (x: number) => steer({ held: true, grab: hand.current.grab, finger: clamp(x, -FINGER_MOST, FINGER_MOST) });

  const release = () => {
    hold(false);
    steer({ held: false, grab: hand.current.grab, finger: hand.current.finger });
  };

  const bind = <K extends keyof Nodes>(key: K) => (node: Nodes[K]) => {
    nodes.current[key] = node;
  };

  const shown = rest.glass;

  return (
    <div
      ref={(node) => {
        root.current = node;
        anchor.current = node;
      }}
      className="ms"
      role="slider"
      tabIndex={0}
      aria-label="Glass button on a test rig: Space presses or lets go, arrow keys move the finger and pull"
      aria-valuemin={-FINGER_MOST - GRAB_MOST}
      aria-valuemax={FINGER_MOST + GRAB_MOST}
      aria-valuenow={0}
      aria-valuetext={RESTING_TEXT}
      data-held={pressing ? "" : undefined}
      onPointerEnter={(event) => {
        if (event.pointerType === "mouse") setHovering(true);
      }}
      onPointerMove={(event) => {
        const x = xAt(event.clientX);
        if (x === null) return;
        if (user.current && event.currentTarget.hasPointerCapture(event.pointerId)) {
          wake();
          pullTo(x);
        } else if (event.pointerType === "mouse" && !user.current) {
          wake();
          hoverTo(x);
        }
      }}
      onPointerDown={(event) => {
        const x = xAt(event.clientX);
        if (x === null) return;
        event.currentTarget.setPointerCapture(event.pointerId);
        wake();
        pressAt(x);
      }}
      onPointerUp={(event) => {
        if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
        if (user.current) release();
      }}
      onPointerCancel={() => {
        if (user.current) release();
      }}
      onPointerLeave={(event) => {
        if (event.pointerType !== "mouse" || user.current) return;
        setHovering(false);
        hoverTo(PROBE_HOME);
      }}
      onFocus={() => setFocused(true)}
      onBlur={() => {
        setFocused(false);
        if (user.current) release();
      }}
      onKeyDown={(event) => {
        const now = hand.current;
        const by = event.shiftKey ? NUDGE * 3 : NUDGE;
        if (event.key === " " || event.key === "Enter") {
          event.preventDefault();
          if (event.repeat) return;
          wake();
          if (user.current) release();
          else pressAt(now.finger);
          return;
        }
        let x: number | null = null;
        if (event.key === "ArrowLeft" || event.key === "ArrowDown") x = now.finger - by;
        else if (event.key === "ArrowRight" || event.key === "ArrowUp") x = now.finger + by;
        else if (event.key === "Home") x = -Infinity;
        else if (event.key === "End") x = Infinity;
        else if (event.key === "Escape") {
          if (user.current) release();
          return;
        }
        if (x === null) return;
        event.preventDefault();
        wake();
        if (user.current) pullTo(x);
        else hoverTo(x);
      }}
    >
      <IsoFigure width={WIDTH} height={HEIGHT} label={props.label}>
        {props.base}
        <g>
          <g>
            {props.rail}
            <g ref={bind("rider")} transform={rest.rider}>
              {props.rider}
            </g>
            <g ref={bind("probe")} transform={rest.probe}>
              {props.block}
            </g>
          </g>
        </g>
        {props.frameBack}
        <g>
          <g>
            <path ref={bind("coilsBack")} className="iso-line ms-coil" data-tone="lo" d={rest.coilsBack} />
            {props.rods}
            <path ref={bind("coilsLeft")} className="iso-line ms-coil" data-tone="mid" d={rest.coilsLeft} />
            <g ref={bind("carriage")} transform={rest.carriage}>
              {props.carriage}
            </g>
          </g>
        </g>
        <g>
          <g ref={bind("glass")} className="iso-solid ms-glass">
            <path ref={bind("fill")} className="iso-fill" d={shown.fill} />
            {[0, 1, 2, 3].map((index) => (
              <path
                key={index}
                ref={(node) => {
                  nodes.current.shades[index] = node;
                }}
                className="iso-shade"
                data-shade={index}
                d={shown.shades?.[index] ?? ""}
              />
            ))}
            <path ref={bind("top")} className="iso-top" d={shown.top} />
            <path ref={bind("crease")} className="iso-line" data-tone="lo" d={shown.crease} />
            <path ref={bind("bevel")} className="iso-line iso-bevel" data-tone="lo" d={shown.bevel ?? ""} />
            <path ref={bind("rim")} className="iso-line ms-rim" data-tone="mid" d={shown.rim} />
            <path ref={bind("shine")} className="iso-line ms-shine" data-tone="hi" d={shown.shine} />
            <path ref={bind("edge")} className="iso-line iso-edge" data-tone="hi" d={shown.outline} />
          </g>
        </g>
        <g>
          <g ref={bind("rest")} style={{ opacity: 0 }}>
            {props.rest}
          </g>
        </g>
        <g>
          <g className="iso-wire ms-pull" data-tone="hi">
            <path ref={bind("pull")} className="iso-line" data-tone="lit" d={rest.pull} />
            <circle ref={bind("grab")} className="iso-wire-end" cx={rest.grab[0]} cy={rest.grab[1]} r={1.2} style={{ opacity: 0 }} />
            <circle ref={bind("touch")} className="iso-wire-end" cx={rest.finger[0]} cy={rest.finger[1]} r={1.2} style={{ opacity: 0 }} />
          </g>
        </g>
        <g>
          <path ref={bind("coilsRightBack")} className="iso-line ms-coil" data-tone="mid" d={rest.coilsRightBack} />
        </g>
        <g>
          <g ref={bind("plunger")} transform={rest.carriage}>
            {props.plunger}
          </g>
        </g>
        <g>
          <path ref={bind("coilsRightFront")} className="iso-line ms-coil" data-tone="mid" d={rest.coilsRightFront} />
        </g>
        {props.frameFront}
        <g>
          <g>
            {props.dial}
            <path ref={bind("needle")} className="iso-line ms-needle" data-tone="hi" d={rest.needle} />
          </g>
        </g>
        <g>
          <g ref={bind("tip")} transform={rest.tip}>
            {props.finger}
          </g>
        </g>
        <g>
          <g ref={bind("arm")} transform={rest.probe}>
            {props.arm}
          </g>
        </g>
      </IsoFigure>
    </div>
  );
}
