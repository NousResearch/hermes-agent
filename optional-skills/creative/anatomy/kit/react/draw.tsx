import * as React from "react";
import { pathOf, type Flat, type KeyMark, type LegendKey, type LineTone, type SolidPaths, type Tone } from "../iso-kit";
import { Stage } from "./live";

interface SolidProps {
  paths: SolidPaths;
  tone?: Tone;
  crease?: Tone | "none";
  halo?: boolean;
  lit?: boolean;
  flat?: boolean;
  className?: string;
  inner?: React.ReactNode;
  children?: React.ReactNode;
}

export function Solid({ paths, tone = "hi", crease = "faint", halo = false, lit = false, flat = false, className, inner, children }: SolidProps) {
  return (
    <g className={className ? `iso-solid ${className}` : "iso-solid"} data-lit={lit ? "" : undefined}>
      {halo ? <path className="iso-halo" d={paths.fill} /> : null}
      <path className="iso-fill" d={paths.fill} />
      {flat ? null : paths.shades.map((d, shade) => (d ? <path key={shade} className="iso-shade" data-shade={shade} d={d} /> : null))}
      {flat ? null : <path className="iso-top" d={paths.top} />}
      {crease === "none" || !paths.crease ? null : <path className="iso-line iso-crease" data-tone={crease} d={paths.crease} />}
      {crease === "none" || !paths.bevel ? null : <path className="iso-line iso-bevel" data-tone={crease} d={paths.bevel} />}
      {inner}
      <path className="iso-line iso-edge" data-tone={tone} d={paths.outline} />
      {children}
    </g>
  );
}

interface LineProps {
  d: string;
  tone?: LineTone;
  dotted?: boolean;
  dashed?: boolean;
  flow?: boolean;
  className?: string;
  free?: boolean | "start" | "end";
}

export function Line({ d, tone = "lo", dotted = false, dashed = false, flow = false, className, free }: LineProps) {
  if (!d) return null;
  return (
    <path
      className={className ? `iso-line ${className}` : "iso-line"}
      data-tone={tone}
      data-dotted={dotted ? "" : undefined}
      data-dashed={dashed ? "" : undefined}
      data-flow={flow ? "" : undefined}
      data-free={free === true ? "" : free || undefined}
      d={d}
    />
  );
}

interface WireProps {
  from: Flat;
  to: Flat;
  via?: Flat[];
  tone?: LineTone;
  dotted?: boolean;
  dashed?: boolean;
  flow?: boolean;
  ends?: "both" | "from" | "to" | "none";
  size?: number;
}

const tenth = (value: number) => Math.round(value * 10) / 10;

export function Wire({ from, to, via = [], tone = "lo", dotted = false, dashed = false, flow = false, ends = "both", size = 1.2 }: WireProps) {
  return (
    <g className="iso-wire" data-tone={tone}>
      <Line d={pathOf([from, ...via, to])} tone={tone} dotted={dotted} dashed={dashed} flow={flow} />
      {ends === "both" || ends === "from" ? <circle className="iso-wire-end" cx={tenth(from[0])} cy={tenth(from[1])} r={size} /> : null}
      {ends === "both" || ends === "to" ? <circle className="iso-wire-end" cx={tenth(to[0])} cy={tenth(to[1])} r={size} /> : null}
    </g>
  );
}

export function Dots({ points, size = 0.5, tone = "mid", pulse = false }: { points: Flat[]; size?: number; tone?: Tone; pulse?: boolean }) {
  return (
    <g className="iso-dots" data-tone={tone}>
      {points.map(([x, y], index) => (
        <circle
          key={index}
          cx={tenth(x)}
          cy={tenth(y)}
          r={size}
          data-pulse={pulse ? "" : undefined}
          style={pulse ? ({ "--pulse-at": `${(index * 137) % 1600}ms` } as React.CSSProperties) : undefined}
        />
      ))}
    </g>
  );
}

export function FaceText({
  transform,
  children,
  size = 3,
  tone = "mid",
  anchor = "start",
}: {
  transform: string;
  children: React.ReactNode;
  size?: number;
  tone?: Tone;
  anchor?: "start" | "middle" | "end";
}) {
  return (
    <text className="iso-face-text" data-tone={tone} transform={transform} fontSize={size} textAnchor={anchor}>
      {children}
    </text>
  );
}

interface IsoFigureProps {
  width: number;
  height: number;
  label: string;
  className?: string;
  children: React.ReactNode;
}

export function IsoFigure({ width, height, label, className, children }: IsoFigureProps) {
  return (
    <Stage className={className ? `iso-stage ${className}` : "iso-stage"}>
      <svg className="iso-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={label} focusable="false">
        {children}
      </svg>
    </Stage>
  );
}

function Swatch({ mark }: { mark: KeyMark }) {
  if (mark === "raised" || mark === "flat" || mark === "lit") {
    return (
      <svg className="iso-swatch" data-mark={mark} viewBox="0 0 16 10" aria-hidden="true" focusable="false">
        <path d={mark === "flat" ? "M2 6.5 8 3.5 14 6.5 8 9.5Z" : "M2 4.5 8 1.5 14 4.5 14 6 8 9 2 6Z"} />
        {mark === "flat" ? null : <path className="iso-swatch-crease" d="M2 4.5 8 7.5 14 4.5" />}
      </svg>
    );
  }
  return (
    <svg className="iso-swatch" data-mark={mark} viewBox="0 0 16 10" aria-hidden="true" focusable="false">
      <path d="M1.5 5h13" />
    </svg>
  );
}

interface PlateProps {
  fig: string;
  title?: string;
  hint?: React.ReactNode;
  readout?: React.ReactNode;
  keys?: LegendKey[];
  caption?: React.ReactNode;
  children: React.ReactNode;
}

export function Plate({ fig, title, hint, readout, keys = [], caption, children }: PlateProps) {
  return (
    <figure className="iso-figure">
      <div className="iso-plate">
        <div className="iso-plate-corners" aria-hidden="true">
          <span className="iso-plate-corner" data-corner="fig">
            {fig}
          </span>
          {title ? (
            <span className="iso-plate-corner" data-corner="title">
              {title}
            </span>
          ) : null}
          {hint ? (
            <span className="iso-plate-corner" data-corner="hint">
              {hint}
            </span>
          ) : null}
          <span className="iso-plate-corner" data-corner="readout" data-readout="">
            {readout}
          </span>
        </div>
        {children}
      </div>
      {keys.length || caption ? (
        <figcaption className="iso-legend">
          {keys.length ? (
            <span className="iso-keys">
              {keys.map((key) => (
                <span className="iso-key" key={key.label}>
                  <Swatch mark={key.mark} />
                  {key.label}
                </span>
              ))}
            </span>
          ) : null}
          {caption ? <span className="iso-caption">{caption}</span> : null}
        </figcaption>
      ) : null}
    </figure>
  );
}
