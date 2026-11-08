import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as k from "../../kit/iso-kit.mjs";
import { kitScript } from "../../scripts/inline-kit.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const theme = process.argv.includes("--light") ? "light" : "dark";

const WIDTH = 600;
const HEIGHT = 440;

const PLATE = { x: -100, y: -84, w: 210, d: 242, r: 10 };
const PLATE_H = 10;
const FOOT_H = 5;

const CASE = { x: -62, y: -62, w: 124, d: 118, r: 8 };
const FRONT = CASE.y + CASE.d;
const SIDE = CASE.x + CASE.w;
const PLINTH_H = 5;
const TOP = 170;
const SEAM = FRONT - 12;

const BEZEL = { x0: -48, x1: 48, z0: 68, z1: 158, r: 6, depth: 1.6 };
const CRT = { x0: -40, x1: 40, z0: 77, z1: 150, r: 8, depth: 5 };
const GLASS_Y = FRONT - BEZEL.depth - CRT.depth;
const CRT_MID = (CRT.z0 + CRT.z1) / 2;
const SLOT = { x0: 8, x1: 46, z0: 45, z1: 50.5, r: 2.4, depth: 4 };

const KB = { x: -76, y: 84, w: 152, d: 58, r: 4 };
const KB_Z = 1.6;
const KB_TOP = 8.6;
const U = 9.2;
const KEYS_X = -U * 7.5;
const KEYS_Y = 88.5;

const ROWS = [
  [["`", 1], ["1", 1], ["2", 1], ["3", 1], ["4", 1], ["5", 1], ["6", 1], ["7", 1], ["8", 1], ["9", 1], ["0", 1], ["-", 1], ["=", 1], ["Backspace", 2]],
  [["Tab", 1.5], ["q", 1], ["w", 1], ["e", 1], ["r", 1], ["t", 1], ["y", 1], ["u", 1], ["i", 1], ["o", 1], ["p", 1], ["[", 1], ["]", 1], ["\\", 1.5]],
  [["CapsLock", 1.75], ["a", 1], ["s", 1], ["d", 1], ["f", 1], ["g", 1], ["h", 1], ["j", 1], ["k", 1], ["l", 1], [";", 1], ["'", 1], ["Enter", 2.25]],
  [["Shift", 2.25], ["z", 1], ["x", 1], ["c", 1], ["v", 1], ["b", 1], ["n", 1], ["m", 1], [",", 1], [".", 1], ["/", 1], ["ShiftRight", 2.75]],
  [["Alt", 1.5], ["Meta", 1.5], [" ", 9], ["NumpadEnter", 1.5], ["AltRight", 1.5]],
];
const ROW_H = [5.6, 5.2, 4.8, 4.4, 4];

const PORT = { y: 28, z: 9 };
const COIL = { x: 80, z: 3.6, r: 3.2, from: 44, to: 84, turns: 17 };

const P = k.fitProjection(
  [...k.boxCorners(PLATE, -PLATE_H - FOOT_H, 0), ...k.boxCorners(CASE, 0, TOP)],
  WIDTH,
  HEIGHT,
  { pad: 30, azimuth: 56 },
);

const at = (point) => k.iso(point, P);
const S = (paths, style) => k.solidSvg(paths, style);
const L = (d, style) => k.lineSvg(d, style);
const D = (points, style) => k.dotsSvg(points, style);
const byDepth = (points) => [...points].sort((a, b) => k.depthOf([a[0], a[1], 0], P) - k.depthOf([b[0], b[1], 0], P));
const planeOf = (y) => k.sideMatrix([0, y], 0, P, "left");
const flat = (x0, z0, x1, z1, r, steps = 6) => k.pathOf(k.roundedPlan({ x: x0, y: z0, w: x1 - x0, d: z1 - z0, r }, steps).map(([u, z]) => [u, -z]), true);
const onFront = (points, y = FRONT) => k.pathOf(points.map(([x, z]) => at([x, y, z])));
const sideSlot = (y, z0, z1, half = 0.8) => k.pathOf(k.roundedPlan({ x: y - half, y: z0, w: half * 2, d: z1 - z0, r: half }, 4).map(([u, z]) => at([SIDE, u, z])), true);

let clips = 0;
function recess({ x0, x1, z0, z1, r, depth }, y, back = "", backClass = "pc-well", inside = "") {
  const id = `pc-clip-${clips++}`;
  const ring = k.roundedPlan({ x: x0, y: z0, w: x1 - x0, d: z1 - z0, r }, 6);
  const cx = (x0 + x1) / 2;
  const cz = (z0 + z1) / 2;
  const { sinA } = k.cameraOf(P);
  const tops = [];
  const sides = [[], [], [], []];
  ring.forEach(([ax, az], index) => {
    const [bx, bz] = ring[(index + 1) % ring.length];
    const length = Math.hypot(bx - ax, bz - az) || 1;
    let [nx, nz] = [(bz - az) / length, -(bx - ax) / length];
    if (nx * (cx - (ax + bx) / 2) + nz * (cz - (az + bz) / 2) < 0) [nx, nz] = [-nx, -nz];
    if (nx <= 0 && nz <= 0) return;
    const quad = k.pathOf([at([ax, y, az]), at([bx, y, bz]), at([bx, y - depth, bz]), at([ax, y - depth, az])], true);
    if (nz >= nx) tops.push(quad);
    else sides[Math.min(3, Math.max(0, Math.floor(((nx * -sinA + 1) / 2) * 4)))].push(quad);
  });
  const shape = flat(x0, z0, x1, z1, r);
  return [
    `<clipPath id="${id}"><path transform="${planeOf(y)}" d="${shape}"/></clipPath>`,
    `<g clip-path="url(#${id})">`,
    `<path class="iso-top" d="${tops.join("")}"/>`,
    ...sides.map((d, shade) => (d.length ? `<path class="iso-shade" data-shade="${shade}" d="${d.join("")}"/>` : "")),
    `<g transform="${planeOf(y - depth)}"><path class="${backClass}" d="${shape}"/>${back}<path class="iso-line" data-tone="faint" d="${shape}"/></g>`,
    inside,
    `</g>`,
    `<g transform="${planeOf(y)}"><path class="iso-line" data-tone="lo" d="${shape}"/></g>`,
  ].join("");
}

function plate() {
  const out = [`<path class="iso-halo" d="${k.haloOf(PLATE, -PLATE_H - FOOT_H, PLATE_H, P)}"/>`];
  for (const [x, y] of byDepth(k.corners(PLATE, 22))) {
    out.push(S(k.cylinder(x, y, 7, -PLATE_H - FOOT_H, 1.6, P, 28), { tone: "lo" }));
    out.push(S(k.cylinder(x, y, 5.4, -PLATE_H - FOOT_H + 1.6, FOOT_H - 1.6, P, 28), { tone: "mid" }));
  }
  out.push(S(k.slabOf(PLATE, -PLATE_H, PLATE_H, P, 8, 1.6), { tone: "mid" }));
  out.push(L(k.planOutline(k.insetPlan(PLATE, 5), 0, P), { tone: "lo" }));
  const screws = k.corners(PLATE, 10);
  out.push(L(screws.map(([x, y]) => k.ring(x, y, 2.2, 0, P, 16)).join(""), { tone: "lo" }));
  out.push(D(screws.map(([x, y]) => at([x, y, 0])), { size: 0.5 }));
  const label = { x: -88, y: 144, w: 40, d: 9, r: 1.2 };
  out.push(S(k.slabOf(label, 0, 0.8, P, 3), { tone: "lo", crease: "none" }));
  out.push(L([30, 22].map((w, i) => k.lineOnTop([label.x + 4, label.y + 3 + i * 3], [label.x + 4 + w, label.y + 3 + i * 3], 0.8, P)).join(""), { tone: "lo", free: true }));
  out.push(D([at([label.x + label.w - 3, label.y + label.d / 2, 0.8])], { size: 0.45 }));
  return out.join("");
}

function screen() {
  const H = 19;
  const left = -H * 0.61;
  const base = CRT_MID + 2;
  const glyph = (points) => k.pathOf(points.map(([gx, gy]) => [left + gx * H, -(base + gy * H)]), true);
  const logo =
    glyph([[0.23, 0], [0.23, 1], [0.43, 1], [0.43, 0]]) +
    glyph([[0.83, 0], [0.83, 1], [1.03, 1], [1.03, 0]]) +
    glyph([[0.43, 0.42], [0.83, 0.42], [0.83, 0.58], [0.43, 0.58]]);
  const lines = [];
  for (let z = CRT.z0 + 1.2; z < CRT.z1; z += 1.6) lines.push(`M${CRT.x0} ${(-z).toFixed(1)}H${CRT.x1}`);
  const glow = flat(-30, CRT.z0 + 8, 30, CRT.z1 - 8, 26, 10);
  const textX = CRT.x0 + 8;
  const textZ = CRT.z0 + 14;
  const crt = [
    `<path class="pc-glow" d="${glow}"/>`,
    `<path class="pc-scan" d="${lines.join("")}"/>`,
    `<path class="pc-logo" d="${logo}"/>`,
    `<text class="pc-text" data-part="text" x="${textX}" y="${-textZ}" font-size="5.2">&gt; </text>`,
    `<rect class="pc-cursor" data-part="cursor" x="${textX + 6.6}" y="${-textZ - 4.2}" width="3" height="5"/>`,
  ].join("");
  const back = [
    `<g data-part="crt" transform="">${crt}</g>`,
    `<path class="pc-beam" data-part="beam" d="M${CRT.x0 + 4} ${-CRT_MID}H${CRT.x1 - 4}" style="opacity:0"/>`,
    `<circle class="pc-dot" data-part="dot" cx="0" cy="${-CRT_MID}" r="1.4" style="opacity:0"/>`,
    `<path class="iso-line" data-tone="lo" d="M${CRT.x0 + 6} ${-(CRT.z1 - 16)}Q${CRT.x0 + 7} ${-(CRT.z1 - 7)} ${CRT.x0 + 16} ${-(CRT.z1 - 6)}"/>`,
  ].join("");
  return recess(BEZEL, FRONT, "", "pc-bezel", recess(CRT, FRONT - BEZEL.depth, back, "pc-glass"));
}

function computer() {
  const out = [`<path class="iso-halo" d="${k.haloOf(CASE, 0, 8, P)}"/>`];
  out.push(S(k.slabOf(k.insetPlan(CASE, 3), 0, PLINTH_H, P, 8), { tone: "lo" }));
  out.push(S(k.slabOf(CASE, PLINTH_H, TOP - PLINTH_H, P, 8, 1.6), { tone: "mid" }));
  const handle = { x: -42, y: CASE.y + 7, w: 84, d: 11, r: 5.5 };
  out.push(L(k.planOutline(handle, TOP, P, 8), { tone: "lo" }));
  out.push(L(k.planOutline(k.insetPlan(handle, 2.2), TOP, P, 8), { tone: "faint" }));
  out.push(L(Array.from({ length: 7 }, (_, i) => k.planOutline({ x: -34, y: CASE.y + 27.3 + i * 3.4, w: 68, d: 1.4, r: 0.7 }, TOP, P, 4)).join(""), { tone: "faint" }));
  out.push(L(k.lineOnTop([CASE.x, SEAM], [SIDE, SEAM], TOP, P) + k.segment([SIDE, SEAM, TOP], [SIDE, SEAM, PLINTH_H], P), { tone: "lo" }));
  const rowOf = (from, to) => Array.from({ length: Math.floor((to - from) / 3.2 + 1e-9) + 1 }, (_, i) => from + i * 3.2);
  out.push(L(rowOf(CASE.y + 10, CASE.y + 52).map((y) => sideSlot(y, TOP - 48, TOP - 14)).join(""), { tone: "faint" }));
  out.push(L(rowOf(CASE.y + 10, CASE.y + 40).map((y) => sideSlot(y, 20, 30)).join(""), { tone: "faint" }));
  out.push(`<g transform="${planeOf(FRONT)}"><path class="iso-line" data-tone="faint" d="${flat(CASE.x + 4, PLINTH_H + 4, SIDE - 4, TOP - 4, 5)}"/></g>`);
  out.push(screen());
  out.push(recess(SLOT, FRONT, `<path class="iso-line" data-tone="lo" d="M${SLOT.x0} ${-(SLOT.z0 + 1.6)}H${SLOT.x1}"/>`, "pc-dark"));
  out.push(D([at([SLOT.x1 + 4, FRONT, SLOT.z0 - 1])], { size: 0.5, tone: "mid" }));
  out.push(`<g transform="${planeOf(FRONT)}">${L(Array.from({ length: 9 }, (_, i) => flat(-46.7 + i * 3, 42, -45.3 + i * 3, 54, 0.7, 4)).join(""), { tone: "lo" })}</g>`);
  out.push(`<g data-part="led">${D([at([-46, FRONT, 36])], { size: 0.8, tone: "hi" })}</g>`);
  out.push(rocker());
  out.push(port());
  return `<g class="pc-computer" data-part="computer">${out.join("")}</g>`;
}

function rocker() {
  const y = CASE.y + 8;
  const out = [S(k.slabOf({ x: SIDE, y, w: 1.2, d: 12, r: 1 }, 12, 15, P, 3), { tone: "mid" })];
  const half = (z, proud) => S(k.slabOf({ x: SIDE + 1.2, y: y + 1.5, w: proud ? 2.4 : 0.8, d: 9, r: 0.6 }, z, 5.6, P, 2), { tone: "mid" });
  out.push(`<g data-rocker="on">${half(13, true)}${half(20.4, false)}</g>`);
  out.push(`<g data-rocker="off" style="display:none">${half(20.4, true)}${half(13, false)}</g>`);
  return out.join("");
}

function port() {
  const out = [];
  out.push(S(k.slabOf({ x: SIDE - 0.4, y: PORT.y - 1, w: 1.4, d: 10, r: 1 }, PORT.z - 1, 9, P, 3), { tone: "lo" }));
  out.push(S(k.slabOf({ x: SIDE + 1, y: PORT.y, w: 4.4, d: 8, r: 1 }, PORT.z, 7, P, 3, 0.4), { tone: "mid" }));
  out.push(S(k.slabOf({ x: SIDE + 5.4, y: PORT.y + 2, w: 5, d: 4, r: 1.6 }, PORT.z + 1.5, 4, P, 4), { tone: "mid" }));
  out.push(D([at([SIDE + 3.2, PORT.y + 1.6, PORT.z + 7]), at([SIDE + 3.2, PORT.y + 6.4, PORT.z + 7])], { size: 0.4 }));
  return out.join("");
}

const keyList = [];
function keyboard() {
  const out = [`<path class="iso-halo" d="${k.haloOf(KB, 0, 4, P)}"/>`];
  for (const [x, y] of byDepth(k.corners(KB, 8))) out.push(S(k.cylinder(x, y, 2.8, 0, KB_Z, P, 16), { tone: "lo" }));
  out.push(S(k.slabOf(KB, KB_Z, KB_TOP - KB_Z, P, 6, 1), { tone: "mid" }));
  const well = { x: KEYS_X - 1.4, y: KEYS_Y - 1.4, w: U * 15 + 2.8, d: U * 5 + 2.8, r: 2 };
  out.push(L(k.planOutline(well, KB_TOP, P, 4), { tone: "lo" }));
  out.push(D([at([KB.x + 4, KB.y + 2.4, KB_TOP]), at([KB.x + KB.w - 4, KB.y + 2.4, KB_TOP])], { size: 0.45 }));
  ROWS.forEach((row, r) => {
    let x = KEYS_X;
    for (const [code, width] of row) {
      const plan = { x: x + 0.7, y: KEYS_Y + r * U + 0.7, w: width * U - 1.4, d: U - 1.4, r: 1.4 };
      const h = ROW_H[r];
      const index = keyList.length;
      keyList.push(code);
      const dish = code === " " ? k.planOutline(k.insetPlan(plan, 1.8), KB_TOP + h, P, 3) : k.planOutline(k.insetPlan(plan, 1.6), KB_TOP + h, P, 3);
      out.push(`<g class="pc-key" data-key="${index}">${S(k.slabOf(plan, KB_TOP, h, P, 3, 0.4), { tone: "mid", crease: "faint" })}${L(dish, { tone: "faint" })}</g>`);
      x += width * U;
    }
  });
  out.push(S(k.slabOf({ x: KB.x + KB.w, y: 90, w: 4, d: 4, r: 1.5 }, 2.5, 4, P, 4), { tone: "mid" }));
  return out.join("");
}

function cable() {
  const out = [];
  const curve = (points) => k.pathOf(points.map((point) => at(point)));
  const start = [COIL.x + COIL.r, COIL.from, COIL.z];
  const end = [COIL.x + COIL.r, COIL.to, COIL.z];
  out.push(`<path class="iso-line pc-cord" data-tone="mid" d="${curve([[SIDE + 10.4, PORT.y + 4, PORT.z + 3.5], [SIDE + 14, PORT.y + 6, PORT.z + 2.4], [SIDE + 17, PORT.y + 9.5, PORT.z - 1], [SIDE + 19.6, PORT.y + 13, COIL.z + 1.6], start])}"/>`);
  const spring = k.coil(COIL.from, COIL.to, [COIL.x, COIL.z], COIL.r, COIL.turns, "y", P);
  out.push(L(spring.back, { tone: "lo" }), L(spring.front, { tone: "mid" }));
  out.push(`<path class="iso-line pc-cord" data-tone="mid" d="${curve([end, [COIL.x + 2.6, COIL.to + 3, COIL.z + 0.2], [KB.x + KB.w + 6.4, 91.4, 4.4], [KB.x + KB.w + 4, 92, 4.5]])}"/>`);
  return out.join("");
}

const svg = k.figureSvg({
  width: WIDTH,
  height: HEIGHT,
  label:
    "An all-in-one desk computer in the spirit of 1984 on a bench plate: a tall case with a CRT set into its front showing the Hermes mark and a typing prompt, a floppy slot and speaker grille below, vents and a rocker switch on the side, and a keyboard joined to a side port by a coiled cord.",
  body: [plate(), computer(), keyboard(), cable()],
});

const PHRASES = ["hello, hermes", "what is a crt?", "draw it isometric"];

const live = `
const P = ${JSON.stringify(P)};
const KEYS = ${JSON.stringify(keyList)};
const PHRASES = ${JSON.stringify(PHRASES)};
const MAX_CHARS = 20, CRT_MID = ${CRT_MID}, TEXT_X = ${CRT.x0 + 8};
const SHIFTED = { "~": "\`", "!": "1", "@": "2", "#": "3", "$": "4", "%": "5", "^": "6", "&": "7", "*": "8", "(": "9", ")": "0", "_": "-", "+": "=", "{": "[", "}": "]", "|": "\\\\", ":": ";", '"': "'", "<": ",", ">": ".", "?": "/" };
const stage = document.querySelector(".iso-stage");
const svg = stage.querySelector("svg");
const keyEls = KEYS.map((_, i) => svg.querySelector('[data-key="' + i + '"]'));
const textEl = svg.querySelector('[data-part="text"]');
const cursor = svg.querySelector('[data-part="cursor"]');
const crt = svg.querySelector('[data-part="crt"]');
const beam = svg.querySelector('[data-part="beam"]');
const dot = svg.querySelector('[data-part="dot"]');
const glass = svg.querySelector(".pc-glass");
const led = svg.querySelector('[data-part="led"]');
const rockerOn = svg.querySelector('[data-rocker="on"]');
const rockerOff = svg.querySelector('[data-rocker="off"]');
const readout = document.querySelector("[data-readout]");
const status = document.querySelector("[data-status]");
const still = matchMedia("(prefers-reduced-motion: reduce)");
const DOWN = 2.2;
const down = axisVector(P, [0, 0, -1]);

const keyState = KEYS.map(() => ({ at: 0, held: 0 }));
const active = new Set();
let line = "", lastKey = "", power = 1, powerClock = 9, keysTyped = 0;
let frame = 0, previous = 0, visible = false, touring = !still.matches, idleTimer = 0;
let tour = { phrase: 0, step: "type", index: 0, wait: 1.4 };

const indexOf = (code) => KEYS.indexOf(code);
function press(code, hold = 0.09) {
  const i = indexOf(code);
  if (i < 0) return;
  keyState[i].held = hold;
  active.add(i);
  run();
}
function type(char) {
  if (!power) return;
  const lower = char.toLowerCase();
  const code = SHIFTED[char] || lower;
  if (indexOf(code) < 0) return;
  if (SHIFTED[char] || char !== lower) press("Shift", 0.14);
  press(code);
  keysTyped++;
  lastKey = char === " " ? "space" : char;
  if (line.length < MAX_CHARS) line += char;
  write();
}
function erase() { if (!power) return; press("Backspace"); lastKey = "backspace"; line = line.slice(0, -1); write(); }
function enter() { if (!power) return; press("Enter"); lastKey = "return"; line = ""; write(); }
function write() {
  textEl.textContent = "> " + line;
  cursor.setAttribute("x", String(TEXT_X + textEl.getComputedTextLength() + 0.4));
  const text = power ? "on · " + line.length + " chars" + (lastKey ? " · key " + lastKey : "") : "off";
  if (readout.textContent !== text) readout.textContent = text;
}
function setPower(next) {
  if (next === power) return;
  power = next;
  powerClock = still.matches ? 9 : 0;
  rockerOn.style.display = power ? "" : "none";
  rockerOff.style.display = power ? "none" : "";
  led.style.opacity = power ? "" : "0.15";
  status.textContent = power ? "Computer on." : "Computer off.";
  write();
  run();
}
const ease = (t) => 1 - Math.pow(1 - Math.min(1, Math.max(0, t)), 3);
function crtFrame() {
  const t = powerClock;
  let sx = 1, sy = 1, beamAt = 0, dotAt = 0, lit = power;
  if (power) {
    if (t < 0.14) { sx = ease(t / 0.14); sy = 0.012; beamAt = 1; }
    else if (t < 0.36) { sy = 0.012 + 0.988 * ease((t - 0.14) / 0.22); beamAt = 1 - ease((t - 0.14) / 0.22); }
  } else {
    if (t < 0.1) { sy = 1 - 0.988 * ease(t / 0.1); beamAt = ease(t / 0.1); lit = 1; }
    else if (t < 0.26) { sy = 0.012; sx = 1 - ease((t - 0.1) / 0.16); beamAt = 1; lit = 1; }
    else { sx = 0; sy = 0; dotAt = Math.max(0, 1 - (t - 0.26) / 0.5); }
  }
  crt.setAttribute("transform", "translate(0 " + -CRT_MID + ") scale(" + sx.toFixed(3) + " " + sy.toFixed(3) + ") translate(0 " + CRT_MID + ")");
  beam.setAttribute("transform", "translate(0 " + -CRT_MID + ") scale(" + Math.max(sx, 0.001).toFixed(3) + " 1) translate(0 " + CRT_MID + ")");
  beam.style.opacity = beamAt.toFixed(2);
  dot.style.opacity = dotAt.toFixed(2);
  glass.toggleAttribute("data-on", !!lit);
  return t > 0.8;
}
function tourStep(dt) {
  tour.wait -= dt;
  if (tour.wait > 0) return;
  const phrase = PHRASES[tour.phrase];
  if (tour.step === "type") {
    if (tour.index < phrase.length) { type(phrase[tour.index++]); tour.wait = 0.09 + Math.random() * 0.11 + (phrase[tour.index - 1] === " " ? 0.08 : 0); }
    else { tour.step = "erase"; tour.wait = 2.4; }
  } else if (tour.step === "erase") {
    if (line.length) { erase(); tour.wait = 0.055; }
    else if (tour.phrase === PHRASES.length - 1) { tour.step = "off"; tour.wait = 0.9; }
    else { tour.phrase++; tour.index = 0; tour.step = "type"; tour.wait = 0.7; }
  } else if (tour.step === "off") { setPower(0); tour.step = "on"; tour.wait = 1.6; }
  else if (tour.step === "on") { setPower(1); tour = { phrase: 0, step: "type", index: 0, wait: 1.4 }; }
}
function tick(now) {
  const dt = previous ? Math.min((now - previous) / 1000, 1 / 30) : 1 / 60;
  previous = now;
  const calm = still.matches;
  if (touring) tourStep(dt);
  powerClock += dt;
  const settledCrt = crtFrame();
  for (const i of active) {
    const s = keyState[i];
    s.held = Math.max(0, s.held - dt);
    const want = s.held > 0 ? DOWN : 0;
    s.at = calm ? want : s.at + (want - s.at) * (1 - Math.exp(-dt / (want ? 0.025 : 0.07)));
    if (Math.abs(want - s.at) < 0.01) s.at = want;
    keyEls[i].setAttribute("transform", "translate(" + (down[0] * s.at).toFixed(2) + " " + (down[1] * s.at).toFixed(2) + ")");
    keyEls[i].toggleAttribute("data-down", s.at > 0.6);
    if (!s.held && s.at === 0) active.delete(i);
  }
  const busy = touring || active.size || !settledCrt;
  frame = visible && busy ? requestAnimationFrame(tick) : 0;
}
function run() { if (!frame && visible) { previous = 0; frame = requestAnimationFrame(tick); } }
function takeOver(switching = false) {
  if (touring) { touring = false; tour = { phrase: 0, step: "type", index: 0, wait: 1.4 }; line = ""; if (!switching) setPower(1); write(); }
  clearTimeout(idleTimer);
  idleTimer = setTimeout(() => { if (still.matches) return; touring = true; line = ""; setPower(1); write(); run(); }, 9000);
}
new IntersectionObserver((entries) => { visible = entries[entries.length - 1].isIntersecting; if (visible) run(); }, { rootMargin: "120px 0px" }).observe(stage);
stage.addEventListener("keydown", (event) => {
  if (event.metaKey || event.ctrlKey || event.altKey) return;
  const key = event.key;
  if (key === "Tab") return;
  takeOver(key === "Escape");
  if (key === "Escape") { event.preventDefault(); setPower(power ? 0 : 1); return; }
  if (key === "Backspace") { event.preventDefault(); erase(); return; }
  if (key === "Enter") { event.preventDefault(); enter(); return; }
  if (key === "Shift") { press(event.code === "ShiftRight" ? "ShiftRight" : "Shift", 0.14); return; }
  if (key.length === 1) { event.preventDefault(); type(key); }
});
svg.addEventListener("click", (event) => {
  const key = event.target.closest("[data-key]");
  if (key) {
    takeOver();
    const code = KEYS[Number(key.dataset.key)];
    if (code.length === 1) type(code);
    else if (code === "Backspace") erase();
    else if (code === "Enter" || code === "NumpadEnter") enter();
    else press(code);
    stage.focus({ preventScroll: true });
    return;
  }
  if (event.target.closest('[data-part="computer"]')) { takeOver(true); setPower(power ? 0 : 1); stage.focus({ preventScroll: true }); }
});
write();
crtFrame();
`;

const css = `<style>
.pc-well{fill:var(--anatomy-shade-1)}
.pc-bezel{fill:var(--anatomy-top)}
.pc-dark{fill:var(--anatomy-shade-0)}
.pc-glass{fill:var(--anatomy-shade-0);transition:fill .5s var(--anatomy-ease)}
.pc-glass[data-on]{fill:var(--anatomy-lit-shade)}
.pc-glow{fill:var(--anatomy-lit);opacity:.07;filter:blur(6px)}
.pc-scan{fill:none;stroke:var(--anatomy-faint);stroke-width:.5px;vector-effect:non-scaling-stroke;opacity:.7}
.pc-logo{fill:var(--anatomy-lit);fill-rule:evenodd}
.pc-text{fill:var(--anatomy-lit);font-family:var(--anatomy-mono);letter-spacing:.02em}
.pc-cursor{fill:var(--anatomy-lit);animation:pc-blink 1.06s steps(1,end) infinite}
.pc-beam{fill:none;stroke:var(--anatomy-lit);stroke-width:1.4px;vector-effect:non-scaling-stroke;stroke-linecap:round}
.pc-dot{fill:var(--anatomy-lit)}
.pc-cord{stroke-width:1.1px}
.pc-key,.pc-computer{cursor:pointer}
.pc-key[data-down] .iso-edge{stroke:var(--anatomy-hi)}
[data-part="led"]{transition:opacity .4s var(--anatomy-ease)}
@keyframes pc-blink{50%{opacity:0}}
@media (prefers-reduced-motion:reduce){.pc-cursor{animation:none}}
.pc-sr{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%);white-space:nowrap}
.iso-stage{touch-action:pan-y}
.iso-stage:focus{outline:none}
.iso-stage:focus-visible{outline:1px solid var(--anatomy-muted);outline-offset:6px}
</style>`;

const body =
  css +
  k.plateHtml({
    fig: "Fig 2",
    title: "Desk computer",
    hint: "Type on it · click it to switch",
    readout: "on · 0 chars",
    keys: [
      { mark: "lit", label: "The screen: the Hermes mark and what you type" },
      { mark: "raised", label: "Fifty-eight keys; each one goes down as it is typed" },
      { mark: "edge", label: "A coiled cord from the keyboard to the side port" },
    ],
    caption:
      "An all-in-one desk computer in the spirit of 1984: a CRT set into a tall case, a floppy slot and speaker grille under it, vents and a rocker switch on the side. Focus it and type: each key you press goes down on the keyboard and the character lands on the screen beneath the Hermes mark. Backspace and Return work, Escape or a click on the case switches it off and the picture collapses to a line and a dot, as a CRT does.",
    body: `<div class="iso-stage" tabindex="0" role="group" aria-label="Desk computer. Type to write on its screen; Escape switches the power.">${svg}<span class="pc-sr" role="status" aria-live="polite" data-status></span></div>`,
  });

const name = theme === "light" ? "desk-computer-light.html" : "desk-computer.html";
writeFileSync(join(HERE, name), k.pageHtml({ title: "Desk computer", theme, body, script: kitScript() + live }));
console.log("wrote", name);
