
const D = __DATA__;
const P = D.P;
const K = { iso, pathOf, solidSvg, lineSvg, dotsSvg };
const PARTS = makeParts(K, G, P, { stick: D.stick });
const stage = document.querySelector(".iso-stage");
const svg = stage.querySelector("svg");
const $ = (sel) => svg.querySelector(sel);
const $$ = (sel) => [...svg.querySelectorAll(sel)];
const at = (p) => iso(p, P);
const readout = document.querySelector("[data-readout]");
const status = document.querySelector("[data-status]");
const still = matchMedia("(prefers-reduced-motion: reduce)");

const el = {
  upper: $('[data-part="stick-upper"]'),
  lower: $('[data-part="stick-lower"]'),
  wire: $('[data-part="wire"]'),
  fall: $('[data-part="coin-fall"]'),
  chute: $('[data-part="coin-chute"]'),
  fly: $('[data-part="coin-fly"]'),
  pile: $('[data-part="pile"]'),
  screen: $('[data-part="screen"]'),
  raster: $('[data-part="raster"]'),
  dot: $('[data-part="dot"]'),
  test: $('[data-mode="test"]'),
  attract: $('[data-mode="attract"]'),
  steps: $$("[data-step]"),
  formation: $('[data-part="formation"]'),
  aliens: $$("[data-alien]"),
  ship: $('[data-part="ship"]'),
  shot: $('[data-part="shot"]'),
  boom: $('[data-part="boom"]'),
  score: $('[data-part="score"]'),
  insert: $('[data-part="insert"]'),
  start: $('[data-part="start"]'),
  credit: $('[data-part="credit"]'),
  meter: $('[data-part="meter"]'),
  glass: $('[data-part="glass"]'),
  stack: $$("[data-stack]"),
  fire: $('[data-button="fire"]'),
  p1: $('[data-button="p1"]'),
  lamps: $$(".ac-lamp"),
  leds: $$(".ac-led"),
  marquee: $$('[data-name="marquee.tube"], [data-name="marquee.plexi"], .ac-marquee'),
};
const routes = {};
for (const g of $$("[data-route]")) (routes[g.dataset.route] ??= []).push(g);
for (const list of Object.values(routes)) list.sort((a, b) => a.dataset.chunk - b.dataset.chunk);

const smooth = (a, b, x) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};
const lerp3 = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
const norm = (v) => {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
};

const GRAV = 1962;
const FLY = 0.9;
const INSERT = 0.34;
const MECH = 0.42;
const DROP_FROM = D.mech.z0 + PARTS.COIN_R;
const FLOOR = D.box.z0 + 0.7 + PARTS.COIN_T / 2;
const HEATER = 2.2;
const TEST = [1.0, 1.7, 2.3, 2.8];
const ATTRACT_AT = 3.6;
const HKHZ = 15.734;

const state = {
  power: 0,
  powerAt: -99,
  offAt: -99,
  credits: 0,
  meter: 4127,
  coins: [],
  pile: 0,
  stack: D.tray.count,
  stick: { x: 0, target: 0 },
  fire: 0,
  fireHeld: 0,
  pulses: [],
  wire: { x: 0, v: 0, push: 0 },
  game: null,
  clock: 0,
};

function newGame() {
  return { step: 0, dir: 1, offset: 0, frame: 0, alive: D_FORMATION.map(() => true), shipX: 112, shot: null, boom: null, score: 0, cool: 1.2, userAt: -99, respawn: 0 };
}
const D_FORMATION = SCREEN.FORMATION;

const STACK_AT = (n) => [D.tray.x, D.tray.y, 1.6 + (n - 1) * D.tray.step + 0.2];
const SLOT_OUT = [D.slot.x, D.slot.y + 9, D.slot.z];
const SLOT_IN = [D.slot.x, D.slot.y - 12, D.slot.z];
const CH = D.chute;
const SN = 0.5;
const CS = Math.sqrt(3) / 2;
const chuteZ = (y) => CH.z0 - CH.slope * (CH.y0 - y);
const ROLL_FROM = D.mech.y0;
const ROLL_A = (2 / 3) * GRAV * SN;
const ROLL_LEN = (ROLL_FROM - CH.y1) / CS;
const T_ROLL = Math.sqrt((2 * ROLL_LEN) / ROLL_A);
const V_END = ROLL_A * T_ROLL;
const END = [D.slot.x, CH.y1 - PARTS.COIN_R * SN, chuteZ(CH.y1) + PARTS.COIN_R * CS];
const EDGE_Z = FLOOR + PARTS.COIN_R;
const T_DROP = (-V_END * SN + Math.sqrt((V_END * SN) ** 2 + 2 * GRAV * (END[2] - EDGE_Z))) / GRAV;
const LAND_Y = END[1] - V_END * CS * T_DROP;
function insertCoin() {
  if (state.stack <= 0) state.stack = D.tray.count;
  const from = STACK_AT(state.stack);
  state.stack--;
  showStack();
  state.coins.push({ t: 0, from, landAt: pileSpot(state.pile + state.coins.length), switched: false, landed: false });
  run();
}
function pileSpot(i) {
  const a = i * 2.39996;
  const r = Math.min(5, 1.1 * Math.sqrt(i));
  const lo = D.box.y0 + 0.7 + PARTS.COIN_R + 0.3;
  const hi = D.box.y1 - 0.7 - PARTS.COIN_R - 0.3;
  return [D.slot.x + Math.cos(a) * r * 1.2, Math.max(lo, Math.min(hi, LAND_Y + Math.sin(a) * r)), FLOOR + Math.floor(i / 12) * PARTS.COIN_T];
}
function coinPose(c) {
  let t = c.t;
  if (t < FLY) {
    const s = smooth(0, 1, t / FLY);
    const mid = [(c.from[0] + SLOT_OUT[0]) / 2, (c.from[1] + SLOT_OUT[1]) / 2, Math.max(c.from[2], SLOT_OUT[2]) + 26];
    const a = lerp3(c.from, mid, s);
    const b = lerp3(mid, SLOT_OUT, s);
    const p = lerp3(a, b, s);
    const turn = smooth(0.15, 0.85, t / FLY);
    return { layer: "fly", p, axis: norm(lerp3([0, 0, 1], [1, 0, 0], turn)), spin: s * 7 };
  }
  t -= FLY;
  if (t < INSERT) return { layer: "fly", clip: "door", p: lerp3(SLOT_OUT, SLOT_IN, smooth(0, 1, t / INSERT)), axis: [1, 0, 0], spin: 7 };
  t -= INSERT;
  if (t < MECH) return { layer: "none", stage: "mech" };
  t -= MECH;
  if (t < T_ROLL) {
    const s = 0.5 * ROLL_A * t * t;
    const y = ROLL_FROM - s * CS;
    const p = [D.slot.x, y - PARTS.COIN_R * SN, chuteZ(y) + PARTS.COIN_R * CS];
    return { layer: "chute", clip: p[1] + PARTS.COIN_R > D.mech.y0 ? "mech" : null, p, axis: [1, 0, 0], spin: 7 + s / PARTS.COIN_R, stage: "roll" };
  }
  t -= T_ROLL;
  const land = c.landAt;
  if (t < T_DROP) {
    const share = t / T_DROP;
    const p = [END[0] + (land[0] - END[0]) * share, END[1] - V_END * CS * t, END[2] - V_END * SN * t - 0.5 * GRAV * t * t];
    p[1] = END[1] + (land[1] - END[1]) * share;
    return { layer: "fall", p, axis: [1, 0, 0], spin: 7 + ROLL_LEN / PARTS.COIN_R + t * 20, stage: "drop" };
  }
  const tl = t - T_DROP;
  const tip = smooth(0, 0.16, tl);
  const lie = [land[0], land[1], land[2] + PARTS.COIN_R * (1 - tip)];
  return { layer: "fall", p: lie, axis: norm(lerp3([1, 0, 0], [0, 0, 1], tip)), spin: 7, settled: tl > 0.22, stage: "drop" };
}

const CLIP_DOOR = (() => {
  const a = at([D.slot.x, D.slot.y, -100]);
  const b = at([D.slot.x, D.slot.y, 300]);
  return `M${a[0]} ${a[1]}L${b[0]} ${b[1]}L${b[0] + 400} ${b[1]}L${a[0] + 400} ${a[1]}Z`;
})();
const CLIP_MECH = (() => {
  const a = at([D.slot.x, D.mech.y0, -100]);
  const b = at([D.slot.x, D.mech.y0, 300]);
  return `M${a[0]} ${a[1]}L${b[0]} ${b[1]}L${b[0] - 400} ${b[1]}L${a[0] - 400} ${a[1]}Z`;
})();
svg.querySelector("defs") || svg.insertAdjacentHTML("afterbegin", "<defs></defs>");
svg.querySelector("defs").insertAdjacentHTML("beforeend", `<clipPath id="ac-clip-door"><path d="${CLIP_DOOR}"/></clipPath><clipPath id="ac-clip-mech"><path d="${CLIP_MECH}"/></clipPath>`);

function drawCoins() {
  const out = { fly: "", chute: "", fall: "" };
  for (const c of state.coins) {
    const pose = coinPose(c);
    if (pose.layer === "none") continue;
    const coin = PARTS.coin(pose.p, pose.axis, pose.spin);
    out[pose.layer] += pose.clip ? `<g clip-path="url(#ac-clip-${pose.clip})">${coin}</g>` : coin;
  }
  for (const key of ["fly", "chute", "fall"]) if (el[key].innerHTML !== out[key]) el[key].innerHTML = out[key];
}
function showStack() {
  el.stack.forEach((g, i) => (g.style.display = i < state.stack ? "" : "none"));
}
function drawPile() {
  let out = "";
  for (let i = 0; i < state.pile; i++) out += PARTS.coin(pileSpot(i), [0, 0, 1], i * 1.7, { lit: false });
  el.pile.innerHTML = out;
}

function wireStep(dt, push) {
  const w = state.wire;
  if (push > w.x) {
    w.x = push;
    w.v = 0;
  } else {
    const k = 900;
    const c = 9;
    for (let i = 0; i < 4; i++) {
      const h = dt / 4;
      w.v += (-k * w.x - c * w.v) * h;
      w.x += w.v * h;
    }
    if (Math.abs(w.x) < 1e-3 && Math.abs(w.v) < 1e-2) (w.x = 0), (w.v = 0);
  }
  const from = D.wire.from;
  const angle = Math.atan2(w.x, D.wire.length);
  const tip = [from[0] + D.wire.length * Math.cos(angle), from[1], from[2] - D.wire.length * Math.sin(angle)];
  el.wire.innerHTML = lineSvg(pathOf([at(from), at(tip)]), { tone: w.x > 0.3 ? "lit" : "hi", free: "end" });
  return w.x !== 0;
}
function wirePush() {
  let push = 0;
  const wz = D.wire.from[2];
  const wy = D.wire.from[1];
  for (const c of state.coins) {
    const pose = coinPose(c);
    if (pose.stage !== "roll") continue;
    const dy = wy - pose.p[1];
    if (Math.abs(dy) >= PARTS.COIN_R) continue;
    const low = pose.p[2] - Math.sqrt(PARTS.COIN_R ** 2 - dy * dy);
    push = Math.max(push, Math.min(1.2, wz - low));
  }
  return Math.max(0, push) * (D.wire.length / (D.slot.x - D.wire.from[0]));
}
function pulse(route, delay = 0, seconds = 0.5) {
  if (!routes[route]) return;
  state.pulses.push({ route, t: -delay, seconds });
}
function drawPulses(dt) {
  const hot = new Set();
  state.pulses = state.pulses.filter((p) => {
    p.t += dt;
    if (p.t < 0) return true;
    const list = routes[p.route];
    const at = (p.t / p.seconds) * (list.length + 1);
    list.forEach((g, i) => Math.abs(i - at + 0.5) < 1.1 && hot.add(g));
    return p.t < p.seconds * 1.25;
  });
  for (const list of Object.values(routes)) for (const g of list) g.toggleAttribute("data-hot", hot.has(g));
  return state.pulses.length > 0;
}

function setPower(on) {
  if (!!on === !!state.power) return;
  state.power = on ? 1 : 0;
  if (on) {
    state.powerAt = state.clock;
    state.game = newGame();
  } else {
    state.offAt = state.clock;
    state.credits = 0;
  }
  el.leds.forEach((g) => g.toggleAttribute("data-on", !!on));
  el.lamps.forEach((g) => g.toggleAttribute("data-on", !!on));
  status.textContent = on ? "Coin accepted. The cabinet wakes and the CRT warms up." : "Switched off.";
  run();
}
function marqueeLevel(t) {
  if (!state.power) return 0;
  if (t < 0.06) return 0;
  if (t < 0.12) return 1;
  if (t < 0.24) return 0;
  if (t < 0.3) return 1;
  if (t < 0.42) return 0.2;
  return 1;
}
function picture() {
  const t = state.clock - state.powerAt;
  const off = state.clock - state.offAt;
  let bright = 0;
  let sx = 1;
  let sy = 1;
  let dot = 0;
  let mode = "none";
  if (state.power) {
    bright = smooth(0.9, HEATER + 0.4, t);
    mode = t < ATTRACT_AT ? (t > 0.9 ? "test" : "none") : t < ATTRACT_AT + 0.08 ? "none" : "attract";
  } else if (off < 1.2 && state.offAt > 0) {
    bright = 1;
    mode = state.lastMode;
    if (off < 0.09) sx = 1 - 0.99 * smooth(0, 0.09, off);
    else if (off < 0.26) (sx = 0.01), (sy = 1 - smooth(0.09, 0.26, off));
    else (sx = 0), (sy = 0);
    dot = off < 0.2 ? 0 : 1 - smooth(0.26, 1.1, off);
  }
  if (state.power) state.lastMode = mode;
  el.raster.setAttribute("transform", `scale(${Math.max(sx, 0.001).toFixed(3)} ${Math.max(sy, 0.004).toFixed(3)})`);
  el.raster.style.opacity = (sx === 0 ? 0 : bright).toFixed(3);
  el.dot.style.opacity = dot.toFixed(3);
  el.test.style.display = mode === "test" ? "" : "none";
  el.attract.style.display = mode === "attract" ? "" : "none";
  el.steps.forEach((s, i) => (s.style.display = t > TEST[i] ? "" : "none"));
  el.glass.toggleAttribute("data-on", bright > 0.05 && sx > 0);
  const m = marqueeLevel(t);
  el.marquee.forEach((g) => g.toggleAttribute("data-on", m > 0.5));
  return { mode, t, busy: state.power ? t < ATTRACT_AT + 0.2 : off < 1.2 };
}

const SHOT_SPEED = 220;
function gameStep(dt, mode) {
  const g = state.game;
  if (!g || mode !== "attract") return;
  g.step += dt;
  if (g.step > 0.32) {
    g.step = 0;
    g.offset += g.dir * 2;
    if (Math.abs(g.offset) >= 16) g.dir = -g.dir;
    g.frame = 1 - g.frame;
    el.formation.setAttribute("transform", `translate(${g.offset} 0)`);
    for (const a of el.aliens) {
      a.children[0].style.display = g.frame ? "none" : "";
      a.children[1].style.display = g.frame ? "" : "none";
    }
  }
  const user = state.clock - g.userAt < 3;
  if (user) g.shipX += state.stick.x * 420 * dt;
  else {
    const live = D_FORMATION.map((a, i) => (g.alive[i] ? a.x + 5 + g.offset : null)).filter((x) => x !== null);
    const target = live.length ? live[Math.floor((state.clock * 0.37) % live.length)] : 112;
    g.shipX += Math.max(-50 * dt, Math.min(50 * dt, target - g.shipX));
  }
  g.shipX = Math.max(14, Math.min(210, g.shipX));
  el.ship.setAttribute("transform", `translate(${g.shipX.toFixed(1)} ${SCREEN.SHIP_Y})`);
  g.cool -= dt;
  if (!g.shot && ((state.fire > 0 && user) || (!user && g.cool < 0))) {
    g.shot = { x: Math.round(g.shipX), y: SCREEN.SHIP_Y - 6 };
    g.cool = 0.9 + Math.random() * 0.8;
  }
  if (g.shot) {
    g.shot.y -= SHOT_SPEED * dt;
    let hit = -1;
    D_FORMATION.forEach((a, i) => {
      if (!g.alive[i] || hit >= 0) return;
      const x = a.x + g.offset;
      if (g.shot.x >= x && g.shot.x <= x + 10 && g.shot.y <= a.y + 8 && g.shot.y + 5 >= a.y) hit = i;
    });
    if (hit >= 0) {
      g.alive[hit] = false;
      el.aliens[hit].style.display = "none";
      const a = D_FORMATION[hit];
      g.boom = { x: a.x + g.offset, y: a.y, t: 0.22 };
      g.score += a.kind === "eye" ? 30 : 10;
      el.score.setAttribute("d", SCREEN.text(String(g.score).padStart(5, "0"), 16, 20));
      g.shot = null;
    } else if (g.shot.y < 30) g.shot = null;
  }
  el.shot.style.display = g.shot ? "" : "none";
  if (g.shot) el.shot.setAttribute("transform", `translate(${g.shot.x} ${g.shot.y.toFixed(1)})`);
  if (g.boom) {
    g.boom.t -= dt;
    el.boom.style.display = g.boom.t > 0 ? "" : "none";
    el.boom.setAttribute("transform", `translate(${g.boom.x} ${g.boom.y})`);
    if (g.boom.t <= 0) g.boom = null;
  }
  if (g.alive.every((a) => !a)) {
    g.respawn += dt;
    if (g.respawn > 1.2) {
      g.respawn = 0;
      g.alive = g.alive.map(() => true);
      el.aliens.forEach((a) => (a.style.display = ""));
    }
  }
  const blink = Math.floor(state.clock / 0.5) % 2 === 0;
  el.insert.style.display = !state.credits && blink ? "" : "none";
  el.start.style.display = state.credits && blink ? "" : "none";
  el.p1?.toggleAttribute("data-blink", state.credits > 0 && blink);
}
let creditShown = -1;
function showCredit() {
  if (creditShown === state.credits) return;
  creditShown = state.credits;
  el.credit.setAttribute("d", SCREEN.text(`CREDIT ${Math.min(9, state.credits)}`, 120, 274));
}

const TILT = 0.21;
let stickShown = null;
function stickStep(dt, calm) {
  const s = state.stick;
  const before = s.x;
  s.x = calm ? s.target : s.x + (s.target - s.x) * (1 - Math.exp(-dt / 0.05));
  if (Math.abs(s.x - s.target) < 1e-3) s.x = s.target;
  const key = s.x.toFixed(3);
  if (key !== stickShown) {
    stickShown = key;
    el.upper.innerHTML = PARTS.stickUpper(s.x * TILT, 0);
    el.lower.innerHTML = PARTS.stickLower(s.x * TILT, 0);
  }
  return s.x !== s.target || before !== s.x;
}
const firePush = axisVector(P, D.stick.nc.map((v) => -v * 1.1));
function fireStep(dt) {
  state.fireHeld = Math.max(0, state.fireHeld - dt);
  const want = state.fireHeld > 0 ? 1 : 0;
  state.fire += (want - state.fire) * (1 - Math.exp(-dt / 0.03));
  if (Math.abs(want - state.fire) < 0.01) state.fire = want;
  el.fire.setAttribute("transform", state.fire ? `translate(${(firePush[0] * state.fire).toFixed(2)} ${(firePush[1] * state.fire).toFixed(2)})` : "");
  el.fire.toggleAttribute("data-down", state.fire > 0.5);
  return state.fire !== want || state.fireHeld > 0;
}

let phase = "";
function say(mode, t) {
  let text;
  const coin = state.coins.find((c) => !c.landed);
  if (coin) {
    const tt = coin.t;
    text = tt < FLY ? "coin · 24.26 mm quarter" : tt < FLY + INSERT ? "coin · into the slot" : tt < FLY + INSERT + MECH ? "mech · checking the coin" : coin.switched ? `coin switch · meter ${String(state.meter).padStart(6, "0")}` : "coin · rolling down the chute";
  } else if (!state.power) text = state.clock - state.offAt < 1.2 && state.offAt > 0 ? "off · picture folds to a line" : `off · ${state.pile ? "insert another coin" : "insert coin"}`;
  else if (mode === "test") text = t < TEST[1] ? "self-test" : t < TEST[2] ? "self-test · RAM OK" : "self-test · RAM OK · ROM OK";
  else if (mode !== "attract") text = `warming · heater ${Math.min(HEATER, t).toFixed(1)} s`;
  else {
    const dir = state.stick.target < 0 ? " · stick ←" : state.stick.target > 0 ? " · stick →" : state.fireHeld > 0 ? " · fire" : "";
    text = `credit ${state.credits} · ${HKHZ} kHz${dir}`;
  }
  if (readout.textContent !== text) readout.textContent = text;
  const now = !state.power ? "off" : mode;
  if (now !== phase) {
    phase = now;
    if (now === "attract") status.textContent = `Attract mode, ${state.credits} credit${state.credits === 1 ? "" : "s"}.`;
    if (now === "test") status.textContent = "Self-test.";
  }
}

let frame = 0;
let last = 0;
let visible = false;
let touring = !still.matches;
let tourClock = 0;
let idle = 0;
const TOUR = [
  [1.4, () => insertCoin()],
  [9.5, () => press("right", 0.5)],
  [10.4, () => press("left", 0.6)],
  [11.4, () => fireButton()],
  [12.6, () => press("right", 0.35)],
  [13.3, () => fireButton()],
  [15.2, () => insertCoin()],
  [24, () => setPower(0)],
  [27.5, null],
];
let tourIndex = 0;
function tourStep(dt) {
  tourClock += dt;
  while (tourIndex < TOUR.length && tourClock >= TOUR[tourIndex][0]) {
    const action = TOUR[tourIndex][1];
    tourIndex++;
    if (action) action();
    else {
      tourClock = 0;
      tourIndex = 0;
      state.pile = 0;
      state.stack = D.tray.count;
      showStack();
      drawPile();
    }
  }
}
let releaseTimer = 0;
function press(dir, seconds) {
  state.stick.target = dir === "left" ? -1 : 1;
  if (state.game) state.game.userAt = state.clock;
  clearTimeout(releaseTimer);
  releaseTimer = setTimeout(() => ((state.stick.target = 0), run()), seconds * 1000);
  run();
}
function fireButton() {
  state.fireHeld = 0.16;
  if (state.game) state.game.userAt = state.clock;
  run();
}

function tick(now) {
  const dt = last ? Math.min((now - last) / 1000, 1 / 30) : 1 / 60;
  last = now;
  const calm = still.matches;
  state.clock += dt;
  if (touring) tourStep(dt);
  for (const c of state.coins) {
    c.t += calm ? 10 : dt;
    const pose = coinPose(c);
    if (!c.switched && pose.stage === "roll" && pose.p[1] < D.wire.from[1]) {
      c.switched = true;
      state.meter++;
      el.meter.textContent = String(state.meter).padStart(6, "0");
      pulse("wire.coin", 0, 0.45);
      pulse("wire.meter", 0, 0.25);
      if (!state.power) {
        setTimeout(() => {
          setPower(1);
          pulse("cable.power", 0, 0.5);
          pulse("cable.video", 0.5, 0.5);
        }, calm ? 0 : 420);
      }
      setTimeout(() => {
        state.credits++;
        showCredit();
        run();
      }, calm ? 0 : 450);
    }
    if (!c.landed && (pose.settled || calm)) {
      c.landed = true;
      state.pile++;
      drawPile();
    }
  }
  state.coins = state.coins.filter((c) => !c.landed);
  drawCoins();
  const wireBusy = wireStep(dt, wirePush());
  const pulsing = drawPulses(dt);
  const { mode, t, busy } = picture();
  gameStep(dt, mode);
  showCredit();
  const sticking = stickStep(dt, calm);
  const firing = fireStep(dt);
  say(mode, t);
  const alive = touring || state.coins.length || wireBusy || pulsing || busy || sticking || firing || (state.power && !calm);
  frame = visible && alive ? requestAnimationFrame(tick) : 0;
}
function run() {
  if (!frame && visible) {
    last = 0;
    frame = requestAnimationFrame(tick);
  }
}
function takeOver() {
  if (touring) {
    touring = false;
  }
  clearTimeout(idle);
  idle = setTimeout(() => {
    if (still.matches) return;
    touring = true;
    tourClock = state.power ? 4 : 0;
    tourIndex = TOUR.findIndex(([t]) => t > tourClock);
    if (state.power) tourIndex = TOUR.findIndex(([t]) => t > 9);
    tourClock = state.power ? 9 : 0;
    run();
  }, 12000);
}
new IntersectionObserver((entries) => {
  visible = entries[entries.length - 1].isIntersecting;
  if (visible) run();
}, { rootMargin: "120px 0px" }).observe(stage);

stage.addEventListener("keydown", (event) => {
  if (event.metaKey || event.ctrlKey || event.altKey) return;
  const key = event.key;
  if (key === "c" || key === "C" || key === "5") {
    event.preventDefault();
    takeOver();
    insertCoin();
  } else if (key === "ArrowLeft" || key === "ArrowRight") {
    event.preventDefault();
    takeOver();
    state.stick.target = key === "ArrowLeft" ? -1 : 1;
    if (state.game) state.game.userAt = state.clock;
    clearTimeout(releaseTimer);
    run();
  } else if (key === " " || key === "Enter") {
    event.preventDefault();
    takeOver();
    state.fireHeld = 0.5;
    if (state.game) state.game.userAt = state.clock;
    run();
  } else if (key === "Escape") {
    event.preventDefault();
    takeOver();
    setPower(0);
  }
});
stage.addEventListener("keyup", (event) => {
  if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
    state.stick.target = 0;
    run();
  }
  if (event.key === " " || event.key === "Enter") state.fireHeld = Math.min(state.fireHeld, 0.06);
});
svg.addEventListener("click", (event) => {
  const hit = event.target.closest('[data-part="tray"], [data-name^="door"]');
  if (hit) {
    takeOver();
    insertCoin();
    stage.focus({ preventScroll: true });
  }
});

showStack();
drawPile();
showCredit();
wireStep(0, 0);
picture();
say("none", 0);
if (still.matches) touring = false;
