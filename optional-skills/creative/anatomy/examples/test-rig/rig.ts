export type Pair = [number, number];
export type Tuning = readonly [number, number];

export interface Spring {
  at: number;
  rate: number;
}

export interface Member {
  swell: Spring;
  lean: Spring;
}

export interface Pose {
  mid: number;
  half: Pair;
  lean: number;
  press: number;
}

export const HALF: Pair = [60, 16];
export const SWELL = 0.08;
export const SWELL_MOST = 6;
export const SWELL_FLOOR = -0.25;
export const RUBBER = 0.55;
const STRETCH = 0.35;
export const STRETCH_MOST = 0.08;
const THIN = 0.5;
const SOFT = 1;
export const GIVE = 14;
export const REACH = 0.15;
export const GIVE_MOST = 0.25;
const QUIET = 0.0005;
const SUB_RATE = 240;
export const FRAME = 1 / 60;

export const PRESS_SPRING: Tuning = [700, 32];
export const HOLD_SPRING: Tuning = [540, 40];
export const RELEASE_SPRING: Tuning = [320, 16];

const LONGEST = Math.max(HALF[0], HALF[1]);
export const SHARE = Math.min(SWELL, SWELL_MOST / LONGEST);
export const GIVEN = Math.min(GIVE, REACH * 2 * LONGEST);
export const ACROSS = (GIVEN * HALF[1]) / LONGEST;

export const zeta = ([stiff, drag]: Tuning) => drag / (2 * Math.sqrt(stiff));

function spring(held: Spring, want: number, dt: number, [stiff, drag]: Tuning, calm: boolean) {
  const damping = calm ? 2 * Math.sqrt(stiff) : drag;
  const steps = Math.max(Math.ceil(dt * SUB_RATE), 1);
  const slice = dt / steps;
  for (let step = 0; step < steps; step++) {
    held.rate += (want - held.at) * stiff * slice - held.rate * damping * slice;
    held.at += held.rate * slice;
  }
  if (Math.abs(want - held.at) < QUIET && Math.abs(held.rate) < QUIET) {
    held.at = want;
    held.rate = 0;
  }
}

export function rubberBand(distance: number, limit: number) {
  return limit <= 0 ? 0 : limit * (1 - 1 / (1 + (RUBBER * distance) / limit));
}

const softened = (value: number) => Math.hypot(value, SOFT) - SOFT;

function stretched(lean: number, half: number, grow: number) {
  const most = STRETCH_MOST * grow * half;
  return most * Math.tanh((STRETCH * softened(lean)) / most);
}

export const grownOf = (member: Member) => 1 + SHARE * Math.max(member.swell.at, SWELL_FLOOR);

export const memberAt = (swell = 0, lean = 0): Member => ({ swell: { at: swell, rate: 0 }, lean: { at: lean, rate: 0 } });

export function step(member: Member, held: boolean, pull: number, dt: number, calm = false) {
  spring(member.swell, held ? 1 : 0, dt, held ? PRESS_SPRING : RELEASE_SPRING, calm);
  const want = held && !calm ? Math.sign(pull) * rubberBand(Math.abs(pull), GIVEN) : 0;
  spring(member.lean, want, dt, held ? HOLD_SPRING : RELEASE_SPRING, calm);
}

export function poseOf(member: Member): Pose {
  const grow = grownOf(member);
  const lean = member.lean.at;
  const stretch = stretched(lean, HALF[0], grow);
  const ratio = 1 + stretch / (grow * HALF[0]);
  return {
    mid: lean - Math.sign(lean) * stretch,
    half: [grow * HALF[0] * ratio, grow * HALF[1] * Math.pow(ratio, -THIN)],
    lean,
    press: Math.min(Math.max(member.swell.at, 0), 1.2),
  };
}

export const carried = (pose: Pose, x: number) => pose.mid + (pose.half[0] / HALF[0]) * x;

export const settled = (member: Member, held: boolean, pull: number) =>
  member.swell.rate === 0 && member.lean.rate === 0 && member.swell.at === (held ? 1 : 0) && (held || member.lean.at === 0) && (held || pull === 0);

export const one = (value: number) => {
  const text = (Math.round(value * 10) / 10).toFixed(1);
  return text === "-0.0" ? "0.0" : text;
};

export const two = (value: number) => (Math.round(value * 100) / 100).toFixed(2);

export const sizeText = (pose: Pose) => `${one(pose.half[0] * 2)} × ${one(pose.half[1] * 2)}`;

export function steady(pull: number) {
  const member = memberAt();
  for (let frame = 0; frame < 360; frame++) step(member, true, pull, FRAME);
  return poseOf(member);
}

function pressRun() {
  const member = memberAt();
  let peak = 0;
  let peakAt = 0;
  for (let frame = 1; frame <= 120; frame++) {
    step(member, true, 0, FRAME);
    const tall = poseOf(member).half[1] * 2;
    if (tall > peak) {
      peak = tall;
      peakAt = frame * FRAME;
    }
  }
  return { peak, peakAt };
}

export const STILL_ENOUGH = 0.1;

function releaseRun(pull: number) {
  const member = memberAt();
  for (let frame = 0; frame < 360; frame++) step(member, true, pull, FRAME);
  let swing = 0;
  let swingAt = 0;
  let dip = 0;
  let still = 0;
  for (let frame = 1; frame <= 300; frame++) {
    step(member, false, 0, FRAME);
    const lean = member.lean.at;
    if (-Math.sign(pull) * lean > swing) {
      swing = -Math.sign(pull) * lean;
      swingAt = frame * FRAME;
    }
    dip = Math.max(dip, (1 - grownOf(member)) * 100);
    const pose = poseOf(member);
    const edges = [pose.mid - pose.half[0], pose.mid + pose.half[0], -pose.half[1], pose.half[1]];
    const rest = [-HALF[0], HALF[0], -HALF[1], HALF[1]];
    const off = Math.max(...edges.map((edge, index) => Math.abs(edge - rest[index]!)));
    if (off > STILL_ENOUGH) still = (frame + 1) * FRAME;
  }
  return { swing, swingAt, dip, still };
}

export const DRAG = 80;
export const PULL_MOST = 110;
export const REST: Pose = { mid: 0, half: [HALF[0], HALF[1]], lean: 0, press: 0 };
export const PRESSED = steady(0);
export const PULLED = steady(-DRAG);
export const PULLED_MOST = steady(-PULL_MOST);
export const PRESS_RUN = pressRun();
export const RELEASE_RUN = releaseRun(-DRAG);
export const DIP_MOST = -SWELL_FLOOR * SHARE * 100;
export const SIDE_GROWTH = SHARE * HALF[0];

export function readoutOf(pose: Pose) {
  return `${sizeText(pose)} · lean ${one(Math.abs(pose.lean))} px`;
}
