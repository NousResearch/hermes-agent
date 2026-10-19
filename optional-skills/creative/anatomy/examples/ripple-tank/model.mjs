export const WATER = { depth: 0.006, gravity: 9.81, tension: 0.0728, density: 998, index: 1.333, viscosity: 1.0e-6 };
export const GLASS = 1.52;
export const RANGE = { frequency: [8, 28], spacing: [20, 80] };
export const BEAT = 0.75;
export const SLOPE = 0.03;
export const REACH = 24;

export function omegaOf(k) {
  const { depth, gravity, tension, density } = WATER;
  return Math.sqrt((gravity * k + (tension / density) * k * k * k) * Math.tanh(k * depth));
}

export function wavenumberOf(frequency) {
  const omega = 2 * Math.PI * frequency;
  let low = 1;
  let high = 20000;
  for (let step = 0; step < 60; step++) {
    const mid = (low + high) / 2;
    if (omegaOf(mid) < omega) low = mid;
    else high = mid;
  }
  return (low + high) / 2;
}

export function waveOf(frequency) {
  const k = wavenumberOf(frequency);
  const lambda = (2 * Math.PI) / k;
  const speed = (2 * Math.PI * frequency) / k;
  const h = 1e-3 * k;
  const group = (omegaOf(k + h) - omegaOf(k - h)) / (2 * h);
  const { viscosity, depth } = WATER;
  const omega = 2 * Math.PI * frequency;
  const damping = 2 * viscosity * k * k + (k * Math.sqrt((viscosity * omega) / 2)) / Math.sinh(2 * k * depth);
  return { frequency, k, lambda, speed, group, damping, decay: damping / group };
}

export function nodalCount(spacing, lambda) {
  return 2 * Math.floor(spacing / lambda + 0.5);
}

export function nodalVertices(spacing, lambda) {
  const out = [];
  for (let m = 0; (m + 0.5) * lambda < spacing; m++) {
    out.push(((m + 0.5) * lambda) / 2, -((m + 0.5) * lambda) / 2);
  }
  return out.sort((a, b) => a - b);
}

export const magnificationOf = (zLamp, zPaper, zWater) => (zLamp - zPaper) / (zLamp - zWater);

export function causticOf(zLamp, zPaper, zWater, zFloor = zWater, zBed = zFloor) {
  const n = WATER.index;
  const bend = (zWater - zFloor) * (1 - 1 / n) + ((zFloor - zBed) * (n - 1)) / GLASS + (zBed - zPaper) * (n - 1);
  return bend / magnificationOf(zLamp, zPaper, zWater);
}

export const amplitudeOf = (kMm) => SLOPE / kMm;

export const focusOf = (kMm, caustic) => caustic * amplitudeOf(kMm) * kMm * kMm;

export function stateOf(frequency, spacingMm, optics) {
  const wave = waveOf(frequency);
  const lambdaMm = wave.lambda * 1000;
  const kMm = wave.k / 1000;
  const amplitude = amplitudeOf(kMm);
  return {
    frequency,
    spacing: spacingMm,
    lambdaMm,
    kMm,
    speed: wave.speed,
    group: wave.group,
    decay: wave.decay / 1000,
    nodal: nodalCount(spacingMm, lambdaMm),
    strobe: frequency - BEAT,
    amplitude,
    focus: focusOf(kMm, optics.caustic),
    projected: lambdaMm * optics.magnification,
  };
}

export const readoutOf = (state) => `f ${state.frequency.toFixed(1)} Hz · λ ${(state.lambdaMm / 10).toFixed(2)} cm · d ${(state.spacing / 10).toFixed(1)} cm · ${state.nodal} nodal lines`;

export function wordsOf(state) {
  return `${state.frequency.toFixed(1)} hertz, wavelength ${(state.lambdaMm / 10).toFixed(2)} centimetres, dippers ${(state.spacing / 10).toFixed(1)} centimetres apart, ${state.nodal} nodal lines`;
}

function clippedRunsOf(points, [x0, y0, x1, y1], project) {
  const runs = [];
  let run = [];
  for (const point of points) {
    if (point[0] >= x0 && point[0] <= x1 && point[1] >= y0 && point[1] <= y1) run.push(project(point));
    else if (run.length) {
      if (run.length > 1) runs.push(run);
      run = [];
    }
  }
  if (run.length > 1) runs.push(run);
  return runs.map((points) => points.map(([x, y], index) => `${index ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`).join("")).join("");
}

export function patternOf(state, { xs, inside, lamp, zWater, zPaper }, project, phase = 0) {
  const lift = (lamp[2] - zPaper) / (lamp[2] - zWater);
  const onPaper = ([x, y]) => project([lamp[0] + (x - lamp[0]) * lift, lamp[1] + (y - lamp[1]) * lift, zPaper]);
  const half = state.spacing / 2;
  const nodal = [];
  for (const vertex of nodalVertices(state.spacing, state.lambdaMm)) {
    const a = Math.abs(vertex);
    const b = Math.sqrt(Math.max(half * half - a * a, 1e-6));
    const points = [];
    for (let index = 0; index <= 240; index++) {
      const t = -4 + (8 * index) / 240;
      points.push([xs + b * Math.sinh(t), Math.sign(vertex) * a * Math.cosh(t)]);
    }
    nodal.push(clippedRunsOf(points, inside, onPaper));
  }
  const crests = [];
  for (const sy of [-half, half]) {
    const reach = Math.max(...[inside[0], inside[2]].flatMap((x) => [inside[1], inside[3]].map((y) => Math.hypot(x - xs, y - sy))));
    for (let r = state.lambdaMm * (1 + (((phase / (2 * Math.PI)) % 1) + 1) % 1); r < reach; r += state.lambdaMm) {
      const points = [];
      const steps = Math.max(24, Math.round(r * 1.2));
      for (let index = 0; index <= steps; index++) {
        const a = (index / steps) * 2 * Math.PI;
        points.push([xs + r * Math.cos(a), sy + r * Math.sin(a)]);
      }
      crests.push(clippedRunsOf(points, inside, onPaper));
    }
  }
  return { nodal: nodal.join(""), crests: crests.join("") };
}
