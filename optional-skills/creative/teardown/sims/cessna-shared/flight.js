// The two sequences the gear lives through, as functions of tau (seconds into the sequence): the takeoff (roll, lift
// off, the gear comes up) and the landing (the gear comes down, the flare, touchdown, rollout). Each gives the
// stroke (0 down, 1 up), the height above the runway, the ground speed and the phase line the console shows.
const smoother = (x) => {
  const u = Math.min(1, Math.max(0, x));
  return u * u * u * (u * (u * 6 - 15) + 10);
};
const ss = (a, b, t) => smoother((t - a) / (b - a));
export const H_MAX = 1.7;
export const V_MAX = 14;

export const SEQ = {
  takeoff: {
    dur: 5,
    from: 'RUNWAY',
    to: 'GEAR UP',
    stroke: (t) => ss(2.3, 4.5, t),
    height: (t) => H_MAX * Math.pow(ss(1.9, 5.0, t), 1.25),
    speed: (t) => V_MAX * ss(0, 2.0, t),
    phases: [
      [0, 'RUNWAY'],
      [0.6, 'TAKEOFF ROLL'],
      [1.9, 'LIFT OFF'],
      [2.3, 'ACTUATOR STROKES AFT'],
      [3.0, 'RACK TURNS THE PINION'],
      [3.6, 'SIDE TUBES SPIN THE LEGS UP'],
      [4.5, 'GEAR UP & LOCKED'],
    ],
  },
  landing: {
    dur: 7,
    from: 'GEAR UP',
    to: 'RUNWAY',
    stroke: (t) => 1 - ss(0.6, 3.0, t),
    height: (t) => H_MAX * (1 - ss(0, 5.2, t)),
    speed: (t) => V_MAX * (1 - ss(5.2, 7.0, t)),
    phases: [
      [0, 'GEAR DOWN SELECTED'],
      [0.6, 'ACTUATOR STROKES FORWARD'],
      [1.3, 'SIDE TUBES SPIN THE LEGS DOWN'],
      [3.0, 'GEAR DOWN & LOCKED'],
      [5.0, 'TOUCHDOWN'],
      [5.5, 'ROLLOUT'],
    ],
  },
};
export const phaseAt = (seq, t) => SEQ[seq].phases.filter(([a]) => a <= t).pop()[1];
/** the mechanism state chip: [title, sub, colour key] */
export const stateAt = (stroke, height) => {
  if (stroke < 0.002) return height < 0.01 ? ['Gear down & locked', 'ON THE RUNWAY', 'ok'] : ['Gear down & locked', `AIRBORNE · ${height.toFixed(2)} M`, 'ok'];
  if (stroke > 0.998) return ['Gear up & locked', `AIRBORNE · ${height.toFixed(2)} M`, 'ok'];
  return ['In transit', height < 0.01 ? 'ON THE RUNWAY' : `AIRBORNE · ${height.toFixed(2)} M`, 'move'];
};
