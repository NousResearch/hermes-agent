import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { TURN } from "../kit/turn.mjs";

const argv = process.argv.slice(2);
if (!argv.length || argv.includes("--help")) {
  console.log(`Time a 3D figure's runtime in Node: emit and order over 360 angles, CPU time, median of 7 rounds.

  node scripts/turn-bench.mjs examples/turning-dial/build.mjs [--rounds 7] [--unit 1]

The build module must export T (from turning()); if it also exports follow(values), every angle is mapped through it.`);
  process.exit(argv.length ? 0 : 1);
}
const take = (flag, fallback) => {
  const at = argv.indexOf(flag);
  return at < 0 ? fallback : Number(argv[at + 1]);
};
const rounds = take("--rounds", 7);
const unit = take("--unit", 1);
const mod = await import(pathToFileURL(resolve(argv[0])).href);
const T = mod.T;
const follow = mod.follow ?? ((values) => values);
const scene = TURN.prepare(T.data, { unit });
const runs = scene.runs;
const frames = runs.map((run) => TURN.frame(run));
const turn = T.groups.find((g) => g.kind === "turn");
const angles = Array.from({ length: 360 }, (_, i) => i + 0.37);
const cams = angles.map((angle) => TURN.cams(scene, follow({ [turn.name]: angle })));
for (let i = 0; i < 20; i++) for (const run of runs) TURN.emit(run, run.group >= 0 ? cams[i][run.group] : cams[i].world, frames[run.index]);
const perPart = runs.map(() => []);
const orderTimes = [];
const frameTimes = [];
const cpuFrames = [];
for (let round = 0; round < rounds; round++) {
  const sums = new Float64Array(runs.length);
  let orderSum = 0;
  const cpu0 = process.cpuUsage();
  const wall0 = performance.now();
  const previous = scene.layers.map((layer) => layer.rest);
  for (let a = 0; a < angles.length; a++) {
    const c = cams[a];
    for (const run of runs) {
      const t0 = performance.now();
      TURN.emit(run, run.group >= 0 ? c[run.group] : c.world, frames[run.index]);
      sums[run.index] += performance.now() - t0;
    }
    const t1 = performance.now();
    scene.layers.forEach((layer, index) => {
      previous[index] = TURN.order(layer, frames, c, previous[index], {}, scene);
    });
    orderSum += performance.now() - t1;
  }
  const cpu = process.cpuUsage(cpu0);
  cpuFrames.push((cpu.user + cpu.system) / 1000 / angles.length);
  frameTimes.push((performance.now() - wall0) / angles.length);
  orderTimes.push(orderSum / angles.length);
  runs.forEach((run, index) => perPart[index].push(sums[index] / angles.length));
}
const median = (list) => list.slice().sort((x, y) => x - y)[Math.floor(list.length / 2)];
const kinds = new Map();
runs.forEach((run, index) => {
  const kind = run.kind === "tube" ? "tube chunk" : run.kind;
  if (!kinds.has(kind)) kinds.set(kind, []);
  kinds.get(kind).push(median(perPart[index]));
});
console.log(`turn-bench ${argv[0]} · ${runs.length} parts · ${angles.length} angles · median of ${rounds} rounds · unit ${unit}`);
for (const [kind, list] of [...kinds].sort((a, b) => b[1].reduce((s, v) => s + v, 0) - a[1].reduce((s, v) => s + v, 0))) {
  const total = list.reduce((s, v) => s + v, 0);
  console.log(`  ${kind.padEnd(11)} ${String(list.length).padStart(4)} parts · ${(total / list.length).toFixed(4)} ms per part · ${total.toFixed(3)} ms per frame · slowest ${Math.max(...list).toFixed(4)} ms`);
}
const slow = runs.map((run, index) => [run.name, median(perPart[index])]).sort((a, b) => b[1] - a[1]).slice(0, 6);
console.log(`  slowest parts: ${slow.map(([name, ms]) => `${name} ${ms.toFixed(3)}`).join(" · ")}`);
console.log(`  order ${median(orderTimes).toFixed(3)} ms per frame (${scene.layers.map((layer) => `${layer.name}: ${layer.parts.length} parts, ${layer.planar.length / 5} planes, ${layer.dynamic.length / 2} dynamic`).join("; ")})`);
console.log(`  frame ${median(frameTimes).toFixed(3)} ms wall · ${median(cpuFrames).toFixed(3)} ms CPU`);
