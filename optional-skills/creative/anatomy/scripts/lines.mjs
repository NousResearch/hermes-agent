import { spawn } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const argv = process.argv.slice(2);
if (!argv.length || argv.includes("--help") || argv.includes("-h")) {
  console.log(`Check that every texture line in a figure ends on something.

  node lines.mjs <file.html | url> [--out marks.png] [--tolerance 1] [--select css] [--wait 1500] [drive.mjs flags]

Every open subpath of every .iso-line (seams, grooves, ribs, rule lines, ticks, needles) must end on another stroke,
a dot or an outline within the tolerance (viewBox px), or under a fill painted after it. A short straight mark (a tick)
needs one rooted end. Outlines, creases, bevels and tube edges are the kit's own and are not checked. Lines drawn with
fadeLineSvg ramp out and pass; lineSvg(d, { free: true | "start" | "end" }) marks an end that is meant to stop (a needle
tip, rule lines standing for printed text). Exits 1 if any end floats; --out writes the card with each one ringed.`);
  process.exit(argv.length ? 0 : 1);
}
const take = (flag, fallback) => {
  const at = argv.indexOf(flag);
  if (at < 0) return fallback;
  const value = argv[at + 1];
  argv.splice(at, 2);
  return value;
};
const out = take("--out", null);
const tolerance = Number(take("--tolerance", 1));
const select = take("--select", null);
const wait = Number(take("--wait", 1500));
const actions = JSON.stringify([["wait", wait], ["lines", out, select, { tolerance }]]);
const child = spawn(process.execPath, [join(dirname(fileURLToPath(import.meta.url)), "drive.mjs"), argv[0], actions, ...argv.slice(1)], { stdio: "inherit" });
child.on("exit", (code) => process.exit(code ?? 1));
