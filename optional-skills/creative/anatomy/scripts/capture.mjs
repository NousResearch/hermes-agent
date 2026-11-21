import { spawn } from "node:child_process";
import { existsSync, mkdtempSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

if (typeof WebSocket === "undefined" && !process.execArgv.includes("--experimental-websocket")) {
  const child = spawn(process.execPath, ["--experimental-websocket", ...process.argv.slice(1)], { stdio: "inherit" });
  child.on("exit", (code) => process.exit(code ?? 1));
} else {
  await main();
}

function usage() {
  console.log(`Capture a figure with headless Chrome.

  node capture.mjs <file.html | file.svg | url> [options]

  --out <png>         output path (default: capture.png in the current directory)
  --width <px>        viewport width (default 1440; use 390 for a phone)
  --height <px>       viewport height (default 1000)
  --scale <n>         device pixel ratio (default 2)
  --wait <ms>         time to let the page settle before the first shot (default 2500)
  --select <css>      clip to this element (default: .iso-plate, else the whole page)
  --zoom <x,y,w,h>    clip a region inside the selected element, in CSS px, for close-up checks
  --zoom-scale <n>    pixel ratio for --zoom shots (default 4)
  --hover <fx,fy>     move the mouse to this fraction of the selected element first (e.g. 0.6,0.5)
  --key <keys>        press keys on the focused figure first, comma separated (e.g. Tab,ArrowRight,7,Escape);
                      Tab is skipped (the figure is focused for you), a comma is written Comma
  --times <ms,...>    extra shots at these delays, each measured from the first shot (not from the previous one),
                      saved as name-<ms>.png; a capture takes 60-200 ms, so delays closer than that run late
  --reduced           emulate prefers-reduced-motion: reduce
  --theme <dark|light>  emulate prefers-color-scheme

Prints the clip box, console errors, uncaught exceptions and whether the page scrolls sideways.`);
}

function parse(argv) {
  const options = { width: 1440, height: 1000, scale: 2, wait: 2500, zoomScale: 4, times: [] };
  const rest = [];
  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index];
    const next = () => argv[++index];
    if (arg === "--out") options.out = next();
    else if (arg === "--width") options.width = Number(next());
    else if (arg === "--height") options.height = Number(next());
    else if (arg === "--scale") options.scale = Number(next());
    else if (arg === "--wait") options.wait = Number(next());
    else if (arg === "--select") options.select = next();
    else if (arg === "--zoom") options.zoom = next().split(",").map(Number);
    else if (arg === "--zoom-scale") options.zoomScale = Number(next());
    else if (arg === "--hover") options.hover = next().split(",").map(Number);
    else if (arg === "--key") options.keys = next().split(",");
    else if (arg === "--times") options.times = next().split(",").map(Number);
    else if (arg === "--reduced") options.reduced = true;
    else if (arg === "--theme") options.theme = next();
    else if (arg === "--help" || arg === "-h") options.help = true;
    else rest.push(arg);
  }
  options.target = rest[0];
  return options;
}

function playwrightShells() {
  const found = [];
  for (const root of [join(homedir(), ".cache/ms-playwright"), join(homedir(), "Library/Caches/ms-playwright")]) {
    if (!existsSync(root)) continue;
    for (const dir of readdirSync(root).filter((d) => d.startsWith("chromium_headless_shell-")).sort().reverse()) {
      for (const sub of ["chrome-headless-shell-linux64/chrome-headless-shell", "chrome-headless-shell-mac-arm64/chrome-headless-shell", "chrome-headless-shell-mac-x64/chrome-headless-shell"]) {
        found.push(join(root, dir, sub));
      }
    }
  }
  return found;
}

function chromePath() {
  const candidates = [
    process.env.CHROME_PATH, ...playwrightShells(),
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  ].filter(Boolean);
  const playwright = join(homedir(), "Library/Caches/ms-playwright");
  if (existsSync(playwright)) {
    for (const dir of readdirSync(playwright).sort().reverse()) {
      for (const sub of ["chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing", "chrome-mac/Chromium.app/Contents/MacOS/Chromium"]) {
        candidates.push(join(playwright, dir, sub));
      }
    }
  }
  return candidates.find((path) => existsSync(path));
}

async function main() {
  const options = parse(process.argv.slice(2));
  if (options.help || !options.target) {
    usage();
    process.exit(options.help ? 0 : 1);
  }
  const url = /^https?:|^file:/.test(options.target) ? options.target : pathToFileURL(resolve(options.target)).href;
  const out = resolve(options.out ?? "capture.png");
  const binary = chromePath();
  if (!binary) {
    console.error("No Chrome found. Set CHROME_PATH to a Chrome or Chromium binary.");
    process.exit(1);
  }
  const profile = mkdtempSync(join(tmpdir(), "iso-capture-"));
  const chrome = spawn(binary, [
    "--headless=new",
    "--remote-debugging-port=0",
    `--user-data-dir=${profile}`,
    "--no-first-run",
    "--no-default-browser-check",
    "--hide-scrollbars",
    "--allow-file-access-from-files",
    "--enable-unsafe-swiftshader",
    "--ignore-gpu-blocklist",
    "about:blank",
  ]);
  const port = await new Promise((done, fail) => {
    let text = "";
    const timer = setTimeout(() => fail(new Error("Chrome did not start")), 20000);
    chrome.stderr.on("data", (chunk) => {
      text += chunk;
      const match = text.match(/DevTools listening on ws:\/\/[^:]+:(\d+)\//);
      if (match) {
        clearTimeout(timer);
        done(Number(match[1]));
      }
    });
  });
  const finish = (code) => {
    chrome.kill("SIGKILL");
    try {
      rmSync(profile, { recursive: true, force: true });
    } catch {}
    process.exit(code);
  };
  try {
    const target = await (await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, { method: "PUT" })).json();
    const socket = new WebSocket(target.webSocketDebuggerUrl);
    let id = 0;
    const pending = new Map();
    const problems = [];
    socket.onmessage = (message) => {
      const data = JSON.parse(message.data);
      if (data.id && pending.has(data.id)) {
        pending.get(data.id)(data);
        pending.delete(data.id);
      } else if (data.method === "Runtime.consoleAPICalled" && (data.params.type === "error" || data.params.type === "warning")) {
        problems.push(`console.${data.params.type}: ${data.params.args.map((arg) => arg.value ?? arg.description ?? "").join(" ").slice(0, 400)}`);
      } else if (data.method === "Runtime.exceptionThrown") {
        const detail = data.params.exceptionDetails;
        problems.push(`exception: ${detail.exception?.description ?? detail.text}`.slice(0, 600));
      }
    };
    await new Promise((done) => (socket.onopen = done));
    const send = (method, params = {}) =>
      new Promise((done) => {
        const call = ++id;
        pending.set(call, done);
        socket.send(JSON.stringify({ id: call, method, params }));
      });
    const evaluate = async (expression) => (await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true })).result?.result?.value;
    const sleep = (ms) => new Promise((done) => setTimeout(done, ms));
    await send("Runtime.enable");
    await send("Page.enable");
    const features = [];
    if (options.reduced) features.push({ name: "prefers-reduced-motion", value: "reduce" });
    if (options.theme) features.push({ name: "prefers-color-scheme", value: options.theme });
    if (features.length) await send("Emulation.setEmulatedMedia", { features });
    await send("Emulation.setDeviceMetricsOverride", { width: options.width, height: options.height, deviceScaleFactor: options.scale, mobile: options.width < 600 });
    await send("Page.navigate", { url });
    await sleep(Math.min(options.wait, 1500));
    await evaluate(`(async()=>{const h=document.documentElement.scrollHeight;for(let y=0;y<h;y+=400){scrollTo(0,y);await new Promise(r=>setTimeout(r,60))}scrollTo(0,0)})()`);
    await sleep(Math.max(0, options.wait - 1500));
    const select = options.select ?? ".iso-plate";
    const boxOf = async () =>
      evaluate(`(()=>{const el=document.querySelector(${JSON.stringify(select)})||document.querySelector("svg")||document.body;el.scrollIntoView({block:"center"});const r=el.getBoundingClientRect();return {x:r.left+scrollX,y:r.top+scrollY,w:r.width,h:r.height,vx:r.left,vy:r.top}})()`);
    let box = await boxOf();
    if (options.hover) {
      await sleep(300);
      box = await boxOf();
      const x = box.vx + box.w * options.hover[0];
      const y = box.vy + box.h * options.hover[1];
      await send("Input.dispatchMouseEvent", { type: "mouseMoved", x: box.vx + 2, y: box.vy + 2 });
      await sleep(60);
      for (let step = 1; step <= 12; step++) {
        await send("Input.dispatchMouseEvent", { type: "mouseMoved", x: box.vx + 2 + ((x - box.vx - 2) * step) / 12, y: box.vy + 2 + ((y - box.vy - 2) * step) / 12 });
        await sleep(30);
      }
      await sleep(1200);
    }
    if (options.keys) {
      await evaluate(`(()=>{const f=document.querySelector('[tabindex="0"]');if(f)f.focus()})()`);
      const named = { " ": "Space", "-": "Minus", "=": "Equal", ",": "Comma", ".": "Period", "/": "Slash", ";": "Semicolon", "'": "Quote", "[": "BracketLeft", "]": "BracketRight", "\\": "Backslash", "`": "Backquote" };
      for (const entry of options.keys) {
        if (entry === "Tab") continue;
        const key = entry === "Comma" ? "," : entry === "Space" ? " " : entry;
        const code = named[key] ?? (/^[0-9]$/.test(key) ? `Digit${key}` : key.length === 1 ? `Key${key.toUpperCase()}` : key);
        await send("Input.dispatchKeyEvent", { type: "keyDown", key, code, text: key.length === 1 ? key : undefined, windowsVirtualKeyCode: key === " " ? 32 : 0 });
        await send("Input.dispatchKeyEvent", { type: "keyUp", key, code });
        await sleep(250);
      }
      await sleep(1200);
    }
    const shoot = async (path, clip, scale) => {
      const shot = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true, clip: { ...clip, scale } });
      writeFileSync(path, Buffer.from(shot.result.data, "base64"));
    };
    const clipOf = () => {
      if (!options.zoom) return { clip: { x: box.x - 6, y: box.y - 6, width: box.w + 12, height: box.h + 12 }, scale: 1 };
      const [zx, zy, zw, zh] = options.zoom;
      return { clip: { x: box.x + zx, y: box.y + zy, width: zw, height: zh }, scale: options.zoomScale / options.scale };
    };
    const first = clipOf();
    await shoot(out, first.clip, first.scale);
    const shotAt = Date.now();
    const readouts = [];
    for (const delay of [...options.times].sort((a, b) => a - b)) {
      await sleep(Math.max(0, shotAt + delay - Date.now()));
      const next = clipOf();
      await shoot(out.replace(/\.png$/, `-${delay}.png`), next.clip, next.scale);
      readouts.push(`+${delay}ms "${await evaluate(`(document.querySelector("[data-readout]")||{}).textContent||""`)}"`);
    }
    const sideways = await evaluate(`document.documentElement.scrollWidth > innerWidth + 1`);
    const readout = await evaluate(`(document.querySelector("[data-readout]")||{}).textContent||""`);
    console.log(`saved ${out}`);
    console.log(`clip ${Math.round(box.x)},${Math.round(box.y)} ${Math.round(box.w)}x${Math.round(box.h)} · readout "${readout}" · scrolls sideways: ${sideways ? "YES" : "no"}`);
    if (readouts.length) console.log(`readout over time: ${readouts.join(" · ")}`);
    console.log(problems.length ? problems.join("\n") : "console: clean");
    socket.close();
    finish(0);
  } catch (error) {
    console.error(error);
    finish(1);
  }
}
