import { spawn } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { LINE_ENDS } from "./line-ends.mjs";

if (typeof WebSocket === "undefined" && !process.execArgv.includes("--experimental-websocket")) {
  const child = spawn(process.execPath, ["--experimental-websocket", ...process.argv.slice(1)], { stdio: "inherit" });
  child.on("exit", (code) => process.exit(code ?? 1));
} else {
  await main();
}

function usage() {
  console.log(`Drive a figure in headless Chrome: move, click, type, wait, screenshot, and build contact sheets.

  node drive.mjs <file.html | url> '<actions as JSON>' [--width 1440] [--height 1100] [--scale 1] [--reduced] [--theme dark|light] [--no-webgl] [--preload file.js]
  node drive.mjs <file.html | url> --actions steps.json

  --reduced     emulate prefers-reduced-motion: reduce
  --theme t     emulate prefers-color-scheme: dark or light (only pages that read the media query follow it)
  --no-webgl    make canvas.getContext("webgl"/"webgl2") return null, to check the SVG fallback
  --preload js  run a script before the page's own scripts (counters, stubs, hooks)
  Kit pages set their theme at build time (body data-theme), so drive the -light.html build for light.
  Put anything with eval strings in an --actions file: inline JSON quoting breaks easily.

Actions run in order. A missing or null selector means the first .iso-plate (the key action focuses [tabindex="0"]).
Fractions are 0..1 across the element.
  ["scroll", sel]                         scroll an element into the middle of the viewport
  ["wait", ms]                            wait
  ["move", sel, fx, fy]                   glide the mouse to a point of the element (fires pointermove)
  ["click", sel, fx?, fy?]                click a point of the element (centre by default)
  ["drag", sel, fx0, fy0, fx1, fy1, steps?]
                                          press at one point of the element, glide to another in steps (20) one frame apart,
                                          release: real pointer events (pointerId 1), so setPointerCapture works
  ["key", key, sel?, code?]               focus sel (default [tabindex="0"]) and press a key ("i", "Escape", "ArrowRight", " ", "7");
                                          pass code for keys that share a name ("Shift" + "ShiftRight", "Enter" + "NumpadEnter")
  ["slow", factor]                        run the page's requestAnimationFrame and performance.now clock at factor x real
                                          time (0.05 = 20x slow motion) so sheets catch 100 ms events; ["slow", 1] restores it
  ["until", js, timeoutMs?]               poll a JS expression until it is truthy (default 10 s), e.g. a readout reaching a phase
  ["eval", js]                            run JS in the page and print the result
  ["shot", out.png, sel?, scale?, [x,y,w,h]?]           screenshot an element, or a region of it in CSS px
  ["lines", out.png?, sel?, options?]     check every texture line's two ends: each must land on a stroke, a dot or an outline
                                          (within options.tolerance, 1 viewBox px), or hide under a fill painted later; a short
                                          straight mark (a tick) needs one rooted end; lines with data-free and fade ramps are
                                          exempt. Prints each floating end in card px; out.png marks them with rings. Any failure
                                          makes the run exit 1
  ["turn", value, group?]                 set a 3D figure's turn through window.__isoTurn (a number sets the first turn group,
                                          group names another), then wait two animation frames
  ["orbit", out.png, from, to, step, sel?, scale?, [x,y,w,h]?, columns?]
                                          a contact sheet across angles of a 3D figure: each frame is turned, waits two
                                          animation frames, is captured and labelled with its angle
  ["sheet", out.png, every, count, sel?, scale?, [x,y,w,h]?, columns?]
                                          take count frames every ms and tile them into one PNG, to catch pops and seams;
                                          each capture also takes 60-200 ms (more on heavy WebGL pages), so combine with ["slow", 0.05] for transitions

Prints the readout after each shot, then console errors and exceptions.`);
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
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  ].filter(Boolean);
  const playwright = join(homedir(), "Library/Caches/ms-playwright");
  if (existsSync(playwright)) {
    for (const dir of readdirSync(playwright).sort().reverse()) {
      for (const sub of ["chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing", "chrome-mac/Chromium.app/Contents/MacOS/Chromium", "chrome-linux/chrome"]) {
        candidates.push(join(playwright, dir, sub));
      }
    }
  }
  return candidates.find((path) => existsSync(path));
}

async function main() {
  const argv = process.argv.slice(2);
  if (!argv.length || argv.includes("--help") || argv.includes("-h")) {
    usage();
    process.exit(argv.length ? 0 : 1);
  }
  const options = { width: 1440, height: 1100, scale: 1, reduced: false, theme: null, actions: null, noWebgl: false, preload: null };
  const rest = [];
  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index];
    if (arg === "--width") options.width = Number(argv[++index]);
    else if (arg === "--height") options.height = Number(argv[++index]);
    else if (arg === "--scale") options.scale = Number(argv[++index]);
    else if (arg === "--reduced") options.reduced = true;
    else if (arg === "--theme") options.theme = argv[++index];
    else if (arg === "--no-webgl") options.noWebgl = true;
    else if (arg === "--preload") options.preload = readFileSync(argv[++index], "utf8");
    else if (arg === "--actions") options.actions = JSON.parse(readFileSync(argv[++index], "utf8"));
    else rest.push(arg);
  }
  const [target, script] = rest;
  const actions = options.actions ?? JSON.parse(script ?? "[]");
  const url = /^https?:|^file:/.test(target) ? target : pathToFileURL(resolve(target)).href;
  const binary = chromePath();
  if (!binary) {
    console.error("No Chrome found. Set CHROME_PATH to a Chrome or Chromium binary.");
    process.exit(1);
  }
  const profile = mkdtempSync(join(tmpdir(), "iso-drive-"));
  const chrome = spawn(binary, ["--headless=new", "--remote-debugging-port=0", `--user-data-dir=${profile}`, "--no-first-run", "--no-default-browser-check", "--hide-scrollbars", "--allow-file-access-from-files", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist", "about:blank"]);
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
    const page = await (await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, { method: "PUT" })).json();
    const socket = new WebSocket(page.webSocketDebuggerUrl);
    let id = 0;
    const pending = new Map();
    const problems = [];
    socket.onmessage = (message) => {
      const data = JSON.parse(message.data);
      if (data.id && pending.has(data.id)) {
        pending.get(data.id)(data);
        pending.delete(data.id);
      } else if (data.method === "Runtime.consoleAPICalled" && (data.params.type === "error" || data.params.type === "warning")) {
        problems.push(`console.${data.params.type}: ${data.params.args.map((arg) => arg.value ?? arg.description ?? "").join(" ").slice(0, 600)}`);
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
    const SLOW = `(()=>{const raf=window.requestAnimationFrame.bind(window);let real=null,virtual=0;window.__driveSpeed=1;window.requestAnimationFrame=(callback)=>raf((now)=>{if(real===null){real=now;virtual=now}virtual+=(now-real)*window.__driveSpeed;real=now;callback(virtual)});const clock=performance.now.bind(performance);let base=null,shown=0;performance.now=()=>{const now=clock();if(base===null){base=now;shown=now}shown+=(now-base)*window.__driveSpeed;base=now;return shown}})();`;
    const NO_GL = `(()=>{const get=HTMLCanvasElement.prototype.getContext;HTMLCanvasElement.prototype.getContext=function(kind,...rest){if(/webgl/.test(String(kind)))return null;return get.call(this,kind,...rest)}})();`;
    const early = [SLOW, options.noWebgl ? NO_GL : "", options.preload ?? ""].filter(Boolean).join("\n");
    await send("Page.addScriptToEvaluateOnNewDocument", { source: early });
    await send("Emulation.setDeviceMetricsOverride", { width: options.width, height: options.height, deviceScaleFactor: options.scale, mobile: options.width < 600 });
    await send("Page.navigate", { url });
    await sleep(1200);
    const boxOf = (selector) => evaluate(`(()=>{const el=document.querySelector(${JSON.stringify(selector ?? ".iso-plate")})||document.body;const r=el.getBoundingClientRect();return {x:r.left,y:r.top,w:r.width,h:r.height,sx:scrollX,sy:scrollY}})()`);
    const readout = () => evaluate(`(document.querySelector("[data-readout]")||{}).textContent||""`);
    const capture = async (selector, scale = 1, region) => {
      const box = await boxOf(selector);
      const x = box.x + box.sx;
      const y = box.y + box.sy;
      const clip = region ? { x: x + region[0], y: y + region[1], width: region[2], height: region[3], scale } : { x: x - 4, y: y - 4, width: box.w + 8, height: box.h + 8, scale };
      const shot = await send("Page.captureScreenshot", { format: "png", clip, captureBeyondViewport: true });
      return shot.result.data;
    };
    const save = (out, base64) => {
      mkdirSync(dirname(resolve(out)), { recursive: true });
      writeFileSync(out, Buffer.from(base64, "base64"));
    };
    let mouse = [4, 4];
    let failed = 0;
    for (const action of actions) {
      const [kind, ...args] = action;
      if (kind === "wait") await sleep(args[0]);
      else if (kind === "scroll") await evaluate(`(()=>{const el=document.querySelector(${JSON.stringify(args[0] ?? ".iso-plate")});if(el)el.scrollIntoView({block:"center"})})()`);
      else if (kind === "move") {
        const box = await boxOf(args[0]);
        const to = [box.x + box.w * args[1], box.y + box.h * args[2]];
        for (let step = 1; step <= 10; step++) {
          await send("Input.dispatchMouseEvent", { type: "mouseMoved", x: mouse[0] + ((to[0] - mouse[0]) * step) / 10, y: mouse[1] + ((to[1] - mouse[1]) * step) / 10 });
          await sleep(16);
        }
        mouse = to;
      } else if (kind === "click") {
        const box = await boxOf(args[0]);
        const to = [box.x + box.w * (args[1] ?? 0.5), box.y + box.h * (args[2] ?? 0.5)];
        await send("Input.dispatchMouseEvent", { type: "mouseMoved", x: to[0], y: to[1] });
        await send("Input.dispatchMouseEvent", { type: "mousePressed", x: to[0], y: to[1], button: "left", clickCount: 1 });
        await send("Input.dispatchMouseEvent", { type: "mouseReleased", x: to[0], y: to[1], button: "left", clickCount: 1 });
        mouse = to;
      } else if (kind === "drag") {
        const box = await boxOf(args[0]);
        const from = [box.x + box.w * args[1], box.y + box.h * args[2]];
        const to = [box.x + box.w * args[3], box.y + box.h * args[4]];
        const steps = Math.max(1, args[5] ?? 20);
        await send("Input.dispatchMouseEvent", { type: "mouseMoved", x: from[0], y: from[1] });
        await send("Input.dispatchMouseEvent", { type: "mousePressed", x: from[0], y: from[1], button: "left", buttons: 1, clickCount: 1 });
        for (let step = 1; step <= steps; step++) {
          await send("Input.dispatchMouseEvent", { type: "mouseMoved", x: from[0] + ((to[0] - from[0]) * step) / steps, y: from[1] + ((to[1] - from[1]) * step) / steps, button: "left", buttons: 1 });
          await sleep(16);
        }
        await send("Input.dispatchMouseEvent", { type: "mouseReleased", x: to[0], y: to[1], button: "left", buttons: 0, clickCount: 1 });
        mouse = to;
      } else if (kind === "key") {
        await evaluate(`(()=>{const el=document.querySelector(${JSON.stringify(args[1] ?? '[tabindex="0"]')});if(el)el.focus()})()`);
        const key = args[0];
        const named = { " ": "Space", "-": "Minus", "=": "Equal", ",": "Comma", ".": "Period", "/": "Slash", ";": "Semicolon", "'": "Quote", "[": "BracketLeft", "]": "BracketRight", "\\": "Backslash", "`": "Backquote" };
        const code = args[2] ?? (named[key] ?? (/^[0-9]$/.test(key) ? `Digit${key}` : key.length === 1 ? `Key${key.toUpperCase()}` : key));
        await send("Input.dispatchKeyEvent", { type: "keyDown", key, code, text: key.length === 1 ? key : undefined, windowsVirtualKeyCode: key === " " ? 32 : 0 });
        await send("Input.dispatchKeyEvent", { type: "keyUp", key, code });
      } else if (kind === "eval") console.log("eval:", JSON.stringify(await evaluate(args[0])));
      else if (kind === "slow") await evaluate(`window.__driveSpeed=${Number(args[0])}`);
      else if (kind === "until") {
        const limit = Date.now() + (args[1] ?? 10000);
        let ok = false;
        while (Date.now() < limit) {
          if (await evaluate(`Boolean(${args[0]})`)) {
            ok = true;
            break;
          }
          await sleep(25);
        }
        if (!ok) console.log(`until: timed out waiting for ${args[0]}`);
      }
      else if (kind === "shot") {
        const [out, selector, scale, region] = args;
        save(out, await capture(selector, scale ?? 1, region));
        console.log(`shot ${out} · readout "${await readout()}"`);
      } else if (kind === "sheet") {
        const [out, every, count, selector, scale, region, columns] = args;
        const frames = [];
        const texts = [];
        for (let index = 0; index < count; index++) {
          if (index) await sleep(every);
          frames.push(await capture(selector, scale ?? 0.5, region));
          texts.push(await readout());
        }
        const tiled = await evaluate(`(async()=>{const frames=${JSON.stringify(frames)};const cols=${columns ?? Math.min(count, 6)};const images=await Promise.all(frames.map(src=>new Promise(done=>{const image=new Image();image.onload=()=>done(image);image.src="data:image/png;base64,"+src})));const w=images[0].width,h=images[0].height,rows=Math.ceil(images.length/cols);const canvas=document.createElement("canvas");canvas.width=w*cols;canvas.height=h*rows;const ctx=canvas.getContext("2d");ctx.fillStyle="#808080";ctx.fillRect(0,0,canvas.width,canvas.height);images.forEach((image,index)=>{ctx.drawImage(image,(index%cols)*w,Math.floor(index/cols)*h);ctx.fillStyle="#ff00aa";ctx.font="bold 12px monospace";ctx.fillText(String(index),(index%cols)*w+6,Math.floor(index/cols)*h+16)});return canvas.toDataURL("image/png").split(",")[1]})()`);
        save(out, tiled);
        console.log(`sheet ${out} · ${count} frames every ${every} ms · readouts ${texts.map((text, index) => `${index}:"${text}"`).join(" ")}`);
      } else if (kind === "turn") {
        const [value, group] = args;
        const call = group ? `window.__isoTurn.set(${JSON.stringify({ [group]: value })})` : `window.__isoTurn.set(${typeof value === "object" ? JSON.stringify(value) : Number(value)})`;
        await evaluate(`(()=>{if(!window.__isoTurn)return false;${call};return new Promise((done)=>requestAnimationFrame(()=>requestAnimationFrame(()=>done(true))))})()`);
      } else if (kind === "orbit") {
        const [out, from, to, step, selector, scale, region, columns] = args;
        const frames = [];
        const labels = [];
        for (let at = from; step > 0 ? at <= to + 1e-9 : at >= to - 1e-9; at += step) {
          const angle = Math.round(at * 1000) / 1000;
          await evaluate(`(()=>{if(!window.__isoTurn)return false;window.__isoTurn.set(${angle});return new Promise((done)=>requestAnimationFrame(()=>requestAnimationFrame(()=>done(true))))})()`);
          frames.push(await capture(selector, scale ?? 0.5, region));
          labels.push(`${angle}°`);
        }
        const tiled = await evaluate(`(async()=>{const frames=${JSON.stringify(frames)};const labels=${JSON.stringify(labels)};const cols=${columns ?? 6};const images=await Promise.all(frames.map(src=>new Promise(done=>{const image=new Image();image.onload=()=>done(image);image.src="data:image/png;base64,"+src})));const w=images[0].width,h=images[0].height,rows=Math.ceil(images.length/cols);const canvas=document.createElement("canvas");canvas.width=w*cols;canvas.height=h*rows;const ctx=canvas.getContext("2d");ctx.fillStyle="#808080";ctx.fillRect(0,0,canvas.width,canvas.height);images.forEach((image,index)=>{ctx.drawImage(image,(index%cols)*w,Math.floor(index/cols)*h);ctx.fillStyle="#ff00aa";ctx.font="bold 12px monospace";ctx.fillText(labels[index],(index%cols)*w+6,Math.floor(index/cols)*h+16)});return canvas.toDataURL("image/png").split(",")[1]})()`);
        save(out, tiled);
        console.log(`orbit ${out} · ${frames.length} frames ${from}°…${to}° every ${step}°`);
      } else if (kind === "lines") {
        const [out, selector, extra] = args;
        const result = await evaluate(`(${LINE_ENDS})(${JSON.stringify({ select: selector ?? null, show: Boolean(out), ...(extra ?? {}) })})`);
        if (!result) {
          console.log("lines: the check did not run");
          failed++;
          continue;
        }
        console.log(`line ends: ${result.problems.length} of ${result.ends} ends on ${result.lines} lines float`);
        for (const problem of result.problems) console.log(`  ${problem.what}  ${problem.end}${problem.mark ? " (a mark: neither end rooted)" : ""}  ${problem.gap == null ? "nothing nearby" : `${problem.gap.toFixed(2)} px from anything`}   at card ${problem.at.join(",")}   length ${problem.length}   d ${problem.d}…${problem.opacity < 1 ? `   opacity ${problem.opacity}` : ""}`);
        if (out) {
          save(out, await capture(selector, 2));
          await evaluate(`(document.querySelector("[data-line-ends]")||{remove(){}}).remove()`);
          console.log(`shot ${out}`);
        }
        failed += result.problems.length;
      } else console.log(`unknown action ${kind}`);
    }
    console.log(problems.length ? problems.join("\n") : "console: clean");
    socket.close();
    finish(failed ? 1 : 0);
  } catch (error) {
    console.error(error);
    finish(1);
  }
}
