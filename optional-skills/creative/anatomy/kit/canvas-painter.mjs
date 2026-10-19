const CP_Q = 64;
const CP_SVG = "http://www.w3.org/2000/svg";

function cpScratch() {
  const c = document.createElement("canvas");
  c.width = c.height = 1;
  return c.getContext("2d");
}

function cpRgba(css, probe) {
  if (!css || css === "none" || css.startsWith("url(")) return null;
  probe.fillStyle = "#000";
  probe.fillStyle = css;
  const out = probe.fillStyle;
  if (out[0] === "#") return [parseInt(out.slice(1, 3), 16), parseInt(out.slice(3, 5), 16), parseInt(out.slice(5, 7), 16), 1];
  const nums = out.match(/-?[\d.]+(e-?\d+)?/g);
  if (!nums) return null;
  if (out.startsWith("color(")) {
    const v = nums.map(Number);
    return [v[0] * 255, v[1] * 255, v[2] * 255, v.length > 3 ? v[3] : 1];
  }
  const v = nums.map(Number);
  return [v[0], v[1], v[2], v.length > 3 ? v[3] : 1];
}

function cpCss(rgba, alpha = 1) {
  const a = rgba[3] * alpha;
  const r = Math.round(rgba[0]);
  const g = Math.round(rgba[1]);
  const b = Math.round(rgba[2]);
  return a >= 1 ? `rgb(${r},${g},${b})` : `rgba(${r},${g},${b},${Math.round(a * 1000) / 1000})`;
}

function cpMatrixOf(text) {
  if (!text) return null;
  const nums = text.match(/-?[\d.]+(e-?\d+)?/g);
  if (!nums) return null;
  const v = nums.map(Number);
  if (text.startsWith("translate")) return [1, 0, 0, 1, v[0], v[1] ?? 0];
  if (text.startsWith("matrix")) return v;
  const m = new DOMMatrix(text);
  return [m.a, m.b, m.c, m.d, m.e, m.f];
}

function cpMul(A, B) {
  if (!A) return B;
  if (!B) return A;
  return [A[0] * B[0] + A[2] * B[1], A[1] * B[0] + A[3] * B[1], A[0] * B[2] + A[2] * B[3], A[1] * B[2] + A[3] * B[3], A[0] * B[4] + A[2] * B[5] + A[4], A[1] * B[4] + A[3] * B[5] + A[5]];
}

function cpStaticMatrix(el) {
  const list = el.transform && el.transform.baseVal;
  if (!list || !list.numberOfItems) return null;
  const m = list.consolidate().matrix;
  return [m.a, m.b, m.c, m.d, m.e, m.f];
}

function cpGeometry(el) {
  const tag = el.localName;
  const num = (name) => parseFloat(el.getAttribute(name) ?? "0") || 0;
  if (tag === "path") return el.getAttribute("d") ? new Path2D(el.getAttribute("d")) : null;
  const p = new Path2D();
  if (tag === "circle") p.arc(num("cx"), num("cy"), num("r"), 0, Math.PI * 2);
  else if (tag === "ellipse") p.ellipse(num("cx"), num("cy"), num("rx"), num("ry"), 0, 0, Math.PI * 2);
  else if (tag === "rect") {
    const rx = num("rx") || num("ry");
    if (rx && p.roundRect) p.roundRect(num("x"), num("y"), num("width"), num("height"), rx);
    else p.rect(num("x"), num("y"), num("width"), num("height"));
  } else if (tag === "line") {
    p.moveTo(num("x1"), num("y1"));
    p.lineTo(num("x2"), num("y2"));
  } else if (tag === "polyline" || tag === "polygon") {
    const v = (el.getAttribute("points") ?? "").trim().split(/[\s,]+/).map(Number);
    for (let i = 0; i + 1 < v.length; i += 2) i ? p.lineTo(v[i], v[i + 1]) : p.moveTo(v[i], v[i + 1]);
    if (tag === "polygon") p.closePath();
  } else return null;
  return p;
}

function cpStyle(el, probe) {
  const cs = getComputedStyle(el);
  if (cs.display === "none" || cs.visibility === "hidden") return null;
  const fill = cpRgba(cs.fill, probe);
  const stroke = cpRgba(cs.stroke, probe);
  const fillOpacity = parseFloat(cs.fillOpacity);
  const strokeOpacity = parseFloat(cs.strokeOpacity);
  const dash = cs.strokeDasharray && cs.strokeDasharray !== "none" ? cs.strokeDasharray.split(/[\s,]+/).map(parseFloat).filter((v) => !Number.isNaN(v)) : null;
  return {
    cs,
    fill: fill ? cpCss(fill, isNaN(fillOpacity) ? 1 : fillOpacity) : null,
    fillRaw: fill ? cpCss(fill) : null,
    stroke: stroke ? cpCss(stroke, isNaN(strokeOpacity) ? 1 : strokeOpacity) : null,
    width: parseFloat(cs.strokeWidth) || 0,
    cap: cs.strokeLinecap || "butt",
    join: cs.strokeLinejoin || "miter",
    miter: parseFloat(cs.strokeMiterlimit) || 4,
    dash,
    dashOffset: parseFloat(cs.strokeDashoffset) || 0,
    rule: cs.fillRule === "evenodd" ? "evenodd" : "nonzero",
    opacity: parseFloat(cs.opacity),
    nonScaling: cs.vectorEffect === "non-scaling-stroke",
  };
}

function cpCompilePart(node, run, probe, unsupported, vary) {
  const ops = [];
  const roleOf = (el) => {
    const name = el === node ? "move" : el.getAttribute("data-r");
    if (name === null || run.role[name] === undefined) return -1;
    return run.role[name];
  };
  const varyOf = (el) => {
    if (!vary || !el.classList) return null;
    for (const cls of el.classList) if (vary[cls]) return vary[cls];
    return null;
  };
  const walk = (el, chain, alphaRoles, alpha, varies = []) => {
    const varied = varyOf(el);
    const ownVaries = varied ? varies.concat([varied]) : varies;
    const r = roleOf(el);
    const type = r >= 0 ? run.roles[r].type : null;
    const tag = el.localName;
    let nextChain = chain;
    if (type === "self" || type === "plane" || type === "xform") nextChain = chain.concat([{ r }]);
    else {
      const m = el === node ? null : cpStaticMatrix(el);
      if (m) nextChain = chain.concat([{ m }]);
    }
    if (tag === "g" || tag === "svg") {
      let groupAlpha = alpha;
      let groupRoles = alphaRoles;
      if (type === "plane") groupRoles = alphaRoles.concat([r]);
      else if (varied) {
      } else if (el !== node) {
        const op = parseFloat(getComputedStyle(el).opacity);
        if (!Number.isNaN(op)) groupAlpha *= op;
      }
      for (const child of el.children) walk(child, nextChain, groupRoles, groupAlpha, ownVaries);
      return;
    }
    if (tag === "title" || tag === "desc") return;
    const style = cpStyle(el, probe);
    if (!style) return;
    let ownAlpha = alpha;
    let ownRoles = alphaRoles;
    if (type === "slot") ownRoles = alphaRoles.concat([r]);
    else if (!varied && !Number.isNaN(style.opacity)) ownAlpha *= style.opacity;
    const op = { tag, r, type, chain: nextChain, alpha: ownAlpha, alphaRoles: ownRoles, varies: ownVaries, style, path: null, size: 0, last: null };
    if (type === "face") {
      const cs = style.cs;
      op.ramp = [0, 1, 2, 3, 4].map((i) => cpRgba(cs.getPropertyValue(`--tn-${i}`).trim(), probe) ?? [0, 0, 0, 1]);
      op.rampKey = op.ramp.map((c) => c.join(",")).join("|");
      op.baseWidth = style.width;
    } else if (type === "dot") {
      op.size = parseFloat(el.getAttribute("r")) || 0.5;
    } else if (type === "path" || type === "slot") {
    } else if (tag === "text") {
      const cs = style.cs;
      op.text = el.textContent;
      op.x = parseFloat(el.getAttribute("x") ?? "0") || 0;
      op.y = parseFloat(el.getAttribute("y") ?? "0") || 0;
      op.font = `${cs.fontStyle} ${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`;
      op.align = cs.textAnchor === "middle" ? "center" : cs.textAnchor === "end" ? "right" : "left";
      op.spacing = cs.letterSpacing === "normal" ? "0px" : cs.letterSpacing;
      op.baseline = cs.dominantBaseline === "central" || cs.dominantBaseline === "middle" ? "middle" : cs.dominantBaseline === "hanging" ? "hanging" : "alphabetic";
    } else {
      op.path = cpGeometry(el);
      if (!op.path) {
        unsupported.add(tag);
        return;
      }
    }
    ops.push(op);
  };
  walk(node, [], [], 1);
  return ops;
}

function cpTrace(ctx, d, arena) {
  ctx.beginPath();
  for (let i = 0; i + 4 < d.length; i += 5) {
    const at = 2 * (d.charCodeAt(i + 1) | (d.charCodeAt(i + 2) << 16));
    const count = d.charCodeAt(i + 3);
    ctx.moveTo(arena[at], arena[at + 1]);
    for (let k = 1; k < count; k++) ctx.lineTo(arena[at + 2 * k], arena[at + 2 * k + 1]);
    if (d.charCodeAt(i + 4)) ctx.closePath();
  }
}

function cpMount(stage, controller, { keepSvg = false, vary = null, only = null, effects = null, below = null, mask = null } = {}) {
  const probe = cpScratch();
  const scene = () => controller.scene();
  const unsupported = new Set();
  const layers = [];
  const programs = [];
  const wanted = only ?? controller.canvasLayers ?? null;
  const svgs = [...stage.querySelectorAll("svg[data-live]")].filter((svg) => !wanted || wanted.includes(svg.dataset.live));
  const first = svgs[0];
  const vb = first.viewBox.baseVal;
  const W = vb.width;
  const H = vb.height;
  const runs = scene().runs;
  const t0 = performance.now();
  for (const svg of svgs) {
    const name = svg.dataset.live;
    const canvas = document.createElement("canvas");
    canvas.className = "iso-layer iso-canvas";
    canvas.dataset.canvasLive = name;
    canvas.setAttribute("aria-hidden", "true");
    svg.before(canvas);
    const ctx = canvas.getContext("2d", { alpha: true, desynchronized: false });
    const container = svg.querySelector(".iso-parts") ?? svg;
    for (const node of container.querySelectorAll(":scope > [data-p]")) {
      const index = Number(node.dataset.p);
      const effect = effects && effects[runs[index].name];
      programs[index] = effect === false ? [] : effect ? [{ effect }] : cpCompilePart(node, runs[index], probe, unsupported, vary);
    }
    layers.push({ name, canvas, ctx, svg });
  }
  const compileMs = performance.now() - t0;
  const effectRuns = effects ? runs.filter((run) => typeof effects[run.name] === "function" && effects[run.name].always !== false && programs[run.index]).map((run) => run.index) : [];
  const view = { zoom: 1, cx: W / 2, cy: H / 2 };
  let cssPerUnit = 1;
  let dpr = window.devicePixelRatio || 1;
  let base = [1, 0, 0, 1, 0, 0];
  const resize = () => {
    forceFull = true;
    const box = stage.getBoundingClientRect();
    dpr = window.devicePixelRatio || 1;
    const cssW = box.width;
    const cssH = box.height;
    const snapX = box.left - Math.round(box.left);
    const snapY = box.top - Math.round(box.top);
    const boxW = Math.ceil(cssW + snapX - 1e-6);
    const boxH = Math.ceil(cssH + snapY - 1e-6);
    const w = Math.max(1, Math.round(boxW * dpr));
    const h = Math.max(1, Math.round(boxH * dpr));
    for (const layer of layers) {
      if (layer.canvas.width !== w) layer.canvas.width = w;
      if (layer.canvas.height !== h) layer.canvas.height = h;
      layer.canvas.style.width = `${boxW}px`;
      layer.canvas.style.height = `${boxH}px`;
      layer.canvas.style.left = `${-snapX}px`;
      layer.canvas.style.top = `${-snapY}px`;
    }
    const fit = Math.min(cssW / W, cssH / H);
    const offX = (cssW - W * fit) / 2;
    const offY = (cssH - H * fit) / 2;
    const s = fit * view.zoom;
    cssPerUnit = s;
    const tx = offX + (W / 2) * fit - view.cx * s;
    const ty = offY + (H / 2) * fit - view.cy * s;
    base = [s * dpr, 0, 0, s * dpr, (tx + snapX) * dpr, (ty + snapY) * dpr];
    const unit = Math.round(Math.pow(2, Math.round(Math.log2(s) * 4) / 4) * 100) / 100;
    if (controller.setUnit) controller.setUnit(unit);
  };
  const faceCache = new Map();
  const faceColour = (op, q) => {
    let table = faceCache.get(op.rampKey);
    if (!table) {
      table = new Array(4 * CP_Q + 1);
      faceCache.set(op.rampKey, table);
    }
    let css = table[q];
    if (css === undefined) {
      const band = Math.min(3, Math.floor(q / CP_Q));
      const f = (q - CP_Q * band) / CP_Q;
      const a = op.ramp[band];
      const b = op.ramp[band + 1];
      css = cpCss([a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f, a[3] + (b[3] - a[3]) * f]);
      table[q] = css;
    }
    return css;
  };
  const stats = { ms: 0, ops: 0, paths: 0, parsed: 0, touched: 0, rects: 0, full: 0 };
  const lastBox = [];
  let painted = false;
  const PAD = 3;
  const rectsOf = (frames, emitted) => {
    const list = [];
    const pad = PAD / cssPerUnit;
    for (const index of emitted) {
      const b = frames[index].box;
      const old = lastBox[index];
      let x0 = b[0];
      let y0 = b[1];
      let x1 = b[2];
      let y1 = b[3];
      if (old) {
        x0 = Math.min(x0, old[0]);
        y0 = Math.min(y0, old[1]);
        x1 = Math.max(x1, old[2]);
        y1 = Math.max(y1, old[3]);
      }
      if (!(x1 >= x0)) continue;
      list.push([x0 - pad, y0 - pad, x1 + pad, y1 + pad]);
    }
    let merged = true;
    while (merged) {
      merged = false;
      for (let i = 0; i < list.length && !merged; i++)
        for (let j = i + 1; j < list.length; j++) {
          const a = list[i];
          const c = list[j];
          if (a[0] <= c[2] && c[0] <= a[2] && a[1] <= c[3] && c[1] <= a[3]) {
            list[i] = [Math.min(a[0], c[0]), Math.min(a[1], c[1]), Math.max(a[2], c[2]), Math.max(a[3], c[3])];
            list.splice(j, 1);
            merged = true;
            break;
          }
        }
    }
    return list.map(([x0, y0, x1, y1]) => {
      const dx0 = Math.floor(base[0] * x0 + base[4]);
      const dy0 = Math.floor(base[3] * y0 + base[5]);
      const dx1 = Math.ceil(base[0] * x1 + base[4]);
      const dy1 = Math.ceil(base[3] * y1 + base[5]);
      return { dx0, dy0, dx1, dy1, x0: (dx0 - base[4]) / base[0] - pad, y0: (dy0 - base[5]) / base[3] - pad, x1: (dx1 - base[4]) / base[0] + pad, y1: (dy1 - base[5]) / base[3] + pad };
    });
  };
  const keepBoxes = (frames, list) => {
    for (const index of list) {
      const b = frames[index].box;
      const slot = lastBox[index] ?? (lastBox[index] = new Float64Array(4));
      slot[0] = b[0];
      slot[1] = b[1];
      slot[2] = b[2];
      slot[3] = b[3];
    }
  };
  const pathOf = (op, d) => {
    if (op.last === d) return op.path;
    op.last = d;
    op.path = d ? new Path2D(d) : null;
    stats.parsed++;
    return op.path;
  };
  const linkMatrix = (link, f) => {
    if (link.m) return link.m;
    const t = f.t[link.r];
    if (t === "\u0002") return f.tm[link.r];
    return cpMatrixOf(t);
  };
  const chainMatrix = (chain, f) => {
    if (chain.length === 1) return linkMatrix(chain[0], f);
    let M = null;
    for (let i = 0; i < chain.length; i++) {
      const m = linkMatrix(chain[i], f);
      if (m) M = cpMul(M, m);
    }
    return M;
  };
  const tableOf = (op) => {
    let table = faceCache.get(op.rampKey);
    if (!table) {
      table = new Array(4 * CP_Q + 1);
      for (let q = 0; q <= 4 * CP_Q; q++) {
        const band = Math.min(3, Math.floor(q / CP_Q));
        const f = (q - CP_Q * band) / CP_Q;
        const a = op.ramp[band];
        const b = op.ramp[band + 1];
        table[q] = cpCss([a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f, a[3] + (b[3] - a[3]) * f]);
      }
      faceCache.set(op.rampKey, table);
    }
    return table;
  };
  const flat = (op) => ({
    kind: op.effect ? 4 : op.tag === "text" ? 3 : op.type === "dot" ? 2 : op.type === "path" || op.type === "slot" || op.type === "face" ? 1 : 0,
    face: op.type === "face",
    r: op.r ?? -1,
    chain: op.chain && op.chain.length ? op.chain : null,
    alpha: op.alpha ?? 1,
    roles: op.alphaRoles && op.alphaRoles.length ? Int32Array.from(op.alphaRoles) : null,
    varies: op.varies && op.varies.length ? op.varies : null,
    fill: op.style ? (op.type === "dot" ? op.style.fillRaw : op.style.fill) : null,
    stroke: op.style ? op.style.stroke : null,
    width: op.style ? op.style.width : 0,
    cap: op.style ? op.style.cap : "butt",
    join: op.style ? op.style.join : "miter",
    miter: op.style ? op.style.miter : 4,
    dash: op.style && op.style.dash ? op.style.dash : null,
    dashOffset: op.style ? op.style.dashOffset : 0,
    evenodd: Boolean(op.style && op.style.rule === "evenodd"),
    nonScaling: Boolean(op.style && op.style.nonScaling),
    table: op.type === "face" ? tableOf(op) : null,
    size: op.size ?? 0,
    path: op.path ?? null,
    last: null,
    text: op.text ?? "",
    x: op.x ?? 0,
    y: op.y ?? 0,
    font: op.font ?? "",
    align: op.align ?? "left",
    spacing: op.spacing ?? "0px",
    baseline: op.baseline ?? "alphabetic",
    effect: op.effect ?? null,
  });
  for (let i = 0; i < programs.length; i++) if (programs[i]) programs[i] = programs[i].map(flat);
  const paintLayer = (layer, frames, rect = null) => {
    const ctx = layer.ctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    if (rect) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(rect.dx0, rect.dy0, rect.dx1 - rect.dx0, rect.dy1 - rect.dy0);
      ctx.clip();
      ctx.clearRect(rect.dx0, rect.dy0, rect.dx1 - rect.dx0, rect.dy1 - rect.dy0);
    } else ctx.clearRect(0, 0, layer.canvas.width, layer.canvas.height);
    const b0 = base[0];
    const b3 = base[3];
    const b4 = base[4];
    const b5 = base[5];
    ctx.setTransform(b0, 0, 0, b3, b4, b5);
    const order = controller.orderIndex(layer.name);
    if (!order) {
      if (rect) ctx.restore();
      return;
    }
    const unitLine = 1 / cssPerUnit;
    const seen = { x0: -b4 / b0 - 4, y0: -b5 / b3 - 4, x1: (layer.canvas.width - b4) / b0 + 4, y1: (layer.canvas.height - b5) / b3 + 4 };
    let curFill = null;
    let curStroke = null;
    let curWidth = -1;
    let curCap = null;
    let curJoin = null;
    let curMiter = -1;
    let curAlpha = 1;
    let curTransformed = false;
    let curBx = b4;
    let curBy = b5;
    let dashed = false;
    let opCount = 0;
    ctx.globalAlpha = 1;
    const drawRun = (index) => {
      const ops = programs[index];
      if (ops === undefined) return;
      const f = frames[index];
      const fb = f.box;
      if (fb[0] > seen.x1 || fb[2] < seen.x0 || fb[1] > seen.y1 || fb[3] < seen.y0) return;
      if (rect !== null) {
        if (fb[0] > rect.x1 || fb[2] < rect.x0 || fb[1] > rect.y1 || fb[3] < rect.y0) return;
        stats.touched++;
      }
      const bx = b4 + b0 * (f.sx || 0);
      const by = b5 + b3 * (f.sy || 0);
      if (bx !== curBx || by !== curBy) {
        curBx = bx;
        curBy = by;
        curTransformed = true;
      }
      for (let o = 0; o < ops.length; o++) {
        const op = ops[o];
        const kind = op.kind;
        if (kind === 4) {
          ctx.setTransform(1, 0, 0, 1, 0, 0);
          ctx.globalAlpha = 1;
          curAlpha = 1;
          op.effect(ctx, layer.canvas.width, layer.canvas.height, base, rect);
          ctx.setTransform(b0, 0, 0, b3, bx, by);
          curTransformed = false;
          curFill = null;
          curStroke = null;
          curWidth = -1;
          curCap = null;
          curJoin = null;
          curMiter = -1;
          continue;
        }
        let alpha = op.alpha;
        const roles = op.roles;
        if (roles !== null) for (let k = 0; k < roles.length; k++) alpha *= f.o[roles[k]];
        const varies = op.varies;
        if (varies !== null) for (let k = 0; k < varies.length; k++) alpha *= varies[k]();
        if (alpha <= 0.001) continue;
        let path = null;
        let token = null;
        let fill = op.fill;
        let stroke = op.stroke;
        let width = op.width;
        if (kind === 1) {
          const d = f.d[op.r];
          if (d === "" || d === undefined) continue;
          if (d.charCodeAt(0) === 1) token = d;
          else {
            path = pathOf(op, d);
            if (path === null) continue;
          }
          if (op.face) {
            fill = stroke = op.table[f.q[op.r]];
            const w = f.w[op.r];
            if (w >= 0) width = w;
          }
        } else if (kind === 2) {
          alpha *= f.o[op.r];
          if (alpha <= 0.001) continue;
        } else path = op.path;
        opCount++;
        if (alpha !== curAlpha) {
          ctx.globalAlpha = alpha;
          curAlpha = alpha;
        }
        const M = op.chain !== null ? chainMatrix(op.chain, f) : null;
        if (kind === 3) {
          const T = cpMul(base, M);
          ctx.setTransform(T[0], T[1], T[2], T[3], T[4], T[5]);
          curTransformed = true;
          ctx.font = op.font;
          ctx.textAlign = op.align;
          ctx.textBaseline = op.baseline;
          if ("letterSpacing" in ctx) ctx.letterSpacing = op.spacing;
          if (fill) {
            ctx.fillStyle = fill;
            curFill = fill;
            ctx.fillText(op.text, op.x, op.y);
          }
          continue;
        }
        let drawPath = path;
        const plain = M === null || (M[0] === 1 && M[1] === 0 && M[2] === 0 && M[3] === 1);
        if (kind === 2) {
          if (M === null) {
            if (curTransformed) {
              ctx.setTransform(b0, 0, 0, b3, bx, by);
              curTransformed = false;
            }
          } else {
            ctx.setTransform(b0 * M[0], b3 * M[1], b0 * M[2], b3 * M[3], b0 * M[4] + bx, b3 * M[5] + by);
            curTransformed = true;
          }
          ctx.beginPath();
          ctx.arc(f.cx[op.r], f.cy[op.r], op.size, 0, 6.283185307179586);
          drawPath = null;
        } else if (M === null) {
          if (curTransformed) {
            ctx.setTransform(b0, 0, 0, b3, bx, by);
            curTransformed = false;
          }
          if (token !== null) cpTrace(ctx, token, f.arena);
        } else if (plain || !(stroke && op.nonScaling)) {
          ctx.setTransform(b0 * M[0], b3 * M[1], b0 * M[2], b3 * M[3], b0 * M[4] + bx, b3 * M[5] + by);
          curTransformed = true;
          if (token !== null) cpTrace(ctx, token, f.arena);
        } else {
          if (curTransformed) {
            ctx.setTransform(b0, 0, 0, b3, bx, by);
            curTransformed = false;
          }
          drawPath = new Path2D();
          drawPath.addPath(path, new DOMMatrix(M));
        }
        if (fill !== null) {
          if (fill !== curFill) {
            ctx.fillStyle = fill;
            curFill = fill;
          }
          if (drawPath !== null) op.evenodd ? ctx.fill(drawPath, "evenodd") : ctx.fill(drawPath);
          else op.evenodd ? ctx.fill("evenodd") : ctx.fill();
        }
        if (stroke !== null && width > 0) {
          if (stroke !== curStroke) {
            ctx.strokeStyle = stroke;
            curStroke = stroke;
          }
          const scaleOf = M === null || plain ? 1 : Math.sqrt(Math.abs(M[0] * M[3] - M[1] * M[2]));
          const lw = op.nonScaling ? (width * unitLine) / (drawPath === path ? scaleOf : 1) : width;
          if (lw !== curWidth) {
            ctx.lineWidth = lw;
            curWidth = lw;
          }
          if (op.cap !== curCap) {
            ctx.lineCap = op.cap;
            curCap = op.cap;
          }
          if (op.join !== curJoin) {
            ctx.lineJoin = op.join;
            curJoin = op.join;
          }
          if (op.miter !== curMiter) {
            ctx.miterLimit = op.miter;
            curMiter = op.miter;
          }
          if (op.dash !== null || dashed) {
            ctx.setLineDash(op.dash !== null ? op.dash.map((v) => v * (op.nonScaling ? unitLine : 1)) : []);
            ctx.lineDashOffset = op.dashOffset;
            dashed = op.dash !== null;
          }
          if (drawPath !== null) ctx.stroke(drawPath);
          else ctx.stroke();
        }
      }
        };
    if (below) {
      for (let at = 0; at < order.length; at++) if (below(order[at])) drawRun(order[at]);
      if (mask) {
        mask(ctx, rect);
        ctx.setTransform(b0, 0, 0, b3, b4, b5);
        curTransformed = false;
        curBx = b4;
        curBy = b5;
        curFill = null;
        curStroke = null;
        curWidth = -1;
        curAlpha = -1;
        ctx.globalAlpha = 1;
      }
      for (let at = 0; at < order.length; at++) if (!below(order[at])) drawRun(order[at]);
    } else for (let at = 0; at < order.length; at++) drawRun(order[at]);
    stats.ops += opCount;
    stats.paths += opCount;
    if (dashed) ctx.setLineDash([]);
    if (rect) ctx.restore();
  };
  let numeric = Boolean(controller.numeric);
  let incremental = false;
  let forceFull = true;
  const paint = () => {
    const t = performance.now();
    stats.ops = 0;
    stats.paths = 0;
    stats.parsed = 0;
    stats.touched = 0;
    stats.rects = 0;
    stats.full = 0;
    const frames = controller.frames();
    const emitted = controller.emitted ? controller.emitted() : null;
    const all = runs.length;
    const full = !incremental || forceFull || !painted || !emitted || emitted.length > all * 0.25;
    if (full) {
      for (const layer of layers) paintLayer(layer, frames);
      stats.full = 1;
      keepBoxes(frames, runs.map((run) => run.index));
    } else if (emitted.length || effectRuns.length) {
      const dirty = effectRuns.length ? emitted.concat(effectRuns) : emitted;
      const rects = rectsOf(frames, dirty);
      stats.rects = rects.length;
      for (const rect of rects) for (const layer of layers) paintLayer(layer, frames, rect);
      keepBoxes(frames, dirty);
    }
    painted = true;
    forceFull = false;
    stats.ms = performance.now() - t;
    return stats;
  };
  resize();
  for (const layer of layers) {
    if (keepSvg) layer.svg.style.visibility = "hidden";
    else layer.svg.remove();
  }
  let watching = null;
  if (typeof ResizeObserver === "function") {
    watching = new ResizeObserver(() => {
      resize();
      paint();
    });
    watching.observe(stage);
  }
  const dprQuery = () => {
    const mq = matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`);
    mq.addEventListener("change", () => {
      resize();
      paint();
      dprQuery();
    }, { once: true });
  };
  dprQuery();
  stage.setAttribute("data-painter", "canvas");
  return {
    paint,
    base: () => base.slice(),
    stats: () => ({ ...stats }),
    compileMs,
    exportSvg({ gl = true, background = true } = {}) {
      controller.snapshot();
      const out = [];
      for (const child of stage.children) {
        const tag = child.localName;
        if (tag === "svg") {
          if (child.style.visibility === "hidden") continue;
          out.push(`<g class="${child.getAttribute("class") || ""}">${child.innerHTML}</g>`);
        } else if (tag === "canvas") {
          const layer = layers.find((l) => l.canvas === child);
          if (layer) out.push(`<g class="${layer.svg.getAttribute("class") || ""}">${layer.svg.innerHTML}</g>`);
          else if (gl && child.width > 1 && getComputedStyle(child).display !== "none") out.push(`<image href="${child.toDataURL("image/png")}" x="0" y="0" width="${W}" height="${H}" preserveAspectRatio="none"/>`);
        }
      }
      const css = [...document.querySelectorAll("style")].map((s) => s.textContent).join("\n");
      const holder = stage.closest("[data-theme]");
      const theme = holder ? holder.getAttribute("data-theme") : "dark";
      const card = getComputedStyle(stage.closest(".iso-plate") || stage).getPropertyValue("--anatomy-card").trim() || "#fff";
      return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}"><style>${css.replace(/<\/style/gi, "")}</style><g class="iso" data-theme="${theme}"><g class="iso-plate" style="padding:0;box-shadow:none">${background ? `<rect width="${W}" height="${H}" fill="${card}"/>` : ""}<g class="${stage.getAttribute("class") || ""}">${out.join("")}</g></g></g></svg>`;
    },
    set incremental(on) {
      incremental = Boolean(on);
      forceFull = true;
    },
    get incremental() {
      return incremental;
    },
    flush() {
      for (const layer of layers) layer.ctx.getImageData(0, 0, 1, 1);
    },
    get numeric() {
      return numeric;
    },
    unsupported: [...unsupported],
    layers: layers.map((l) => l.name),
    ops: programs.reduce((sum, ops) => sum + (ops ? ops.length : 0), 0),
    zoom(z, cx = W / 2, cy = H / 2) {
      view.zoom = z;
      view.cx = cx;
      view.cy = cy;
      resize();
      return paint();
    },
    destroy() {
      if (watching) watching.disconnect();
    },
  };
}

export const CANVAS2D = { mount: cpMount };
