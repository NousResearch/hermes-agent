function lineEnds({ select = null, tolerance = 1, root: rootSlack = 2, mark = 7, least = 0.3, cover = 0.5, show = false } = {}) {
  const plate = (select && document.querySelector(select)) || document.querySelector(".iso-plate") || document.body;
  const box = plate.getBoundingClientRect();
  const style = document.createElement("style");
  style.textContent = `${select || ".iso-plate"} *{pointer-events:none!important}${select || ".iso-plate"} svg,${select || ".iso-plate"} svg *{pointer-events:visiblePainted!important}`;
  document.head.append(style);
  const skipClass = ["iso-edge", "iso-bevel", "iso-crease", "tb-edge"];
  const svgs = [...plate.querySelectorAll("svg")];
  const shapes = svgs.flatMap((svg) => [...svg.querySelectorAll("path, line, polyline, polygon, circle, ellipse, rect")]).filter((el) => !el.closest("defs, clipPath, mask, pattern, marker, symbol"));
  const opacityOf = (el) => {
    let alpha = 1;
    for (let node = el; node && node.nodeType === 1; node = node.parentElement) {
      const cs = getComputedStyle(node);
      if (cs.display === "none") return 0;
      alpha *= Number(cs.opacity);
      if (node === plate) break;
    }
    return getComputedStyle(el).visibility === "hidden" ? 0 : alpha;
  };
  const numbers = (text) => (text.match(/-?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?/gi) || []).map(Number);
  const subpathsOf = (d) => {
    const out = [];
    let current = null;
    let x = 0;
    let y = 0;
    let sx = 0;
    let sy = 0;
    let control = null;
    const push = (px, py) => {
      current.points.push([px, py]);
      x = px;
      y = py;
    };
    const start = (px, py) => {
      current = { points: [[px, py]], closed: false };
      out.push(current);
      x = sx = px;
      y = sy = py;
    };
    const cubic = (x1, y1, x2, y2, px, py) => {
      const x0 = x;
      const y0 = y;
      for (let i = 1; i <= 8; i++) {
        const t = i / 8;
        const u = 1 - t;
        push(u * u * u * x0 + 3 * u * u * t * x1 + 3 * u * t * t * x2 + t * t * t * px, u * u * u * y0 + 3 * u * u * t * y1 + 3 * u * t * t * y2 + t * t * t * py);
      }
      control = [x2, y2];
    };
    const quad = (x1, y1, px, py) => {
      const x0 = x;
      const y0 = y;
      for (let i = 1; i <= 6; i++) {
        const t = i / 6;
        const u = 1 - t;
        push(u * u * x0 + 2 * u * t * x1 + t * t * px, u * u * y0 + 2 * u * t * y1 + t * t * py);
      }
      control = [x1, y1];
    };
    for (const [, command, args] of d.matchAll(/([MmLlHhVvCcSsQqTtAaZz])([^MmLlHhVvCcSsQqTtAaZz]*)/g)) {
      const v = numbers(args);
      const rel = command === command.toLowerCase();
      const C = command.toUpperCase();
      const ox = () => (rel ? x : 0);
      const oy = () => (rel ? y : 0);
      if (C === "Z") {
        if (current) {
          current.closed = true;
          push(sx, sy);
        }
        current = null;
        continue;
      }
      if (C === "M") {
        for (let i = 0; i + 1 < v.length; i += 2) {
          if (i === 0) start(v[0] + ox(), v[1] + oy());
          else push(v[i] + ox(), v[i + 1] + oy());
        }
        control = null;
        continue;
      }
      if (!current) start(x, y);
      if (C === "L") for (let i = 0; i + 1 < v.length; i += 2) push(v[i] + ox(), v[i + 1] + oy());
      else if (C === "H") for (const value of v) push(value + ox(), y);
      else if (C === "V") for (const value of v) push(x, value + oy());
      else if (C === "C") for (let i = 0; i + 5 < v.length; i += 6) cubic(v[i] + ox(), v[i + 1] + oy(), v[i + 2] + ox(), v[i + 3] + oy(), v[i + 4] + ox(), v[i + 5] + oy());
      else if (C === "S")
        for (let i = 0; i + 3 < v.length; i += 4) {
          const reflected = control ? [2 * x - control[0], 2 * y - control[1]] : [x, y];
          cubic(reflected[0], reflected[1], v[i] + ox(), v[i + 1] + oy(), v[i + 2] + ox(), v[i + 3] + oy());
        }
      else if (C === "Q") for (let i = 0; i + 3 < v.length; i += 4) quad(v[i] + ox(), v[i + 1] + oy(), v[i + 2] + ox(), v[i + 3] + oy());
      else if (C === "T")
        for (let i = 0; i + 1 < v.length; i += 2) {
          const reflected = control ? [2 * x - control[0], 2 * y - control[1]] : [x, y];
          quad(reflected[0], reflected[1], v[i] + ox(), v[i + 1] + oy());
        }
      else if (C === "A") for (let i = 0; i + 6 < v.length; i += 7) push(v[i + 5] + ox(), v[i + 6] + oy());
      if (C !== "C" && C !== "S" && C !== "Q" && C !== "T") control = null;
    }
    return out;
  };
  const geometryOf = (el) => {
    const tag = el.tagName.toLowerCase();
    const num = (name) => Number(el.getAttribute(name) || 0);
    if (tag === "path") return subpathsOf(el.getAttribute("d") || "");
    if (tag === "line") return [{ points: [[num("x1"), num("y1")], [num("x2"), num("y2")]], closed: false }];
    if (tag === "polyline" || tag === "polygon") {
      const v = numbers(el.getAttribute("points") || "");
      const points = [];
      for (let i = 0; i + 1 < v.length; i += 2) points.push([v[i], v[i + 1]]);
      if (tag === "polygon" && points.length) points.push(points[0]);
      return [{ points, closed: tag === "polygon" }];
    }
    if (tag === "rect") {
      const [x, y, w, h] = [num("x"), num("y"), num("width"), num("height")];
      return [{ points: [[x, y], [x + w, y], [x + w, y + h], [x, y + h], [x, y]], closed: true }];
    }
    return [];
  };
  const cell = 6;
  const grid = new Map();
  const add = (gx, gy, entry) => {
    const id = `${gx},${gy}`;
    if (!grid.has(id)) grid.set(id, []);
    grid.get(id).push(entry);
  };
  const lines = [];
  let scale = 1;
  for (const el of shapes) {
    const cs = getComputedStyle(el);
    const alpha = opacityOf(el);
    if (alpha < 0.1) continue;
    const ctm = el.getScreenCTM();
    if (!ctm) continue;
    const map = ([px, py]) => [ctm.a * px + ctm.c * py + ctm.e, ctm.b * px + ctm.d * py + ctm.f];
    const tag = el.tagName.toLowerCase();
    const stroked = cs.stroke !== "none" && parseFloat(cs.strokeWidth) > 0 && Number(cs.strokeOpacity) > 0.05;
    const filled = cs.fill !== "none" && Number(cs.fillOpacity) > 0.05;
    if (tag === "circle" || tag === "ellipse") {
      if (!stroked && !filled) continue;
      const c = map([Number(el.getAttribute("cx") || 0), Number(el.getAttribute("cy") || 0)]);
      const r = Number(el.getAttribute("r") || el.getAttribute("rx") || 0) * Math.hypot(ctm.a, ctm.b);
      const entry = { el, kind: "dot", c, r };
      for (let gx = Math.floor((c[0] - r) / cell); gx <= Math.floor((c[0] + r) / cell); gx++) for (let gy = Math.floor((c[1] - r) / cell); gy <= Math.floor((c[1] + r) / cell); gy++) add(gx, gy, entry);
      continue;
    }
    if (!stroked) continue;
    const owner = el.closest("svg");
    const viewCtm = owner && owner.getScreenCTM();
    if (viewCtm) scale = Math.hypot(viewCtm.a, viewCtm.b) || scale;
    const subpaths = geometryOf(el).map((sub, index) => ({ ...sub, index, flat: sub.points.map(map) }));
    subpaths.forEach((sub) => {
      for (let i = 0; i < sub.flat.length - 1; i++) {
        const a = sub.flat[i];
        const b = sub.flat[i + 1];
        const entry = { el, kind: "segment", a, b, sub: sub.index, i };
        for (let gx = Math.floor(Math.min(a[0], b[0]) / cell); gx <= Math.floor(Math.max(a[0], b[0]) / cell); gx++) for (let gy = Math.floor(Math.min(a[1], b[1]) / cell); gy <= Math.floor(Math.max(a[1], b[1]) / cell); gy++) add(gx, gy, entry);
      }
    });
    const checked = el.classList.contains("iso-line") && !skipClass.some((name) => el.classList.contains(name)) && alpha >= least;
    if (checked) lines.push({ el, subpaths, alpha, free: el.getAttribute("data-free") });
  }
  const near = (p, reach) => {
    const out = new Set();
    for (let gx = Math.floor((p[0] - reach) / cell); gx <= Math.floor((p[0] + reach) / cell); gx++) for (let gy = Math.floor((p[1] - reach) / cell); gy <= Math.floor((p[1] + reach) / cell); gy++) for (const entry of grid.get(`${gx},${gy}`) || []) out.add(entry);
    return out;
  };
  const segmentDistance = (p, a, b) => {
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    const span = dx * dx + dy * dy;
    const t = span > 0 ? Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / span)) : 0;
    return Math.hypot(p[0] - a[0] - dx * t, p[1] - a[1] - dy * t);
  };
  const lengthOf = (flat) => flat.reduce((sum, p, i) => (i ? sum + Math.hypot(p[0] - flat[i - 1][0], p[1] - flat[i - 1][1]) : 0), 0);
  const hidden = (el, p) => {
    const stack = document.elementsFromPoint(p[0], p[1]);
    const at = stack.indexOf(el);
    const above = at < 0 ? stack : stack.slice(0, at);
    return above.some((node) => {
      if (!(node instanceof SVGGeometryElement) || node.contains(el)) return false;
      const cs = getComputedStyle(node);
      return cs.fill !== "none" && Number(cs.fillOpacity) * opacityOf(node) >= cover && node.isPointInFill(new DOMPoint(...inverse(node, p)));
    });
  };
  const inverse = (node, p) => {
    const m = node.getScreenCTM().inverse();
    return [m.a * p[0] + m.c * p[1] + m.e, m.b * p[0] + m.d * p[1] + m.f];
  };
  const clearance = (line, sub, end, p, reach) => {
    let best = Infinity;
    const own = sub.flat;
    const skipNear = 2 * reach;
    const along = [0];
    for (let i = 1; i < own.length; i++) along.push(along[i - 1] + Math.hypot(own[i][0] - own[i - 1][0], own[i][1] - own[i - 1][1]));
    const total = along[along.length - 1];
    for (const entry of near(p, reach * 4)) {
      if (entry.kind === "dot") {
        best = Math.min(best, Math.max(0, Math.hypot(p[0] - entry.c[0], p[1] - entry.c[1]) - entry.r));
        continue;
      }
      if (entry.el === line.el && entry.sub === sub.index) {
        const from = along[entry.i];
        const to = along[entry.i + 1];
        const gapFrom = end === "start" ? from : total - to;
        if (gapFrom < skipNear) continue;
      }
      best = Math.min(best, segmentDistance(p, entry.a, entry.b));
    }
    return best;
  };
  const describe = (el) => {
    const item = el.closest("[data-part], [data-name], [data-item], .it");
    const tag = item && item !== el ? (item.getAttribute("data-part") || item.getAttribute("data-name") || item.getAttribute("data-item") || `it k ${item.getAttribute("data-k") || ""}`) : "";
    return `${el.getAttribute("class")}${el.getAttribute("data-tone") ? ` ${el.getAttribute("data-tone")}` : ""}${tag ? ` in ${tag}` : ""}`;
  };
  const problems = [];
  let ends = 0;
  const slack = tolerance * scale;
  const rooted = rootSlack * scale;
  for (const line of lines) {
    for (const sub of line.subpaths) {
      if (sub.closed || sub.flat.length < 2) continue;
      const first = sub.flat[0];
      const last = sub.flat[sub.flat.length - 1];
      if (Math.hypot(first[0] - last[0], first[1] - last[1]) < 0.05 * scale) continue;
      const length = lengthOf(sub.flat);
      if (length < 0.2 * scale) continue;
      const result = {};
      for (const [end, p] of [["start", first], ["end", last]]) {
        if (line.free === "" || line.free === "both" || line.free === end) {
          result[end] = 0;
          continue;
        }
        ends++;
        result[end] = hidden(line.el, p) ? 0 : clearance(line, sub, end, p, Math.max(slack, rooted));
      }
      const straight = sub.flat.length === 2 || sub.flat.every((q) => segmentDistance(q, first, last) < 0.15 * scale);
      const isMark = straight && length <= mark * scale;
      const fail = (end) => result[end] !== undefined && result[end] > slack;
      let bad = [];
      if (isMark) {
        const best = Math.min(result.start ?? Infinity, result.end ?? Infinity);
        if (!(best <= rooted)) bad = ["start", "end"].filter((end) => result[end] !== undefined && result[end] > 0);
      } else bad = ["start", "end"].filter(fail);
      for (const end of bad) {
        const p = end === "start" ? first : last;
        problems.push({ what: describe(line.el), d: (line.el.getAttribute("d") || "").slice(0, 36), end, mark: isMark, gap: result[end] / scale, at: [Math.round((p[0] - box.left) * 10) / 10, Math.round((p[1] - box.top) * 10) / 10], client: p, opacity: Math.round(line.alpha * 100) / 100, length: Math.round((length / scale) * 10) / 10 });
      }
    }
  }
  style.remove();
  problems.sort((a, b) => b.gap - a.gap);
  if (show) {
    const ns = "http://www.w3.org/2000/svg";
    const overlay = document.createElementNS(ns, "svg");
    overlay.setAttribute("data-line-ends", "");
    overlay.setAttribute("style", `position:fixed;left:0;top:0;width:${innerWidth}px;height:${innerHeight}px;pointer-events:none;z-index:2147483647;overflow:visible`);
    for (const problem of problems) {
      const ring = document.createElementNS(ns, "circle");
      ring.setAttribute("cx", problem.client[0]);
      ring.setAttribute("cy", problem.client[1]);
      ring.setAttribute("r", 5);
      ring.setAttribute("style", "fill:none;stroke:#ff0050;stroke-width:1.5");
      overlay.append(ring);
    }
    document.body.append(overlay);
  }
  return { ends, lines: lines.length, problems: problems.map(({ client, ...rest }) => rest) };
}

export const LINE_ENDS = lineEnds.toString();
