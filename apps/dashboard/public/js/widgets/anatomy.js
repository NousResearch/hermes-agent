// 3D Anatomy Explorer — adaptive-fidelity anatomical model for the Health page.
// Tier B (procedural three.js 3D) on WebGL devices, Tier C (interactive 2D SVG
// body) everywhere else, sharing one data layer (structures + condition map)
// and one highlight bus (window "hub:anatomy-highlight"). Selecting a structure
// can hand it to the SA MedBot via the existing "hub:medbot-ask" bridge.
// See ANATOMY.md for the full design.

import { h, clear } from "../utils.js";

const SVGNS = "http://www.w3.org/2000/svg";
const svgEl = (tag, attrs = {}) => {
  const el = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
};

let DATA = null;
async function loadData() {
  if (DATA) return DATA;
  const [structures, conditions] = await Promise.all([
    fetch("/anatomy/structures.json").then((r) => r.json()),
    fetch("/anatomy/conditions.json").then((r) => r.json()),
  ]);
  const byId = Object.fromEntries(structures.structures.map((s) => [s.id, s]));
  DATA = { layers: structures.layers, structures: structures.structures, byId,
    conditions: conditions.conditions };
  return DATA;
}

// Detect the best renderer for this device (or honour a manual override).
function detectTier(quality) {
  if (quality === "2d") return "2d";
  let webgl = false;
  try {
    const c = document.createElement("canvas");
    webgl = !!(c.getContext("webgl2") || c.getContext("webgl"));
  } catch { webgl = false; }
  if (quality === "3d") return webgl ? "3d" : "2d";
  // auto
  if (!webgl) return "2d";
  const mem = navigator.deviceMemory || 4;
  const cores = navigator.hardwareConcurrency || 4;
  return (mem >= 2 && cores >= 2) ? "3d" : "2d";
}

// Front-view 2D schematic. Each entry → an SVG shape tagged with its structure.
// Coordinates in a 240×480 viewBox. Anterior organs hidden in the back view.
const ANTERIOR = new Set(["heart", "lungs", "liver", "stomach", "intestines",
  "bladder", "spleen", "pancreas"]);
function build2DRegions() {
  const R = [];
  const add = (id, tag, attrs) => R.push({ id, el: svgEl(tag, attrs) });
  // skin silhouette (single clickable body outline)
  add("skin", "path", { d:
    "M120 20 C104 20 96 32 96 48 C96 60 102 68 110 72 L96 84 C84 88 78 100 78 120 "
    + "L70 96 L58 100 L66 150 L74 150 L82 128 L82 250 L96 250 L100 300 L96 452 L114 452 "
    + "L118 300 L122 300 L126 452 L144 452 L140 300 L144 250 L158 250 L158 128 L166 150 "
    + "L174 150 L182 100 L170 96 L162 120 C162 100 156 88 144 84 L130 72 C138 68 144 60 "
    + "144 48 C144 32 136 20 120 20 Z", class: "an2-skin" });
  // skeleton
  add("skull", "ellipse", { cx: 120, cy: 50, rx: 22, ry: 26, class: "an2-bone" });
  add("spine", "rect", { x: 116, y: 84, width: 8, height: 172, rx: 3, class: "an2-bone" });
  add("ribcage", "ellipse", { cx: 120, cy: 140, rx: 42, ry: 48, fill: "none", "stroke-width": 4, class: "an2-bone-outline" });
  add("pelvis", "ellipse", { cx: 120, cy: 272, rx: 40, ry: 22, fill: "none", "stroke-width": 5, class: "an2-bone-outline" });
  add("arm_bones", "path", { d: "M92 96 L64 150 M148 96 L176 150", "stroke-width": 6, class: "an2-bone-outline", fill: "none" });
  add("leg_bones", "path", { d: "M108 292 L104 448 M132 292 L136 448", "stroke-width": 7, class: "an2-bone-outline", fill: "none" });
  // organs
  add("brain", "ellipse", { cx: 120, cy: 46, rx: 17, ry: 13, class: "an2-organ" });
  add("lungs", "path", { d: "M90 120 a14 22 0 1 0 28 0 a14 22 0 1 0 -28 0 Z M122 120 a14 22 0 1 0 28 0 a14 22 0 1 0 -28 0 Z", class: "an2-organ" });
  add("heart", "ellipse", { cx: 113, cy: 118, rx: 12, ry: 14, class: "an2-organ" });
  add("liver", "ellipse", { cx: 100, cy: 168, rx: 22, ry: 14, class: "an2-organ" });
  add("stomach", "ellipse", { cx: 140, cy: 165, rx: 15, ry: 12, class: "an2-organ" });
  add("spleen", "ellipse", { cx: 154, cy: 178, rx: 8, ry: 10, class: "an2-organ" });
  add("pancreas", "ellipse", { cx: 120, cy: 182, rx: 20, ry: 6, class: "an2-organ" });
  add("kidneys", "path", { d: "M96 198 a8 12 0 1 0 16 0 a8 12 0 1 0 -16 0 Z M128 198 a8 12 0 1 0 16 0 a8 12 0 1 0 -16 0 Z", class: "an2-organ" });
  add("intestines", "ellipse", { cx: 120, cy: 220, rx: 30, ry: 26, class: "an2-organ" });
  add("bladder", "ellipse", { cx: 120, cy: 268, rx: 13, ry: 11, class: "an2-organ" });
  add("thyroid", "ellipse", { cx: 120, cy: 90, rx: 10, ry: 5, class: "an2-organ" });
  add("trachea", "rect", { x: 116, y: 94, width: 8, height: 16, rx: 3, class: "an2-organ" });
  add("gallbladder", "ellipse", { cx: 90, cy: 176, rx: 5, ry: 7, class: "an2-organ" });
  add("diaphragm", "path", { d: "M80 150 Q120 138 160 150", fill: "none", "stroke-width": 4, class: "an2-organ" });
  return R;
}

export default {
  type: "anatomy",
  title: "Anatomy Explorer",
  icon: "🧍",
  defaultSize: "xl",

  render(body, ctx) {
    const { store } = ctx;
    if (!store.state.anatomy) store.state.anatomy = {};
    const S = store.state.anatomy;
    S.quality = S.quality || "auto";
    S.view = S.view || "front";
    S.layers = S.layers || { skin: false, muscle: false, skeleton: true, organ: true };

    let engine = null;      // active renderer with { highlight(ids), setLayer(id,on), select(id), dispose() }
    let selectHandler = () => {};

    const persist = () => store.update((s) => { s.anatomy = S; }, "anatomy");

    const draw = async () => {
      clear(body).append(h("div.widget-loading", {}, "LOADING ANATOMY…"));
      let data;
      try { data = await loadData(); }
      catch (err) { clear(body).append(h("div.widget-error", {}, `Anatomy data unavailable: ${err.message}`)); return; }

      const tier = detectTier(S.quality);

      // ---- shell: controls rail + viewport + info panel ----
      const info = h("div.an-info", {}, h("div.muted.small", {}, "Select a structure, or search a condition."));
      const showStructure = (id) => {
        const st = data.byId[id];
        if (!st) return;
        clear(info).append(
          h("div.an-info-name", {}, st.name),
          h("div.an-info-layer.muted.small", {}, data.layers.find((l) => l.id === st.layer)?.name || st.layer),
          h("div.an-info-blurb.small", {}, st.blurb),
          h("button.btn.btn-tiny.an-ask", { type: "button",
            onclick: () => window.dispatchEvent(new CustomEvent("hub:medbot-ask",
              { detail: { text: `Explain the ${st.name.toLowerCase()} — key clinical points for a South African context.` } })),
          }, "Ask SA MedBot about this"));
        S.selected = id; persist();
      };
      selectHandler = showStructure;

      // layer toggles
      const layerRow = h("div.an-layers");
      for (const layer of data.layers) {
        const on = !!S.layers[layer.id];
        const btn = h("button.an-layer", { type: "button",
          class: `an-layer ${on ? "on" : ""}`, "aria-pressed": String(on),
          "data-layer": layer.id,
        }, h("span.an-swatch", { style: `background:${layer.color}` }), layer.name);
        btn.addEventListener("click", () => {
          S.layers[layer.id] = !S.layers[layer.id];
          btn.classList.toggle("on", S.layers[layer.id]);
          btn.setAttribute("aria-pressed", String(S.layers[layer.id]));
          engine?.setLayer(layer.id, S.layers[layer.id]);
          persist();
        });
        layerRow.append(btn);
      }

      // condition search
      const condSelect = h("select.select.an-cond", { "aria-label": "Condition" },
        h("option", { value: "" }, "Highlight a condition…"),
        ...data.conditions.map((c) => h("option", { value: c.slug }, c.name)));
      condSelect.addEventListener("change", () => {
        const cond = data.conditions.find((c) => c.slug === condSelect.value);
        if (!cond) { engine?.highlight([]); return; }
        applyCondition(cond);
      });

      // search-to-structure
      const focusStructure = (id) => {
        if (!data.byId[id]) return;
        engine?.highlight([id]); engine?.focus?.(id); showStructure(id);
      };
      const listId = `an-struct-${Math.random().toString(36).slice(2)}`;
      const search = h("input.input.an-search", { type: "search", list: listId,
        placeholder: "Find a structure…", "aria-label": "Find a structure" });
      const datalist = h("datalist", { id: listId },
        ...data.structures.map((s) => h("option", { value: s.name })));
      const runSearch = () => {
        const q = search.value.trim().toLowerCase();
        if (!q) return;
        const hit = data.structures.find((s) => s.name.toLowerCase() === q)
          || data.structures.find((s) => s.name.toLowerCase().includes(q));
        if (hit) focusStructure(hit.id);
      };
      search.addEventListener("change", runSearch);
      search.addEventListener("keydown", (e) => { if (e.key === "Enter") runSearch(); });

      // view presets + ghost skin
      const VIEWS = [["front", "Front"], ["back", "Back"], ["left", "L"], ["right", "R"], ["top", "Top"]];
      const viewRow = h("div.an-views");
      for (const [v, label] of VIEWS) {
        const b = h("button.btn.btn-tiny.an-view", { type: "button", "data-view": v }, label);
        b.addEventListener("click", () => { S.view = v; persist(); engine?.setView?.(v); });
        viewRow.append(b);
      }
      const resetBtn = h("button.btn.btn-tiny.an-view", { type: "button" }, "Reset");
      resetBtn.addEventListener("click", () => { S.view = "front"; persist(); engine?.setView?.("reset"); });
      viewRow.append(resetBtn);

      const ghost = h("label.an-ghost", {},
        h("input", { type: "checkbox", checked: !!S.ghost }), "Ghost skin");
      ghost.querySelector("input").addEventListener("change", (e) => {
        S.ghost = e.target.checked; persist(); engine?.setGhost?.(S.ghost);
      });

      // quality selector + high-detail (Tier A) loader
      const qual = h("select.select.an-quality", { "aria-label": "Render quality" },
        ...[["auto", "Auto"], ["3d", "3D"], ["2d", "2D"]].map(([v, l]) =>
          h("option", { value: v, selected: S.quality === v }, l)));
      qual.addEventListener("change", () => { S.quality = qual.value; persist(); draw(); });

      const hdBtn = h("button.btn.btn-tiny.an-hd", { type: "button" }, "Load high-detail model");
      const hdNote = h("div.muted.small", {});
      hdBtn.addEventListener("click", async () => {
        hdBtn.disabled = true; hdNote.textContent = "Checking for high-detail model…";
        const ok = await engine?.loadHighDetail?.((msg) => { hdNote.textContent = msg; });
        hdBtn.disabled = false;
        if (!ok) hdBtn.textContent = "Load high-detail model";
      });

      const viewport = h("div.an-viewport", { class: `an-viewport tier-${tier}` });

      const rail = h("div.an-rail", {},
        h("div.an-rail-group", {}, h("div.an-rail-label", {}, "LAYERS"), layerRow),
        h("div.an-rail-group", {}, h("div.an-rail-label", {}, "FIND"), search, datalist),
        h("div.an-rail-group", {}, h("div.an-rail-label", {}, "VIEW"), viewRow,
          tier === "3d" ? ghost : null),
        h("div.an-rail-group", {}, h("div.an-rail-label", {}, "LEARN"), condSelect),
        h("div.an-rail-group", {}, h("div.an-rail-label", {}, "QUALITY"),
          qual, tier === "3d" ? h("div.an-hd-wrap", {}, hdBtn, hdNote) : null,
          tier === "2d" && S.quality !== "2d"
            ? h("div.muted.small", {}, "3D unavailable on this device — showing 2D.") : null),
        info,
        h("div.muted.small.an-note", {}, "Educational model · verify clinically. three.js (MIT)."));

      clear(body).append(h("div.an-wrap", {}, rail, viewport));

      const applyCondition = (cond) => {
        engine?.highlight(cond.structures);
        clear(info).append(
          h("div.an-info-name", {}, cond.name),
          h("div.small", {}, "Typically involves: ",
            cond.structures.map((sid, i) => h("span", {},
              i ? ", " : "", h("b", {}, data.byId[sid]?.name || sid))))
          , h("button.btn.btn-tiny.an-ask", { type: "button",
            onclick: () => window.dispatchEvent(new CustomEvent("hub:medbot-ask",
              { detail: { text: `${cond.name}: pathophysiology and South African management essentials.` } })),
          }, "Ask SA MedBot about this"));
      };
      ctx._applyCondition = applyCondition;

      // build the active renderer
      if (tier === "3d") {
        try { engine = await build3D(viewport, data, S, showStructure); }
        catch (err) { engine = build2D(viewport, data, S, showStructure); }
      } else {
        engine = build2D(viewport, data, S, showStructure);
      }
      // apply persisted layer visibility
      for (const layer of data.layers) engine.setLayer(layer.id, !!S.layers[layer.id]);
      if (S.selected) showStructure(S.selected);
    };

    // external highlight bus (condition or explicit structures)
    if (highlightHandler) window.removeEventListener("hub:anatomy-highlight", highlightHandler);
    highlightHandler = (ev) => {
      const d = ev.detail || {};
      loadData().then((data) => {
        let ids = d.structures;
        let cond = null;
        if (d.slug) {
          cond = data.conditions.find((c) => c.slug === d.slug
            || c.name.toLowerCase() === String(d.slug).toLowerCase());
        }
        if (!cond && d.text) {
          // match the first condition whose name/slug appears in free text
          const t = String(d.text).toLowerCase();
          cond = data.conditions.find((c) => t.includes(c.slug.replace(/-/g, " "))
            || t.includes(c.name.toLowerCase().replace(/\s*\(.*?\)\s*/g, "").trim()));
        }
        if (cond && ctx._applyCondition) { ctx._applyCondition(cond); return; }
        if (cond) ids = cond.structures;
        if (ids && engine) engine.highlight(ids);
      });
    };
    window.addEventListener("hub:anatomy-highlight", highlightHandler);

    ctx.onRefresh(draw);
    draw();
  },
};

let highlightHandler = null;

// ---------------------------------------------------------------------------
// Tier C — 2D interactive SVG body
// ---------------------------------------------------------------------------
function build2D(viewport, data, S, onSelect) {
  clear(viewport);
  const regions = build2DRegions();
  const svg = svgEl("svg", { viewBox: "0 0 240 480", class: "an2", role: "img",
    "aria-label": "Anatomical body map" });
  const shapes = {}; // id → [elements]
  for (const { id, el } of regions) {
    const st = data.byId[id];
    if (!st) continue;
    el.classList.add("an2-region", `layer-${st.layer}`);
    el.dataset.structure = id;
    el.addEventListener("click", () => { highlight([id]); onSelect(id); });
    (shapes[id] = shapes[id] || []).push(el);
    svg.append(el);
  }
  viewport.append(svg);

  const setLayer = (layerId, on) => {
    for (const st of data.structures) {
      if (st.layer !== layerId) continue;
      const anterior = ANTERIOR.has(st.id);
      const visible = on && !(S.view === "back" && anterior);
      for (const el of shapes[st.id] || []) el.style.display = visible ? "" : "none";
    }
  };
  const highlight = (ids) => {
    const set = new Set(ids);
    for (const [id, els] of Object.entries(shapes))
      for (const el of els) el.classList.toggle("an2-hot", set.has(id));
  };
  const setView = (name) => {
    // 2D supports front/back; other presets fall back to front.
    S.view = (name === "back") ? "back" : "front";
    svg.classList.toggle("an2-back", S.view === "back");
    for (const l of data.layers) setLayer(l.id, !!S.layers[l.id]);
  };
  const focus = (id) => {
    const el = (shapes[id] || [])[0];
    el?.scrollIntoView?.({ block: "nearest", behavior: "smooth" });
  };
  return { setLayer, highlight, select: onSelect, setView, focus,
    setGhost() {}, async loadHighDetail(report) { report?.("High-detail model needs the 3D renderer."); return false; },
    dispose() {} };
}

// ---------------------------------------------------------------------------
// Tier B — procedural three.js 3D
// ---------------------------------------------------------------------------
async function build3D(viewport, data, S, onSelect) {
  const THREE = await import("three");
  clear(viewport);

  const width = viewport.clientWidth || 360;
  const height = viewport.clientHeight || 460;
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(width, height);
  viewport.append(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
  camera.position.set(0, 0.1, 6);
  scene.add(new THREE.AmbientLight(0xffffff, 0.75));
  const key = new THREE.DirectionalLight(0xffffff, 0.9); key.position.set(3, 5, 5); scene.add(key);
  const rim = new THREE.DirectionalLight(0x88aaff, 0.4); rim.position.set(-4, 2, -3); scene.add(rim);

  const pivot = new THREE.Group(); scene.add(pivot);
  const groups = {}; // layer → THREE.Group
  for (const l of data.layers) { groups[l.id] = new THREE.Group(); pivot.add(groups[l.id]); }
  const meshes = []; // pickable
  const colorOf = (id) => data.layers.find((l) => l.id === id)?.color || "#cccccc";

  const mat = (hex, opts = {}) => new THREE.MeshStandardMaterial({
    color: new THREE.Color(hex), roughness: 0.7, metalness: 0.05, ...opts });
  const add = (layer, id, geo, material, pos, scale) => {
    const m = new THREE.Mesh(geo, material);
    if (pos) m.position.set(...pos);
    if (scale) m.scale.set(...scale);
    m.userData.structure = id;
    groups[layer].add(m); meshes.push(m);
    return m;
  };
  const sphere = (r) => new THREE.SphereGeometry(r, 24, 16);
  const capsule = (r, len) => new THREE.CapsuleGeometry(r, len, 6, 16);

  // skin (translucent body) + muscle (opaque inset)
  const skinMat = mat(colorOf("skin"), { transparent: true, opacity: 0.28 });
  const muscleMat = mat(colorOf("muscle"), { roughness: 0.85 });
  for (const [layer, material, s] of [["skin", skinMat, 1.0], ["muscle", muscleMat, 0.9]]) {
    add(layer, layer === "skin" ? "skin" : "musculature", capsule(0.62 * s, 1.5), material, [0, 0.35, 0]);   // torso
    add(layer, layer === "skin" ? "skin" : "musculature", sphere(0.42 * s), material, [0, 1.45, 0]);          // head
    add(layer, layer === "skin" ? "skin" : "musculature", capsule(0.16 * s, 1.1), material, [-0.82, 0.55, 0], null); // arms
    add(layer, layer === "skin" ? "skin" : "musculature", capsule(0.16 * s, 1.1), material, [0.82, 0.55, 0]);
    add(layer, layer === "skin" ? "skin" : "musculature", capsule(0.2 * s, 1.4), material, [-0.28, -1.15, 0]);       // legs
    add(layer, layer === "skin" ? "skin" : "musculature", capsule(0.2 * s, 1.4), material, [0.28, -1.15, 0]);
  }
  // skeleton
  const bone = mat(colorOf("skeleton"));
  add("skeleton", "skull", sphere(0.34), bone, [0, 1.45, 0]);
  add("skeleton", "spine", new THREE.CylinderGeometry(0.07, 0.07, 1.5, 10), bone, [0, 0.35, -0.05]);
  add("skeleton", "ribcage", sphere(0.5), mat(colorOf("skeleton"), { transparent: true, opacity: 0.5, wireframe: true }), [0, 0.75, 0], [1, 0.9, 0.7]);
  add("skeleton", "pelvis", new THREE.TorusGeometry(0.32, 0.1, 8, 16), bone, [0, -0.35, 0], [1, 0.7, 0.6]);
  add("skeleton", "arm_bones", new THREE.CylinderGeometry(0.06, 0.06, 1.2, 8), bone, [-0.82, 0.55, 0]);
  add("skeleton", "arm_bones", new THREE.CylinderGeometry(0.06, 0.06, 1.2, 8), bone, [0.82, 0.55, 0]);
  add("skeleton", "leg_bones", new THREE.CylinderGeometry(0.08, 0.07, 1.5, 8), bone, [-0.28, -1.15, 0]);
  add("skeleton", "leg_bones", new THREE.CylinderGeometry(0.08, 0.07, 1.5, 8), bone, [0.28, -1.15, 0]);
  // organs
  const organ = (id, r, pos, scale) => add("organ", id, sphere(r), mat(colorOf("organ")), pos, scale);
  organ("brain", 0.28, [0, 1.5, 0.02]);
  organ("lungs", 0.2, [-0.22, 0.85, 0.05], [1, 1.4, 0.8]);
  organ("lungs", 0.2, [0.22, 0.85, 0.05], [1, 1.4, 0.8]);
  organ("heart", 0.16, [-0.06, 0.82, 0.16]);
  organ("liver", 0.22, [-0.2, 0.42, 0.16], [1.3, 0.8, 0.8]);
  organ("stomach", 0.16, [0.22, 0.45, 0.16]);
  organ("spleen", 0.1, [0.34, 0.4, 0.05]);
  organ("pancreas", 0.08, [0, 0.35, 0.1], [2, 0.6, 0.6]);
  organ("kidneys", 0.1, [-0.24, 0.15, -0.15], [1, 1.4, 0.8]);
  organ("kidneys", 0.1, [0.24, 0.15, -0.15], [1, 1.4, 0.8]);
  organ("intestines", 0.34, [0, -0.05, 0.12], [1, 0.8, 0.7]);
  organ("bladder", 0.13, [0, -0.5, 0.14]);
  organ("trachea", 0.05, [0, 1.15, 0.12], [1, 2.4, 1]);
  organ("thyroid", 0.07, [0, 1.18, 0.16], [1.4, 0.6, 0.8]);
  organ("gallbladder", 0.06, [-0.28, 0.36, 0.18]);
  organ("diaphragm", 0.36, [0, 0.6, 0.05], [1, 0.16, 0.8]);

  // remember base emissive for highlight restore
  for (const m of meshes) m.userData.baseEmissive = m.material.emissive?.getHex?.() ?? 0x000000;

  const setLayer = (id, on) => { if (groups[id]) groups[id].visible = on; };
  const HOT = new THREE.Color("#4fd1ff");
  const highlight = (ids) => {
    const set = new Set(ids);
    for (const m of meshes) {
      const hot = set.has(m.userData.structure);
      if (m.material.emissive) m.material.emissive.set(hot ? HOT : m.userData.baseEmissive || 0x000000);
      m.material.emissiveIntensity = hot ? 0.9 : 1;
    }
  };

  // ---- minimal orbit + zoom + pick ----
  let rotX = 0.1, rotY = 0, dist = 6, dragging = false, moved = 0, lx = 0, ly = 0, pinch = 0;
  const dom = renderer.domElement;
  const onDown = (x, y) => { dragging = true; moved = 0; lx = x; ly = y; };
  const onMove = (x, y) => {
    if (!dragging) return;
    const dx = x - lx, dy = y - ly; lx = x; ly = y; moved += Math.abs(dx) + Math.abs(dy);
    rotY += dx * 0.01; rotX = Math.max(-1.2, Math.min(1.2, rotX + dy * 0.01));
  };
  const ray = new THREE.Raycaster(); const ptr = new THREE.Vector2();
  const pick = (x, y) => {
    const rect = dom.getBoundingClientRect();
    ptr.x = ((x - rect.left) / rect.width) * 2 - 1;
    ptr.y = -((y - rect.top) / rect.height) * 2 + 1;
    ray.setFromCamera(ptr, camera);
    const hits = ray.intersectObjects(meshes.filter((m) => m.parent.visible), false);
    if (hits.length) { const id = hits[0].object.userData.structure; highlight([id]); onSelect(id); }
  };
  dom.addEventListener("pointerdown", (e) => { onDown(e.clientX, e.clientY); dom.setPointerCapture?.(e.pointerId); });
  dom.addEventListener("pointermove", (e) => onMove(e.clientX, e.clientY));
  dom.addEventListener("pointerup", (e) => { dragging = false; if (moved < 6) pick(e.clientX, e.clientY); });
  dom.addEventListener("wheel", (e) => { e.preventDefault(); dist = Math.max(3, Math.min(12, dist + Math.sign(e.deltaY) * 0.5)); }, { passive: false });
  dom.addEventListener("touchmove", (e) => {
    if (e.touches.length === 2) {
      const d = Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY);
      if (pinch) dist = Math.max(3, Math.min(12, dist - (d - pinch) * 0.01)); pinch = d;
    }
  }, { passive: true });
  dom.addEventListener("touchend", () => { pinch = 0; });

  // view presets (rotate the whole body to a canonical angle) + ghost skin
  const VIEW_ANGLES = {
    front: [0.1, 0], back: [0.1, Math.PI], left: [0.1, -Math.PI / 2],
    right: [0.1, Math.PI / 2], top: [-Math.PI / 2, 0], reset: [0.1, 0],
  };
  const setView = (name) => {
    const a = VIEW_ANGLES[name] || VIEW_ANGLES.front;
    rotX = a[0]; rotY = a[1]; if (name === "reset") dist = 6;
  };
  const setGhost = (on) => { skinMat.opacity = on ? 0.07 : 0.28; skinMat.needsUpdate = true; };
  const focus = (id) => {
    const m = meshes.find((mm) => mm.userData.structure === id);
    if (!m) return;
    // rotate so the structure faces the camera and zoom in a little
    rotY = Math.atan2(m.position.x, m.position.z || 0.001) * 0 + (m.position.x < 0 ? 0.4 : -0.4);
    rotX = 0.1; dist = 4.5;
  };

  // Tier A — load a high-detail GLB atlas on demand (pluggable; see ANATOMY.md).
  const MODEL_URL = "/anatomy/models/body.glb";
  const loadHighDetail = async (report) => {
    try {
      const status = await fetch("/api/anatomy/model").then((r) => r.json()).catch(() => ({}));
      if (!status.available) { report?.("No high-detail model installed. See ANATOMY.md to add one."); return false; }
      report?.("Loading high-detail model…");
      const { GLTFLoader } = await import("../vendor/three/GLTFLoader.js");
      const gltf = await new GLTFLoader().loadAsync(MODEL_URL);
      // replace procedural body; map named nodes to structures + layers
      pivot.clear(); for (const l of data.layers) { groups[l.id] = new THREE.Group(); pivot.add(groups[l.id]); }
      meshes.length = 0;
      gltf.scene.traverse((o) => {
        if (!o.isMesh) return;
        const id = (o.name || "").toLowerCase();
        const st = data.byId[id];
        const layer = st ? st.layer : "organ";
        o.userData.structure = st ? id : (o.name || "structure");
        o.userData.baseEmissive = 0x000000;
        (groups[layer] || groups.organ).add(o); meshes.push(o);
      });
      pivot.add(gltf.scene);
      for (const l of data.layers) setLayer(l.id, !!S.layers[l.id]);
      report?.("High-detail model loaded.");
      return true;
    } catch (err) {
      report?.(`Couldn't load model: ${err.message}`);
      return false;
    }
  };

  if (anatRaf) cancelAnimationFrame(anatRaf);
  setView(S.view || "front");
  if (S.ghost) setGhost(true);
  const loop = () => {
    if (!dom.isConnected) return; // self-terminate on unmount
    pivot.rotation.x = rotX; pivot.rotation.y = rotY;
    camera.position.z = dist;
    const w = viewport.clientWidth || width, hgt = viewport.clientHeight || height;
    if (dom.width !== Math.floor(w * renderer.getPixelRatio()) ) { renderer.setSize(w, hgt); camera.aspect = w / hgt; camera.updateProjectionMatrix(); }
    renderer.render(scene, camera);
    anatRaf = requestAnimationFrame(loop);
  };
  loop();

  return { setLayer, highlight, select: onSelect, setView, setGhost, focus, loadHighDetail,
    dispose() { if (anatRaf) cancelAnimationFrame(anatRaf); renderer.dispose(); } };
}

let anatRaf = null;
