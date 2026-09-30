"""Standalone Mission Control Web Server (samagent/ui_server.py).

Mounts the exact same ``plugins/samagent/dashboard/plugin_api.py`` FastAPI router at
``/api/plugins/samagent`` and serves ``plugins/samagent/dashboard/dist/{index.js,style.css}``
with a zero-CDN ``window.__HERMES_PLUGIN_SDK__`` host shim so Mission Control can be previewed
directly on ``0.0.0.0:8080`` as well as inside the Hermes Dashboard.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
import uvicorn

from plugins.samagent.dashboard.plugin_api import router as samagent_api_router

REPO_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = REPO_ROOT / "plugins" / "samagent" / "dashboard"
UI_BUNDLE_DIR = DASHBOARD_DIR / "ui_bundle"


def _ensure_dist_materialized() -> Path:
    dist_dir = DASHBOARD_DIR / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("index.js", "style.css"):
        src = UI_BUNDLE_DIR / fname
        dst = dist_dir / fname
        if src.exists():
            shutil.copy2(src, dst)
    return UI_BUNDLE_DIR


_ensure_dist_materialized()

app = FastAPI(title="SamAgent Local Platform & VS Code Studio", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(samagent_api_router, prefix="/api/plugins/samagent")


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "service": "samagent-mission-control"}


@app.get("/dashboard-plugins/samagent/dist/style.css")
def serve_css() -> FileResponse:
    _ensure_dist_materialized()
    return FileResponse(
        UI_BUNDLE_DIR / "style.css",
        media_type="text/css",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.get("/dashboard-plugins/samagent/dist/index.js")
def serve_js() -> FileResponse:
    _ensure_dist_materialized()
    return FileResponse(
        UI_BUNDLE_DIR / "index.js",
        media_type="application/javascript",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


_STANDALONE_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SamAgent — Codex Desktop</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="/dashboard-plugins/samagent/dist/style.css" />
  <style>body { margin: 0; background: #ffffff; font-family: 'Inter', -apple-system, sans-serif; overflow: hidden; }</style>
</head>
<body>
  <div id="samagent-standalone-root"></div>
  <script>
  // Self-contained zero-CDN __SAMAGENT_PLUGIN_SDK__ host shim for standalone Codex Desktop
  (function () {
    let states = [];
    let stateIdx = 0;
    let effects = [];
    let effectIdx = 0;
    let rootComponent = null;
    let renderScheduled = false;

    function scheduleRender() {
      if (renderScheduled) return;
      renderScheduled = true;
      Promise.resolve().then(renderRoot);
    }

    function useState(init) {
      const i = stateIdx++;
      if (states.length <= i) {
        states.push(typeof init === "function" ? init() : init);
      }
      const setState = function (next) {
        const val = typeof next === "function" ? next(states[i]) : next;
        if (val !== states[i]) {
          states[i] = val;
          scheduleRender();
        }
      };
      return [states[i], setState];
    }

    function useCallback(fn) {
      return fn;
    }

    function useEffect(fn, deps) {
      const i = effectIdx++;
      const prev = effects[i];
      let changed = !prev || !deps;
      if (!changed && Array.isArray(deps) && Array.isArray(prev.deps)) {
        changed = deps.length !== prev.deps.length || deps.some((d, idx) => d !== prev.deps[idx]);
      }
      if (changed) {
        effects[i] = { deps: deps, ran: false, fn: fn };
      }
    }

    function createElement(tag, props) {
      const children = [];
      for (let i = 2; i < arguments.length; i++) {
        const c = arguments[i];
        if (Array.isArray(c)) {
          c.forEach((item) => children.push(item));
        } else {
          children.push(c);
        }
      }
      return { tag: tag, props: props || {}, children: children };
    }

    function buildDom(vnode) {
      if (vnode === null || vnode === undefined || vnode === false || vnode === true) {
        return document.createTextNode("");
      }
      if (typeof vnode === "string" || typeof vnode === "number") {
        return document.createTextNode(String(vnode));
      }
      if (typeof vnode.tag === "function") {
        return buildDom(vnode.tag(vnode.props));
      }
      const el = document.createElement(vnode.tag);
      const p = vnode.props || {};
      Object.keys(p).forEach((k) => {
        const v = p[k];
        if (v === null || v === undefined || v === false) {
          if (k === "disabled") el.removeAttribute("disabled");
          return;
        }
        if (k === "className") {
          el.className = v;
        } else if (k === "style" && typeof v === "object") {
          Object.assign(el.style, v);
        } else if (k === "dangerouslySetInnerHTML" && v && typeof v.__html === "string") {
          el.innerHTML = v.__html;
        } else if (k.startsWith("on") && typeof v === "function") {
          const rawEvt = k.slice(2);
          let evt = rawEvt.toLowerCase();
          if (rawEvt === "Change" && (vnode.tag === "input" || vnode.tag === "textarea")) {
            evt = "input";
          }
          el.addEventListener(evt, v);
        } else if (k === "disabled") {
          el.disabled = Boolean(v);
          if (v) el.setAttribute("disabled", "");
          else el.removeAttribute("disabled");
        } else if (k === "checked") {
          el.checked = !!v;
        } else if (k === "value") {
          el.value = v;
        } else if (k === "readOnly") {
          el.readOnly = !!v;
        } else if (k !== "key") {
          el.setAttribute(k, String(v));
        }
      });
      (vnode.children || []).forEach((ch) => {
        el.appendChild(buildDom(ch));
      });
      if (vnode.tag === "select" && p.value !== undefined) {
        el.value = p.value;
      }
      return el;
    }

    function renderRoot() {
      renderScheduled = false;
      if (!rootComponent) return;
      const active = document.activeElement;
      let activeTag = active ? active.tagName : null;
      let activePlaceholder = active ? active.getAttribute("placeholder") : null;
      let activeClass = active ? active.className : null;
      let activeStart = active && active.selectionStart !== undefined ? active.selectionStart : 0;
      let activeEnd = active && active.selectionEnd !== undefined ? active.selectionEnd : 0;

      stateIdx = 0;
      effectIdx = 0;
      const host = document.getElementById("samagent-standalone-root");
      const tree = rootComponent({});
      const dom = buildDom(tree);
      host.innerHTML = "";
      host.appendChild(dom);

      if (active && (activeTag === "INPUT" || activeTag === "TEXTAREA")) {
        const candidates = host.querySelectorAll(activeTag.toLowerCase());
        for (let c of candidates) {
          if ((activePlaceholder && c.getAttribute("placeholder") === activePlaceholder) || (activeClass && c.className === activeClass)) {
            c.focus();
            try { c.setSelectionRange(activeStart, activeEnd); } catch (e) {}
            break;
          }
        }
      }

      effects.forEach((ef) => {
        if (ef && !ef.ran) {
          ef.ran = true;
          ef.fn();
        }
      });
    }

    const sdkObj = {
      React: { createElement: createElement, useState: useState, useEffect: useEffect, useCallback: useCallback },
      hooks: { useState: useState, useEffect: useEffect, useCallback: useCallback },
      components: {},
      utils: {},
    };
    const pluginsObj = {
      register: function (name, comp) {
        rootComponent = comp;
        scheduleRender();
      },
    };

    window.__SAMAGENT_PLUGIN_SDK__ = sdkObj;
    window.__HERMES_PLUGIN_SDK__ = sdkObj;
    window.__SAMAGENT_PLUGINS__ = pluginsObj;
    window.__HERMES_PLUGINS__ = pluginsObj;
  })();
  </script>
  <script src="/dashboard-plugins/samagent/dist/index.js"></script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def serve_index() -> HTMLResponse:
    return HTMLResponse(_STANDALONE_HTML)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run SamAgent Mission Control server")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--reload", action="store_true", help="Enable hot-reload on code edits")
    args = ap.parse_args()
    if args.reload:
        uvicorn.run(
            "samagent.ui_server:app",
            host=args.host,
            port=args.port,
            reload=True,
            reload_dirs=[str(REPO_ROOT / "samagent"), str(REPO_ROOT / "plugins" / "samagent")],
            log_level="info",
        )
    else:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
