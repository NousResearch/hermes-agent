#!/usr/bin/env python
"""
cdp_browser.py — composed one-pass CDP driver (ego-lite mechanisms, ported).

Steals from citrolabs/ego-lite and applies them to our raw-CDP collector stack:
  1. CODE-BASE not CLI-base: a JSON *steps script* runs in ONE websocket
     connection — snapshot/focus/type/click/upload/wait/navigate/capture as
     ops, no agent round-trip between steps.  (ego: "agent writes JS calling
     tools, run in one pass")
  2. SPACES: `spaces` subcommand runs N named tabs concurrently, each with
     its own steps script — parallel isolated contexts in one browser.
  3. STRONG SNAPSHOT: `snapshot` op returns a compact semantic view
     (interactive elements + visible text, stable selectors) instead of a
     giant AX tree — token-cheap for the agent.

Usage:
  python cdp_browser.py list
  python cdp_browser.py run <steps.json> [--tab <tab-id-or-auto>] [--port 9333]
  python cdp_browser.py spaces <spaces.json> [--port 9333]

Steps ops (JSON array):
  {"op":"open_tab","url":"https://..."}           # create + attach new tab
  {"op":"navigate","url":"https://..."}           # navigate attached tab
  {"op":"snapshot","max":40}                      # compact semantic snapshot
  {"op":"eval","expr":"...","ret":true}           # Runtime.evaluate (ret=returnByValue)
  {"op":"focus","sel":"[contenteditable=true]"}   # focus element by CSS selector
  {"op":"type","text":"..."}                      # Input.insertText (fires native input events)
  {"op":"click","sel":"button:has-text?"}         # el.click() by CSS selector
  {"op":"click_coord","x":100,"y":200}            # Input.dispatchMouseEvent press+release
  {"op":"upload","sel":"input[type=file]","file":"C:/path"}  # DOM.setFileInputFiles
  {"op":"wait","ms":5000}                         # sleep
  {"op":"capture","out":"C:/shot.png"}            # Page.captureScreenshot
  {"op":"close"}                                  # close attached tab
  {"op":"echo","msg":"..."}                       # debug marker

spaces.json:
  {"spaces":[{"name":"a","tab":"new","url":"https://a","steps":[...]},
             {"name":"b","tab":"gemini","steps":[...]}]}
  tab: "new" (open fresh tab w/ url) | "gemini" (existing gemini tab) |
       exact targetId | "auto" (first page tab)
"""
import asyncio, json, sys, time, argparse, re

import websockets

DEFAULT_PORT = 9333


def _ws_url(port):
    import urllib.request
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=4) as r:
        return json.load(r)["webSocketDebuggerUrl"]


def _tabs(port):
    import urllib.request
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=4) as r:
        return json.load(r)


class CdpSession:
    """One attached target (tab). Ops run as coroutines over a shared ws."""

    def __init__(self, ws, session_id=None):
        self.ws = ws
        self.sid = session_id
        self._id = 100

    async def attach_to(self, target_id):
        """Re-attach this session to a different target (for open_tab mid-script)."""
        await self.ws.send(json.dumps({"id": self._id + 1, "method": "Target.attachToTarget", "params": {"targetId": target_id, "flatten": True}}))
        self._id += 1
        cid = self._id
        while True:
            m = json.loads(await self.ws.recv())
            if m.get("id") == cid and "result" in m:
                self.sid = m["result"]["sessionId"]
                return target_id
            if m.get("method") == "Target.attachedToTarget" and m["params"].get("targetId") == target_id:
                self.sid = m["params"]["sessionId"]
                return target_id

    async def cmd(self, method, params=None, target_id=None):
        self._id += 1
        cid = self._id
        msg = {"id": cid, "method": method, "params": params or {}}
        if self.sid:
            msg["sessionId"] = self.sid
        elif target_id:
            msg["sessionId"] = target_id
        await self.ws.send(json.dumps(msg))
        while True:
            m = json.loads(await self.ws.recv())
            if m.get("id") == cid:
                if "error" in m:
                    raise RuntimeError(f"{method}: {m['error']}")
                return m.get("result", {})

    async def eval_js(self, expr, ret=True):
        r = await self.cmd("Runtime.evaluate", {"expression": expr, "returnByValue": ret})
        if ret:
            return r.get("result", {}).get("value")
        return r

    # ---------- ops ----------
    async def op_snapshot(self, max_el=40):
        """Compact semantic snapshot: interactive elements + visible text."""
        expr = """
        (() => {
          const out = {url: location.href, title: document.title, els: [], text: ''};
          const seen = new Set();
          const q = (sel) => [...document.querySelectorAll(sel)];
          const pick = (el) => {
            if (!el || seen.has(el)) return null;
            seen.add(el);
            const r = el.getBoundingClientRect();
            if (r.width < 4 && r.height < 4) return null;
            const tag = el.tagName.toLowerCase();
            const txt = (el.innerText || el.value || el.getAttribute('aria-label') || el.getAttribute('placeholder') || '').trim().replace(/\\s+/g,' ').slice(0,80);
            let role = tag;
            if (el.getAttribute('role')) role = el.getAttribute('role');
            else if (tag==='button') role='button';
            else if (tag==='a') role='link';
            else if (tag==='input'||tag==='textarea') role=tag;
            else if (el.getAttribute('contenteditable')==='true') role='editable';
            // stable-ish selector: try id, then data-test-id, then n-th of tag
            let sel = tag;
            if (el.id) sel += '#' + CSS.escape(el.id);
            else if (el.getAttribute('data-test-id')) sel += '[data-test-id="'+el.getAttribute('data-test-id')+'"]';
            else {
              const parent = el.parentElement;
              if (parent) {
                const sibs = [...parent.children].filter(c=>c.tagName===el.tagName);
                if (sibs.length>1) sel += ':nth-of-type('+(sibs.indexOf(el)+1)+')';
              }
            }
            return {role, tag, txt, sel, x:Math.round(r.x), y:Math.round(r.y), w:Math.round(r.width), h:Math.round(r.height)};
          };
          const sels = ['button','a','input','textarea','select','[contenteditable="true"]','[role="button"]','[role="textbox"]','[role="option"]','[role="menuitem"]','[role="tab"]','[role="checkbox"]','[role="radio"]','[role="link"]'];
          for (const s of sels) for (const el of q(s)) {
            const e = pick(el); if (e && e.txt) out.els.push(e);
            if (out.els.length >= __MAX__) break;
          }
          // keep a short body-text digest for context
          const body = document.body ? (document.body.innerText||'') : '';
          out.text = body.replace(/\\s+/g,' ').slice(0,400);
          return JSON.stringify(out);
        })()
        """.replace("__MAX__", str(max_el))
        v = await self.eval_js(expr)
        try:
            return json.loads(v)
        except Exception:
            return {"error": "snapshot parse failed", "raw": str(v)[:200]}

    async def op_focus(self, sel):
        r = await self.eval_js(f"(()=>{{const el=document.querySelector({json.dumps(sel)}); if(!el) return 'notfound'; el.focus(); return 'ok';}})()")
        await asyncio.sleep(0.3)
        return r

    async def op_type(self, text):
        await self.cmd("Input.insertText", {"text": text})
        await asyncio.sleep(0.3)
        return f"inserted {len(text)} chars"

    async def op_click(self, sel):
        return await self.eval_js(f"(()=>{{const el=document.querySelector({json.dumps(sel)}); if(!el) return 'notfound'; el.click(); return 'ok';}})()")

    async def op_click_coord(self, x, y):
        for typ, btn in (("mousePressed", "left"), ("mouseReleased", "left")):
            await self.cmd("Input.dispatchMouseEvent", {"type": typ, "x": x, "y": y, "button": btn, "clickCount": 1})
        await asyncio.sleep(0.3)
        return f"clicked {x},{y}"

    async def op_upload(self, sel, file):
        await self.cmd("DOM.enable", {})
        root = (await self.cmd("DOM.getDocument", {"depth": 0}))["root"]["nodeId"]
        nid = None
        for attempt in range(8):
            r = await self.cmd("DOM.querySelector", {"nodeId": root, "selector": sel})
            nid = r.get("nodeId")
            if nid:
                break
            await asyncio.sleep(1.0)
        if not nid:
            return "notfound"
        await self.cmd("DOM.setFileInputFiles", {"nodeId": nid, "files": [file]})
        await asyncio.sleep(1.0)
        return f"uploaded {file}"

    async def op_wait(self, ms):
        await asyncio.sleep(ms / 1000.0)
        return f"waited {ms}ms"

    async def op_capture(self, out):
        r = await self.cmd("Page.captureScreenshot", {"format": "png"})
        with open(out, "wb") as f:
            f.write(__import__("base64").b64decode(r["data"]))
        return f"saved {out}"

    async def op_eval(self, expr, ret):
        if ret:
            return await self.eval_js(expr)
        await self.eval_js(expr, ret=False)
        return "ok"

    async def op_navigate(self, url):
        await self.cmd("Page.navigate", {"url": url})
        await asyncio.sleep(2.0)
        return f"navigated {url}"

    async def run_steps(self, steps):
        results = []
        for i, st in enumerate(steps):
            op = st.get("op")
            if op == "open_tab":
                tid = await open_new_tab(self.ws, st.get("url", "about:blank"))
                await self.attach_to(tid)
                self._target_id = tid
                # createTarget URL is async — wait for it to settle, then force-navigate if given
                await asyncio.sleep(1.5)
                url = st.get("url")
                if url and url != "about:blank":
                    await self.op_navigate(url)
                results.append({"step": i, "op": op, "result": f"opened+attached {tid}"})
                continue
            if op == "close":
                try:
                    await self.ws.send(json.dumps({"id": self._id + 1, "method": "Target.closeTarget", "params": {"targetId": getattr(self, "_target_id", "")}}))
                    self._id += 1
                    results.append({"step": i, "op": op, "result": "closed"})
                except Exception as e:
                    results.append({"step": i, "op": op, "error": str(e)})
                continue
            fn = {
                "snapshot": lambda: self.op_snapshot(st.get("max", 40)),
                "focus": lambda: self.op_focus(st["sel"]),
                "type": lambda: self.op_type(st["text"]),
                "click": lambda: self.op_click(st["sel"]),
                "click_coord": lambda: self.op_click_coord(st["x"], st["y"]),
                "upload": lambda: self.op_upload(st["sel"], st["file"]),
                "wait": lambda: self.op_wait(st["ms"]),
                "capture": lambda: self.op_capture(st["out"]),
                "eval": lambda: self.op_eval(st["expr"], st.get("ret", True)),
                "navigate": lambda: self.op_navigate(st["url"]),
                "echo": lambda: st.get("msg", ""),
            }.get(op)
            if not fn:
                results.append({"step": i, "op": op, "error": "unknown op"})
                continue
            try:
                r = await fn()
                results.append({"step": i, "op": op, "result": r if not isinstance(r, dict) or "error" in r else {"snapshot": r}})
            except Exception as e:
                results.append({"step": i, "op": op, "error": str(e)})
        return results


async def attach(ws, target_id):
    """Attach to a target, return (session, target_id)."""
    await ws.send(json.dumps({"id": 1, "method": "Target.attachToTarget", "params": {"targetId": target_id, "flatten": True}}))
    sid = None
    while sid is None:
        m = json.loads(await ws.recv())
        if m.get("id") == 1 and "result" in m:
            sid = m["result"]["sessionId"]
        elif m.get("method") == "Target.attachedToTarget" and m["params"].get("targetId") == target_id:
            sid = m["params"]["sessionId"]
    return CdpSession(ws, sid), target_id


async def pick_target(port, spec):
    """Resolve 'gemini' | 'new' | targetId | 'auto' -> (target_id, opened_new)."""
    tabs = _tabs(port)
    pages = [t for t in tabs if t.get("type") == "page"]
    if spec in ("auto", None, ""):
        for t in pages:
            if "gemini.google.com" in t.get("url", ""):
                return t["id"], False
        return pages[0]["id"], False if pages else None
    if spec == "gemini":
        for t in pages:
            if "gemini.google.com" in t.get("url", ""):
                return t["id"], False
        raise RuntimeError("no gemini tab found")
    if spec == "new":
        return None, False  # caller opens
    # exact targetId
    for t in tabs:
        if t.get("id") == spec:
            return t["id"], False
    raise RuntimeError(f"target {spec} not found")


async def open_new_tab(ws, url):
    r = await ws.send(json.dumps({"id": 2, "method": "Target.createTarget", "params": {"url": url or "about:blank"}}))
    # need to read response; use a raw round trip
    while True:
        m = json.loads(await ws.recv())
        if m.get("id") == 2:
            return m["result"]["targetId"]


async def main_run(args):
    steps = json.load(open(args.steps, encoding="utf-8")) if args.steps.endswith(".json") else json.loads(args.steps)
    async with websockets.connect(_ws_url(args.port), max_size=None) as ws:
        tid, _ = await pick_target(args.port, args.tab)
        if tid is None:
            tid = await open_new_tab(ws, None)
        sess, _ = await attach(ws, tid)
        results = await sess.run_steps(steps)
        print(json.dumps(results, ensure_ascii=False, indent=1)[:60000])


async def main_spaces(args):
    spec = json.load(open(args.spaces, encoding="utf-8"))
    async with websockets.connect(_ws_url(args.port), max_size=None) as ws:
        out = {}
        for sp in spec.get("spaces", []):
            name = sp["name"]
            if sp.get("tab") == "new":
                tid = await open_new_tab(ws, sp.get("url", "about:blank"))
                await asyncio.sleep(1.5)
            else:
                try:
                    tid, _ = await pick_target(args.port, sp.get("tab", "auto"))
                except RuntimeError as e:
                    out[name] = {"error": str(e)}
                    continue
            sess, _ = await attach(ws, tid)
            sess._target_id = tid
            url = sp.get("url")
            if url and url != "about:blank":
                await sess.op_navigate(url)
            res = await sess.run_steps(sp.get("steps", []))
            out[name] = res
        print(json.dumps(out, ensure_ascii=False, indent=1)[:8000])


async def main_list(args):
    tabs = _tabs(args.port)
    for t in tabs:
        if t.get("type") == "page":
            print(f"{t['id']} | {t.get('title','')[:50]} | {t.get('url','')[:70]}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    p_list = sub.add_parser("list")
    p_list.add_argument("--port", type=int, default=DEFAULT_PORT)
    p_run = sub.add_parser("run")
    p_run.add_argument("steps")
    p_run.add_argument("--tab", default="auto")
    p_run.add_argument("--port", type=int, default=DEFAULT_PORT)
    p_sp = sub.add_parser("spaces")
    p_sp.add_argument("spaces")
    p_sp.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = p.parse_args()
    if args.cmd == "list":
        asyncio.run(main_list(args))
    elif args.cmd == "run":
        asyncio.run(main_run(args))
    elif args.cmd == "spaces":
        asyncio.run(main_spaces(args))


if __name__ == "__main__":
    main()
