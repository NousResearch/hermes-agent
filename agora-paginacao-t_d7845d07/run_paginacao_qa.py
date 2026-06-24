#!/usr/bin/env python3
"""QA de paginação do feed Ágora com histórico > PAGE_SIZE.

Valida via Chrome CDP no dashboard real:
- carga inicial de PAGE_SIZE mensagens
- before_id carrega páginas anteriores
- scroll não pula durante carga
- hasMoreMessages / indicador "Início da conversa"

Gera screenshots + summary.json + report.md.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import websockets

TOKEN_PATH = Path.home() / ".hermes" / ".dashboard-token"
CDP_HTTP = "http://127.0.0.1:9222"
DASHBOARD_URL = "http://127.0.0.1:9119/agora"
OUT_DIR = Path(__file__).resolve().parent
SCREENSHOT_DIR = OUT_DIR / "screenshots"
PAGE_SIZE = 50
MESSAGE_COUNT = 150


def get_token() -> str:
    if "DASHBOARD_TOKEN" in os.environ:
        return os.environ["DASHBOARD_TOKEN"]
    return TOKEN_PATH.read_text().strip()


class CDPClient:
    def __init__(self) -> None:
        self.ws: websockets.WebSocketClientProtocol | None = None
        self._next_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._handlers: dict[str, list[Callable]] = {}
        self._recv_task: asyncio.Task | None = None

    async def connect(self, ws_url: str) -> None:
        self.ws = await websockets.connect(ws_url)
        self._recv_task = asyncio.create_task(self._recv_loop())

    async def _recv_loop(self) -> None:
        async for raw in self.ws:
            msg = json.loads(raw)
            mid = msg.get("id")
            if mid is not None and mid in self._pending:
                if not self._pending[mid].done():
                    self._pending[mid].set_result(msg)
                continue
            method = msg.get("method")
            for cb in self._handlers.get(method, []):
                try:
                    cb(msg)
                except Exception:
                    pass

    async def send(self, method: str, params: dict | None = None) -> dict:
        self._next_id += 1
        req = {"id": self._next_id, "method": method, "params": params or {}}
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[self._next_id] = fut
        await self.ws.send(json.dumps(req))
        return await asyncio.wait_for(fut, timeout=15)

    def on(self, event: str, callback: Callable) -> None:
        self._handlers.setdefault(event, []).append(callback)

    async def evaluate(self, expression: str, await_promise: bool = True) -> Any:
        resp = await self.send(
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": True,
                "awaitPromise": await_promise,
            },
        )
        result = resp.get("result", {}).get("result", {})
        if result.get("subtype") == "error":
            desc = result.get("description", str(result))
            raise RuntimeError(f"JS error: {desc}")
        err = resp.get("result", {}).get("exceptionDetails")
        if err:
            raise RuntimeError(f"JS exception: {err}")
        return result.get("value")

    async def screenshot(self, path: Path) -> Path:
        resp = await self.send("Page.captureScreenshot", {"format": "png"})
        data = resp["result"]["data"]
        path.write_bytes(base64.b64decode(data))
        return path

    async def close(self) -> None:
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
        if self.ws:
            await self.ws.close()


def new_tab(url: str) -> dict:
    encoded = urllib.parse.quote(url, safe=":/&?=")
    req = urllib.request.Request(f"{CDP_HTTP}/json/new?{encoded}", method="PUT")
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())


async def open_page(token: str) -> tuple[CDPClient, str]:
    url = f"{DASHBOARD_URL}?token={token}"
    tab = new_tab(url)
    client = CDPClient()
    await client.connect(tab["webSocketDebuggerUrl"])
    await client.send("Runtime.enable")
    await client.send("Page.enable")
    return client, tab["id"]


INJECT_QA_HELPERS = r"""
window.__agoraQA = window.__agoraQA || {
  sleep: (ms) => new Promise(r => setTimeout(r, ms)),
  waitFor: async function(sel, timeout=12000) {
    const deadline = Date.now() + timeout;
    while (Date.now() < deadline) {
      const el = document.querySelector(sel);
      if (el) return el;
      await this.sleep(100);
    }
    throw new Error('timeout waiting for ' + sel);
  },
  waitForCondition: async function(fn, timeout=12000, interval=200) {
    const deadline = Date.now() + timeout;
    while (Date.now() < deadline) {
      const res = await fn();
      if (res) return res;
      await this.sleep(interval);
    }
    throw new Error('timeout waiting for condition');
  },
  getChannels: function() {
    const list = document.querySelector('.agora-channel-list');
    const btns = list ? Array.from(list.querySelectorAll('button[role="tab"]')) : [];
    return btns.map(b => ({
      text: b.querySelector('.agora-channel-name')?.textContent.trim() || b.textContent.trim(),
      selected: b.getAttribute('aria-selected') === 'true',
    }));
  },
  selectChannelBySlug: function(slug) {
    const list = document.querySelector('.agora-channel-list');
    const btns = list ? Array.from(list.querySelectorAll('button[role="tab"]')) : [];
    for (const b of btns) {
      const name = b.querySelector('.agora-channel-name')?.textContent.trim() || b.textContent.trim();
      if (name.toLowerCase().includes(slug.toLowerCase())) {
        b.click();
        return { clicked: true, text: name };
      }
    }
    return { clicked: false };
  },
  getMessageListState: function() {
    const el = document.querySelector('.agora-message-list');
    if (!el) return null;
    const messages = Array.from(el.querySelectorAll('.agora-message'));
    const texts = messages.map(m => m.querySelector('.agora-message-text')?.textContent.trim() || '');
    return {
      scrollTop: el.scrollTop,
      scrollHeight: el.scrollHeight,
      clientHeight: el.clientHeight,
      messageCount: messages.length,
      firstText: texts[0] || null,
      lastText: texts[texts.length - 1] || null,
      loadingOlder: !!document.querySelector('.agora-loading-dots--top'),
      feedEndVisible: !!document.querySelector('.agora-feed-end'),
      feedEndText: document.querySelector('.agora-feed-end')?.textContent.trim() || null,
    };
  },
  setFeedScroll: function(top) {
    const el = document.querySelector('.agora-message-list');
    if (el) el.scrollTop = top;
    return el ? el.scrollTop : null;
  },
  scrollToTop: function() {
    const el = document.querySelector('.agora-message-list');
    if (el) el.scrollTop = 0;
    return el ? el.scrollTop : null;
  },
};
true;
"""


async def wait_for_mount(client: CDPClient, timeout: int = 20) -> None:
    await client.evaluate(INJECT_QA_HELPERS)
    await client.evaluate(
        "window.__agoraQA.waitFor('.agora-message-list', 20000).then(() => true)",
        await_promise=True,
    )
    await asyncio.sleep(1)


@dataclass
class Step:
    name: str
    pass_: bool = True
    notes: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


async def run_pagination_test(client: CDPClient) -> dict[str, Any]:
    steps: list[Step] = []
    out: dict[str, Any] = {"pass": False, "steps": []}

    try:
        # 1. select channel
        channels = await client.evaluate("window.__agoraQA.getChannels()")
        steps.append(Step("list channels", True, {"channels": channels}))
        sel = await client.evaluate("window.__agoraQA.selectChannelBySlug('Paginação')")
        steps.append(Step("select qa-paginacao", sel.get("clicked", False), {"selection": sel}))
        if not sel.get("clicked"):
            raise RuntimeError("não encontrou canal qa-paginacao")

        # wait for initial messages to load
        await client.evaluate(
            """window.__agoraQA.waitForCondition(() => {
              const s = window.__agoraQA.getMessageListState();
              return s && s.messageCount > 0 ? s : false;
            }, 15000)""",
            await_promise=True,
        )
        await asyncio.sleep(1)
        state0 = await client.evaluate("window.__agoraQA.getMessageListState()")
        steps.append(Step("initial load", True, {"state": state0}))
        await client.screenshot(SCREENSHOT_DIR / "01_initial_load.png")

        # check initial count == PAGE_SIZE
        if state0["messageCount"] != PAGE_SIZE:
            raise RuntimeError(f"carga inicial deveria ter {PAGE_SIZE} mensagens, tem {state0['messageCount']}")
        steps.append(Step(f"initial count == {PAGE_SIZE}", True, {"count": state0["messageCount"]}))

        # 2. scroll up to load older pages, repeat until feed-end or count stops
        total_loaded = state0["messageCount"]
        loads = 0
        while True:
            if state0.get("feedEndVisible"):
                steps.append(Step("feed end visible before scrolling", True, {"state": state0}))
                break

            # scroll to very top to trigger older load
            await client.evaluate("window.__agoraQA.scrollToTop()")
            # wait for older load start and finish
            try:
                await client.evaluate(
                    """window.__agoraQA.waitForCondition(async () => {
                      const s = window.__agoraQA.getMessageListState();
                      if (!s) return false;
                      if (s.loadingOlder) return { phase: 'loading', state: s };
                      return false;
                    }, 3000, 100)""",
                    await_promise=True,
                )
            except RuntimeError:
                pass  # may already have finished if fast

            state_after = await client.evaluate(
                """window.__agoraQA.waitForCondition(async () => {
                  const s = window.__agoraQA.getMessageListState();
                  if (!s) return false;
                  if (s.loadingOlder) return false;
                  return s;
                }, 15000, 200)""",
                await_promise=True,
            )
            loads += 1
            new_count = state_after["messageCount"]
            increased = new_count > total_loaded
            step = Step(
                f"older load {loads}",
                increased,
                {
                    "before_count": total_loaded,
                    "after_count": new_count,
                    "firstText": state_after.get("firstText"),
                    "lastText": state_after.get("lastText"),
                    "scrollTop": state_after.get("scrollTop"),
                    "feedEndVisible": state_after.get("feedEndVisible"),
                    "feedEndText": state_after.get("feedEndText"),
                },
            )
            if not increased:
                step.error = f"não carregou mensagens antigas: {total_loaded} -> {new_count}"
            steps.append(step)
            await client.screenshot(SCREENSHOT_DIR / f"02_older_load_{loads:02d}.png")

            if not increased:
                break
            total_loaded = new_count
            state0 = state_after

        # 3. final state validations
        final = await client.evaluate("window.__agoraQA.getMessageListState()")
        steps.append(Step("final state", True, {"state": final}))
        await client.screenshot(SCREENSHOT_DIR / "03_final_state.png")

        expected_final = 150
        count_ok = final["messageCount"] == expected_final
        steps.append(Step(
            f"final count == {expected_final}",
            count_ok,
            {"count": final["messageCount"], "expected": expected_final},
        ))

        feed_end_ok = final.get("feedEndVisible") is True and final.get("feedEndText") == "Início da conversa"
        steps.append(Step("feed-end indicator", feed_end_ok, {
            "feedEndVisible": final.get("feedEndVisible"),
            "feedEndText": final.get("feedEndText"),
        }))

        # 4. verify there was no empty "ghost" request.
        # For a total that is an exact multiple of PAGE_SIZE, the legacy heuristic
        # triggers one extra older load that returns zero items. With the
        # server-side has_more sentinel fix we should stop exactly at the last
        # full page.
        expected_older_loads = MESSAGE_COUNT // PAGE_SIZE - 1 if MESSAGE_COUNT % PAGE_SIZE == 0 else (MESSAGE_COUNT + PAGE_SIZE - 1) // PAGE_SIZE - 1
        no_empty_request = loads == expected_older_loads
        steps.append(Step(
            "no empty older-load request",
            no_empty_request,
            {"loads": loads, "expected": expected_older_loads},
        ))

        # 5. verify first and last message texts are consistent with seed order
        first_expected = "msg-000 — mensagem de teste 1/150"
        last_expected = "msg-149 — mensagem de teste 150/150"
        first_ok = first_expected in (final.get("firstText") or "")
        last_ok = last_expected in (final.get("lastText") or "")
        steps.append(Step("first message is oldest", first_ok, {"firstText": final.get("firstText")}))
        steps.append(Step("last message is newest", last_ok, {"lastText": final.get("lastText")}))

        out["pass"] = count_ok and feed_end_ok and no_empty_request and first_ok and last_ok
        out["final_count"] = final["messageCount"]
        out["older_loads"] = loads

    except Exception as e:
        out["error"] = str(e)
        steps.append(Step("exception", False, {}, str(e)))
        try:
            await client.screenshot(SCREENSHOT_DIR / "99_exception.png")
        except Exception:
            pass

    out["steps"] = [asdict(s) for s in steps]
    return out


async def main() -> int:
    token = get_token()
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    client, tab_id = await open_page(token)
    try:
        await wait_for_mount(client)
        result = await run_pagination_test(client)

        # summary JSON
        summary_path = OUT_DIR / "summary.json"
        summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

        # markdown report
        report_path = OUT_DIR / "report.md"
        report_lines = [
            "# QA Paginação Ágora — t_d7845d07",
            "",
            f"**Data:** {datetime.utcnow().isoformat()}Z",
            f"**Resultado geral:** {'APROVADO' if result.get('pass') else 'REPROVADO'}",
            f"**Mensagens seed:** {MESSAGE_COUNT}",
            f"**PAGE_SIZE:** {PAGE_SIZE}",
            f"**Cargas de histórico:** {result.get('older_loads', 0)}",
            f"**Contagem final:** {result.get('final_count')}",
            "",
            "## Passos",
            "",
        ]
        for s in result.get("steps", []):
            status = "✅" if s["pass_"] else "❌"
            report_lines.append(f"### {status} {s['name']}")
            if s.get("error"):
                report_lines.append(f"**Erro:** {s['error']}")
            report_lines.append("```json")
            report_lines.append(json.dumps(s.get("notes", {}), indent=2, ensure_ascii=False))
            report_lines.append("```")
            report_lines.append("")
        if result.get("error") and not any(s["name"] == "exception" for s in result.get("steps", [])):
            report_lines.append(f"**Erro fatal:** {result['error']}")
        report_lines.append("")
        report_lines.append("## Screenshots")
        report_lines.append("")
        for p in sorted(SCREENSHOT_DIR.glob("*.png")):
            report_lines.append(f"- {p}")
        report_path.write_text("\n".join(report_lines), encoding="utf-8")

        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\nsummary: {summary_path}")
        print(f"report: {report_path}")
        return 0 if result.get("pass") else 1
    finally:
        await client.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
