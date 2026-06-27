#!/usr/bin/env python3
"""QA da UX Admin/Settings full-screen do Ágora via Chrome CDP.

Gera evidências duráveis (screenshots + JSON) mostrando:
- tela base da Ágora
- tela Admin ocupando a área central inteira
- criação de canal funcionando
- retorno para a Ágora
"""

import asyncio
import base64
import json
import os
import time
import urllib.request

import websockets

CDP_URL = "http://127.0.0.1:9222"
DASHBOARD_URL = "http://127.0.0.1:9119/agora"
EVIDENCE_DIR = "/home/felipi/.hermes/hermes-agent/plugins/agora/dashboard/.task-t_6522f3e7-evidence"


def get_hermes_tab():
    req = urllib.request.Request(f"{CDP_URL}/json/list")
    tabs = json.loads(urllib.request.urlopen(req, timeout=10).read())
    for tab in tabs:
        if "Hermes Agent - Dashboard" in tab.get("title", "") or "/agora" in tab.get("url", ""):
            return tab
    raise RuntimeError("Aba Hermes Agent - Dashboard não encontrada")


async def run():
    tab = get_hermes_tab()
    ws_url = tab["webSocketDebuggerUrl"]
    ts = int(time.time())
    slug = f"fs-qa-{ts}"
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    notes = {"ts": ts, "slug_criado": slug}

    async with websockets.connect(ws_url, origin=None) as ws:
        _id = 0

        async def send(method, params=None):
            nonlocal _id
            _id += 1
            await ws.send(json.dumps({"id": _id, "method": method, "params": params or {}}))
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("id") == _id:
                    return msg

        async def evaluate(expression, await_promise=False):
            resp = await send(
                "Runtime.evaluate",
                {"expression": expression, "returnByValue": True, "awaitPromise": await_promise},
            )
            return resp.get("result", {}).get("result", {}).get("value")

        async def screenshot(name):
            path = os.path.join(EVIDENCE_DIR, name)
            resp = await send("Page.captureScreenshot", {"format": "png"})
            with open(path, "wb") as f:
                f.write(base64.b64decode(resp["result"]["data"]))
            return path

        async def wait_for(selector, timeout=8):
            deadline = time.time() + timeout
            while time.time() < deadline:
                found = await evaluate(f"!!document.querySelector({json.dumps(selector)})")
                if found:
                    return True
                await asyncio.sleep(0.2)
            return False

        # Garante bundle atualizado limpando SW/cache
        await send("Runtime.enable")
        await evaluate(
            """
            (async () => {
                const regs = await navigator.serviceWorker.getRegistrations();
                await Promise.all(regs.map(r => r.unregister()));
                const keys = await caches.keys();
                await Promise.all(keys.map(k => caches.delete(k)));
                return 'cleaned';
            })()
            """,
            await_promise=True,
        )

        await send("Page.enable")
        await send("Page.navigate", {"url": DASHBOARD_URL})

        # Aguarda a Ágora montar
        ready = False
        for _ in range(50):
            ready = await evaluate(
                "!!(document.querySelector('.agora') && document.querySelector('.agora-header') && document.querySelector('.agora-layout'))"
            )
            if ready:
                break
            await asyncio.sleep(0.2)
        notes["agora_ready"] = ready
        if not ready:
            await screenshot("00-failed-ready.png")
            raise RuntimeError("Ágora não montou a tempo")

        await asyncio.sleep(0.4)
        await screenshot("00-agora-base.png")

        # Abre a tela Admin (ou usa se já estiver aberta)
        btn_active = await evaluate(
            """
            (() => {
                const b = document.querySelector('.agora-header__admin button');
                return b ? b.getAttribute('aria-pressed') === 'true' : false;
            })()
            """
        )
        if not btn_active:
            await evaluate(
                """
                (() => { const b = document.querySelector('.agora-header__admin button'); if (b) b.click(); return true; })()
                """
            )
        await asyncio.sleep(0.6)
        admin_open = await wait_for(".agora-screen--admin", timeout=8)
        notes["admin_open_after_click"] = admin_open
        await screenshot("01-admin-fullscreen.png")
        if not admin_open:
            raise RuntimeError("Tela Admin não abriu")

        # Preenche o formulário de criação de canal
        await evaluate(
            f"""
            ((slug) => {{
                const nativeSet = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
                const inputs = Array.from(document.querySelectorAll('.agora-admin-form input, .agora-screen--admin input'));
                const byPH = (re) => inputs.find(i => re.test(i.placeholder || ''));
                const nameInput = byPH(/nome do canal/i);
                const slugInput = byPH(/nome-do-canal/i);
                const descInput = byPH(/descrição opcional|optional/i);
                if (nameInput) {{ nativeSet.call(nameInput, 'Fullscreen QA {ts}'); nameInput.dispatchEvent(new Event('input', {{bubbles:true}})); }}
                if (slugInput) {{ nativeSet.call(slugInput, slug); slugInput.dispatchEvent(new Event('input', {{bubbles:true}})); }}
                if (descInput) {{ nativeSet.call(descInput, 'Canal criado pela validação full-screen'); descInput.dispatchEvent(new Event('input', {{bubbles:true}})); }}
                return {{name: !!nameInput, slug: !!slugInput, desc: !!descInput}};
            }})('{slug}')
            """
        )
        await asyncio.sleep(0.4)
        await screenshot("02-admin-form-filled.png")

        # Submete o formulário (sucesso fecha o Admin e volta à Ágora)
        await evaluate(
            """
            (() => {
                const form = document.querySelector('.agora-admin-form');
                const btn = form ? Array.from(form.querySelectorAll('button, [type="submit"]')).find(b => /criar/i.test(b.textContent)) : null;
                if (btn) btn.click(); else if (form) form.requestSubmit();
                return true;
            })()
            """
        )

        returned = False
        for _ in range(50):
            returned = await evaluate(
                "!!document.querySelector('.agora-layout') && !document.querySelector('.agora-screen--admin')"
            )
            if returned:
                break
            await asyncio.sleep(0.2)
        notes["returned_to_agora_after_create"] = returned
        await asyncio.sleep(0.5)
        await screenshot("03-channel-created-return.png")
        if not returned:
            raise RuntimeError("Não retornou para a Ágora após criar canal")

        # Verifica que o novo canal aparece na lista e está selecionado
        channel_check = await evaluate(
            f"""
            (() => {{
                const items = Array.from(document.querySelectorAll('.agora-channel-list .agora-channel'));
                const found = items.some(el => el.textContent.includes('Fullscreen QA {ts}') || el.textContent.includes({json.dumps(slug)}));
                const active = document.querySelector('.agora-channel--active .agora-channel-name');
                return {{found: found, selected: active ? active.textContent.trim() : null, total: items.length}};
            }})()
            """
        )
        notes["new_channel_in_list"] = channel_check

        # Estado final
        notes["final_state"] = await evaluate(
            """
            (() => ({
                url: location.href,
                hasAgora: !!document.querySelector('.agora'),
                hasAdminScreen: !!document.querySelector('.agora-screen--admin'),
                hasLayout: !!document.querySelector('.agora-layout'),
                adminButtonVisible: (() => {
                    const b = document.querySelector('.agora-header__admin button');
                    return b ? !!(b.offsetWidth || b.offsetHeight) : false;
                })(),
                roundedCornersInAgora: (() => {
                    let count = 0;
                    document.querySelectorAll('.agora, .agora *').forEach(el => {
                        const r = parseFloat(window.getComputedStyle(el).borderRadius);
                        if (r > 0) count++;
                    });
                    return count;
                })(),
                channelCount: document.querySelectorAll('.agora-channel-list .agora-channel').length,
                selectedChannel: (document.querySelector('.agora-channel--active .agora-channel-name') || {}).textContent || null,
            }))()
            """
        )

    evidence_path = os.path.join(EVIDENCE_DIR, "evidence.json")
    with open(evidence_path, "w") as f:
        json.dump(notes, f, indent=2, ensure_ascii=False)
    print(json.dumps({"ok": True, "evidence_dir": EVIDENCE_DIR, "slug": slug, "notes": notes}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(run())
