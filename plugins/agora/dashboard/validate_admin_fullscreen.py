#!/usr/bin/env python3
"""Valida a UX de Admin/Settings full-screen do Ágora via Chrome CDP.

Reutiliza a aba "Hermes Agent - Dashboard", limpa cache, recarrega o plugin
com cache-bust, abre a tela Admin, cria um canal e volta para a Ágora.
Gera screenshots + JSON em EVIDENCE_DIR.
"""

import asyncio
import base64
import json
import os
import sys
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
        title = tab.get("title", "")
        url = tab.get("url", "")
        if "Hermes Agent - Dashboard" in title or "/agora" in url:
            return tab
    raise RuntimeError("Aba Hermes Agent - Dashboard não encontrada")


async def run():
    tab = get_hermes_tab()
    ws_url = tab["webSocketDebuggerUrl"]
    ts = int(time.time())
    slug = f"fs-qa-{ts}"
    os.makedirs(EVIDENCE_DIR, exist_ok=True)

    async with websockets.connect(ws_url, origin=None) as ws:
        _id = 0

        async def send(method, params=None):
            nonlocal _id
            _id += 1
            payload = {"id": _id, "method": method, "params": params or {}}
            await ws.send(json.dumps(payload))
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("id") == _id:
                    return msg

        async def evaluate(expression, await_promise=False):
            resp = await send(
                "Runtime.evaluate",
                {
                    "expression": expression,
                    "returnByValue": True,
                    "awaitPromise": await_promise,
                },
            )
            result = resp.get("result", {}).get("result", {})
            return result.get("value")

        async def screenshot(name):
            path = os.path.join(EVIDENCE_DIR, name)
            resp = await send("Page.captureScreenshot", {"format": "png"})
            data = resp["result"]["data"]
            with open(path, "wb") as f:
                f.write(base64.b64decode(data))
            return path

        async def poll(condition_js, timeout=10):
            deadline = time.time() + timeout
            while time.time() < deadline:
                if await evaluate(condition_js):
                    return True
                await asyncio.sleep(0.2)
            return False

        # Limpa Service Worker e cache storage para garantir bundle atualizado
        await send("Runtime.enable")
        await evaluate("""
            (async () => {
                const regs = await navigator.serviceWorker.getRegistrations();
                await Promise.all(regs.map(r => r.unregister()));
                const keys = await caches.keys();
                await Promise.all(keys.map(k => caches.delete(k)));
                return 'cleaned';
            })()
        """, await_promise=True)

        await send("Page.enable")
        await send("Page.navigate", {"url": DASHBOARD_URL})

        ready = False
        for _ in range(80):
            ready = await evaluate(
                """(() => {
                    const app = document.querySelector('.agora');
                    return !!(app && app.querySelector('.agora-header') && app.querySelector('.agora-layout'));
                })()"""
            )
            if ready:
                break
            await asyncio.sleep(0.25)

        observacoes = {"ts": ts, "slug_criado": slug, "agora_ready": bool(ready)}
        await screenshot("00-agora-base.png")
        if not ready:
            raise RuntimeError("Ágora não montou dentro do tempo esperado")

        # Verifica botão Admin
        admin_button = await evaluate(
            """(() => {
                const btns = Array.from(document.querySelectorAll('button, [role="button"]'));
                const btn = btns.find(b => /Admin/i.test(b.textContent));
                return {found: !!btn, text: btn ? btn.textContent.trim() : null, rect: btn ? btn.getBoundingClientRect() : null};
            })()"""
        )
        observacoes["admin_button"] = admin_button

        if not admin_button or not admin_button.get("found"):
            raise RuntimeError("Botão Admin não encontrado")

        # Garante que a tela Admin esteja aberta
        admin_button_state = await evaluate(
            """(() => {
                const btn = document.querySelector('.agora-header__admin button');
                return btn ? {found: true, active: btn.getAttribute('aria-pressed') === 'true'} : {found: false};
            })()"""
        )
        observacoes["admin_button_state_before"] = admin_button_state

        if admin_button_state and not admin_button_state.get("active"):
            await evaluate(
                """(() => {
                    const btn = document.querySelector('.agora-header__admin button');
                    if (btn) btn.click();
                    return true;
                })()"""
            )

        await asyncio.sleep(0.5)

        admin_open = await poll(
            "() => !!document.querySelector('.agora-screen--admin')", timeout=10
        )
        observacoes["admin_screen_open"] = admin_open
        await asyncio.sleep(0.3)
        await screenshot("01-admin-fullscreen.png")
        if not admin_open:
            raise RuntimeError("Tela Admin não abriu")

        # Preenche formulário
        await evaluate(
            f"""(() => {{
                const inputs = Array.from(document.querySelectorAll('input'));
                const labels = Array.from(document.querySelectorAll('label, .agora-admin-label'));
                const findInput = (regex) => inputs.find(i => regex.test(i.placeholder || i.name || (i.labels[0] && i.labels[0].textContent) || ''));
                const nameInput = findInput(/Nome/i);
                const slugInput = findInput(/slug/i);
                const descInput = findInput(/descrição|description/i);
                if (nameInput) {{ nameInput.value = 'Fullscreen QA {ts}'; nameInput.dispatchEvent(new Event('input', {{bubbles:true}})); }}
                if (slugInput) {{ slugInput.value = '{slug}'; slugInput.dispatchEvent(new Event('input', {{bubbles:true}})); }}
                if (descInput) {{ descInput.value = 'Canal criado pela validação full-screen'; descInput.dispatchEvent(new Event('input', {{bubbles:true}})); }}
                return {{name: !!nameInput, slug: !!slugInput, desc: !!descInput}};
            }})()"""
        )
        await asyncio.sleep(0.3)

        # Submete
        await evaluate(
            """(() => {
                const form = document.querySelector('.agora-admin-form');
                const submitBtn = form ? Array.from(form.querySelectorAll('button, [type="submit"]')).find(b => /criar|create/i.test(b.textContent)) : null;
                if (submitBtn) { submitBtn.click(); }
                else if (form) { form.requestSubmit(); }
                return true;
            })()"""
        )

        created = await poll(
            f"() => Array.from(document.querySelectorAll('.agora-admin-channel-item__slug')).some(s => s.textContent.includes('{slug}'))",
            timeout=10,
        )
        observacoes["channel_created_in_list"] = created
        await asyncio.sleep(0.4)
        await screenshot("02-channel-created.png")
        if not created:
            raise RuntimeError("Canal criado não apareceu na lista Admin")

        # Clica em voltar
        await evaluate(
            """(() => {
                const btn = Array.from(document.querySelectorAll('.agora-screen--admin button, .agora-screen--admin [role="button"]'))
                    .find(b => /voltar|fechar|back|close|retornar/i.test(b.textContent));
                if (btn) btn.click();
                return true;
            })()"""
        )

        returned = await poll(
            "() => !!document.querySelector('.agora-layout') && !document.querySelector('.agora-screen--admin')",
            timeout=5,
        )
        observacoes["returned_to_agora"] = returned
        await asyncio.sleep(0.4)
        await screenshot("03-back-to-agora.png")
        if not returned:
            raise RuntimeError("Não retornou à Ágora")

        # Coleta estado final legível
        observacoes["final_classes"] = await evaluate(
            """(() => ({
                hasAgora: !!document.querySelector('.agora'),
                hasAdminScreen: !!document.querySelector('.agora-screen--admin'),
                hasLayout: !!document.querySelector('.agora-layout'),
                adminButtonActive: !!document.querySelector('.agora-header-btn--active'),
                adminButtonVisible: (() => {
                    const btn = document.querySelector('.agora-header__admin button');
                    if (!btn) return false;
                    const rect = btn.getBoundingClientRect();
                    return rect.width > 0 && rect.height > 0;
                })(),
                noRoundedInAgora: Array.from(document.querySelectorAll('.agora *')).filter(el => {
                    const s = window.getComputedStyle(el);
                    return parseFloat(s.borderRadius) > 0;
                }).length,
                channelCount: document.querySelectorAll('.agora-channel-list .agora-channel').length,
                adminChannelItems: document.querySelectorAll('.agora-admin-channel-item').length,
                selectedChannel: (document.querySelector('.agora-channel--active .agora-channel-name') || {}).textContent || null,
            }))()"""
        )

        evidence_path = os.path.join(EVIDENCE_DIR, "evidence.json")
        with open(evidence_path, "w") as f:
            json.dump(observacoes, f, indent=2, ensure_ascii=False)

        print(json.dumps({"ok": True, "evidence_dir": EVIDENCE_DIR, "observacoes": observacoes}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(run())
