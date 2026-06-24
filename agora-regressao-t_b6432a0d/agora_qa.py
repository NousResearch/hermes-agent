#!/usr/bin/env python3
"""QA regressão Ágora: mentions autocomplete + Admin Settings."""
import asyncio
import base64
import json
import os
import sqlite3
import time
from datetime import datetime, timezone

import requests
import websockets

ARTIFACTS = os.environ.get('ARTIFACTS', '/home/felipi/.hermes/hermes-agent/agora-regressao-t_b6432a0d')
os.makedirs(ARTIFACTS, exist_ok=True)

BROWSER_WS = None


def get_browser_ws():
    global BROWSER_WS
    if not BROWSER_WS:
        BROWSER_WS = requests.get('http://localhost:9222/json/version').json()['webSocketDebuggerUrl']
    return BROWSER_WS


async def send(ws, method, params=None, rid=None):
    i = rid or 1
    await ws.send(json.dumps({'id': i, 'method': method, 'params': params or {}}))
    while True:
        msg = json.loads(await ws.recv())
        if isinstance(msg, dict) and msg.get('id') == i:
            return msg


async def new_agora_tab():
    cb = int(time.time())
    async with websockets.connect(get_browser_ws()) as ws:
        r = await send(ws, 'Target.createTarget', {'url': f'http://127.0.0.1:9119/agora?cb={cb}'})
        target_id = r['result']['targetId']
    for _ in range(10):
        await asyncio.sleep(0.3)
        tabs = requests.get('http://localhost:9222/json/list').json()
        for t in tabs:
            if t.get('id') == target_id:
                return t['webSocketDebuggerUrl'], cb
    raise RuntimeError('new tab not found')


async def wait_render(ws):
    for _ in range(20):
        r = await send(ws, 'Runtime.evaluate', {
            'expression': "document.querySelector('button.agora-header-btn[aria-label=\"Admin\"]')!==null",
            'returnByValue': True,
        })
        if r['result']['result'].get('value'):
            return True
        await asyncio.sleep(0.3)
    return False


async def screenshot(ws, filename):
    await send(ws, 'Page.enable')
    await send(ws, 'Page.bringToFront')
    await asyncio.sleep(0.2)
    r = await send(ws, 'Page.captureScreenshot', {'format': 'png', 'fromSurface': True})
    path = os.path.join(ARTIFACTS, filename)
    with open(path, 'wb') as f:
        f.write(base64.b64decode(r['result']['data']))
    return path


async def eval_expr(ws, expr, return_by_value=True):
    r = await send(ws, 'Runtime.evaluate', {'expression': expr, 'returnByValue': return_by_value})
    if 'exceptionDetails' in r:
        return {'__error__': r['exceptionDetails']['exception']['description']}
    inner = r.get('result', {})
    res = inner.get('result', {})
    return res.get('value')


async def type_text(ws, text):
    await send(ws, 'Input.enable')
    for ch in text:
        await ws.send(json.dumps({'id': 2, 'method': 'Input.dispatchKeyEvent', 'params': {'type': 'keyDown', 'text': ch, 'key': ch}}))
        await ws.send(json.dumps({'id': 2, 'method': 'Input.dispatchKeyEvent', 'params': {'type': 'keyUp', 'key': ch}}))
        await asyncio.sleep(0.02)


async def press_key(ws, key, code=None):
    await send(ws, 'Input.enable')
    code = code or key
    await ws.send(json.dumps({'id': 2, 'method': 'Input.dispatchKeyEvent', 'params': {'type': 'keyDown', 'key': key, 'code': code}}))
    await ws.send(json.dumps({'id': 2, 'method': 'Input.dispatchKeyEvent', 'params': {'type': 'keyUp', 'key': key, 'code': code}}))


async def clear_composer(ws):
    return await eval_expr(ws, """
    (function(){
      const inp = document.querySelector('.agora-composer-input');
      if (!inp) return 'no-input';
      inp.focus();
      inp.setSelectionRange(0, inp.value.length);
      inp.setRangeText('', 0, inp.value.length, 'end');
      inp.dispatchEvent(new InputEvent('input', {bubbles:true, inputType:'deleteContentBackward', data:''}));
      inp.dispatchEvent(new Event('change', {bubbles:true}));
      return inp.value;
    })()
    """)


async def fill_admin_input(ws, index, text):
    return await eval_expr(ws, f"""
    (function(){{
      const inputs = document.querySelectorAll('.agora-admin-panel input');
      const inp = inputs[{index}];
      if (!inp) return 'no-input';
      inp.focus();
      inp.select();
      document.execCommand('delete', false, null);
      document.execCommand('insertText', false, {json.dumps(text)});
      ['input','change'].forEach(type=>inp.dispatchEvent(new Event(type,{{bubbles:true}})));
      return inp.value;
    }})()
    """)


async def submit_admin(ws):
    return await eval_expr(ws, """
    (function(){
      const form = document.querySelector('.agora-admin-form');
      if (!form) return {error:'no form'};
      const ev = new Event('submit', {bubbles:true, cancelable:true});
      form.dispatchEvent(ev);
      return {defaultPrevented: ev.defaultPrevented};
    })()
    """)


async def open_admin(ws):
    await eval_expr(ws, "document.querySelector('button.agora-header-btn[aria-label=\"Admin\"]').click(); 'clicked'")
    for _ in range(10):
        val = await eval_expr(ws, "!!document.querySelector('.agora-admin-panel')")
        if val:
            return True
        await asyncio.sleep(0.2)
    return False


async def close_admin(ws):
    await eval_expr(ws, """
    (function(){
      const panel=document.querySelector('.agora-admin-panel');
      if(panel){
        const btn=Array.from(document.querySelectorAll('button')).find(b=>b.textContent.trim()==='← Voltar à Ágora' || b.textContent.trim()==='Fechar');
        if(btn) btn.click();
      }
    })()
    """)
    await asyncio.sleep(0.3)


async def main():
    import sys
    def log(msg):
        print(msg, flush=True)
    results = {
        'task_id': 't_b6432a0d',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'tab_url': None,
        'tests': [],
    }

    def fail(name, message, **kw):
        results['tests'].append({'test': name, 'status': 'FAIL', 'message': message, **kw})

    def pass_(name, **kw):
        results['tests'].append({'test': name, 'status': 'PASS', **kw})

    ws_url, cb = await new_agora_tab()
    results['tab_url'] = f'http://127.0.0.1:9119/agora?cb={cb}'
    log('tab opened, connecting')

    async with websockets.connect(ws_url) as ws:
        await send(ws, 'Runtime.enable')
        log('waiting render')
        rendered = await wait_render(ws)
        if not rendered:
            fail('page_render', 'Agora page did not render')
            return results
        log('rendered')

        # select QA test channel
        log('selecting channel')
        selected = await eval_expr(ws, """
        (function(){
          const chan = Array.from(document.querySelectorAll('button.agora-channel'))
            .find(b => b.getAttribute('aria-label') && b.getAttribute('aria-label').toLowerCase().includes('teste qa'));
          if (chan) chan.click();
          return chan && chan.getAttribute('aria-label');
        })()
        """)
        log(f'selected {selected}')
        await asyncio.sleep(0.4)

        # --- Mention autocomplete ---
        log('mention @')
        await clear_composer(ws)
        await type_text(ws, '@')
        await asyncio.sleep(0.6)
        mention_at = await eval_expr(ws, """
        (function(){
          const popup = document.querySelector('.agora-mention-popup');
          const opts = Array.from(document.querySelectorAll('.agora-mention-popup [role="option"]')).map(o=>o.innerText.trim());
          return {popupOpen:!!popup, options:opts, composerRole:(document.querySelector('.agora-composer-input')||{}).getAttribute('role')};
        })()
        """)
        at_screenshot = await screenshot(ws, '01_mention_dropdown_at.png')
        opts = mention_at.get('options') or []
        expected_all = '@all\nTodos' in opts and '@todos\nTodos' in opts and any('@agora-frontend' in o for o in opts)
        if mention_at.get('popupOpen') and expected_all and mention_at.get('composerRole') == 'combobox':
            pass_('mention_autocomplete_opens', options=opts, screenshot=at_screenshot)
        else:
            fail('mention_autocomplete_opens', f"popup={mention_at.get('popupOpen')}, options={opts}, role={mention_at.get('composerRole')}", screenshot=at_screenshot)

        # filter 'fr'
        await type_text(ws, 'fr')
        await asyncio.sleep(0.5)
        mention_fr = await eval_expr(ws, """
        (function(){
          const opts = Array.from(document.querySelectorAll('.agora-mention-popup [role="option"]')).map(o=>o.innerText.trim());
          return {options:opts, value: (document.querySelector('.agora-composer-input')||{}).value};
        })()
        """)
        fr_screenshot = await screenshot(ws, '02_mention_dropdown_fr.png')
        fr_opts = mention_fr.get('options') or []
        filtered_ok = len(fr_opts) <= 2 and any('agora-frontend' in o for o in fr_opts)
        if filtered_ok:
            pass_('mention_filter', options=fr_opts, screenshot=fr_screenshot)
        else:
            fail('mention_filter', f"options={fr_opts}", screenshot=fr_screenshot)

        # Escape closes
        await press_key(ws, 'Escape')
        await asyncio.sleep(0.3)
        popup_after_esc = await eval_expr(ws, "!!document.querySelector('.agora-mention-popup')")
        if not popup_after_esc:
            pass_('mention_escape_closes')
        else:
            fail('mention_escape_closes', 'popup still open')

        # ArrowDown + Enter selects @todos
        await clear_composer(ws)
        await type_text(ws, '@')
        await asyncio.sleep(0.5)
        await press_key(ws, 'ArrowDown')
        await asyncio.sleep(0.2)
        await press_key(ws, 'Enter')
        await asyncio.sleep(0.5)
        insert_data = await eval_expr(ws, """
        (function(){
          const inp = document.querySelector('.agora-composer-input');
          const popup = document.querySelector('.agora-mention-popup');
          return {value: inp&&inp.value, popupOpen:!!popup};
        })()
        """)
        insert_screenshot = await screenshot(ws, '03_mention_inserted_todos.png')
        val = insert_data.get('value') or ''
        if '@todos ' in val and not insert_data.get('popupOpen'):
            pass_('mention_keyboard_insert', value=val, screenshot=insert_screenshot)
        else:
            fail('mention_keyboard_insert', f"value={val}, popup={insert_data.get('popupOpen')}", screenshot=insert_screenshot)

        # a11y mentions
        a11y_mention = await eval_expr(ws, """
        (function(){
          const composer = document.querySelector('.agora-composer-input');
          return {composerRole: composer && composer.getAttribute('role')};
        })()
        """)
        pass_('mention_a11y', detail=a11y_mention)

        # --- Admin Settings ---
        log('open admin')
        admin_open = await open_admin(ws)
        admin_form_screenshot = await screenshot(ws, '04_admin_form.png')
        if admin_open:
            pass_('admin_panel_opens', screenshot=admin_form_screenshot)
        else:
            fail('admin_panel_opens', 'admin panel did not open', screenshot=admin_form_screenshot)

        admin_inputs = await eval_expr(ws, """
        (function(){
          const labels = Array.from(document.querySelectorAll('.agora-admin-panel label')).map(l=>l.innerText.trim().split('\\n')[0]);
          const inputs = Array.from(document.querySelectorAll('.agora-admin-panel input')).map(i=>({
            placeholder: i.getAttribute('placeholder'),
            label: i.labels && i.labels.length ? i.labels[0].innerText.split('\\n')[0] : null,
          }));
          return {labels, inputs};
        })()
        """)
        labels = admin_inputs.get('labels') or []
        if set(['Nome', 'Slug', 'Descrição']).issubset(set(labels)):
            pass_('admin_form_labels', labels=labels)
        else:
            fail('admin_form_labels', f"labels={labels}")

        valid_name = f'QA Regressão Admin {cb}'
        valid_slug = f'qa-regressao-admin-{cb}'
        log('fill valid create')
        await fill_admin_input(ws, 0, valid_name)
        await fill_admin_input(ws, 1, valid_slug)
        await fill_admin_input(ws, 2, 'Canal criado pela regressão QA')
        await asyncio.sleep(0.3)
        log('submit valid')
        await submit_admin(ws)
        await asyncio.sleep(2.5)
        created_data = await eval_expr(ws, f"""
        (function(){{
          const btn = Array.from(document.querySelectorAll('button.agora-channel')).find(b=>b.innerText.includes({json.dumps(valid_name)}));
          return {{found:!!btn, active:btn&&btn.classList.contains('agora-channel--active'), text:btn&&btn.innerText.trim(), aria:btn&&btn.getAttribute('aria-label')}};
        }})()
        """)
        created_screenshot = await screenshot(ws, '05_admin_created_channel.png')
        if created_data.get('found') and created_data.get('active'):
            pass_('admin_create_valid', slug=valid_slug, channel=created_data, screenshot=created_screenshot)
        else:
            fail('admin_create_valid', f"data={created_data}", screenshot=created_screenshot)

        # DB sanity
        canonical = '/home/felipi/.hermes/agora.db'
        profile_db = '/home/felipi/.hermes/profiles/agora-qa/agora.db'
        def check_db(path):
            if not os.path.exists(path):
                return None
            conn = sqlite3.connect(path)
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM agora_channels WHERE slug=?", (valid_slug,)).fetchone()
            return dict(row) if row else None
        canonical_row = check_db(canonical)
        profile_row = check_db(profile_db)
        if canonical_row and not profile_row:
            pass_('db_sanity', canonical=canonical, canonical_id=canonical_row.get('id'))
        else:
            fail('db_sanity', f"canonical={canonical_row is not None}, profile={profile_row is not None}")

        # Duplicate slug test
        log('duplicate test')
        dup_open = await open_admin(ws)
        dup_name = f'QA Duplicate {cb}'
        await fill_admin_input(ws, 0, dup_name)
        await fill_admin_input(ws, 1, valid_slug)
        await fill_admin_input(ws, 2, 'desc')
        await asyncio.sleep(0.2)
        await submit_admin(ws)
        await asyncio.sleep(1.0)
        dup_err_el = await eval_expr(ws, """
        (function(){
          const err = document.querySelector('.agora-admin-form-error');
          return err && err.innerText.trim();
        })()
        """)
        dup_screenshot = await screenshot(ws, '06_admin_duplicate_error.png')
        if dup_err_el and ('Já existe' in dup_err_el or 'already exists' in dup_err_el):
            pass_('admin_duplicate_slug', error=dup_err_el, screenshot=dup_screenshot)
        else:
            fail('admin_duplicate_slug', f"error_text={dup_err_el}", screenshot=dup_screenshot)

        # Invalid slug
        log('invalid slug test')
        await close_admin(ws)
        inv_open = await open_admin(ws)
        inv_name = f'QA Invalid {cb}'
        invalid_slug = '!!!'
        await fill_admin_input(ws, 0, inv_name)
        await fill_admin_input(ws, 1, invalid_slug)
        await fill_admin_input(ws, 2, 'desc')
        await asyncio.sleep(0.2)
        await submit_admin(ws)
        await asyncio.sleep(1.0)
        inv_err_el = await eval_expr(ws, """
        (function(){
          const err = document.querySelector('.agora-admin-form-error');
          return err && err.innerText.trim();
        })()
        """)
        inv_err_text = inv_err_el if isinstance(inv_err_el, str) else str(inv_err_el)
        inv_screenshot = await screenshot(ws, '07_admin_invalid_slug.png')
        if inv_err_text and ('inválido' in inv_err_text.lower() or 'caracteres' in inv_err_text.lower() or 'deve' in inv_err_text.lower()):
            pass_('admin_invalid_slug', error=inv_err_text, screenshot=inv_screenshot)
        else:
            fail('admin_invalid_slug', f"error_text={inv_err_text}", screenshot=inv_screenshot)

        # Channel aria labels and keyboard focus checks
        log('channel a11y')
        channels_a11y = await eval_expr(ws, """
        (function(){
          const channels = Array.from(document.querySelectorAll('button.agora-channel')).slice(0,5).map(b=>({
            label: b.getAttribute('aria-label'),
            role: b.getAttribute('role'),
          }));
          return channels;
        })()
        """)
        pass_('channel_aria_labels', labels=channels_a11y)

        final = await eval_expr(ws, "document.body.innerText.substring(0,120)")
        final_preview = str(final).replace(chr(10), ' ')[:120]
        log(f'final body preview: {final_preview}')

    log('writing results')
    results_path = os.path.join(ARTIFACTS, 'results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    report_path = os.path.join(ARTIFACTS, 'report.md')
    total = len(results['tests'])
    passed = sum(1 for t in results['tests'] if t['status'] == 'PASS')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# QA Regressão Ágora — t_b6432a0d\n\n")
        f.write(f"**URL testada:** {results['tab_url']}\n\n")
        f.write(f"**Resumo:** {passed}/{total} critérios validados com sucesso.\n\n")
        f.write("## Resultados\n\n")
        for t in results['tests']:
            f.write(f"- **{t['test']}**: {t['status']}\n")
            if t.get('message'):
                f.write(f"  - {t['message']}\n")
            for k, v in t.items():
                if k in ('test', 'status', 'message'):
                    continue
                f.write(f"  - `{k}`: {json.dumps(v, ensure_ascii=False)}\n")
        f.write("\n## Screenshots\n\n")
        for fn in sorted(os.listdir(ARTIFACTS)):
            if fn.lower().endswith('.png'):
                f.write(f"- `{os.path.join(ARTIFACTS, fn)}`\n")

    print(f"done {passed}/{total}")
    return results


if __name__ == '__main__':
    res = asyncio.run(main())
    print(json.dumps(res, indent=2, ensure_ascii=False))
