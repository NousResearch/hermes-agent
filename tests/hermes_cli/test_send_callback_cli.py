"""Real CLI entry point with isolated disk configuration and an HTTP fixture."""
import argparse
import json

import httpx
import pytest


def test_callback_cli_send_from_disk_config(tmp_path, monkeypatch, capsys):
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / 'config.yaml').write_text('''platforms:
  wecom_callback:
    enabled: true
    extra:
      corp_id: fixture-corp
      corp_secret: fixture-secret
      agent_id: '123'
''', encoding='utf8')
    from plugins.platforms.wecom import callback_adapter as cb
    from hermes_cli.send_cmd import cmd_send
    requests = []
    clients = []
    original = httpx.AsyncClient

    def handle(request):
        requests.append(request)
        if request.url.path.endswith('/gettoken'):
            assert request.url.params['corpid'] == 'fixture-corp'
            return httpx.Response(200, json={'errcode': 0, 'access_token': 'fixture-token', 'expires_in': 7200})
        assert json.loads(request.content)['text']['content'] == 'cli hello'
        return httpx.Response(200, json={'errcode': 0, 'msgid': 'cli-receipt'})

    def client(**kw):
        obj = original(transport=httpx.MockTransport(handle), **kw)
        clients.append(obj)
        return obj

    monkeypatch.setattr(cb.httpx, 'AsyncClient', client)
    with pytest.raises(SystemExit) as exited:
        cmd_send(argparse.Namespace(to='wecom_callback:alice', message='cli hello', json=True))
    assert exited.value.code == 0, capsys.readouterr()
    assert requests and all(c.is_closed for c in clients)
    assert 'cli-receipt' in capsys.readouterr().out
