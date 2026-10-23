import pytest
from email.mime.text import MIMEText
from types import SimpleNamespace
from unittest.mock import Mock
from gateway.email_reply_policy import enforce_email_reply,EmailHeld

@pytest.mark.parametrize('decision',['allow','hold','invalid','missing','raises'])
def test_smtp_policy_and_durable_draft(tmp_path,monkeypatch,decision):
    import hermes_cli.config as cfg
    import hermes_cli.plugins as plugins
    import hermes_constants
    monkeypatch.setattr(cfg,'load_config',lambda:{'email':{'required_reply_policy':'review'}})
    monkeypatch.setattr(hermes_constants,'get_hermes_home',lambda:tmp_path)
    def evaluate(e):
        assert e.recipients==('recipient',)
        if decision=='raises':raise ValueError('failed')
        return decision
    p=SimpleNamespace(evaluate=evaluate)
    manager=SimpleNamespace(discover_and_load=lambda:None,_email_reply_policies={} if decision=='missing' else {'review':p})
    monkeypatch.setattr(plugins,'get_plugin_manager',lambda:manager)
    msg=MIMEText('body');msg['To']='recipient'
    transport=Mock()
    def send():enforce_email_reply(msg);transport()
    if decision=='allow':send();transport.assert_called_once()
    else:
        with pytest.raises((RuntimeError,ValueError)) as error:send()
        transport.assert_not_called()
        if decision=='hold':
            assert isinstance(error.value,EmailHeld)
            path=tmp_path/'drafts'/(error.value.reference+'.eml')
            assert path.read_bytes()==msg.as_bytes()
            assert path.stat().st_mode&0o777==0o600

def test_no_policy_preserves_default(monkeypatch):
    import hermes_cli.config as cfg
    monkeypatch.setattr(cfg,'load_config',lambda:{})
    enforce_email_reply(MIMEText('normal'))

def test_all_smtp_sends_share_policy_choke():
    from pathlib import Path
    import ast
    path=Path(__file__).resolve().parents[2]/'plugins/platforms/email/adapter.py'
    tree=ast.parse(path.read_text());fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_smtp_send')
    calls=[getattr(n.func,'id',getattr(n.func,'attr','')) for n in ast.walk(fn) if isinstance(n,ast.Call)]
    assert calls.index('enforce_email_reply')<calls.index('_connect_smtp')
