"""Installed plugin → native aiohttp routing/middleware → real canonical SQLite.

No listener or mocked auth/effect. make_mocked_request is only the HTTP transport
boundary; app._handle executes the native resolved router and full middleware.
"""
import asyncio
import base64
from contextlib import closing
import hashlib
import hmac
import json
from pathlib import Path
from types import SimpleNamespace
import time
from urllib.parse import urlencode

from aiohttp import web
from aiohttp.test_utils import make_mocked_request
from multidict import CIMultiDict
import pytest
import pytest_asyncio
pytest.importorskip(
    "hermes_telegram_experience",
    reason="optional external-plugin integration; exercised by plugin repository CI",
)
import yaml

from gateway.config import Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.run import GatewayRunner
from gateway.telegram_init_data import InitDataDenied, verify_init_data
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_surface as source
from hermes_cli.plugins import get_plugin_manager
from plugins.platforms.telegram.adapter import TelegramAdapter

TOKEN = '123456:synthetic-token-for-slice-four-only'
ORIGIN = 'https://tasks.example.test'
PREFIX = '/apps/hermes-telegram-experience/'


def signed(*, token=TOKEN, actor=42, stamp=None, user=None, extra=None):
    # Independent fixture construction: legacy hmac.new API and explicit sorted
    # list, not production parsing/verifier or any production signing helper.
    data = {'auth_date': str(int(time.time()) if stamp is None else stamp),
            'query_id': 'synthetic-query',
            'user': json.dumps({'id': actor, 'first_name': 'Synthetic', 'is_bot': False} if user is None else user,
                               separators=(',', ':'))}
    data.update(extra or {})
    key = hmac.new(b'WebAppData', token.encode(), hashlib.sha256).digest()
    digest = hmac.new(key, '\n'.join(key + '=' + data[key] for key in sorted(data)).encode(), hashlib.sha256).hexdigest()
    return urlencode({**data, 'hash': digest})


def selector(profile, board, tid, incarnation):
    return base64.urlsafe_b64encode(json.dumps([profile, board, tid, incarnation], separators=(',', ':')).encode()).decode().rstrip('=')


async def http(app, path=PREFIX+'detail', *, method='GET', headers=None):
    # Preserve duplicate headers and HTTP case-insensitivity at the transport boundary.
    incoming = CIMultiDict(headers or {})
    incoming.setdefault('Host', 'tasks.example.test')
    request = make_mocked_request(method, path, headers=incoming, app=app)
    try:
        return await app._handle(request)
    except web.HTTPException as response:
        return response


def config(r, **changes):
    settings = dict(enabled=True, durable_cards=True, task_detail=True, public_origin=ORIGIN,
                    bot_username='SyntheticTaskBot', app_short_name='taskdetail',
                    scope=dict(
                        routes=[dict(profile='default', platform='telegram', chat_id='-100', thread_id='7')],
                        task_resources=[dict(board='default', task_id=r.tid),
                                        dict(board='default', task_id=r.btid),
                                        dict(board='other', task_id=r.tid)],
                    ))
    settings.update(changes.pop('settings', {}))
    r.config = dict(plugins=dict(enabled=['hermes-telegram-experience'], entries={
        'hermes-telegram-experience': dict(settings=settings)}), kanban=dict(read_grants=r.grants))
    r.config['kanban'].update(changes)
    (r.home/'config.yaml').write_text(yaml.safe_dump(r.config))


@pytest_asyncio.fixture
async def rig(tmp_path, monkeypatch):
    home = tmp_path/'.hermes'
    home.mkdir()
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(home))
    monkeypatch.delenv('HERMES_KANBAN_DB', raising=False)
    path = kb.kanban_db_path(board='default')
    with closing(kbc.connect(path)) as conn:
        tid = kb.create_task(conn, title='A synthetic task <img src=x onerror=alert(1)>', body='PRIVATE BODY MUST NOT LEAK')
        btid = kb.create_task(conn, title='B different selected task')
        incarnation = source.get_task_source(conn, tid).task_incarnation
        bincarnation = source.get_task_source(conn, btid).task_incarnation
        conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    grants = [dict(actor=42, bot_id=123456, profile='default', board='default', task_id=t,
                   task_incarnation=i, permissions=['read']) for t, i in ((tid, incarnation), (btid, bincarnation))]
    r = SimpleNamespace(home=home, path=path, tid=tid, btid=btid, incarnation=incarnation,
                        grants=grants, encoded=selector('default','default',tid,incarnation),
                        bencoded=selector('default','default',btid,bincarnation))
    config(r)
    r.manager = get_plugin_manager()
    r.manager.discover_and_load(force=True)
    import hermes_telegram_experience
    import sys
    assert Path(hermes_telegram_experience.__file__).is_relative_to(Path(sys.prefix))
    assert 'site-packages' in hermes_telegram_experience.__file__
    r.service = r.manager._task_read_registration
    r.telegram = TelegramAdapter(PlatformConfig(enabled=True, token=TOKEN, typing_indicator=False))
    r.runner = GatewayRunner.__new__(GatewayRunner)
    r.runner.config = SimpleNamespace(multiplex_profiles=False, profile_routes=[])
    r.runner.adapters = {Platform.TELEGRAM: r.telegram}
    r.runner._profile_adapters = {}
    r.runner._profile_failed_platforms = {}
    r.runner._primary_profile_name = 'default'
    r.api = APIServerAdapter(PlatformConfig(enabled=True, extra=dict(key='synthetic-api-key-no-task-authority', cors_origins=[ORIGIN])))
    r.api.gateway_runner = r.runner
    r.app = r.api.create_http_application()
    r.app.freeze()
    r.headers = {'X-Telegram-Init-Data': signed(), 'X-Hermes-Task-Selector': r.encoded, 'Origin': ORIGIN}
    yield r
    r.manager.unload()
    r.api._response_store.close()


def db_state(r):
    with closing(__import__('sqlite3').connect(r.path.as_uri()+'?mode=ro', uri=True)) as conn:
        return '\n'.join(conn.iterdump())


async def produce_launch(r, monkeypatch):
    """Real watcher/card/adapter; only Telegram send and watcher clock are fixtures."""
    import importlib.util
    import re
    from urllib.parse import parse_qs, urlsplit
    from hermes_cli import kanban_db_notify as notify
    spec = importlib.util.spec_from_file_location('card_transport_fixture', Path(__file__).with_name('test_kanban_cards_integration.py'))
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    bot = fixture.Bot()
    r.telegram._bot = bot
    r.telegram.gateway_runner = r.runner
    r.runner._kanban_notifier_profile = 'default'
    r.runner._kanban_dispatcher_lock_handle = object()
    r.runner._running = True
    with closing(kbc.connect(r.path)) as conn:
        notify.add_notify_sub(conn, task_id=r.tid, platform='telegram', chat_id='-100',
                              thread_id='7', notifier_profile='default', chat_type='group')
    real_sleep = asyncio.sleep
    async def initial(delay):
        if delay != 5:
            await real_sleep(delay)
    async def end(interval):
        r.runner._running = False
    with monkeypatch.context() as clock:
        clock.setattr(asyncio, 'sleep', initial)
        clock.setattr(r.runner, '_sleep_between_ticks', end)
        await r.runner._kanban_notifier_watcher(interval=1)
    sources = tuple(r.manager._task_card_registration.sources)
    if sources:
        await asyncio.wait_for(asyncio.gather(*(s.task for s in sources)), 8)
    assert len(bot.sent) == 1
    message = bot.sent[0]
    assert message['message_thread_id'] == 7
    assert 'web_app' not in str(message.get('reply_markup', ''))
    urls = re.findall(r'https://t\.me/[A-Za-z0-9_/]+\?startapp=[A-Za-z0-9_-]+', message['text'])
    assert len(urls) == 1, message['text']
    link = urlsplit(urls[0])
    assert link.path == '/SyntheticTaskBot/taskdetail'
    encoded = parse_qs(link.query, strict_parsing=True)['startapp'][0]
    assert encoded == r.encoded
    assert json.loads(base64.urlsafe_b64decode(encoded + '=' * (-len(encoded) % 4))) == ['default', 'default', r.tid, r.incarnation]
    return encoded


@pytest.mark.asyncio
async def test_canonical_watcher_to_launch_to_native_read(rig, monkeypatch):
    r = rig
    encoded = await produce_launch(r, monkeypatch)
    before = db_state(r)
    response = await http(r.app, headers={**r.headers, 'X-Hermes-Task-Selector': encoded})
    assert response.status == 200 and json.loads(response.text)['selector'] == encoded
    assert db_state(r) == before
    with closing(__import__('sqlite3').connect(r.path.as_uri()+'?mode=ro', uri=True)) as conn:
        assert conn.execute('SELECT count(*) FROM kanban_action_records').fetchone()[0] == 0


@pytest.mark.asyncio
async def test_installed_native_authorized_and_read_only(rig):
    r = rig
    before = db_state(r)
    files_before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in r.path.parent.glob('kanban.db*') if p.is_file()}
    for extra in ({}, {'If-None-Match': '*', 'If-Modified-Since': 'Wed, 01 Jan 2099 00:00:00 GMT'}):
        response = await http(r.app, headers={**r.headers, **extra})
        assert response.status == 200, response.text
        data = json.loads(response.text)
        assert set(data) == {'selector','title','status','incarnation','revision','updated_at','fetched_at'}
        with closing(__import__('sqlite3').connect(r.path.as_uri()+'?mode=ro', uri=True)) as conn:
            canonical_status = conn.execute('SELECT status FROM tasks WHERE id=?',(r.tid,)).fetchone()[0]
        assert data['selector'] == r.encoded and data['status'] == canonical_status
        assert 'PRIVATE' not in response.text and 'body' not in data
        assert response.headers['Cache-Control'] == 'no-store' and 'ETag' not in response.headers
    for path in ('','app.js','app.css'):
        response = await http(r.app, PREFIX+path)
        assert response.status == 200 and response.body
        assert response.headers['Cache-Control'] == 'no-store'
        assert 'default-src' in response.headers['Content-Security-Policy']
    # Native agent/session routes still require API key, not Telegram or cookies.
    for headers in ({}, r.headers, {'Cookie':'session=admin'}):
        response = await http(r.app, '/api/sessions', headers=headers)
        assert response.status == 401
    assert db_state(r) == before
    files_after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in r.path.parent.glob('kanban.db*') if p.is_file()}
    assert files_after == files_before


@pytest.mark.asyncio
@pytest.mark.parametrize('case', ['missing','bitflip','wrong_bot','duplicate','malformed_percent','oversize','expired','future','no_user','bot','bool_id','string_id','negative_id','null_user','json_duplicate','unknown_reserved','bad_hash','bad_stamp','wrong_actor'])
async def test_init_data_denials_do_not_read_or_mutate(rig, monkeypatch, case):
    r = rig
    values = {
        'missing':'', 'bitflip':signed().replace('Synthetic','Synthetix'),
        'wrong_bot':signed(token='999999:synthetic-token-for-slice-four-only'),
        'duplicate':signed()+'&auth_date=1', 'malformed_percent':signed()+'&start_param=%ZZ',
        'oversize':'x'*8193, 'expired':signed(stamp=int(time.time())-1000),
        'future':signed(stamp=int(time.time())+1000), 'no_user':signed(extra={'user':''}),
        'bot':signed(user={'id':42,'is_bot':True}), 'bool_id':signed(actor=True),
        'string_id':signed(actor='42'), 'negative_id':signed(actor=-42),
        'null_user':signed(extra={'user':'null'}),
        'json_duplicate':signed(extra={'user':'{"id":42,"id":43}'}),
        'unknown_reserved':signed(extra={'actor':'42'}), 'bad_hash':signed()[:-5],
        'bad_stamp':signed(stamp='NaN'), 'wrong_actor':signed(actor=43),
    }
    before = db_state(r)
    files = {p.name: p.read_bytes() for p in r.path.parent.glob('kanban.db*')}
    def forbidden(_):
        pytest.fail('Unauthorized request reached resource projection')
    monkeypatch.setattr(r.service, '_project', forbidden)
    response = await http(r.app, headers={**r.headers, 'X-Telegram-Init-Data':values[case]})
    expected = {'error': 'expired'} if case == 'expired' else {'error': 'unavailable'}
    assert response.status == 403 and json.loads(response.text) == expected
    assert db_state(r) == before
    assert {p.name: p.read_bytes() for p in r.path.parent.glob('kanban.db*')} == files


@pytest.mark.asyncio
@pytest.mark.parametrize('case', ['profile','board','task','incarnation','traversal','absent_grant','ambiguous','permission','revoked','bad_age','api_key','cookie','query','fields','method_override','foreign_origin','encoded_path','unknown_path','POST','PUT','DELETE','PATCH','HEAD','OPTIONS'])
async def test_authority_and_route_matrix(rig, case):
    r = rig
    headers = dict(r.headers)
    path = PREFIX+'detail'
    method = case if case in ('POST','PUT','DELETE','PATCH','HEAD','OPTIONS') else 'GET'
    expected = 403
    if case in ('profile','board','task','incarnation','traversal'):
        v = ['default','default',r.tid,r.incarnation]
        index, value = {'profile':(0,'other'),'board':(1,'other'),'task':(2,'t_deadbeef'),
                        'incarnation':(3,999),'traversal':(1,'../../default')}[case]
        v[index] = value
        headers['X-Hermes-Task-Selector'] = selector(*v)
    elif case in ('absent_grant','ambiguous','permission','revoked','bad_age'):
        if case == 'ambiguous': r.grants += [r.grants[0].copy()]
        elif case == 'permission': r.grants[0]['permissions'] = ['unblock_needs_input']
        elif case == 'bad_age': pass
        else: r.grants = []
        config(r, **({'read_max_age_seconds':float('inf')} if case == 'bad_age' else {}))
    elif case == 'api_key': headers = {'Authorization':'Bearer synthetic-api-key-no-task-authority'}
    elif case == 'cookie': headers = {'Cookie':'session=admin'}
    elif case in ('query','fields'): path += '?'+('initData=do-not-log' if case=='query' else 'fields=body')
    elif case == 'method_override': headers['X-HTTP-Method-Override'] = 'POST'
    elif case == 'foreign_origin': headers['Origin'] = 'https://foreign.example.test'
    elif case == 'encoded_path': path = PREFIX+'%64etail'
    elif case == 'unknown_path': path = PREFIX+'../private'; expected=404
    if method not in ('GET','OPTIONS'): expected=405
    if method == 'OPTIONS': expected=200
    before = db_state(r)
    response = await http(r.app,path,method=method,headers=headers)
    assert response.status == expected, response.text
    assert r.tid not in (response.text or '') and 'PRIVATE' not in (response.text or '')
    assert db_state(r) == before


@pytest.mark.asyncio
@pytest.mark.parametrize('change',['revoke','unload','replace','epoch','delete','recreate','profile_switch','credential','expiry','board_rebind'])
async def test_awaited_read_rechecks_all_fences(rig, monkeypatch, change):
    r = rig
    import threading
    started, release = threading.Event(), threading.Event()
    real = r.service._project
    def paused(value):
        result = real(value)
        started.set()
        assert release.wait(5)
        return result
    monkeypatch.setattr(r.service,'_project',paused)
    task = asyncio.create_task(http(r.app,headers=r.headers))
    assert await asyncio.to_thread(started.wait,5)
    if change == 'revoke': r.grants=[];config(r)
    elif change == 'unload': r.manager.unload()
    elif change == 'replace': r.api.create_http_application()
    elif change == 'epoch': r.telegram._live_todo_epoch='replacement'
    elif change == 'profile_switch': r.runner._primary_profile_name='other';r.runner.adapters={}
    elif change == 'credential': r.telegram.config.token='654321:another-synthetic-token-for-slice-four'
    elif change == 'expiry':
        monkeypatch.setattr('gateway.task_read.time', SimpleNamespace(time=lambda: int(time.time()) + 1000))
    elif change == 'board_rebind':
        import shutil
        replacement = r.path.with_name('replacement.db')
        shutil.copy2(r.path, replacement)
        replacement.replace(r.path)
    else:
        with closing(kbc.connect(r.path)) as conn:
            if change == 'delete': conn.execute('DELETE FROM tasks WHERE id=?',(r.tid,))
            else: conn.execute("INSERT INTO task_events(task_id,kind,created_at) VALUES(?,'created',?)",(r.tid,int(time.time())))
            conn.commit()
    release.set()
    response = await task
    assert response.status == 403 and 'synthetic task' not in response.text


def test_signature_field_and_independent_openssl_vector(tmp_path):
    # OpenSSL derives both HMAC stages independently, including newer signature field.
    import subprocess
    data = 'auth_date=1700000000\nsignature=syntheticEd25519Field\nuser={"id":42}'
    secret = subprocess.check_output(['/usr/bin/openssl','dgst','-sha256','-mac','HMAC','-macopt','key:WebAppData','-binary'],input=TOKEN.encode())
    digest = subprocess.check_output(['/usr/bin/openssl','dgst','-sha256','-mac','HMAC','-macopt','hexkey:'+secret.hex(),'-binary'],input=data.encode()).hex()
    raw = urlencode({'auth_date':'1700000000','signature':'syntheticEd25519Field','user':'{"id":42}','hash':digest})
    assert verify_init_data(raw,TOKEN,now=1700000001)['actor'] == 42
    with pytest.raises(InitDataDenied):
        verify_init_data(raw.replace('syntheticEd25519Field','different'),TOKEN,now=1700000001)
