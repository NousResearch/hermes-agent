"""Exercise the candidate imports with a real clean process, not live scripts."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('explicit', [False, True])
def test_report_import_resolves_profile_home_without_writes(tmp_path, explicit):
    env = dict(os.environ, HOME=str(tmp_path), PYTHONPATH=str(ROOT))
    env.pop('HERMES_HOME', None)
    expected = tmp_path/'.hermes'
    if explicit:
        expected = tmp_path/'selected-profile'
        env['HERMES_HOME'] = str(expected)
    code = (
        'import sys; from pathlib import Path; '
        f'sys.path.insert(0, {str(ROOT / "scripts" / "content_engine")!r}); '
        'import x_manager_report; '
        f'assert x_manager_report.REPORT_DIR == Path({str(expected)!r}) / "document_cache" / "x-manager"'
    )
    result = subprocess.run([sys.executable, '-c', code], env=env, cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not expected.exists()


@pytest.mark.parametrize('complete', [True, False])
def test_rendered_review_has_source_bound_expiry_and_correct_argument(tmp_path, monkeypatch, complete):
    import hashlib
    import importlib.util
    import json
    from datetime import datetime, timedelta, timezone
    import x_manager as xm
    from cron.delivery_expiry import attachment_deadline, DeliveryExpired

    home = tmp_path/'profile'
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(xm, 'DB_PATH', home/'test.db')
    created = datetime.now(timezone.utc) - timedelta(hours=5)
    tid = str((int(created.timestamp()*1000)-1288834974657)<<22)
    source = dict(id=tid, url=f'https://x.com/Sahil_Saghir/status/{tid}',
                  created_at=created.isoformat(), origin='for_you')
    # Synthetic, isolated approval fixture, never a production corpus approval.
    corpus = home/'research/x-voice/sahil-public-corpus-2026.json'
    corpus.parent.mkdir(parents=True)
    corpus.write_text(json.dumps([{**source, 'text': 'Small queues keep failures visible.'}]))
    Path(str(corpus)+'.approval.json').write_text(json.dumps({
        'approved': True, 'sha256': hashlib.sha256(corpus.read_bytes()).hexdigest()}))
    fields = ('claim', 'evidence', 'mechanism', 'position') if complete else ('', '', '', '')
    # Short replies can omit the argument pack; expiry applies to both shapes.
    body = 'The queue limit is documented in the config.' if complete else 'Source: ' + source['url']
    item = xm.XArtifact('isolated', xm.LANE_REPLY, 'sahil_twitter', body,
                        xm.ArgumentPack(*fields, {'sources': [source]}))
    xm.stage_for_approval(item)
    spec = importlib.util.spec_from_file_location('candidate_report', ROOT/'scripts/content_engine/x_manager_report.py')
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    path = report.render_report([item], lane='reply', title='Synthetic approval test')
    assert ('<details data-section="argument">' in path.read_text()) is complete
    assert path.name.endswith('.expiry.html')
    assert attachment_deadline([(str(path), False)]) == created + timedelta(hours=6)
    from types import SimpleNamespace
    from cron.scheduler_delivery import _standalone_send
    sent = []
    async def capture(*args, **kwargs):
        sent.append(kwargs['media_files'])
        return {'success': True, 'message_id': 'isolated'}
    monkeypatch.setattr('tools.send_message_tool._send_to_platform', capture)
    target = SimpleNamespace(job={'id': 'isolated'}, where='discord:123', platform='discord',
                             pconfig=None, chat_id='123', thread_id=None)
    result, error = _standalone_send(target, '', [(str(path), False)])
    assert result['success'] and error is None
    assert len(sent) == 1
    # A delayed re-send consults its self-contained receipt, not the rendering clock.
    import cron.delivery_expiry as expiry
    class Later(datetime):
        @classmethod
        def now(cls, tz=None):
            return created + timedelta(hours=6, seconds=1)
    monkeypatch.setattr(expiry, 'datetime', Later)
    with pytest.raises(DeliveryExpired, match='expired'):
        attachment_deadline([(str(path), False)])
    result, error = _standalone_send(target, '', [(str(path), False)])
    assert result is None and 'expired' in error
    assert len(sent) == 1
