import tempfile,pathlib,time
from unittest.mock import patch,Mock
import pytest,yaml
import agent.model_metadata as m
import agent.bedrock_adapter as b

@pytest.fixture
def home(monkeypatch):
 with tempfile.TemporaryDirectory() as d:
  monkeypatch.setattr(m,'_get_context_cache_path',lambda:pathlib.Path(d)/'cache.yaml')
  monkeypatch.setattr(m,'_bedrock_catalog_input_limit',lambda model:None)
  yield pathlib.Path(d)/'cache.yaml'

def test_failed_probe_not_cached(home):
 with patch.object(b,'probe_bedrock_context_length',return_value=None) as probe,patch.object(b,'resolve_bedrock_region',return_value='eu-central-1'):
  assert m.get_model_context_length('not-a-known-model',provider='bedrock')==128000
  assert not home.exists()

def test_success_probe_cached(home):
 with patch.object(b,'probe_bedrock_context_length',return_value=500000) as probe,patch.object(b,'resolve_bedrock_region',return_value='eu-central-1'):
  assert m.get_model_context_length('not-a-known-model',provider='bedrock')==500000
  assert m.get_model_context_length('not-a-known-model',provider='bedrock')==500000
  assert probe.call_count==1

def test_catalog_precedes_hand_cache(home):
 m.save_context_length('new-model','bedrock://',128000)
 with patch.object(m,'_bedrock_catalog_input_limit',return_value=922000),patch.object(b,'probe_bedrock_context_length') as probe:
  assert m.get_model_context_length('new-model',provider='bedrock')==922000
  probe.assert_not_called()

def test_legacy_stale(home):
 home.write_text(yaml.safe_dump({'context_lengths':{'x@https://x/v1':100}}))
 assert not m._context_cache_is_fresh('x','https://x/v1')

def test_ttl(home):
 m.save_context_length('x','https://x/v1',100)
 assert m._context_cache_is_fresh('x','https://x/v1')
 with patch.object(m.time,'time',return_value=time.time()+3601):assert not m._context_cache_is_fresh('x','https://x/v1')

def test_info_exact_membership():
 with patch.object(m,'fetch_endpoint_model_metadata',return_value={'known':{'name':'known'}}),patch.object(m,'_query_litellm_model_info',return_value=1000000) as info:
  assert m._resolve_endpoint_context_length('known','https://x/v1','key')==1000000
  assert m._resolve_endpoint_context_length('unknown','https://x/v1','key') is None
  assert info.call_count==1

def test_info_auth_and_max():
 m._ensure_requests();r=Mock();r.json.return_value={'data':[{'model_name':'x','model_info':{'max_input_tokens':524288}},{'model_name':'x','model_info':{'max_input_tokens':229376}},{'model_name':'other','model_info':{'max_input_tokens':99999999}}]}
 with patch.object(m.requests,'get',return_value=r) as get:
  assert m._query_litellm_model_info('x','https://x/v1','secret')==524288
  assert get.call_args.kwargs['headers']=={'Authorization':'Bearer secret'}
  assert get.call_args.args[0]=='https://x/model/info'
  r.close.assert_called_once()

def test_config_override_keeps_precedence(home):
 with patch.object(m,'_bedrock_catalog_input_limit',return_value=922000):
  assert m.get_model_context_length('x',provider='bedrock',config_context_length=200000)==200000
