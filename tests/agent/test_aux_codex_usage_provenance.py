"""Preserve upstream cache/reasoning details through the actual auxiliary adapter."""
import asyncio
from types import SimpleNamespace as S
import pytest
from agent.auxiliary_client import _CodexCompletionsAdapter,_AsyncCompletionsAdapter
from agent.usage_pricing import normalize_usage

@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket
    def forbidden(*args,**kwargs):raise AssertionError('Network forbidden in telemetry regression tests')
    monkeypatch.setattr(socket.socket,'connect',forbidden)

def adapter(raw):
    final=S(output=[S(type='message',content=[S(type='output_text',text='ok')])],usage=raw)
    return _CodexCompletionsAdapter(S(base_url='http://127.0.0.1:18324/v1',responses=S(create=lambda **kw:final)),'gpt-6-astra')

@pytest.mark.parametrize('as_dict',[True,False])
@pytest.mark.parametrize('async_mode',[True,False])
def test_actual_adapter_keeps_live_compression_usage(as_dict,async_mode):
    # Actual Tokyo repeat #2 totals; a separate synthetic reasoning count checks its path.
    raw={'input_tokens':6358,'output_tokens':23,'total_tokens':6381,
         'input_tokens_details':{'cached_tokens':6144,'cache_write_tokens':0},
         'output_tokens_details':{'reasoning_tokens':7}}
    wrapped=adapter(raw if as_dict else S(**raw))
    kwargs={'messages':[{'role':'user','content':'Synthetic completed calibration.'}]}
    response=asyncio.run(_AsyncCompletionsAdapter(wrapped).create(**kwargs)) if async_mode else wrapped.create(**kwargs)
    usage=normalize_usage(response.usage)
    assert usage.cache_read_tokens==6144
    assert usage.input_tokens==214 and usage.output_tokens==23 and usage.reasoning_tokens==7
    assert response.usage.total_tokens==6381 and response.choices[0].message.content=='ok'
    assert raw['input_tokens_details']['cached_tokens']==6144

@pytest.mark.parametrize('as_dict',[True,False])
@pytest.mark.parametrize('shape',['missing','null','zero'])
def test_detail_presence_is_preserved(as_dict,shape):
    raw={'input_tokens':12,'output_tokens':3,'total_tokens':15}
    if shape!='missing':
        raw['input_tokens_details']=None if shape=='null' else {'cached_tokens':0}
        raw['output_tokens_details']=None if shape=='null' else {'reasoning_tokens':0}
    usage=adapter(raw if as_dict else S(**raw)).create(messages=[{'role':'user','content':'Synthetic.'}]).usage
    if shape=='missing':
        assert not hasattr(usage,'prompt_tokens_details') and not hasattr(usage,'completion_tokens_details')
    elif shape=='null':
        assert usage.prompt_tokens_details is None and usage.completion_tokens_details is None
    else:
        assert usage.prompt_tokens_details=={'cached_tokens':0}
        assert usage.completion_tokens_details=={'reasoning_tokens':0}

def test_explicit_cache_write_remains_separate():
    raw=S(input_tokens=6000,output_tokens=24,total_tokens=6024,
          input_tokens_details=S(cached_tokens=4096,cache_write_tokens=128),
          output_tokens_details=S(reasoning_tokens=3))
    usage=normalize_usage(adapter(raw).create(messages=[{'role':'user','content':'Synthetic.'}]).usage)
    assert (usage.input_tokens,usage.cache_read_tokens,usage.cache_write_tokens)==(1776,4096,128)

def test_missing_usage_stays_missing():
    assert adapter(None).create(messages=[{'role':'user','content':'Synthetic.'}]).usage is None

@pytest.mark.parametrize('explicit_null',[False,True])
def test_sdk_unset_is_distinct_from_explicit_null(explicit_null):
    from openai.types.responses.response_usage import ResponseUsage
    fields={'input_tokens':12,'output_tokens':3,'total_tokens':15}
    if explicit_null:fields.update(input_tokens_details=None,output_tokens_details=None)
    raw=ResponseUsage.model_construct(**fields)
    result=adapter(raw).create(messages=[{'role':'user','content':'Synthetic.'}]).usage
    if explicit_null:assert result.prompt_tokens_details is None
    else:assert not hasattr(result,'prompt_tokens_details')

@pytest.mark.parametrize('cached,output',[(3968,24),(6144,23)])
def test_sdk_stream_preserves_actual_terminal_buckets(cached,output):
    import json
    import httpx
    from openai import OpenAI
    def handler(request):
        assert request.url.host=='127.0.0.1'
        item={'type':'message','id':'fixture-message','role':'assistant','status':'completed',
              'content':[{'type':'output_text','text':'ok','annotations':[]}]}
        usage={'input_tokens':6358,'input_tokens_details':{'cached_tokens':cached,'cache_write_tokens':0},
               'output_tokens':output,'output_tokens_details':{'reasoning_tokens':0},'total_tokens':6358+output}
        events=[{'type':'response.output_item.done','output_index':0,'item':item},
                {'type':'response.completed','response':{'id':'fixture','object':'response','model':'gpt-6-astra',
                   'status':'completed','output':[item],'usage':usage}}]
        data=''.join('data: '+json.dumps(event)+'\n\n' for event in events)
        return httpx.Response(200,headers={'content-type':'text/event-stream'},stream=httpx.ByteStream(data.encode()))
    with OpenAI(api_key='synthetic',base_url='http://127.0.0.1:18324/v1',max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
        result=_CodexCompletionsAdapter(client,'gpt-6-astra').create(messages=[{'role':'user','content':'Synthetic.'}])
        normalized=normalize_usage(result.usage)
        assert (normalized.input_tokens,normalized.cache_read_tokens,normalized.output_tokens)==(6358-cached,cached,output)
