"""Private Responses endpoint for Hindsight, with one native Hermes OAuth owner.

No token export and no API-key/model fallback. Hindsight keeps its existing
Responses provider; Hermes owns refresh under its native cross-process lock.
"""
import asyncio
import hmac
import json
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from agent.secret_scope import get_secret
from agent.codex_headers import codex_cloudflare_headers
from hermes_cli.auth_codex import resolve_codex_runtime_credentials

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


def codex_request(body):
    from deploy.railway.hindsight_settings import current
    if body.get('model') != current()['llm_model']:
        raise ValueError('This private endpoint is reserved for the configured Hindsight model.')
    instructions = body.get('instructions') or ''
    incoming = body.get('input', [])
    if isinstance(incoming, str):
        incoming = [{'role':'user','content':incoming}]
    items = []
    for item in incoming:
        if item.get('role') in {'system','developer'}:
            content = item.get('content','')
            if isinstance(content,list):
                content = '\n'.join(part.get('text','') for part in content)
            instructions += '\n' + content
        else:
            items.append(item)
    payload = {key:body[key] for key in ('model','reasoning','text','tools','tool_choice','parallel_tool_calls') if key in body}
    payload.update(input=items,instructions=instructions or 'Return the requested structured answer.',store=False,stream=True)
    return payload


async def infer(payload):
    for attempt in range(2):
        credentials = await asyncio.to_thread(resolve_codex_runtime_credentials, force_refresh=bool(attempt))
        token, base = credentials['api_key'], credentials['base_url'].rstrip('/')
        headers = {**codex_cloudflare_headers(token, base_url=base), 'Authorization':'Bearer '+token, 'Content-Type':'application/json'}
        async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=30)) as client:
            async with client.stream('POST',base+'/responses',json=payload,headers=headers) as response:
                if response.status_code == 401 and not attempt:
                    await response.aread()
                    continue
                if response.status_code >= 400:
                    # Do not echo upstream bodies containing identifiers or request text.
                    raise HTTPException(response.status_code, 'Codex inference failed; inspect server logs and account availability.',
                                        headers={'Retry-After':response.headers.get('retry-after','60')} if response.status_code==429 else None)
                async for line in response.aiter_lines():
                    if not line.startswith('data: '):
                        continue
                    data = line[6:]
                    if data == '[DONE]':
                        continue
                    event = json.loads(data)
                    if event.get('type') == 'response.completed':
                        return event['response']
                    if event.get('type') in {'response.failed','error'}:
                        raise HTTPException(502,'Codex did not complete the Hindsight request.')
    raise HTTPException(502,'Codex stream ended without a completed response.')


@app.post('/v1/responses')
async def responses(request: Request):
    expected = get_secret('HINDSIGHT_INFERENCE_KEY', '')
    provided = request.headers.get('authorization','').removeprefix('Bearer ')
    if not expected or not hmac.compare_digest(expected,provided):
        raise HTTPException(401,'Unauthorized')
    raw = await request.body()
    if len(raw)>8*1024*1024:
        raise HTTPException(413,'Request too large')
    try:
        body = json.loads(raw)
        if body.get('stream'):
            raise ValueError('Hindsight must use non-streaming Responses requests.')
        payload = codex_request(body)
    except (ValueError,TypeError,AttributeError) as exc:
        raise HTTPException(400,str(exc))
    return JSONResponse(await infer(payload))


@app.get('/v1/settings')
async def runtime_settings(request: Request):
    _require_service_key(request)
    from deploy.railway.hindsight_settings import current
    return await asyncio.to_thread(current)


@app.post('/v1/settings/status')
async def runtime_status(request: Request):
    _require_service_key(request)
    body = await request.json()
    if body.get('state') not in {'starting', 'ready', 'error'} or not isinstance(body.get('revision'), str):
        raise HTTPException(400, 'Invalid status')
    def save():
        import time
        from hermes_constants import get_hermes_home
        from utils import atomic_json_write
        atomic_json_write(get_hermes_home() / 'hindsight_runtime.json',
                          {'state': body['state'], 'revision': body['revision'], 'updated_at': time.time()})
    await asyncio.to_thread(save)
    return {'ok': True}


def _require_service_key(request):
    expected = get_secret('HINDSIGHT_INFERENCE_KEY', '')
    provided = request.headers.get('authorization', '').removeprefix('Bearer ')
    if not expected or not hmac.compare_digest(expected, provided):
        raise HTTPException(401, 'Unauthorized')


@app.get('/health')
async def health():
    return {'status':'ok'}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app,host='::',port=8879)
