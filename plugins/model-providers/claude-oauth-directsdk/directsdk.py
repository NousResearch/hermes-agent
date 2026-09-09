"""Request-scoped Claude Code transport with host-owned HTTP admission."""
from __future__ import annotations

import asyncio
import copy
import json
import math
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import weakref
from types import SimpleNamespace

try:
    from .admission import Admission
    from .model_catalog import native_model
except ImportError:
    from admission import Admission
    from model_catalog import native_model

CARRIER = 'claude-oauth-directsdk.native_assistant'
PREFIX = 'mcp__hermes__'


class Object(SimpleNamespace):
    def model_dump(self, **_):
        def unpack(v):
            if isinstance(v, Object):
                return {k: unpack(x) for k, x in vars(v).items()}
            if isinstance(v, list):
                return [unpack(x) for x in v]
            return copy.deepcopy(v)
        return unpack(self)


def obj(value):
    if isinstance(value, dict):
        return Object(**{k: copy.deepcopy(v) if k in ('reasoning_details', 'native_usage') else obj(v) for k, v in value.items()})
    if isinstance(value, list):
        return [obj(v) for v in value]
    return value


def projection(message):
    calls = []
    for tc in message.get('tool_calls') or []:
        f = tc['function']
        args = f['arguments']
        calls.append({'id': tc['id'], 'name': f['name'],
                      'input': json.loads(args) if isinstance(args, str) else args})
    return {'content': message.get('content') or '', 'tool_calls': calls}


def content_blocks(content):
    if content is None:
        return []
    if isinstance(content, str):
        return [{'type': 'text', 'text': content}] if content else []
    result = []
    for block in content:
        kind = block.get('type')
        if kind == 'text':
            result.append(copy.deepcopy(block))
        elif kind == 'image_url':
            url = block['image_url']['url']
            if not url.startswith('data:') or ';base64,' not in url:
                raise ValueError('Only base64 data image_url inputs are supported')
            media, data = url[5:].split(';base64,', 1)
            result.append({'type': 'image', 'source': {'type': 'base64', 'media_type': media, 'data': data}})
        elif kind in ('image', 'document', 'tool_result'):
            result.append(copy.deepcopy(block))
        else:
            raise ValueError(f'Unsupported content block: {kind}')
    return result


def prepare_history(messages):
    system, frames = [], []
    for message in messages:
        role = message.get('role')
        if role in ('system', 'developer'):
            if frames:
                raise ValueError('System/developer messages must precede conversation history')
            if not isinstance(message.get('content'), str):
                raise ValueError('System content must be text')
            system.append(message['content'])
            continue
        if role == 'assistant':
            details = message.get('reasoning_details') or []
            carriers = [d for d in details if isinstance(d, dict) and d.get('type') == CARRIER]
            if carriers:
                if len(carriers) != 1 or carriers[0].get('version') != 1:
                    raise ValueError('Unsupported native assistant carrier version')
                carrier = carriers[0]
                expected = carrier['projection']
                native_text = ''.join(block.get('text', '') for item in carrier['messages']
                                      for block in item['content'] if block.get('type') == 'text')
                if projection(message) == expected and native_text == expected['content']:
                    for native in carrier['messages']:
                        frames.append({'type': 'assistant', 'message': copy.deepcopy(native)})
                    continue
                # Host compaction/hooks own visible history. Never restore stale
                # pre-edit blocks or attach their signatures to rewritten content.
            blocks = content_blocks(message.get('content'))
            for call in projection(message)['tool_calls']:
                blocks.append({'type': 'tool_use', 'id': call['id'], 'name': PREFIX + call['name'], 'input': call['input']})
        elif role == 'tool':
            role = 'user'
            blocks = [{'type': 'tool_result', 'tool_use_id': message['tool_call_id'],
                       'content': message.get('content') if isinstance(message.get('content'), str) else content_blocks(message.get('content'))}]
            if message.get('is_error') is not None:
                blocks[0]['is_error'] = bool(message['is_error'])
        elif role == 'user':
            blocks = content_blocks(message.get('content'))
        else:
            raise ValueError(f'Unsupported message role: {role}')
        if frames and frames[-1]['type'] == role and role == 'user':
            frames[-1]['message']['content'].extend(blocks)
        else:
            frames.append({'type': role, 'message': {'role': role, 'content': blocks}})
    if not frames or frames[-1]['type'] != 'user' or not frames[-1]['message']['content']:
        raise ValueError('History must end in a nonempty user/tool-result message; assistant prefill is unsupported')
    return '\n\n'.join(system), frames


def request_body(kwargs):
    allowed = {'model', 'messages', 'tools', 'stream', 'stream_options', 'max_tokens', 'max_completion_tokens',
               'temperature', 'top_p', 'stop', 'extra_body', 'timeout', 'tool_choice', 'parallel_tool_calls', 'n', 'response_format'}
    unknown = set(kwargs) - allowed
    if unknown:
        raise ValueError('Unsupported request parameters: ' + ', '.join(sorted(unknown)))
    if kwargs.get('n', 1) != 1 or kwargs.get('tool_choice', 'auto') not in ('auto', None):
        raise ValueError('Only n=1 and tool_choice=auto are supported')
    if kwargs.get('parallel_tool_calls') is False:
        raise ValueError('parallel_tool_calls=False is unsupported')
    if kwargs.get('stream_options') not in (None, {}, {'include_usage': True}, {'include_usage': False}):
        raise ValueError('Unsupported stream_options')
    extra = kwargs.get('extra_body')
    if extra is None:
        extra = {}
    if not isinstance(extra, dict):
        raise ValueError('extra_body must be an object')
    unknown_extra = set(extra) - {'max_tokens', 'temperature', 'top_p', 'stop_sequences', 'reasoning', 'response_format'}
    if unknown_extra:
        raise ValueError('Unsupported extra_body fields: ' + ', '.join(sorted(unknown_extra)))
    body = copy.deepcopy(extra)
    reasoning = body.pop('reasoning', None)
    if reasoning is not None:
        if not isinstance(reasoning, dict) or set(reasoning) - {'enabled', 'effort'}:
            raise ValueError('reasoning supports enabled and effort only')
        if 'enabled' in reasoning and type(reasoning['enabled']) is not bool:
            raise ValueError('reasoning.enabled must be boolean')
        from agent.reasoning_effort import clamp_effort
        effort = clamp_effort(reasoning.get('effort'), ('none', 'low', 'medium', 'high', 'xhigh', 'max'))
        if effort not in (None, 'none', 'low', 'medium', 'high', 'xhigh', 'max'):
            raise ValueError('Unsupported native reasoning effort')
        if reasoning.get('enabled') is False or effort == 'none':
            body['thinking'] = {'type': 'disabled'}
            # Native clear-thinking context edits are invalid when thinking is disabled.
            body['context_management'] = {'edits': []}
        else:
            if reasoning.get('enabled') is True:
                body['thinking'] = {'type': 'adaptive'}
            if effort:
                body['output_config'] = {'effort': effort}
    response_format = kwargs.get('response_format', body.pop('response_format', None))
    if response_format and response_format.get('type') != 'text':
        if response_format.get('type') != 'json_schema':
            raise ValueError('Only json_schema structured output is supported')
        schema = response_format.get('json_schema', {}).get('schema')
        if not isinstance(schema, dict):
            raise ValueError('response_format requires a JSON Schema object')
        body.setdefault('output_config', {})['format'] = {'type': 'json_schema', 'schema': schema}
    for key in ('max_tokens', 'temperature', 'top_p'):
        if kwargs.get(key) is not None:
            body[key] = kwargs[key]
    if kwargs.get('max_completion_tokens') is not None:
        if 'max_tokens' in body:
            raise ValueError('Specify only one output-token limit')
        body['max_tokens'] = kwargs['max_completion_tokens']
    if kwargs.get('stop') is not None:
        stop = kwargs['stop']
        body['stop_sequences'] = [stop] if isinstance(stop, str) else stop
    for key in ('temperature', 'top_p'):
        if key in body and (isinstance(body[key], bool) or not isinstance(body[key], (int, float)) or not math.isfinite(body[key]) or not 0 <= body[key] <= 1):
            raise ValueError(f'{key} must be finite and between zero and one')
        # Subscription models reject sampling controls, including Hermes' title
        # generator default. Match the host's sampling-forbidden model behavior.
        body.pop(key, None)
    if 'max_tokens' in body and (type(body['max_tokens']) is not int or body['max_tokens'] < 1):
        raise ValueError('max_tokens must be a positive integer')
    if 'stop_sequences' in body and (not isinstance(body['stop_sequences'], list) or not all(isinstance(x, str) and x for x in body['stop_sequences'])):
        raise ValueError('stop_sequences must be a list of nonempty strings')
    manifest, tools, names = [], [], set()
    for tool in kwargs.get('tools') or []:
        if tool.get('type') != 'function':
            raise ValueError('Only function tools are supported')
        f = tool['function']
        name = f['name']
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,50}', name) or name in names:
            raise ValueError('Tool names must be unique ASCII identifiers of at most 50 characters')
        if f.get('strict'):
            raise ValueError('Strict function schemas are unsupported')
        names.add(name)
        schema, description = f.get('parameters', {'type': 'object'}), f.get('description', '')
        if not isinstance(schema, dict) or not isinstance(description, str):
            raise ValueError('Tool schema must be an object and description a string')
        manifest.append({'name': name, 'description': description, 'inputSchema': schema})
        tools.append({'name': PREFIX + name, 'description': description, 'input_schema': schema})
    body['tools'] = tools
    encoded = json.dumps(body, separators=(',', ':'), allow_nan=False)
    return encoded, manifest, names


class Request:
    def __init__(self, client):
        self.client, self.process = client, None
        self.stream = None
        self.cancelled = threading.Event()
        self.admission = None
        self.lock = threading.Lock()

    def cancel(self):
        self.cancelled.set()
        if self.admission is not None:
            self.admission.abort()
        with self.lock:
            if self.process is not None:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)  # windows-footgun: ok — spawn rejects non-POSIX before creating a process
                except ProcessLookupError:
                    pass
                except PermissionError:
                    # Darwin can report EPERM for an already-exited group during
                    # repeated cancellation. Never hide denial for a live child.
                    if self.process.poll() is None:
                        raise

    def spawn(self, command, *, stdin=subprocess.DEVNULL, **kwargs):
        with self.lock:
            if self.cancelled.is_set():
                raise RuntimeError('Claude request cancelled')
            self.process = subprocess.Popen(command, stdin=stdin, start_new_session=True, **kwargs)
        return self.process


class Stream:
    def __init__(self, iterator, request):
        self.iterator, self.request = iterator, request
        self._advancing = threading.Lock()
        request.stream = weakref.ref(self)
    def __iter__(self):
        return self
    def __next__(self):
        with self._advancing:
            return next(self.iterator)
    def close(self):
        self.request.cancel()
        # An active consumer unwinds itself after cancellation. A paused/unstarted
        # generator has no active owner and can be finalized here.
        if self._advancing.acquire(blocking=False):
            try:
                self.iterator.close()
                with self.request.client._lock:
                    self.request.client._requests.discard(self.request)
            finally:
                self._advancing.release()
    def __enter__(self):
        return self
    def __exit__(self, *_):
        self.close()


class AsyncStream:
    def __init__(self, stream):
        self.stream = stream
    def __aiter__(self):
        return self
    async def __anext__(self):
        def advance():
            try:
                return True, next(self.stream)
            except StopIteration:
                return False, None
        try:
            present, item = await asyncio.to_thread(advance)
        except asyncio.CancelledError:
            self.stream.close()
            raise
        if not present:
            raise StopAsyncIteration
        return item
    async def aclose(self):
        self.stream.close()
    async def __aenter__(self):
        return self
    async def __aexit__(self, *_):
        await self.aclose()


class Client:
    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(self, command=None, args=None, env=None, timeout=180, **_):
        # Hermes snapshots routing metadata from client-shaped objects; this is not a credential.
        self.api_key = 'external-process'
        self.base_url = 'process://claude-oauth-directsdk'
        self.env = dict(env) if env is not None else None
        source_env = self.env if self.env is not None else os.environ
        command = command or source_env.get('CLAUDE_OAUTH_DIRECTSDK_COMMAND') or 'claude'
        self.command = ([command] if isinstance(command, str) else list(command)) + list(args or [])
        self.timeout = timeout if isinstance(timeout, (int, float)) else 180
        self._lock, self._requests, self._closed = threading.Lock(), set(), False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def cancel(self):
        """Fast cross-thread cancellation: signal owned groups; never close caller-thread FDs."""
        with self._lock:
            requests = tuple(self._requests)
        for request in requests:
            request.cancel()

    def close(self):
        with self._lock:
            self._closed = True
            requests = tuple(self._requests)
        for request in requests:
            stream = request.stream() if request.stream else None
            if stream is not None:
                stream.close()
            else:
                request.cancel()

    def create(self, **kwargs):
        # Hermes' auxiliary seam returns this same object and awaits create.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            return self._acreate(**kwargs)
        return self._create(**kwargs)

    async def _acreate(self, **kwargs):
        task = asyncio.create_task(asyncio.to_thread(self._create, **kwargs))
        try:
            result = await task
            return AsyncStream(result) if kwargs.get('stream') else result
        except asyncio.CancelledError:
            self.cancel()
            raise

    def _create(self, **kwargs):
        body, manifest, names = request_body(kwargs)
        system, frames = prepare_history(kwargs.get('messages', []))
        if os.name != 'posix':
            raise RuntimeError('This native process-tree transport is currently qualified only on POSIX')
        if not isinstance(kwargs.get('model'), str) or not kwargs['model']:
            raise ValueError('model is required')
        request = Request(self)
        with self._lock:
            if self._closed:
                raise RuntimeError('Claude client is closed')
            self._requests.add(request)
        stream = Stream(self._run(request, kwargs, body, manifest, names, system, frames), request)
        if kwargs.get('stream'):
            return stream
        try:
            for chunk in stream:
                if hasattr(chunk, '_response'):
                    return chunk._response
            raise RuntimeError('Native response missing')
        finally:
            stream.close()

    def _run(self, request, kwargs, body, manifest, names, system, frames):
        p = None
        reader = None
        try:
            timeout = kwargs.get('timeout', self.timeout)
            timeout = getattr(timeout, 'read', timeout)
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                raise ValueError('timeout must be positive seconds')
            with tempfile.TemporaryDirectory(prefix='claude-directsdk-') as tmp:
                root = Path(tmp)
                (root / 'tools.json').write_text(json.dumps(manifest), encoding='utf-8')
                mcp = {'mcpServers': {'hermes': {'command': sys.executable, 'args': [str(Path(__file__).with_name('inert_mcp.py')), str(root / 'tools.json')]}}}
                env = dict(self.env if self.env is not None else os.environ)
                if self.env is None:
                    conflicts = [key for key in ('ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_BASE_URL', 'ANTHROPIC_FOUNDRY_API_KEY') if env.get(key)]
                    conflicts += [key for key in ('CLAUDE_CODE_USE_BEDROCK', 'CLAUDE_CODE_USE_VERTEX', 'CLAUDE_CODE_USE_FOUNDRY') if env.get(key, '').lower() not in ('', '0', 'false', 'no', 'off')]
                    if conflicts:
                        raise ValueError('OAuth provider refuses conflicting native auth/backend overrides: ' + ', '.join(conflicts))
                config = env.pop('CLAUDE_OAUTH_DIRECTSDK_CONFIG_DIR', None)
                if config:
                    env['CLAUDE_CONFIG_DIR'] = config
                env.pop('CLAUDE_CODE_EXTRA_BODY', None)
                env.update(ENABLE_TOOL_SEARCH='false', CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC='1', CLAUDE_CODE_MAX_RETRIES='0', DISABLE_AUTO_COMPACT='1', DISABLE_COMPACT='1')
                # Hermes owns budgets; native's replayed reminder invalidates cached history.
                env['CLAUDE_CODE_TOTAL_TOKENS_REMINDER'] = 'off'
                request.admission = Admission(env.get('ANTHROPIC_BASE_URL', 'https://api.anthropic.com'), timeout)
                env['ANTHROPIC_BASE_URL'] = request.admission.url
                # Native settings apply env inside the process, avoiding execve's
                # per-argument/environment-string limit for full Hermes schemas.
                (root / 'settings.json').write_text(json.dumps({'env': {'CLAUDE_CODE_EXTRA_BODY': body}}), encoding='utf-8')
                (root / 'system.md').write_text(system, encoding='utf-8')
                if 'max_tokens' in json.loads(body):
                    env['CLAUDE_CODE_MAX_OUTPUT_TOKENS'] = str(json.loads(body)['max_tokens'])
                command = self.command + ['-p', '--model', native_model(kwargs['model']), '--input-format', 'stream-json', '--output-format', 'stream-json', '--verbose', '--include-partial-messages', '--tools', '', '--system-prompt-file', str(root / 'system.md'), '--settings', str(root / 'settings.json'), '--setting-sources', '', '--strict-mcp-config', '--disable-slash-commands', '--max-turns', '1', '--permission-mode', 'dontAsk', '--no-session-persistence', '--mcp-config', json.dumps(mcp)]
                p = request.spawn(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding='utf-8', cwd=tmp, env=env)
                events = queue.Queue()
                def read():
                    try:
                        for line in p.stdout:
                            events.put(json.loads(line))
                    except Exception as error:
                        events.put(error)
                    finally:
                        # The consumer may close while paused at a yielded chunk.
                        # Reaping belongs to this owner thread, never cancel().
                        p.wait()
                        events.put(None)
                reader = threading.Thread(target=read, daemon=True)
                reader.start()
                deadline = time.monotonic() + timeout
                def receive():
                    nonlocal deadline
                    while True:
                        if request.cancelled.is_set():
                            raise RuntimeError('Claude request cancelled')
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError('Claude request timed out')
                        try:
                            event = events.get(timeout=min(remaining, .2))
                        except queue.Empty:
                            continue
                        if isinstance(event, Exception):
                            raise RuntimeError('Invalid native stream-json output') from event
                        deadline = time.monotonic() + timeout
                        return event
                for index, frame in enumerate(frames):
                    frame = copy.deepcopy(frame)
                    if frame['type'] == 'user' and index < len(frames) - 1:
                        frame['shouldQuery'] = False
                    p.stdin.write(json.dumps(frame, allow_nan=False) + '\n')
                    p.stdin.flush()
                    if frame.get('shouldQuery') is False:
                        while True:
                            ack = receive()
                            if ack is None:
                                raise RuntimeError('Native exited before replay acknowledgment')
                            if ack.get('type') == 'result':
                                if ack.get('num_turns') != 0 or ack.get('is_error'):
                                    raise RuntimeError('Native history replay not supported: expected zero-turn acknowledgment')
                                break
                p.stdin.close()
                assistants, results, stopped, emitted = [], [], False, ''
                native_error = None
                while True:
                    event = receive()
                    if event is None:
                        break
                    kind = event.get('type')
                    if kind == 'assistant':
                        if event.get('error') or event.get('message', {}).get('error'):
                            detail = '\n'.join(b.get('text', '') for b in event.get('message', {}).get('content', []) if b.get('type') == 'text')
                            native_error = detail
                        else:
                            assistants.append(event['message'])
                    elif kind == 'result':
                        results.append(event)
                    elif kind == 'stream_event':
                        native = event['event']
                        if native['type'] == 'message_stop':
                            stopped = True
                        delta = native.get('delta', {})
                        if delta.get('type') == 'text_delta':
                            emitted += delta['text']
                            yield self._chunk(kwargs['model'], {'content': delta['text']})
                        elif delta.get('type') == 'thinking_delta':
                            yield self._chunk(kwargs['model'], {'reasoning_content': delta['thinking']})
                p.wait(timeout=max(.1, deadline-time.monotonic()))
                reader.join(timeout=1)
                if request.cancelled.is_set():
                    raise RuntimeError('Claude request cancelled')
                admission = request.admission
                if admission.used:
                    if admission.status != 200 or not admission.capture.complete:
                        raise RuntimeError('Incomplete upstream response' + (': ' + native_error if native_error else ''))
                    assistants = [admission.capture.message]
                    stopped = True
                partial_arguments = admission.capture.incomplete_arguments if admission.used else {}
                native_failure_handled = admission.denied or bool(partial_arguments) or (admission.used and assistants[0].get('stop_reason') == 'refusal')
                if native_error and not native_failure_handled:
                    raise RuntimeError('Native API error: ' + native_error)
                if len(results) != 1 or not assistants or not stopped:
                    raise RuntimeError('Incomplete native response: assistant, message_stop and one result required')
                final = results[0]
                blocks = [b for a in assistants for b in a['content']]
                calls = []
                for index, block in enumerate(blocks):
                    if block.get('type') == 'tool_use':
                        name = block['name']
                        if not name.startswith(PREFIX) or name[len(PREFIX):] not in names:
                            raise RuntimeError('Native returned a tool outside the current host inventory')
                        arguments = partial_arguments[index] if index in partial_arguments else json.dumps(block['input'], separators=(',', ':'), allow_nan=False)
                        calls.append({'id': block['id'], 'type': 'function', 'function': {'name': name[len(PREFIX):], 'arguments': arguments}})
                boundary = bool(calls) and final.get('subtype') == 'error_max_turns' and p.returncode == 1
                if not boundary and not native_failure_handled and (p.returncode != 0 or final.get('is_error') or final.get('subtype') != 'success'):
                    raise RuntimeError('Native request failed: ' + str(final.get('subtype')))
                usage = assistants[0]['usage'] if admission.used else final.get('usage')
                if not isinstance(usage, dict) or not all(isinstance(usage.get(k), (int, float)) for k in ('input_tokens', 'output_tokens')):
                    raise RuntimeError('Native result missing complete token usage')
                text = ''.join(b.get('text', '') for b in blocks if b.get('type') == 'text')
                if emitted != text:
                    if text.startswith(emitted):
                        yield self._chunk(kwargs['model'], {'content': text[len(emitted):]})
                    else:
                        raise RuntimeError('Native final text differs from incremental stream')
                message = {'role': 'assistant', 'content': text or None, 'tool_calls': calls or None,
                           'reasoning_content': ''.join(b.get('thinking', '') for b in blocks if b.get('type') == 'thinking') or None}
                # Incomplete tool arguments are a host retry signal, never a valid
                # signed replay block. Hermes' length path will not execute them.
                carriers = [] if partial_arguments else [{'type': CARRIER, 'version': 1, 'messages': assistants, 'projection': projection(message)}]
                message['reasoning_details'] = carriers
                inp = usage['input_tokens'] + usage.get('cache_read_input_tokens', 0) + usage.get('cache_creation_input_tokens', 0)
                normalized_usage = {'prompt_tokens': inp, 'completion_tokens': usage['output_tokens'], 'total_tokens': inp + usage['output_tokens'], 'prompt_tokens_details': {'cached_tokens': usage.get('cache_read_input_tokens', 0)}, 'cache_creation_input_tokens': usage.get('cache_creation_input_tokens', 0), 'native_usage': usage,
                                    'completion_tokens_details': {'reasoning_tokens': usage.get('output_tokens_details', {}).get('thinking_tokens', 0)},
                                    'native_cost': {'total_cost_usd': final.get('total_cost_usd'), 'modelUsage': final.get('modelUsage')}}
                normalized_usage['native_admission'] = {'upstream_requests': int(admission.used), 'blocked_requests': admission.denied, 'request_id': admission.request_id}
                # Context pressure is not output exhaustion. Preserve this stop
                # so Hermes compresses canonical history rather than appending
                # output-continuation prompts to an already-full context.
                context_exhausted = any(a.get('stop_reason') == 'model_context_window_exceeded' for a in assistants)
                finish = 'model_context_window_exceeded' if context_exhausted else ('length' if partial_arguments else ('tool_calls' if calls else ('length' if any(a.get('stop_reason') == 'max_tokens' for a in assistants) else 'stop')))
                response = obj({'id': assistants[-1].get('id', 'claude-native'), 'model': kwargs['model'], 'object': 'chat.completion', 'choices': [{'index': 0, 'finish_reason': finish, 'message': message}], 'usage': normalized_usage})
                chunk = self._chunk(kwargs['model'], {'content': None, 'tool_calls': [dict(tc, index=i) for i, tc in enumerate(calls)] or None, 'reasoning_details': carriers}, finish, normalized_usage)
                chunk._response = response
                yield chunk
        finally:
            request.cancel()
            if request.admission is not None:
                request.admission.close()
            if p is not None:
                p.wait(timeout=5)
                if reader is not None:
                    reader.join(timeout=5)
                for pipe in (p.stdin, p.stdout):
                    if pipe and not pipe.closed:
                        pipe.close()
            with self._lock:
                self._requests.discard(request)

    @staticmethod
    def _chunk(model, delta, finish=None, usage=None):
        return obj({'id': 'claude-native', 'model': model, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'content': None, 'tool_calls': None, 'reasoning_details': None, **delta}, 'finish_reason': finish}], 'usage': usage})
