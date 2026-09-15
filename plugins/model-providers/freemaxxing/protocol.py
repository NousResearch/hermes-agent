"""Completion validation and atomic SSE replay: partial tool calls never escape."""
from __future__ import annotations

import json

import httpx

from .policy import (Budget, TransientError, _MAX_RESPONSE_BODY_BYTES,
                     _MAX_SSE_EVENT_BYTES, _MAX_SSE_LINE_BYTES)


def loads(raw):
    def invalid(_value):
        raise ValueError('non-finite JSON value')
    try:
        return json.loads(raw, parse_constant=invalid)
    except (ValueError, UnicodeError, TypeError) as exc:
        raise TransientError('invalid upstream JSON') from exc


def validate_completion(payload):
    if not isinstance(payload, dict) or 'error' in payload:
        raise TransientError('upstream returned an error envelope')
    choices = payload.get('choices')
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise TransientError('upstream returned no single completion')
    choice = choices[0]
    message = choice.get('message')
    if not isinstance(message, dict) or message.get('role', 'assistant') != 'assistant':
        raise TransientError('upstream returned an invalid assistant message')
    if choice.get('finish_reason') not in {'stop', 'length', 'tool_calls', 'content_filter'}:
        raise TransientError('upstream completion has no terminal reason')
    tools = message.get('tool_calls', [])
    if tools is None:
        tools = []
    if not isinstance(tools, list):
        raise TransientError('invalid tool-call envelope')
    seen = set()
    for tool in tools:
        if not isinstance(tool, dict):
            raise TransientError('invalid tool call')
        function = tool.get('function')
        identity = tool.get('id')
        if (not isinstance(identity, str) or not identity or identity in seen or
                tool.get('type') != 'function' or not isinstance(function, dict) or
                not isinstance(function.get('name'), str) or not function['name'] or
                not isinstance(function.get('arguments'), str) or
                not isinstance(loads(function['arguments']), dict)):
            raise TransientError('incomplete or invalid tool call')
        seen.add(identity)
    content = message.get('content') or message.get('refusal')
    if not tools and (not isinstance(content, str) or not content.strip()):
        raise TransientError('upstream returned no usable final content')
    if tools and choice.get('finish_reason') not in {'tool_calls', 'stop'}:
        raise TransientError('upstream truncated a tool-call response')
    return payload


class StreamValidator:
    def __init__(self):
        self.message = {'role': 'assistant', 'content': ''}
        self.tools = {}
        self.finish = None
        self.done = False
        self.identity = None

    def event(self, data):
        if data.strip() == b'[DONE]':
            self.message['tool_calls'] = [self.tools[i] for i in sorted(self.tools)]
            validate_completion({'choices': [{'message': self.message,
                                              'finish_reason': self.finish}]})
            self.done = True
            return
        chunk = loads(data)
        if not isinstance(chunk, dict) or 'error' in chunk:
            raise TransientError('upstream SSE error')
        choices = chunk.get('choices')
        if choices == [] and 'usage' in chunk:
            return
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise TransientError('invalid SSE choices')
        identity = chunk.get('id')
        if identity:
            if self.identity and self.identity != identity:
                raise TransientError('completion identity changed mid-stream')
            self.identity = identity
        choice = choices[0]
        if choice.get('index', 0) != 0:
            raise TransientError('unexpected SSE choice index')
        delta = choice.get('delta')
        if not isinstance(delta, dict):
            raise TransientError('invalid SSE delta')
        if self.finish is not None and any(delta.get(k) for k in ('content', 'tool_calls', 'refusal')):
            raise TransientError('content appeared after terminal SSE event')
        for field in ('content', 'refusal'):
            value = delta.get(field)
            if value is not None:
                if not isinstance(value, str):
                    raise TransientError('invalid SSE text')
                self.message[field] = self.message.get(field, '') + value
        calls = delta.get('tool_calls', [])
        if calls is None:
            calls = []
        if not isinstance(calls, list):
            raise TransientError('invalid streamed tool calls')
        for call in calls:
            if not isinstance(call, dict) or type(call.get('index')) is not int or not 0 <= call['index'] < 128:
                raise TransientError('invalid streamed tool index')
            tool = self.tools.setdefault(call['index'], {'id': '', 'type': 'function',
                                                        'function': {'name': '', 'arguments': ''}})
            for field in ('id', 'type'):
                if call.get(field):
                    if not isinstance(call[field], str):
                        raise TransientError('invalid streamed tool identity')
                    if field == 'id' and tool['id'] and tool['id'] != call[field]:
                        raise TransientError('tool identity changed mid-stream')
                    tool[field] = call[field]
            function = call.get('function', {})
            if not isinstance(function, dict):
                raise TransientError('invalid streamed function')
            for field in ('name', 'arguments'):
                value = function.get(field)
                if value is not None:
                    if not isinstance(value, str):
                        raise TransientError('invalid streamed function field')
                    tool['function'][field] += value
        if choice.get('finish_reason') is not None:
            if self.finish is not None and self.finish != choice['finish_reason']:
                raise TransientError('conflicting SSE termination')
            self.finish = choice['finish_reason']


def read_stream(response, budget: Budget):
    """Return one complete validated stream or fail before downstream HTTP 200.

    Atomic replay intentionally trades first-token latency for transparent
    failover after any upstream interruption. It does not splice generations.
    """
    validator = StreamValidator()
    raw, pending, event = bytearray(), bytearray(), []
    event_size = 0
    try:
        for chunk in response.iter_raw():
            budget.remaining()
            if len(raw) + len(chunk) > _MAX_RESPONSE_BODY_BYTES:
                raise TransientError('SSE response exceeds total byte bound')
            raw.extend(chunk)
            pending.extend(chunk)
            offset = 0
            while (newline := pending.find(b'\n', offset)) >= 0:
                line = pending[offset:newline]
                offset = newline + 1
                if len(line) > _MAX_SSE_LINE_BYTES:
                    raise TransientError('SSE line exceeds byte bound')
                line = line.rstrip(b'\r')
                if not line:
                    if event:
                        validator.event(b'\n'.join(event))
                    event, event_size = [], 0
                    if validator.done:
                        # Do not forward bytes following the terminal event in
                        # the same network chunk (a second generation/error).
                        return bytes(raw[:len(raw) - len(pending) + offset])
                else:
                    event_size += len(line)
                    if event_size > _MAX_SSE_EVENT_BYTES:
                        raise TransientError('SSE event exceeds byte bound')
                    if line.startswith(b'data:'):
                        event.append(bytes(line[5:]).lstrip(b' '))
            del pending[:offset]
            if len(pending) > _MAX_SSE_LINE_BYTES:
                raise TransientError('unterminated SSE line exceeds byte bound')
        raise TransientError('upstream stream ended without valid terminal completion')
    except (httpx.HTTPError, OSError) as exc:
        raise TransientError('upstream stream interrupted') from exc
    finally:
        response.close()
