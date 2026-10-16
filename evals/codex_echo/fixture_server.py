#!/usr/bin/env python3
"""Offline Codex JSON-RPC peer; no provider or authentication calls."""
import json
import os
import sys


def send(value):
    print(json.dumps(value), flush=True)


for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request:
        continue
    method = request["method"]
    result = {}
    if method == "thread/start":
        result = {"thread": {"id": "fixture-thread"}}
    elif method == "turn/start":
        result = {"turn": {"id": "fixture-turn"}}
    send({"jsonrpc": "2.0", "id": request["id"], "result": result})
    if method != "turn/start":
        continue
    text = request["params"]["input"][0]["text"]
    mode = os.environ.get("ECHO_FIXTURE_MODE", "echo")
    items = []
    if mode != "assistant_only":
        items.append({"type": "userMessage", "id": "input", "content": [
            {"type": "text", "text": "different" if mode == "different" else text}]})
    items.append({"type": "agentMessage", "id": "reply", "text": "fixture reply"})
    if mode == "later_equal":
        items.append({"type": "userMessage", "id": "steer", "content": [{"type": "text", "text": text}]})
    for item in items:
        send({"method": "item/completed", "params": {
            "threadId": "fixture-thread", "turnId": "fixture-turn", "item": item}})
    send({"method": "turn/completed", "params": {
        "threadId": "fixture-thread", "turn": {"id": "fixture-turn", "status": "completed", "error": None}}})
