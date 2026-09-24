"""Local stdio peer for the permission-denial wire contract; no provider access."""
import json
import sys


def send(message):
    print(json.dumps(message), flush=True)


def note(method, **params):
    send({"method": method, "params": {"threadId": "thread-1", "turnId": "turn-1", **params}})


for line in sys.stdin:
    message = json.loads(line)
    method = message.get("method")
    rid = message.get("id")
    if method == "initialize":
        send({"id": rid, "result": {"userAgent": "permission-fixture"}})
    elif method == "thread/start":
        send({"id": rid, "result": {"thread": {"id": "thread-1"}}})
    elif method == "turn/start":
        send({"id": rid, "result": {"turn": {"id": "turn-1"}}})
        send({"id": "permission-1", "method": "item/permissions/requestApproval", "params": {
            "threadId": "thread-1", "turnId": "turn-1", "itemId": "item-1",
            "permissions": {"network": {"enabled": True}}, "reason": "needs network",
        }})
    elif rid == "permission-1":
        # A deny must be a valid PermissionsRequestApprovalResponse and grant nothing.
        if message.get("result") != {"permissions": {}}:
            note("turn/completed", turn={"id": "turn-1", "status": "failed", "error": {
                "message": "invalid permission denial: " + json.dumps(message),
            }})
        else:
            note("item/completed", item={"id": "answer-1", "type": "agentMessage", "text": "DENIED-CLEANLY"})
            note("turn/completed", turn={"id": "turn-1", "status": "completed", "error": None})
