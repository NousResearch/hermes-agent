"""Literal, active-only input with a session-lifetime receipt ledger.

The inbox lock serializes admission, incorporation and closure. It is independent
of the interactive redirect/steer buffers, which may survive into another turn.
"""
from __future__ import annotations

import threading
import uuid


CAPABILITY = "ai.nous/active-turn-guidance"
VERSION = 1


class GuidanceInbox:
    def __init__(self, instance_id: str, session_id: str):
        self.instance_id = instance_id
        self.session_id = session_id
        self.lock = threading.Lock()
        self.turn_id = ""
        self.prompt_id = ""
        self.closed = "rejected_turn_ended"
        self.receipts: dict[str, dict] = {}

    def start(self, prompt_id: str) -> dict:
        with self.lock:
            self.turn_id = str(uuid.uuid4())
            self.prompt_id = prompt_id
            self.closed = ""
            return self.identity()

    def identity(self) -> dict:
        return {"runtimeInstanceId": self.instance_id, "sessionId": self.session_id,
                "turnId": self.turn_id, "promptId": self.prompt_id}

    def request(self, params: dict, *, lookup: bool = False) -> dict:
        with self.lock:
            identity = {"runtimeInstanceId": params.get("runtimeInstanceId"),
                        "sessionId": params.get("sessionId"), "turnId": params.get("expectedTurnId"),
                        "messageId": params.get("messageId")}
            reject = lambda why: {**identity, "disposition": why, "state": "rejected"}
            if params.get("runtimeInstanceId") != self.instance_id:
                return reject("rejected_instance_mismatch")
            if params.get("sessionId") != self.session_id:
                return reject("rejected_session_mismatch")
            message_id = params.get("messageId")
            if not isinstance(message_id, str) or not message_id or len(message_id) > 128:
                return reject("invalid_input")
            previous = self.receipts.get(message_id)
            if previous is not None:
                if previous["turnId"] != params.get("expectedTurnId") or (
                    not lookup and previous["text"] != params.get("text")
                ):
                    return reject("message_id_conflict")
                return self._public(previous)
            if lookup:
                return reject("unknown_message")
            text = params.get("text")
            if not isinstance(text, str) or not text.strip() or len(text.encode("utf-8")) > 262144:
                return reject("invalid_input")
            if params.get("expectedTurnId") != self.turn_id:
                return reject("rejected_turn_mismatch")
            if self.closed:
                return reject(self.closed)
            # Never evict a receipt and make a previously used identity admissible again.
            if len(self.receipts) >= 4096:
                return reject("receipt_capacity_reached")
            receipt = {**identity, "disposition": "accepted_active", "state": "pending_incorporation",
                       "sequence": len(self.receipts) + 1, "text": text}
            self.receipts[message_id] = receipt
            return self._public(receipt)

    @staticmethod
    def _public(receipt: dict) -> dict:
        return {k: v for k, v in receipt.items() if k != "text"}

    def incorporate(self, messages: list[dict]) -> None:
        with self.lock:
            if self.closed or not messages or messages[-1].get("role") not in {"tool", "assistant"}:
                return
            pending = [r for r in self.receipts.values()
                       if r["turnId"] == self.turn_id and r["state"] == "pending_incorporation"]
            if not pending:
                return
            # One new user row keeps both the cached prefix and role alternation intact.
            messages.append({"role": "user", "content": "\n\n".join(r["text"] for r in pending)})
            for receipt in pending:
                receipt["state"] = "incorporated"

    def snapshot(self) -> dict:
        with self.lock:
            return {**self.identity(), "state": "ended" if self.closed else "started",
                    "receipts": [self._public(r) for r in self.receipts.values()]}

    def close(self, *, cancelled: bool = False) -> list[dict]:
        with self.lock:
            if not self.closed:
                self.closed = "rejected_cancelled" if cancelled else "rejected_turn_ended"
                for receipt in self.receipts.values():
                    if receipt["turnId"] == self.turn_id and receipt["state"] == "pending_incorporation":
                        receipt["state"] = "cancelled_before_incorporation" if cancelled else "unconsumed_at_turn_end"
            return [self._public(r) for r in self.receipts.values() if r["turnId"] == self.turn_id]


def incorporate_active_guidance(agent, messages: list[dict]) -> None:
    inbox = getattr(agent, "_active_turn_guidance", None)
    if isinstance(inbox, GuidanceInbox):
        inbox.incorporate(messages)
