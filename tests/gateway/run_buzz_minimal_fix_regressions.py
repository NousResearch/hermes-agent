#!/usr/bin/env python3
"""Dependency-light regressions for the evidence-corrected Buzz fix."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path


SELF = "9fd5c7ba6d3ef224da78f541e0fcb9c50f72cc63edb19aae76ac6a0474dfa860"
OWNER = "a" * 64
CHANNEL = "ccc2bc1a-7a82-5a8f-8c4e-57a070cbe7cd"
ROOT = "b5c066e2e384da259f91c054259c10f44918506e5e277fe8087ed26e5def1d38"
TRIGGER = "54e64063eceaa8e260fa842a9ead13d5803d6c4136da939f9270156de8fec9ad"


def documented_nip10_tags(reply_to: str | None) -> list[list[str]]:
    """Return the observed Buzz NIP-10 contract, without scalar collapse."""
    tags = [["h", CHANNEL]]
    if reply_to == TRIGGER:
        tags.extend([
            ["e", ROOT, "", "root"],
            ["e", TRIGGER, "", "reply"],
        ])
    elif reply_to:
        tags.append(["e", reply_to, "", "reply"])
    return tags


class DocumentedNip10Cli:
    """Offline Buzz CLI boundary double using distinct root/reply markers."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str | None]] = []
        self.events: list[dict] = []
        self._sequence = 0

    async def __call__(self, args, *, input_text=None):
        self.calls.append((list(args), input_text))
        if args[:2] == ["messages", "send"]:
            self._sequence += 1
            event_id = f"event-{self._sequence}"
            reply_to = args[args.index("--reply-to") + 1] if "--reply-to" in args else None
            self.events.append({
                "id": event_id,
                "pubkey": SELF,
                "content": input_text or "",
                "created_at": 1001,
                "kind": 9,
                "tags": documented_nip10_tags(reply_to),
            })
            return 0, json.dumps({"accepted": True, "event_id": event_id}), ""
        if args[:2] == ["messages", "get"]:
            return 0, json.dumps(self.events), ""
        if args[:2] == ["reactions", "add"]:
            return 0, json.dumps({"accepted": True}), ""
        return 0, "[]", ""


async def wait_for_adapter_tasks(adapter) -> None:
    for _ in range(100):
        tasks = [task for task in getattr(adapter, "_background_tasks", ()) if not task.done()]
        if not tasks:
            return
        await asyncio.gather(*tasks)
    raise AssertionError("adapter background task did not finish")


async def run(source_root: Path) -> dict:
    sys.path.insert(0, str(source_root))
    from gateway.config import PlatformConfig
    from tests.gateway._plugin_adapter_loader import load_plugin_adapter

    module = load_plugin_adapter("buzz")
    module._DELIVERY_READBACK_DELAY = 0
    Adapter = module.BuzzAdapter

    def make_adapter() -> object:
        config = PlatformConfig(
            enabled=True,
            typing_indicator=False,
            extra={"relay_url": "https://offline.invalid"},
        )
        adapter = Adapter(config)
        adapter._self_pubkey = SELF
        adapter._self_npub = module.hex_to_npub(SELF) or ""
        adapter._display_name = "QA"
        adapter._user_names[OWNER] = "Owner"
        adapter._channel_state[CHANNEL] = {
            "chat_type": "group",
            "last_ts": 0,
            "seen": OrderedDict(),
        }
        return adapter

    results: list[dict] = []

    async def case(name, operation) -> None:
        try:
            await operation()
        except Exception as exc:
            results.append({
                "name": name,
                "status": "FAIL",
                "detail": f"{type(exc).__name__}: {exc}",
            })
        else:
            results.append({"name": name, "status": "PASS"})

    async def nested_reply_preserves_root_and_parent() -> None:
        adapter = make_adapter()
        cli = DocumentedNip10Cli()
        adapter._run_cli = cli
        result = await adapter.send(CHANNEL, "nested final", reply_to=TRIGGER)
        assert result.success is True
        tags = cli.events[0]["tags"]
        assert ["e", ROOT, "", "root"] in tags
        assert ["e", TRIGGER, "", "reply"] in tags

    async def root_marker_cannot_satisfy_reply() -> None:
        adapter = make_adapter()
        cli = DocumentedNip10Cli()
        cli.events.append({
            "id": "root-only-event",
            "pubkey": SELF,
            "content": "invalid root-only reply",
            "created_at": 1001,
            "kind": 9,
            "tags": [["h", CHANNEL], ["e", TRIGGER, "", "root"]],
        })
        adapter._run_cli = cli
        verified, diagnostic = await adapter._read_back_delivery(
            chat_id=CHANNEL,
            event_id="root-only-event",
            content="invalid root-only reply",
            reply_target=TRIGGER,
        )
        assert verified is False
        assert diagnostic == "mismatch:anchor"

    async def correlated_turn_uses_exact_trigger() -> None:
        adapter = make_adapter()
        cli = DocumentedNip10Cli()
        adapter._run_cli = cli
        outcomes = []

        async def handler(_event):
            outcomes.append(await adapter._send_with_retry(
                CHANNEL,
                "correlated final",
                reply_to=ROOT,
                metadata={"thread_id": ROOT, "notify": True},
            ))

        adapter._message_handler = handler
        await adapter._dispatch_message(
            text="do the task",
            chat_id=CHANNEL,
            chat_type="group",
            user_id=OWNER,
            user_name="Owner",
            message_id=TRIGGER,
            created_at=1000,
        )
        await wait_for_adapter_tasks(adapter)
        sends = [call for call in cli.calls if call[0][:2] == ["messages", "send"]]
        assert outcomes and outcomes[0].success is True
        assert sends[0][0][sends[0][0].index("--reply-to") + 1] == TRIGGER
        assert ["e", ROOT, "", "root"] in cli.events[0]["tags"]
        assert ["e", TRIGGER, "", "reply"] in cli.events[0]["tags"]

    async def only_terminal_task_message_publishes() -> None:
        adapter = make_adapter()
        cli = DocumentedNip10Cli()
        adapter._run_cli = cli
        outcomes = []

        async def handler(_event):
            for label in (
                "status callback",
                "working heartbeat",
                "tool progress",
                "compression warning",
                "onboarding notice",
            ):
                outcomes.append(await adapter.send(
                    CHANNEL, label, reply_to=ROOT, metadata={}
                ))
            outcomes.append(await adapter._send_with_retry(
                CHANNEL, "one final", reply_to=ROOT, metadata={"notify": True}
            ))
            outcomes.append(await adapter._send_with_retry(
                CHANNEL, "second final", reply_to=ROOT, metadata={"notify": True}
            ))

        adapter._message_handler = handler
        await adapter._dispatch_message(
            text="do the task",
            chat_id=CHANNEL,
            chat_type="group",
            user_id=OWNER,
            user_name="Owner",
            message_id=TRIGGER,
            created_at=1000,
        )
        await wait_for_adapter_tasks(adapter)
        sends = [call for call in cli.calls if call[0][:2] == ["messages", "send"]]
        assert len(sends) == 1
        assert all(result.success is False for result in outcomes[:5])
        assert outcomes[5].success is True
        assert outcomes[6].success is False

        from gateway.run import _prepare_gateway_status_message

        class BuzzPlatform:
            value = "buzz"

        assert _prepare_gateway_status_message(
            BuzzPlatform(), "warning", "Codex context compaction notice"
        ) is None

    async def post_validation_config_mutation_fails_closed() -> None:
        adapter = make_adapter()
        cli = DocumentedNip10Cli()
        adapter._run_cli = cli
        outcomes = []
        outbound_boundary_index = []
        config_path = Path(os.environ["HERMES_HOME"]) / "config.yaml"
        local_image = Path(os.environ["HERMES_HOME"]) / "fixture.png"
        local_image.write_bytes(b"synthetic image bytes")

        async def handler(_event):
            outbound_boundary_index.append(len(cli.calls))
            config_path.write_text(
                "model: openai-codex:gpt-5.6-terra\nchanged: true\n",
                encoding="utf-8",
            )
            outcomes.append(await adapter._send_with_retry(
                CHANNEL, "final", reply_to=ROOT, metadata={"notify": True}
            ))
            outcomes.append(await adapter.send_image(
                CHANNEL, str(local_image), caption="image", reply_to=TRIGGER
            ))
            outcomes.append(await adapter.send_reaction(CHANNEL, TRIGGER, "seen"))

        adapter._message_handler = handler
        await adapter._dispatch_message(
            text="do the task",
            chat_id=CHANNEL,
            chat_type="group",
            user_id=OWNER,
            user_name="Owner",
            message_id=TRIGGER,
            created_at=1000,
        )
        await wait_for_adapter_tasks(adapter)
        assert len(outcomes) == 3
        assert all(not getattr(result, "success", result) for result in outcomes)
        outbound_calls = [
            call
            for call in cli.calls[outbound_boundary_index[0]:]
            if call[0][:2] in (["messages", "send"], ["reactions", "add"])
        ]
        assert not outbound_calls
        assert adapter.configuration_integrity_state == "FAILED_CLOSED"

    async def buzz_first_contact_is_nonwriting() -> None:
        from gateway.run import _gateway_profile_config_writes_allowed

        class BuzzPlatform:
            value = "buzz"

        class TelegramPlatform:
            value = "telegram"

        assert _gateway_profile_config_writes_allowed(BuzzPlatform()) is False
        assert _gateway_profile_config_writes_allowed(TelegramPlatform()) is True

    await case("documented_nested_reply_root_and_parent_markers", nested_reply_preserves_root_and_parent)
    await case("root_only_marker_rejected_for_reply", root_marker_cannot_satisfy_reply)
    await case("correlated_task_uses_exact_trigger", correlated_turn_uses_exact_trigger)
    await case("buzz_nonterminal_events_suppressed", only_terminal_task_message_publishes)
    await case("post_validation_config_mutation_fails_closed", post_validation_config_mutation_fails_closed)
    await case("buzz_first_contact_onboarding_is_nonwriting", buzz_first_contact_is_nonwriting)
    failed = sum(item["status"] == "FAIL" for item in results)
    return {
        "schema_version": 1,
        "result": "PASS" if failed == 0 else "FAIL",
        "passed": len(results) - failed,
        "failed": failed,
        "tests": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="g2-1-fix-003-home-") as isolated_home:
        config_path = Path(isolated_home) / "config.yaml"
        config_path.write_text(
            "model: openai-codex:gpt-5.6-terra\n",
            encoding="utf-8",
        )
        os.environ["HERMES_HOME"] = isolated_home
        result = asyncio.run(run(args.source_root.resolve()))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
