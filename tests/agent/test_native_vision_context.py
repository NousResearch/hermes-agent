from __future__ import annotations

import asyncio

import pytest

from agent.delegation_context import delegated_child_context
from agent.native_vision_context import (
    is_native_image_attached,
    scoped_native_image_refs,
)


def test_scope_normalizes_file_urls_and_resets_after_success(tmp_path):
    image = tmp_path / "image.png"

    with scoped_native_image_refs([f"file://{image}"]):
        assert is_native_image_attached(str(image)) is True

    assert is_native_image_attached(str(image)) is False


def test_scope_resets_after_exception(tmp_path):
    image = tmp_path / "image.png"

    with pytest.raises(RuntimeError, match="turn failed"):
        with scoped_native_image_refs([str(image)]):
            assert is_native_image_attached(str(image)) is True
            raise RuntimeError("turn failed")

    assert is_native_image_attached(str(image)) is False


def test_delegated_child_does_not_inherit_parent_attachment(tmp_path):
    image = tmp_path / "image.png"

    with scoped_native_image_refs([str(image)]):
        assert is_native_image_attached(str(image)) is True
        with delegated_child_context():
            assert is_native_image_attached(str(image)) is False
        assert is_native_image_attached(str(image)) is True


def test_cancelled_task_does_not_leak_attachment(tmp_path):
    image = tmp_path / "image.png"
    entered = asyncio.Event()

    async def cancelled_turn():
        with scoped_native_image_refs([str(image)]):
            entered.set()
            await asyncio.Event().wait()

    async def scenario():
        task = asyncio.create_task(cancelled_turn())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert is_native_image_attached(str(image)) is False

    asyncio.run(scenario())


def test_concurrent_turns_keep_attachment_refs_isolated(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    both_entered = asyncio.Event()
    entered = 0
    lock = asyncio.Lock()

    async def turn(own, other):
        nonlocal entered
        with scoped_native_image_refs([str(own)]):
            async with lock:
                entered += 1
                if entered == 2:
                    both_entered.set()
            await both_entered.wait()
            assert is_native_image_attached(str(own)) is True
            assert is_native_image_attached(str(other)) is False

    async def scenario():
        await asyncio.gather(turn(first, second), turn(second, first))

    asyncio.run(scenario())
