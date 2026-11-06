"""The local-model task pins exist only inside their owner-admitted worker turn."""

import pytest

from agent.delegation_context import (
    owned_local_gpu_task_context,
    scoped_owned_local_gpu_task_context,
)


def test_local_gpu_task_context_is_scoped_and_restored_after_failure():
    assert owned_local_gpu_task_context() == {}

    with pytest.raises(RuntimeError, match="turn failed"):
        with scoped_owned_local_gpu_task_context({
            "capability_id": "fixed-json.v1",
            "task_contract_sha256": "a" * 64,
        }):
            assert owned_local_gpu_task_context() == {
                "capability_id": "fixed-json.v1",
                "task_contract_sha256": "a" * 64,
            }
            raise RuntimeError("turn failed")

    assert owned_local_gpu_task_context() == {}
