import pytest

from gateway.run import (
    _build_runtime_awareness_dispatch_command,
    _runtime_awareness_approval_run_id,
)


def test_runtime_awareness_approval_phrase_extracts_exact_run_id():
    assert _runtime_awareness_approval_run_id('APPROVE runtime-awareness repair 20260520T155326Z') == '20260520T155326Z'
    assert _runtime_awareness_approval_run_id('approve runtime-awareness repair 20260520T155326Z') == '20260520T155326Z'
    assert _runtime_awareness_approval_run_id('APPROVE repair 20260520T155326Z') is None


def test_runtime_awareness_dispatch_command_uses_active_script_path():
    cmd = _build_runtime_awareness_dispatch_command('20260520T155326Z')
    assert cmd == [
        '/home/alcoo/.hermes/scripts/runtime-awareness-approval-dispatch.sh',
        'APPROVE',
        'runtime-awareness',
        'repair',
        '20260520T155326Z',
    ]
