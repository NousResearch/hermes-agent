from hermes_os_integration.delegation import delegate_task


def test_delegation_dry_run_stores_hermes_os_reference():
    result = delegate_task({
        "task_id": "task-1",
        "project_id": "project-1",
        "task_type": "docs",
        "prompt": "Write docs",
        "working_directory": "/tmp",
        "dry_run": True,
    })

    assert result.request.agent_kind == "documentation"
    assert result.response.status == "dry_run"
    assert result.persisted_outputs[0].startswith("hermes-os://")
