"""Validate the assigned profile before the dispatcher launches its worker client."""


def validate_worker_model(task, *, board=None):
    from gateway.run import _load_gateway_config
    from gateway.session_kanban import resolve_worker_model
    from hermes_cli import kanban_db_dispatch as dispatch
    from hermes_cli.profiles import normalize_profile_name, resolve_profile_env
    from hermes_state_runtime import RuntimeStoreError

    profile = normalize_profile_name(task.assignee)
    try:
        with dispatch._worker_profile_scope(resolve_profile_env(profile)):
            resolve_worker_model(task, _load_gateway_config())
    except RuntimeStoreError as exc:
        message = exc.detail or exc.reason
        with dispatch._open_worker_log(task, board) as log:
            log.write(f"Kanban worker refused: {message}\n".encode())
        raise RuntimeError(message) from exc


def call_spawn_fn(spawn_fn, task, workspace: str, board: str | None) -> int | None:
    """Back-compat: older spawn_fn signatures (and test stubs) accept only
    ``(task, workspace)``; pass ``board`` only when the callable supports it."""
    if spawn_fn is None:
        from hermes_cli import kanban_db_dispatch as dispatch
        validate_worker_model(task, board=board)
        spawn_fn = dispatch._default_spawn
    import inspect
    try:
        sig = inspect.signature(spawn_fn)
        if "board" in sig.parameters:
            return spawn_fn(task, workspace, board=board)
        return spawn_fn(task, workspace)
    except (TypeError, ValueError):
        return spawn_fn(task, workspace)
