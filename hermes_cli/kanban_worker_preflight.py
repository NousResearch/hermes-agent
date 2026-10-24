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
        from hermes_cli.kanban_worker_log import open_worker_log
        message = exc.detail or exc.reason
        with open_worker_log(task, board) as log:
            log.write(f"Kanban worker refused: {message}\n".encode())
        raise RuntimeError(message) from exc
