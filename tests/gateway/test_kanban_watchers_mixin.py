"""Kanban dispatcher tick logging and alerting (PR-added tests; the pre-existing module was purged upstream in 61fa0f47ec)."""


def test_dispatch_tick_log_counts_and_names_parent_satisfied_sticky(caplog):
    import logging

    from gateway.kanban_watchers_dispatcher import _log_spawn_results, logger
    from hermes_cli.kanban_db_dispatch import DispatchResult

    result = DispatchResult(parent_satisfied_sticky=["t_beta", "t_alpha"])
    with caplog.at_level(logging.INFO, logger=logger.name):
        assert _log_spawn_results([("board", result)]) is False

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "parents_done_sticky=2 (t_alpha, t_beta)" in message
        for message in messages
    )
