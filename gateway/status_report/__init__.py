"""Status reports: a gateway tells a URL chosen by its hosting deployment how it is doing.

Off unless the launch environment carries ``GATEWAY_STATUS_REPORT_URL``. Siblings: ``target`` (the
URL and its credential), ``payload`` (the report body: contract v1, ``status`` cut to an allow-list),
``transport`` (one POST over a fresh connection, the reply classified), ``reporter`` (the schedule,
the backoff, the one-sender lock and the goodbye before a self-suspend). The contract and its test
vectors are in ``tests/fixtures/status_report/v1``.

Nothing is imported here: the gateway imports ``reporter`` only when the stamp is set.
"""
