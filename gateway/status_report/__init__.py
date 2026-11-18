"""Status reports: a gateway tells a URL chosen by its hosting deployment how it is doing.

Off unless the launch environment carries ``GATEWAY_STATUS_REPORT_URL``. Siblings: ``target`` (the
URL and its credential), ``payload`` (the report body: contract v1, ``status`` cut to an allow-list),
``transport`` (one POST over a fresh connection, the reply classified), ``reporter`` (the schedule,
the backoff, the one-sender lock and the goodbye before a self-suspend). The contract and its test
vectors are in ``tests/fixtures/status_report/v1``.
"""

from gateway.status_report.reporter import INTERVAL_S, StatusReporter, status_reporter_from_launch_env
from gateway.status_report.target import STATUS_REPORT_URL_ENV

__all__ = ["INTERVAL_S", "STATUS_REPORT_URL_ENV", "StatusReporter", "status_reporter_from_launch_env"]
