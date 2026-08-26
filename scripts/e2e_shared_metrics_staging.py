"""Live staging E2E for the shared-metrics exporter.

Sends REAL packages through the REAL sender to the REAL staging ingest
service, then reports what the service acknowledged. Uses a throwaway
HERMES_HOME so the operator's own telemetry state is untouched.

Usage:
    .venv/bin/python scripts/e2e_shared_metrics_staging.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

STAGING = "https://telemetry.staging-nousresearch.com/v1/telemetry"


def main() -> int:
    scratch = Path(tempfile.mkdtemp(prefix="hermes-telemetry-e2e-"))
    os.environ["HERMES_HOME"] = str(scratch)

    # Staging is selected by writing config into the THROWAWAY profile, not by
    # an environment override: a runtime env var that can retarget consented
    # telemetry would be a consent hazard in production.
    (scratch / "config.yaml").write_text(
        "telemetry:\n"
        "  shared_metrics:\n"
        "    enabled: true\n"
        "    send: true\n"
        f"    endpoint: {STAGING}\n"
    )

    from hermes_cli.observability.shared_metrics import SharedMetricsStore
    from hermes_cli.observability.shared_metrics_send_config import (
        resolve_send_config,
    )
    from hermes_cli.observability.shared_metrics_sender import SharedMetricsSender

    # Resolve through the real config path so this exercises what a user gets.
    import yaml

    resolved = resolve_send_config(
        yaml.safe_load((scratch / "config.yaml").read_text())
    )
    if not resolved.send or resolved.endpoint != STAGING:
        print(f"FAIL: config did not resolve to staging: {resolved}")
        return 1

    store = SharedMetricsStore(
        database_path=scratch / "metrics.sqlite3",
        outbox_directory=scratch / "outbox",
    )

    today = datetime.now(timezone.utc).date().isoformat()
    real_install_id = str(uuid.uuid4())
    packages = []

    # Two packages for today's period: the "head" and a later "tail", which is
    # the real shape the outbox produces and the case the period gate exists
    # for. One is large enough to exercise gzip.
    for index, metric_count in ((0, 3), (1, 140)):
        package_id = str(uuid.uuid4())
        payload = {
            "schema_version": "hermes.shared_metrics.v2",
            "package_id": package_id,
            "install_id": real_install_id,
            "generated_at": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            "period_start": f"{today}T00:00:00Z",
            "period_end": f"{today}T23:59:59Z",
            "resource": {
                "hermes_version": "e2e-test",
                "os_family": "macos",
                "architecture": "arm64",
                "install_method": "git",
            },
            "metrics": [
                {
                    "name": f"hermes.e2e.metric.{i}",
                    "type": "counter",
                    "dimensions": {"outcome": "ok", "surface": "e2e"},
                    "value": i + 1,
                }
                for i in range(metric_count)
            ],
        }
        with store._connection() as connection:
            connection.execute(
                """
                INSERT INTO package_outbox(
                    package_id, period_start, period_end, payload_json,
                    created_at, exported_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    package_id,
                    f"{today}T00:00:00Z",
                    f"{today}T23:59:59Z",
                    json.dumps(payload),
                    f"{today}T0{index}:00:00Z",
                    f"{today}T0{index}:00:01Z",
                ),
            )
        packages.append((package_id, metric_count))

    print(f"scratch HERMES_HOME : {scratch}")
    print(f"endpoint            : {STAGING}")
    print(f"local install_id    : {real_install_id}")
    print(f"packages queued     : {len(packages)}")
    for package_id, count in packages:
        print(f"  - {package_id}  ({count} metrics)")
    print()

    outcome = SharedMetricsSender(store, resolved.endpoint).send_pending()
    print(f"outcome: sent={outcome.sent} rejected={outcome.rejected} "
          f"deferred={outcome.deferred}")
    print()

    failures = []
    with store._connection() as connection:
        rows = connection.execute(
            """
            SELECT package_id, send_state, sent_at, send_attempts,
                   sent_install_id, last_error
            FROM package_outbox ORDER BY created_at
            """
        ).fetchall()

    for row in rows:
        print(f"package        : {row[0]}")
        print(f"  send_state   : {row[1]}")
        print(f"  sent_at      : {row[2]}")
        print(f"  attempts     : {row[3]}")
        print(f"  transmitted  : {row[4]}")
        print(f"  last_error   : {row[5]}")
        if row[1] != "sent":
            failures.append(f"{row[0]} is {row[1]}: {row[5]}")
        if row[4] == real_install_id:
            failures.append(f"{row[0]} LEAKED the real install_id")
        if not row[4] or len(str(row[4])) != 64:
            failures.append(f"{row[0]} has a malformed derived id")
        print()

    if failures:
        print("FAILURES:")
        for failure in failures:
            print(f"  ✗ {failure}")
        return 1

    print("PASS: every package acknowledged 202 with a derived identifier.")
    print()
    print("Verify the objects in S3 with the package ids above:")
    print("  aws s3 ls --recursive "
          "s3://hermes-agent-telemetry-staging-767397871023-us-west-2-an/raw/ "
          "| tail -20")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
