"""The lock that serializes a config.yaml read-modify-write SPAN across writers on worker threads.

``config._CONFIG_LOCK`` covers each load/save call, not the span between them. The dashboard's
``_CONFIG_MUTATION_LOCK`` is this lock, so profile provisioning and dashboard handlers never drop
each other's writes.
"""

from __future__ import annotations

import threading

CONFIG_MUTATION_LOCK = threading.RLock()
