#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


async def _main() -> None:
    logging.basicConfig(
        level=os.environ.get("HERMES_SIMPLEX_SIDECAR_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    from gateway.calls.native.aiortc_engine import run_simplex_aiortc_sidecar

    await run_simplex_aiortc_sidecar()


if __name__ == "__main__":
    asyncio.run(_main())
