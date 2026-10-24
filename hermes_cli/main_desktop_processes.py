"""Read-only desktop bundle process scans; uncertainty keeps a live bundle untouched."""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def processes_running_from(tree: Path) -> Optional[list[int]]:
    """return pids running from ``tree``, or None when the process scan is uncertain."""
    try:
        import psutil
        tree = tree.resolve()
        proc_iter = psutil.process_iter(["pid", "exe", "name", "cmdline"])
    except Exception:
        logger.warning("could not initialize desktop process scan for %s", tree, exc_info=True)
        return None

    pids = []
    try:
        for proc in proc_iter:
            try:
                info = proc.info
            except getattr(psutil, "NoSuchProcess", ()):
                continue
            except Exception:
                logger.warning("could not read desktop process details for %s", tree, exc_info=True)
                return None
            try:
                pid = info.get("pid")
                if pid is None:
                    return None
                exe = info.get("exe")
                if not exe:
                    name = str(info.get("name") or "").casefold()
                    cmdline = info.get("cmdline") or []
                    tree_text = os.path.normcase(str(tree))
                    mentions_tree = any(tree_text in os.path.normcase(str(arg)) for arg in cmdline)
                    if name.startswith("hermes") or mentions_tree:
                        return None
                    if not name and not cmdline:
                        return None
                    continue
                exe_path = Path(exe).resolve()
            except Exception:
                logger.warning("could not resolve desktop process identity for %s", tree, exc_info=True)
                return None
            if exe_path == tree or tree in exe_path.parents:
                pids.append(int(pid))
    except Exception:
        logger.warning("could not finish desktop process scan for %s", tree, exc_info=True)
        return None
    return pids
