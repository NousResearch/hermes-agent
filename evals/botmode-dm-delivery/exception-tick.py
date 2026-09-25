"""Controlled filesystem fault; run only against the native disposable sandbox."""
import json
import os
from pathlib import Path

from cron import scheduler_delivery as delivery
from cron.bot_chat_delivery import _root
from cron.scheduler import tick

root = _root()
original_which = delivery.shutil.which
original_is_dir = Path.is_dir
armed = False
raised = 0


def resolve_cli(*args, **kwargs):
    global armed
    armed = raised == 0
    return original_which(*args, **kwargs)


def is_dir(path):
    global armed, raised
    if armed and path == Path(os.environ["HERMES_HOME"]) / "profiles" / "beta":
        armed = False
        raised += 1
        raise PermissionError("controlled target traversal denied after discovery")
    return original_is_dir(path)


delivery.shutil.which = resolve_cli
Path.is_dir = is_dir
try:
    tick(verbose=False)
    tick(verbose=False)
finally:
    Path.is_dir = original_is_dir
    delivery.shutil.which = original_which
    records = [json.loads(p.read_text(encoding="utf-8")) for p in root.glob("*.json") if p.name != "broken.json"]
    Path(os.environ["BOT_DM_EXCEPTION_RECEIPT"]).write_text(json.dumps({"raised": raised, "records": records}, indent=2), encoding="utf-8")
