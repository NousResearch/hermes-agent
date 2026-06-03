# Intentional source orphan (Python sidecar not wired to TS/JS main app)
# Models shape of mixed-lang repo with orphans per UA-P5-003
import json
import os

def legacy_task_dump():
    # Would have been called from old node bridge, now orphaned
    return {"tasks": []}

if __name__ == "__main__":
    print(json.dumps(legacy_task_dump()))
