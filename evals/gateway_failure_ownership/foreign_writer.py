"""Independent durable writer used only by the loopback ownership probe."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
from hermes_state import SessionDB

db = SessionDB(db_path=Path(sys.argv[2]))
db.append_message(
    sys.argv[3], "user", "foreign independent input", observed=False,
    display_metadata={"gateway_input_owner": "independent-process-owner"},
)
db.close()
