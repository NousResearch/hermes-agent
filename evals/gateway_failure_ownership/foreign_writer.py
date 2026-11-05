"""Independent durable writer used only by the loopback ownership probe."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
from persistence.conversation.messages import NewMessage
from persistence_assembly import open_persistence, profile_scope

store = open_persistence(profile_scope(home=Path(sys.argv[2]).parent), eager=True)
store.messages.append(sys.argv[3], NewMessage(
    role="user", content="foreign independent input", observed=False,
    display_metadata={"gateway_input_owner": "independent-process-owner"},
))
store.close()
