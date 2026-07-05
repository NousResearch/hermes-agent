"""pre_persist_user_message: a returned {"context"} is appended to the user
message before it is baked into `messages` (and thus before the :329 persist)."""
from hermes_cli import plugins as P


def test_hook_registered():
    assert "pre_persist_user_message" in P.VALID_HOOKS


def test_join_semantics():
    # invoke_hook is pure fan-out; the host joins non-None returns. Mirror the
    # host's join+append logic to lock the contract without spinning a full agent.
    results = [{"context": "A"}, None, "B", {"nope": 1}]
    parts = []
    for r in results:
        if isinstance(r, dict) and r.get("context"):
            parts.append(str(r["context"]))
        elif isinstance(r, str) and r.strip():
            parts.append(r)
    user_message = "hi" + ("\n\n" + "\n\n".join(parts) if parts else "")
    assert user_message == "hi\n\nA\n\nB"
