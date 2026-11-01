"""Release a closed conversation's entries from the per-message token-estimate memo.

``agent.model_metadata._MSG_TOKENS_CACHE`` keys hold every string they fingerprint, so a long-lived process
kept each closed session's message bodies resident until 4096 newer messages pushed them out. Two release
paths cover it:

* by owner: every entry records the agent whose turn inserted it (``current_token_memo_owner``), so
  ``close()`` drops the request-time strings no history holds (the ephemeral system-prompt concat, prefill
  rows, prompt-caching and skill-split slices, MoA advisor folds);
* by value: ``forget_message_token_memo`` drops entries holding the long strings of given messages, for
  entries inserted outside a turn and for the bodies compression replaced.
"""

from __future__ import annotations

from typing import Any


def current_token_memo_owner() -> int | None:
    """The owner new memo entries record: the agent whose turn runs in this context (``AIAgent.run_conversation``
    binds it via ``bind_subagent_parent``), if any. An id reused by a later agent only widens what that agent's
    close() evicts, never what an estimate returns."""
    from agent.subagent_lifecycle import get_active_subagent_parent

    agent = get_active_subagent_parent()
    return None if agent is None else id(agent)


def _long_strings(value: Any) -> set[str]:
    found: set[str] = set()
    stack = [value]
    while stack:
        value = stack.pop()
        if type(value) is dict:
            stack.extend(value.values())
        elif type(value) is list or type(value) is tuple:
            stack.extend(value)
        elif type(value) is str and len(value) > 64:  # short values ("assistant", tool names) are shared
            found.add(value)
    return found


def _key_holds(key: tuple, owned: set[str]) -> bool:
    """True when fingerprint ``key`` (nested tuples, string leaves) holds a string in ``owned``."""
    stack = [key]
    while stack:
        value = stack.pop()
        if type(value) is tuple:
            stack.extend(value)
        elif type(value) is str and value in owned:
            return True
    return False


def forget_message_token_memo(messages: Any, *, keep: Any = None, owner: Any = None) -> int:
    """Evict memo entries holding any long string of ``messages`` that ``keep`` does not also hold, plus
    every entry ``owner``'s turns inserted; return how many.

    Eviction never changes an estimate: a later lookup recomputes the same value.
    """
    from agent.model_metadata import _MSG_TOKENS_CACHE

    owned = _long_strings(messages)
    if owned and keep is not None:
        owned -= _long_strings(keep)
    owner_id = None if owner is None else id(owner)
    if not owned and owner_id is None:
        return 0
    # Fingerprint keys are nested tuples whose string leaves are the message's own values.
    stale = [key for key, (_text, _images, entry_owner) in list(_MSG_TOKENS_CACHE.items())
             if (owner_id is not None and entry_owner == owner_id)
             or (owned and _key_holds(key, owned))]
    for key in stale:
        _MSG_TOKENS_CACHE.pop(key, None)
    return len(stale)
