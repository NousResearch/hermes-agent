"""``FooteredReply`` must survive being copied, hashed-free and pickled.

Publishing a finished reply on a session's event stream deep-copies it. ``FooteredReply``
subclasses ``str`` but its ``__new__`` takes ``footer`` positionally, so str's default
``__reduce__`` could not rebuild an instance: the copy raised

    TypeError: FooteredReply.__new__() missing 1 required positional argument: 'footer'

and the finished reply was never delivered to the chat. Copying is identity for an
immutable value, and pickling must round-trip both the text and the footer.
"""

import copy
import pickle

from gateway.runtime_footer import FooteredReply

FOOTER = "model-x · 29%"


def test_shallow_copy_is_identity():
    reply = FooteredReply("hello", FOOTER)
    assert copy.copy(reply) is reply


def test_deepcopy_does_not_raise():
    reply = FooteredReply("hello", FOOTER)
    duplicate = copy.deepcopy(reply)          # raised TypeError before the fix
    assert str(duplicate) == str(reply)
    assert duplicate.runtime_footer == FOOTER


def test_pickle_round_trip_preserves_text_and_footer():
    reply = FooteredReply("hello", FOOTER)
    restored = pickle.loads(pickle.dumps(reply))
    assert type(restored) is FooteredReply
    assert str(restored) == str(reply)
    assert str(restored).endswith("\n\n" + FOOTER)
    assert restored.runtime_footer == FOOTER


def test_empty_footer_adds_no_separator_and_is_stable():
    """The call site guards against an empty footer, but a round-trip must not grow the text."""
    reply = FooteredReply("no footer", "")
    assert str(reply) == "no footer"
    once = pickle.loads(pickle.dumps(reply))
    twice = pickle.loads(pickle.dumps(once))
    assert str(once) == str(reply) == str(twice)
