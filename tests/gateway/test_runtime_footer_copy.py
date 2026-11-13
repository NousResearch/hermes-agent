"""``FooteredReply`` is a str subclass whose ``__new__`` needs a footer, so the default
copy/pickle reconstruction (``cls.__new__(cls)``) raised TypeError. The gateway deepcopies
every event payload in ``gateway.session_events.publish``, which turned that into a failed
``message.complete`` publication and a viewer stuck on "working"."""
import pickle
from copy import copy, deepcopy

from gateway.runtime_footer import FooteredReply

FOOTER = "gpt-5.4 · 68% · ~/projects/hermes"


def _reply():
    return FooteredReply("Here it is", FOOTER)


def test_deepcopy_of_a_payload_holding_the_reply_keeps_the_footer():
    reply = _reply()
    frame = deepcopy({"text": reply, "content": reply})
    assert frame["text"] == f"Here it is\n\n{FOOTER}"
    assert frame["text"].runtime_footer == FOOTER


def test_copy_returns_the_immutable_reply_itself():
    reply = _reply()
    assert copy(reply) is reply
    assert deepcopy(reply) is reply


def test_pickle_roundtrip_keeps_the_footer_once():
    reply = pickle.loads(pickle.dumps(_reply()))
    assert isinstance(reply, FooteredReply)
    assert str(reply) == f"Here it is\n\n{FOOTER}"
    assert reply.runtime_footer == FOOTER
