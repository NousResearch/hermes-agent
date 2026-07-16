from gateway.stream_reconcile import should_reconcile_partial

# A partial is on screen (already_sent + message_id) and NO delivery flag is set:
# the exact flood-control duplication scenario.
BASE = dict(
    failed=False,
    transformed=False,
    streamed=False,
    previewed=False,
    content_delivered=False,
    already_sent=True,
    message_id="123",
)


def test_reconcile_when_partial_on_screen_but_delivery_unconfirmed():
    assert should_reconcile_partial("Full answer", **BASE) is True


def test_no_reconcile_when_delivery_already_confirmed():
    for flag in ("streamed", "previewed", "content_delivered"):
        kw = dict(BASE)
        kw[flag] = True
        assert should_reconcile_partial("Full answer", **kw) is False, flag


def test_no_reconcile_when_nothing_on_screen():
    kw = dict(BASE)
    kw["already_sent"] = False
    assert should_reconcile_partial("Full answer", **kw) is False
    kw = dict(BASE)
    kw["message_id"] = None
    assert should_reconcile_partial("Full answer", **kw) is False


def test_no_reconcile_for_empty_failed_or_transformed():
    assert should_reconcile_partial("", **BASE) is False
    assert should_reconcile_partial("(empty)", **BASE) is False
    kw = dict(BASE)
    kw["failed"] = True
    assert should_reconcile_partial("Full answer", **kw) is False
    kw = dict(BASE)
    kw["transformed"] = True
    assert should_reconcile_partial("Full answer", **kw) is False
