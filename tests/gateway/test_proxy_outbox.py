import time

import pytest

from gateway import proxy_outbox


@pytest.fixture(autouse=True)
def _isolated_outbox(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"gateway": {"proxy_outbox_platforms": ["matrix", "telegram"]}},
    )


def test_configured_platforms_are_fronted_without_hardcoding():
    assert proxy_outbox.fronts_platform("matrix")
    assert proxy_outbox.fronts_platform("telegram")
    assert not proxy_outbox.fronts_platform("api_server")


def test_native_send_timeout_fits_inside_lease():
    assert proxy_outbox.NATIVE_SEND_TIMEOUT_SECONDS < proxy_outbox.LEASE_SECONDS


def test_text_is_leased_only_to_a_gateway_that_owns_the_platform():
    matrix_id = proxy_outbox.enqueue(
        platform="matrix", chat_id="!room:example.org", content="matrix"
    )
    telegram_id = proxy_outbox.enqueue(
        platform="telegram", chat_id="123", content="telegram"
    )

    telegram_items = proxy_outbox.lease(platforms={"telegram"})
    assert [item["delivery_id"] for item in telegram_items] == [telegram_id]
    assert telegram_items[0]["platform"] == "telegram"
    assert proxy_outbox.acknowledge(telegram_id, attempt=1, success=True)

    matrix_items = proxy_outbox.lease(platforms={"matrix"})
    assert [item["delivery_id"] for item in matrix_items] == [matrix_id]


def test_expired_unacknowledged_lease_fails_closed_without_redelivery():
    delivery_id = proxy_outbox.enqueue(
        platform="matrix", chat_id="!room:example.org", content="once"
    )
    assert proxy_outbox.lease(platforms={"matrix"})[0]["attempt"] == 1
    with proxy_outbox._DB_LOCK, proxy_outbox._transaction() as conn:
        conn.execute(
            "UPDATE proxy_outbox SET lease_until=? WHERE delivery_id=?",
            (time.time() - 1, delivery_id),
        )

    assert proxy_outbox.delivery_result(delivery_id) == (
        False,
        "delivery outcome unknown after consumer lease expired",
    )
    assert proxy_outbox.lease(platforms={"matrix"}) == []


def test_explicit_failure_is_terminal_without_duplicate_retry():
    delivery_id = proxy_outbox.enqueue(
        platform="matrix", chat_id="!room:example.org", content="retry"
    )
    assert proxy_outbox.lease(platforms={"matrix"})[0]["attempt"] == 1
    assert proxy_outbox.acknowledge(
        delivery_id, attempt=1, success=False, error="offline"
    )

    assert proxy_outbox.delivery_result(delivery_id) == (False, "offline")
    assert proxy_outbox.lease(platforms={"matrix"}) == []


def test_stale_ack_cannot_confirm_delivery():
    delivery_id = proxy_outbox.enqueue(
        platform="matrix", chat_id="!room:example.org", content="race"
    )
    assert proxy_outbox.lease(platforms={"matrix"})[0]["attempt"] == 1

    assert not proxy_outbox.acknowledge(delivery_id, attempt=2, success=True)
    assert proxy_outbox.delivery_result(delivery_id) is None
    assert proxy_outbox.acknowledge(delivery_id, attempt=1, success=True)


def test_capacity_never_discards_active_delivery(monkeypatch):
    monkeypatch.setattr(proxy_outbox, "MAX_ITEMS", 2)
    proxy_outbox.enqueue(platform="matrix", chat_id="a", content="first")
    proxy_outbox.enqueue(platform="matrix", chat_id="b", content="second")
    with pytest.raises(RuntimeError, match="outbox is full"):
        proxy_outbox.enqueue(platform="matrix", chat_id="c", content="third")
