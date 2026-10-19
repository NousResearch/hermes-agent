"""After a use: Telegram confirmation, partial-failure report, restart choice and order, and
the sentinel value / raw token never leaking anywhere observable."""

import logging

import pytest

from gateway.secret_link_store import LINK_TTL_SECONDS
from tests.gateway.secret_link_helpers import (
    SENTINEL, Sent, clock_fixture, env_text, form_nonce, get_form, hermes_root_fixture, link_path,
    make_profile, new_link, restarts_fixture, secret_client, store_fixture, submit)
from tests.waiting import async_wait_until

pytestmark = pytest.mark.asyncio


async def test_success_sends_only_the_fixed_word_and_key(hermes_root, store, restarts):
    sent = Sent()
    token = new_link(store, notify=sent)
    async with secret_client() as client:
        _r, _h, nonce = await get_form(client, token)
        resp, _ = await submit(client, token, nonce, profiles=("alpha", "beta"))
    assert resp.status == 200
    await async_wait_until(lambda: sent.texts)
    assert sent.texts == ["Saved STRIPE_SECRET_KEY."]


async def test_partial_failure_reports_by_name_uses_the_token_and_restarts_only_written(hermes_root, store, restarts):
    make_profile(hermes_root, "gamma")
    (hermes_root / "profiles" / "beta" / ".env").mkdir()  # a write here fails for real
    restarts.running = {"alpha", "beta", "gamma"}
    sent = Sent()
    token = new_link(store, notify=sent)
    record = store._records[0]
    async with secret_client() as client:
        _r, _h, nonce = await get_form(client, token)
        resp, html = await submit(client, token, nonce, profiles=("gamma", "beta", "alpha"), replace=True)
    assert resp.status == 500
    assert f"STRIPE_SECRET_KEY={SENTINEL}" in env_text(hermes_root / "profiles" / "alpha")
    assert env_text(hermes_root / "profiles" / "gamma") == ""
    assert "Written: alpha." in html and "Failed: beta." in html and "Not tried: gamma." in html
    await async_wait_until(lambda: sent.texts and restarts)
    assert sent.texts == ["STRIPE_SECRET_KEY was not saved to all profiles. Written: alpha. Failed: beta. "
                          "Not tried: gamma."]
    assert record.state == "used"
    assert restarts == ["alpha"]


async def test_restart_only_affected_profiles_whose_gateway_runs(hermes_root, store, restarts):
    restarts.running = {"alpha", "beta"}
    sent = Sent()
    token = new_link(store, notify=sent)
    async with secret_client() as client:
        _r, _h, nonce = await get_form(client, token)
        resp, html = await submit(client, token, nonce, profiles=("alpha",))
    assert resp.status == 200 and "alpha" in html
    await async_wait_until(lambda: restarts)
    assert restarts == ["alpha"]


async def test_no_running_gateway_means_no_restart(hermes_root, store, restarts):
    sent = Sent()
    token = new_link(store, notify=sent)
    async with secret_client() as client:
        _r, _h, nonce = await get_form(client, token)
        await submit(client, token, nonce, profiles=("alpha", "beta"))
    await async_wait_until(lambda: sent.texts)
    assert restarts == []


async def test_confirmation_goes_out_before_any_restart_and_the_owner_restarts_last(hermes_root, store, restarts):
    events = []
    restarts.running = {"alpha", "default"}

    async def notify(text):
        events.append(("send", text))

    import hermes_cli.web_server_gateway as wsg
    real = wsg._restart_gateway_after

    def ordered_restart(profile, *, what, label):
        events.append(("restart", profile))
        return real(profile, what=what, label=label)

    wsg._restart_gateway_after = ordered_restart
    try:
        token = new_link(store, notify=notify, owner_home=hermes_root / "profiles" / "alpha")
        async with secret_client() as client:
            _r, _h, nonce = await get_form(client, token)
            await submit(client, token, nonce, profiles=("alpha", "default"))
        await async_wait_until(lambda: len(events) == 3)
    finally:
        wsg._restart_gateway_after = real
    assert events == [("send", "Saved STRIPE_SECRET_KEY."), ("restart", "default"), ("restart", "alpha")]


async def test_sentinel_and_raw_token_never_leak(hermes_root, store, restarts, clock, caplog):
    caplog.set_level(logging.DEBUG)
    restarts.running = {"alpha"}
    sent = Sent()
    seen: list[str] = []
    tokens: list[str] = []

    async def record(resp_and_body):
        resp, body = resp_and_body
        seen.append(body)
        seen.append(repr(dict(resp.headers)))

    (hermes_root / "profiles" / "beta" / ".env").write_text("STRIPE_SECRET_KEY=old\n")
    async with secret_client() as client:
        # success
        tokens.append(new_link(store, notify=sent))
        _r, _h, nonce = await get_form(client, tokens[-1])
        await record(await submit(client, tokens[-1], nonce, profiles=("alpha",)))
        # conflict, then validation errors
        tokens.append(new_link(store, notify=sent))
        _r, _h, nonce = await get_form(client, tokens[-1])
        resp_body = await submit(client, tokens[-1], nonce, profiles=("beta",))
        await record(resp_body)
        nonce = form_nonce(resp_body[1])
        resp_body = await submit(client, tokens[-1], nonce, key="1BAD", profiles=("beta",))
        await record(resp_body)
        await record(await submit(client, tokens[-1], form_nonce(resp_body[1]), value=SENTINEL + "\n"))
        clock.now += 61  # fresh rate window
        # write error (nothing written) and partial failure
        (make_profile(hermes_root, "gamma") / ".env").mkdir()
        tokens.append(new_link(store, notify=sent))
        _r, _h, nonce = await get_form(client, tokens[-1])
        resp_body = await submit(client, tokens[-1], nonce, profiles=("gamma",), replace=True)
        await record(resp_body)
        await record(await submit(client, tokens[-1], form_nonce(resp_body[1]), profiles=("alpha", "gamma"),
                                  replace=True))
        # expired
        tokens.append(new_link(store, notify=sent))
        _r, _h, nonce = await get_form(client, tokens[-1])
        clock.now += LINK_TTL_SECONDS + 1
        await record(await submit(client, tokens[-1], nonce))
        await async_wait_until(lambda: len(sent.texts) >= 2)
    observed = "\n".join(seen + sent.texts + [caplog.text] + [repr(r.__dict__) for r in caplog.records])
    assert SENTINEL not in observed
    for token in tokens:
        assert token not in caplog.text
    for path in hermes_root.rglob("*"):
        if path.is_file() and path.name != ".env":
            assert SENTINEL.encode() not in path.read_bytes(), path
