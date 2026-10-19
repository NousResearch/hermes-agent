"""Secret-link HTTP routes against a real aiohttp client: form, submit, conflicts, validation,
token states, CSRF/origin checks, limits and response headers."""

import asyncio
import os
import stat
from urllib.parse import urlencode

import pytest

from gateway.secret_link_store import LINK_TTL_SECONDS
from tests.gateway.secret_link_helpers import (
    BASE, SENTINEL, assert_secret_headers, clock_fixture, env_text, form_nonce, get_form, hermes_root_fixture,
    link_path, make_profile, new_link, restarts_fixture, secret_client, store_fixture, submit)

pytestmark = pytest.mark.asyncio


def _profile(root, name):
    return root if name == "default" else root / "profiles" / name


async def test_get_shows_the_form_and_does_not_consume_the_token(hermes_root, store):
    token = new_link(store, suggested="STRIPE_SECRET_KEY")
    async with secret_client() as client:
        resp, html, nonce = await get_form(client, token)
        assert resp.status == 200
        assert_secret_headers(resp)
        assert 'name="key"' in html and 'value="STRIPE_SECRET_KEY"' in html
        assert 'type="password"' in html and 'name="secret"' in html
        for attr in ('autocomplete="off"', 'autocapitalize="off"', 'spellcheck="false"'):
            assert attr in html
        assert 'name="all_profiles"' in html and "All profiles" in html
        for name in ("default", "alpha", "beta"):
            assert f'name="profile" value="{name}"' in html
        assert "<script" not in html.lower() and "http://" not in html and "src=" not in html
        assert nonce
        again, _html, _nonce = await get_form(client, token)
        assert again.status == 200


async def test_valid_submit_writes_only_the_selected_profile(hermes_root, store, restarts):
    alpha, beta = _profile(hermes_root, "alpha"), _profile(hermes_root, "beta")
    (alpha / ".env").write_text("# keep\nA=1\n")
    os.chmod(alpha / ".env", 0o600)
    token = new_link(store)
    async with secret_client() as client:
        _resp, _html, nonce = await get_form(client, token)
        resp, html = await submit(client, token, nonce, profiles=("alpha",))
    assert resp.status == 200
    assert_secret_headers(resp)
    assert env_text(alpha) == f"# keep\nA=1\nSTRIPE_SECRET_KEY={SENTINEL}\n"
    assert stat.S_IMODE((alpha / ".env").stat().st_mode) == 0o600
    assert env_text(beta) == "" and env_text(hermes_root) == ""
    assert SENTINEL not in html and "STRIPE_SECRET_KEY" in html and "alpha" in html


async def test_all_profiles_includes_a_profile_created_after_the_get(hermes_root, store, restarts):
    token = new_link(store)
    async with secret_client() as client:
        _resp, _html, nonce = await get_form(client, token)
        make_profile(hermes_root, "gamma")
        resp, _ = await submit(client, token, nonce, profiles=(), all_profiles=True)
    assert resp.status == 200
    for name in ("default", "alpha", "beta", "gamma"):
        assert f"STRIPE_SECRET_KEY={SENTINEL}" in env_text(_profile(hermes_root, name))


async def test_used_unknown_malformed_and_expired_tokens_are_identical_404s(hermes_root, store, restarts, clock):
    used = new_link(store)
    expired = new_link(store)
    async with secret_client() as client:
        _r, _h, nonce = await get_form(client, used)
        ok, _ = await submit(client, used, nonce)
        assert ok.status == 200
        before = env_text(_profile(hermes_root, "alpha"))
        second, second_body = await submit(client, used, nonce, key="OTHER_KEY")
        assert env_text(_profile(hermes_root, "alpha")) == before

        _r, _h, exp_nonce = await get_form(client, expired)
        clock.now += LINK_TTL_SECONDS + 1
        late, late_body = await submit(client, expired, exp_nonce, key="LATE_KEY")
        assert "LATE_KEY" not in env_text(_profile(hermes_root, "alpha"))
        assert all(record.state != "open" for record in store._records)

        unknown, unknown_body = await submit(client, "A" * 43, nonce)
        malformed, malformed_body = await submit(client, "bad", nonce)
        get_unknown = await client.get(link_path("A" * 43))
        get_malformed = await client.get(link_path("x/y"))
        bodies = {second_body, late_body, unknown_body, malformed_body,
                  await get_unknown.text(), await get_malformed.text()}
    assert {r.status for r in (second, late, unknown, malformed, get_unknown, get_malformed)} == {404}
    assert len(bodies) == 1
    for resp in (second, late, unknown, malformed, get_unknown, get_malformed):
        assert_secret_headers(resp)


async def test_expired_get_removes_the_record(hermes_root, store, clock):
    token = new_link(store)
    clock.now += LINK_TTL_SECONDS + 1
    async with secret_client() as client:
        resp = await client.get(link_path(token))
    assert resp.status == 404
    assert store._records == []


async def test_conflict_names_profiles_hides_values_and_keeps_the_token(hermes_root, store, restarts):
    alpha = _profile(hermes_root, "alpha")
    (alpha / ".env").write_text("STRIPE_SECRET_KEY=existing-old-value\n")
    token = new_link(store)
    async with secret_client() as client:
        _r, _h, nonce = await get_form(client, token)
        resp, html = await submit(client, token, nonce, profiles=("alpha", "beta"))
        assert resp.status == 409
        assert_secret_headers(resp)
        assert "alpha" in html and 'name="replace"' in html
        assert SENTINEL not in html and "existing-old-value" not in html
        assert 'name="secret" value=""' in html or 'name="secret" value' not in html
        assert env_text(alpha) == "STRIPE_SECRET_KEY=existing-old-value\n"
        assert env_text(_profile(hermes_root, "beta")) == ""
        assert form_nonce(html) and form_nonce(html) != nonce


async def test_a_new_key_name_after_a_conflict_writes_it_and_keeps_the_old_key(hermes_root, store, restarts):
    """AC-11: the other way out of a conflict is a different key name; the old key stays as it was."""
    alpha = _profile(hermes_root, "alpha")
    (alpha / ".env").write_text("# keep\nSTRIPE_SECRET_KEY=existing-old-value\nOTHER=1\n")
    token = new_link(store)
    async with secret_client() as client:
        _r, _h, nonce = await get_form(client, token)
        conflict, html = await submit(client, token, nonce, profiles=("alpha",))
        renamed, _ = await submit(client, token, form_nonce(html), key="STRIPE_SECRET_KEY_2", profiles=("alpha",))
    assert conflict.status == 409
    assert renamed.status == 200
    assert env_text(alpha) == ("# keep\nSTRIPE_SECRET_KEY=existing-old-value\nOTHER=1\n"
                               f"STRIPE_SECRET_KEY_2={SENTINEL}\n")


async def test_replace_overwrites_after_a_conflict(hermes_root, store, restarts):
    alpha = _profile(hermes_root, "alpha")
    (alpha / ".env").write_text("STRIPE_SECRET_KEY=old\n")
    token = new_link(store)
    async with secret_client() as client:
        _r, _h, nonce = await get_form(client, token)
        _resp, html = await submit(client, token, nonce)
        resp, _ = await submit(client, token, form_nonce(html), replace=True)
    assert resp.status == 200
    assert env_text(alpha) == f"STRIPE_SECRET_KEY={SENTINEL}\n"


@pytest.mark.parametrize("key", ["PATH", "LD_PRELOAD", "1BAD", "A-B", ""])
async def test_refused_key_names_return_400_and_write_nothing(hermes_root, store, key):
    token = new_link(store)
    async with secret_client() as client:
        _r, _h, nonce = await get_form(client, token)
        resp, html = await submit(client, token, nonce, key=key, profiles=("alpha",))
    assert resp.status == 400
    assert SENTINEL not in html
    assert env_text(_profile(hermes_root, "alpha")) == ""


async def test_administrator_pinned_key_returns_400(hermes_root, store, tmp_path, monkeypatch):
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / ".env").write_text("PINNED_KEY=admin\n")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    token = new_link(store)
    async with secret_client() as client:
        _r, _h, nonce = await get_form(client, token)
        resp, _ = await submit(client, token, nonce, key="PINNED_KEY")
    assert resp.status == 400
    assert env_text(_profile(hermes_root, "alpha")) == ""


@pytest.mark.parametrize("profile", ["../x", "a/b", "default/..", "nosuch"])
async def test_bad_profile_names_return_400_and_write_nothing(hermes_root, store, profile):
    token = new_link(store)
    async with secret_client() as client:
        _r, _h, nonce = await get_form(client, token)
        resp, _ = await submit(client, token, nonce, profiles=("alpha", profile))
    assert resp.status == 400
    for name in ("default", "alpha", "beta"):
        assert env_text(_profile(hermes_root, name)) == ""


@pytest.mark.platforms("posix")
async def test_symlinked_env_outside_the_profile_returns_400(hermes_root, store, tmp_path):
    outside = tmp_path / "outside.env"
    outside.write_text("A=1\n")
    (_profile(hermes_root, "beta") / ".env").symlink_to(outside)
    token = new_link(store)
    async with secret_client() as client:
        _r, _h, nonce = await get_form(client, token)
        resp, _ = await submit(client, token, nonce, profiles=("alpha", "beta"))
    assert resp.status == 400
    assert outside.read_text() == "A=1\n"
    assert env_text(_profile(hermes_root, "alpha")) == ""


async def test_no_profile_selected_returns_400(hermes_root, store):
    token = new_link(store)
    async with secret_client() as client:
        _r, _h, nonce = await get_form(client, token)
        resp, _ = await submit(client, token, nonce, profiles=())
    assert resp.status == 400


@pytest.mark.parametrize("headers", [
    {"Origin": "https://evil.example.com"},
    {"Origin": None},
    {"Origin": None, "Referer": "https://evil.example.com/secret-link/x"},
    {"Origin": "null"},
])
async def test_wrong_or_missing_origin_returns_403(hermes_root, store, headers):
    token = new_link(store)
    async with secret_client() as client:
        _r, _h, nonce = await get_form(client, token)
        resp, _ = await submit(client, token, nonce, headers=headers)
    assert resp.status == 403
    assert_secret_headers(resp)
    assert env_text(_profile(hermes_root, "alpha")) == ""


async def test_matching_referer_is_accepted_when_origin_is_absent(hermes_root, store, restarts):
    token = new_link(store)
    async with secret_client() as client:
        _r, _h, nonce = await get_form(client, token)
        resp, _ = await submit(client, token, nonce, headers={"Origin": None, "Referer": f"{BASE}/secret-link/{token}"})
    assert resp.status == 200


async def test_wrong_or_stale_nonce_returns_403(hermes_root, store):
    token = new_link(store)
    async with secret_client() as client:
        _r, _h, stale = await get_form(client, token)
        await get_form(client, token)  # rotates the nonce
        resp, _ = await submit(client, token, stale)
        wrong, _ = await submit(client, token, "x" * 43)
    assert resp.status == 403 and wrong.status == 403
    assert env_text(_profile(hermes_root, "alpha")) == ""


async def test_wrong_content_type_and_oversized_body_return_400(hermes_root, store):
    token = new_link(store)
    async with secret_client() as client:
        await get_form(client, token)
        json_resp = await client.post(link_path(token), json={"secret": SENTINEL},
                                      headers={"Origin": BASE})
        big = await client.post(link_path(token), data=b"a=" + b"b" * (16 * 1024),
                                headers={"Origin": BASE, "Content-Type": "application/x-www-form-urlencoded"})
    assert json_resp.status == 400 and big.status == 400
    assert_secret_headers(big)


async def _chunked(body: bytes, size: int):
    for start in range(0, len(body), size):
        yield body[start:start + size]
        await asyncio.sleep(0)  # let the server see each chunk on its own


async def test_chunked_form_body_is_read_in_full(hermes_root, store, restarts):
    token = new_link(store)
    async with secret_client() as client:
        _r, _h, nonce = await get_form(client, token)
        body = urlencode([("nonce", nonce), ("key", "STRIPE_SECRET_KEY"), ("secret", SENTINEL),
                          ("profile", "alpha")]).encode("ascii")
        resp = await client.post(link_path(token), data=_chunked(body, 7), headers={
            "Origin": BASE, "Content-Type": "application/x-www-form-urlencoded"})
    assert resp.status == 200
    assert env_text(_profile(hermes_root, "alpha")) == f"STRIPE_SECRET_KEY={SENTINEL}\n"


async def test_chunked_body_over_the_cap_returns_400(hermes_root, store):
    token = new_link(store)
    async with secret_client() as client:
        await get_form(client, token)
        resp = await client.post(link_path(token), data=_chunked(b"a=" + b"b" * (16 * 1024), 1024), headers={
            "Origin": BASE, "Content-Type": "application/x-www-form-urlencoded"})
    assert resp.status == 400
    assert_secret_headers(resp)


@pytest.mark.parametrize("value", ["", "a\nb", "a\rb", "a\0b", "x" * 8193], ids=["empty", "lf", "cr", "nul", "long"])
async def test_empty_long_or_line_breaking_values_return_400(hermes_root, store, value):
    token = new_link(store)
    async with secret_client() as client:
        _r, _h, nonce = await get_form(client, token)
        resp, html = await submit(client, token, nonce, value=value)
    assert resp.status == 400
    assert env_text(_profile(hermes_root, "alpha")) == ""
    if value:
        assert value not in html


async def test_value_of_exactly_8192_bytes_is_accepted(hermes_root, store, restarts):
    token = new_link(store)
    async with secret_client() as client:
        _r, _h, nonce = await get_form(client, token)
        resp, _ = await submit(client, token, nonce, value="v" * 8192)
    assert resp.status == 200


async def test_eleventh_request_in_a_minute_returns_429(hermes_root, store):
    async with secret_client() as client:
        statuses = [(await client.get(link_path("A" * 43))).status for _ in range(10)]
        limited = await client.get(link_path("A" * 43))
    assert statuses == [404] * 10
    assert limited.status == 429
    assert_secret_headers(limited)


async def test_sixth_failed_submit_finds_the_token_expired(hermes_root, store, clock):
    token = new_link(store)
    async with secret_client() as client:
        _r, _h, nonce = await get_form(client, token)
        statuses = []
        for _ in range(5):
            clock.now += 13  # stay under the per-client limit
            resp, html = await submit(client, token, nonce, key="1BAD")
            statuses.append(resp.status)
            nonce = form_nonce(html) or nonce
        clock.now += 13
        sixth, _ = await submit(client, token, nonce)
    assert statuses == [400] * 5
    assert sixth.status == 404
    assert env_text(_profile(hermes_root, "alpha")) == ""


async def test_two_parallel_submits_write_exactly_once(hermes_root, store, restarts):
    token = new_link(store)
    async with secret_client() as client:
        _r, _h, nonce = await get_form(client, token)
        first, second = await asyncio.gather(
            submit(client, token, nonce, value="first-value"),
            submit(client, token, nonce, value="second-value"))
    assert sorted([first[0].status, second[0].status]) == [200, 404]
    text = env_text(_profile(hermes_root, "alpha"))
    assert text.count("STRIPE_SECRET_KEY=") == 1


async def test_internal_error_returns_500_with_the_headers(hermes_root, store, monkeypatch):
    import gateway.secret_link_routes as routes

    def boom(*_a, **_k):
        raise RuntimeError(SENTINEL)

    monkeypatch.setattr(routes, "render_form", boom)
    token = new_link(store)
    async with secret_client() as client:
        resp = await client.get(link_path(token))
        body = await resp.text()
    assert resp.status == 500
    assert SENTINEL not in body
    assert_secret_headers(resp)


@pytest.mark.parametrize("method", ["PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def test_unsupported_methods_return_405_with_the_headers(hermes_root, store, method):
    token = new_link(store)
    async with secret_client() as client:
        resp = await client.request(method, link_path(token))
    assert resp.status == 405
    assert_secret_headers(resp)
