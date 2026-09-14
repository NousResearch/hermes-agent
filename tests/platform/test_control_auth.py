"""Who may call the control plane, over what transport.

Three audit findings meet here, and they are one piece of work rather than three: a bearer
token is worth little without TLS, TLS is worth little without identity, and identity is
worth little if the token was visible in ``ps`` the whole time.
"""

from __future__ import annotations

import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from nova.control.api import ControlAPI
from nova.control.auth import (
    LOCAL_ADMIN,
    ROLES,
    ROUTE_ROLES,
    Principal,
    PrincipalStore,
    hash_token,
    new_token,
)
from nova.control.server import build_server
from nova.errors import NovaError


def principals_file(tmp_path, *entries) -> Path:
    """A principals file from ``(name, role, token)`` triples."""
    lines = ["principals:"]
    for name, role, token in entries:
        lines += [f"  - name: {name}", f"    role: {role}",
                  f"    token_sha256: {hash_token(token)}"]
    path = tmp_path / "control-principals.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# -- tokens are never stored --------------------------------------------------


def test_the_store_holds_digests_and_never_tokens(tmp_path):
    """A leaked principals file should be an inconvenience, not an incident."""
    token = new_token()
    path = principals_file(tmp_path, ("ops", "viewer", token))
    assert token not in path.read_text(encoding="utf-8")


def test_a_pasted_token_is_refused_with_an_explanation(tmp_path):
    """Pasting the token where the digest goes is the likely mistake, and a silent 401
    loop is the worst way to learn about it."""
    path = tmp_path / "p.yaml"
    path.write_text(
        "principals:\n  - name: ops\n    role: viewer\n"
        f"    token_sha256: {new_token()}\n",
        encoding="utf-8",
    )
    with pytest.raises(NovaError, match="64-character SHA-256"):
        PrincipalStore.load(path)


def test_generated_tokens_are_unguessable_and_unique():
    tokens = {new_token() for _ in range(200)}
    assert len(tokens) == 200
    assert all(len(token) >= 40 for token in tokens)


# -- authentication -----------------------------------------------------------


def test_a_valid_token_resolves_its_principal(tmp_path):
    token = new_token()
    store = PrincipalStore.load(principals_file(tmp_path, ("ops", "viewer", token)))
    principal = store.authenticate(token)
    assert principal is not None and principal.name == "ops" and principal.role == "viewer"


@pytest.mark.parametrize("wrong", ["", "nope", "Bearer x"])
def test_an_invalid_token_resolves_to_nobody(tmp_path, wrong):
    store = PrincipalStore.load(principals_file(tmp_path, ("ops", "viewer", new_token())))
    assert store.authenticate(wrong) is None


def test_revoking_one_principal_leaves_the_others(tmp_path):
    keep, drop = new_token(), new_token()
    store = PrincipalStore.load(
        principals_file(tmp_path, ("keep", "viewer", keep), ("drop", "admin", drop))
    )
    assert store.authenticate(drop) is not None

    reduced = PrincipalStore.load(principals_file(tmp_path, ("keep", "viewer", keep)))
    assert reduced.authenticate(keep) is not None
    assert reduced.authenticate(drop) is None


@pytest.mark.parametrize(
    "entry,match",
    [
        ({"name": "", "role": "viewer"}, "no name"),
        ({"name": "x", "role": "wizard"}, "not one of"),
    ],
)
def test_a_malformed_principal_is_refused(tmp_path, entry, match):
    path = tmp_path / "p.yaml"
    path.write_text(
        "principals:\n"
        f"  - name: {entry['name']}\n    role: {entry['role']}\n"
        f"    token_sha256: {'a' * 64}\n",
        encoding="utf-8",
    )
    with pytest.raises(NovaError, match=match):
        PrincipalStore.load(path)


def test_duplicate_names_are_refused(tmp_path):
    with pytest.raises(NovaError, match="duplicate principal"):
        PrincipalStore.load(
            principals_file(tmp_path, ("ops", "viewer", new_token()), ("ops", "admin", new_token()))
        )


def test_an_absent_file_is_an_empty_store_not_an_error(tmp_path):
    assert not PrincipalStore.load(tmp_path / "absent.yaml").configured


# -- roles --------------------------------------------------------------------


def test_a_viewer_sees_operational_state_and_not_governance():
    viewer = Principal("ops", "viewer")
    assert viewer.may("/agents") and viewer.may("/tasks") and viewer.may("/health")
    assert not viewer.may("/policy")
    assert not viewer.may("/decisions")
    assert not viewer.may("/budget")


def test_an_admin_sees_everything():
    admin = Principal("sec", "admin")
    assert all(admin.may(route) for route in ROUTE_ROLES)


def test_an_undeclared_route_requires_admin():
    """A new endpoint should have to be declared viewer-readable, rather than becoming
    readable by everyone because someone forgot the route table existed."""
    assert not Principal("ops", "viewer").may("/a-route-added-later")
    assert Principal("sec", "admin").may("/a-route-added-later")


def test_every_declared_route_names_a_real_role():
    assert set(ROUTE_ROLES.values()) <= set(ROLES)


# -- bind refusals ------------------------------------------------------------


@pytest.fixture
def api(bundle, runtime):
    return ControlAPI(bundle, runtime)


def test_a_public_bind_without_principals_is_refused(api):
    with pytest.raises(NovaError, match="no principals file"):
        build_server(api, host="0.0.0.0", port=0)


def test_a_public_bind_without_tls_is_refused(api, tmp_path):
    store = PrincipalStore.load(principals_file(tmp_path, ("ops", "viewer", new_token())))
    with pytest.raises(NovaError, match="without TLS"):
        build_server(api, host="0.0.0.0", port=0, principals=store)


def test_acknowledging_a_tls_proxy_permits_a_public_bind(api, tmp_path):
    """The acknowledgement is required rather than assumed: "there is probably a load
    balancer" is exactly the assumption that ships a token in cleartext."""
    store = PrincipalStore.load(principals_file(tmp_path, ("ops", "viewer", new_token())))
    server = build_server(
        api, host="0.0.0.0", port=0, principals=store, behind_tls_proxy=True
    )
    server.server_close()


def test_a_loopback_bind_needs_neither(api):
    build_server(api, host="127.0.0.1", port=0).server_close()


# -- over a real socket -------------------------------------------------------


def serve_in_thread(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server.server_address[1]


def status(port: int, path: str, token: str = "") -> int:
    request = urllib.request.Request(f"http://127.0.0.1:{port}/platform/v1{path}")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        return urllib.request.urlopen(request, timeout=5).status
    except urllib.error.HTTPError as exc:
        return exc.code


def test_loopback_is_admin_even_with_principals_configured(api, tmp_path):
    """Someone on this host can already read the principals file, the bundle and every
    profile off disk. Demanding a token from them protects nothing and breaks the
    dashboard, which is a browser and cannot send one."""
    store = PrincipalStore.load(principals_file(tmp_path, ("ops", "viewer", new_token())))
    server = build_server(api, host="127.0.0.1", port=0, principals=store)
    try:
        port = serve_in_thread(server)
        assert status(port, "/health") == 200
        assert status(port, "/policy") == 200  # admin-only, allowed locally
    finally:
        server.shutdown()
        server.server_close()


def test_behind_a_proxy_loopback_is_not_trusted(api, tmp_path):
    """A proxy terminating TLS on this host makes every forwarded request look local.
    Trusting loopback there would mean anyone on the internet is a local admin."""
    token = new_token()
    store = PrincipalStore.load(principals_file(tmp_path, ("ops", "viewer", token)))
    server = build_server(
        api, host="127.0.0.1", port=0, principals=store, behind_tls_proxy=True
    )
    try:
        port = serve_in_thread(server)
        assert status(port, "/health") == 401
        assert status(port, "/health", token) == 200
        assert status(port, "/policy", token) == 403
    finally:
        server.shutdown()
        server.server_close()


def test_a_role_refusal_is_403_not_404(api, tmp_path):
    """The caller is authenticated and the route exists. Pretending otherwise makes a
    permissions problem look like a bug."""
    token = new_token()
    store = PrincipalStore.load(principals_file(tmp_path, ("ops", "viewer", token)))
    server = build_server(
        api, host="127.0.0.1", port=0, principals=store, behind_tls_proxy=True
    )
    try:
        port = serve_in_thread(server)
        assert status(port, "/decisions", token) == 403
    finally:
        server.shutdown()
        server.server_close()


def test_a_read_route_is_not_writable_by_any_principal(api, tmp_path):
    """Phase 8 opened POST, but only onto declared write routes. An admin posting to a
    read route gets nowhere — which is the point of keeping the two tables apart.

    The example used to be ``/agents``, which is now a declared write route (creating an
    agent, the way ``POST /automations`` declares one). ``/policy`` replaces it: the
    compiled policy is readable by an admin and is not writable by anyone, because it is
    compiled from the bundle rather than edited in place.
    """
    store = PrincipalStore.load(principals_file(tmp_path, ("ops", "admin", new_token())))
    server = build_server(api, host="127.0.0.1", port=0, principals=store)
    try:
        port = serve_in_thread(server)
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/platform/v1/policy",
            method="POST",
            data=b"{}",
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=5)
        assert caught.value.code == 404
    finally:
        server.shutdown()
        server.server_close()


# -- TLS ----------------------------------------------------------------------


def test_a_bad_certificate_fails_at_construction_not_at_first_request(api, tmp_path):
    store = PrincipalStore.load(principals_file(tmp_path, ("ops", "viewer", new_token())))
    with pytest.raises(NovaError, match="TLS certificate"):
        build_server(
            api, host="127.0.0.1", port=0, principals=store,
            tls_certfile=str(tmp_path / "absent.pem"),
        )


def test_the_local_admin_is_named_rather_than_anonymous():
    """So an access log distinguishes "nobody configured auth" from "someone
    authenticated"."""
    assert LOCAL_ADMIN.role == "admin"
    assert LOCAL_ADMIN.via == "loopback"
