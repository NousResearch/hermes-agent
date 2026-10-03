"""After a group moves, its copies are retired by the host they follow, never by the one they left.

Real stores and room identity keys per installation (``test_hosted_room_custody_lineage``): the
participant's copy follows a verified move, its operator's enrollment stays bound to the first home,
and the successor inherits the obligation.
"""

import sqlite3
from contextlib import closing
from types import SimpleNamespace

import pytest

from gateway import hosted_room_identity as identity
from gateway import hosted_room_replica_retirement as retirement
from gateway import hosted_room_replication as publisher
from gateway import hosted_rooms as rooms
from gateway.hosted_rooms_common import open_sqlite
from tests.gateway.fixtures.passive_copy import member
from tests.gateway.test_hosted_room_custody_lineage import (  # noqa: F401
    ROOM, acting, configure, copy_to, enroll, message, move, net, verify)

SECRET = b"first-home-retirement-secret-of-32-bytes"


@pytest.fixture
def moved(net):
    """home hosts; the participant enrolled its copy's retirement; second continues the group at epoch 2."""
    home, second, participant = net["home"], net["second"], net["third"]
    members = [{"member_id": "writer", "handle": "writer", "profile": "default",
                "target": {"kind": "local", "profile": "default"}}, member("reviewer", target=participant.install_id)]
    with acting(home):
        rooms.create_room(home.db, room_id=ROOM, name="Workshop", members=members, authority_gateway_id=home.install_id)
    message(home, "m1", 1)
    enroll(home, second)
    enroll(home, participant, successor=False)
    configure(home)
    with acting(home):
        entry = retirement.prepare_home_enrollment(
            home.db, room_id=ROOM, target_install_id=participant.install_id, endpoint=participant.endpoint,
            local_gateway_id=home.install_id, secret=SECRET)
    copy_to(home, participant)
    with acting(participant):
        retirement.enroll_target(participant.db, enrollment=entry, target_install_id=participant.install_id)
    copy_to(home, second)
    move(second, home, from_epoch=1, to_epoch=2)
    message(second, "m2", 2)
    copy_to(second, participant, verify=verify)  # the copy follows the verified move
    return net, entry


def report(participant, authority):
    with acting(participant):
        return retirement.current_target_enrollment(participant.db, room_id=ROOM, authority_gateway_id=authority[0],
                                                    authority_epoch=authority[1])


def signed(signer, body):
    with acting(signer):
        return identity.sign(retirement.AUTHORITY_NOTICE_DOMAIN, body)


def retire(participant, payload, value):
    with acting(participant):
        return retirement.retire_copy(participant.db, payload=payload, value=value,
                                      local_gateway_id=participant.install_id)


def test_the_host_a_copy_follows_retires_it_after_a_move(moved):
    net, entry = moved
    second, participant = net["second"], net["third"]
    # The participant tells the copy's verified authority about its enrollment, and nobody else.
    reported = report(participant, (second.install_id, 2))
    assert reported == {**entry, "state": "active"}
    assert report(participant, (second.install_id, 3)) is None
    assert report(participant, (net["fresh"].install_id, 2)) is None
    with acting(second):
        inherited = retirement.inherit_home_enrollment(
            second.db, enrollment=reported, endpoint=participant.endpoint + "/p/default",
            local_gateway_id=second.install_id, proof_grant="grant-that-authenticated-the-probe")
        assert (inherited["authority_gateway_id"], inherited["authority_epoch"]) == (second.install_id, 2)
        assert retirement.inherit_home_enrollment(
            second.db, enrollment=reported, endpoint=participant.endpoint, local_gateway_id=second.install_id,
            proof_grant="grant-that-authenticated-the-probe") == inherited
        with pytest.raises(retirement.RetirementConflictError, match="disband has not completed"):
            retirement.materialize_notice(second.db, enrollment_id=entry["enrollment_id"],
                                          local_gateway_id=second.install_id, secret_loader=lambda: b"")
        rooms.disband_room(second.db, room_id=ROOM, expected_gateway_id=second.install_id, expected_epoch=2)
        outgoing = retirement.materialize_notice(
            second.db, enrollment_id=entry["enrollment_id"], local_gateway_id=second.install_id,
            secret_loader=lambda: b"")  # an inherited notice never needs the first home's key
    assert (outgoing.endpoint, outgoing.proof_grant) == (participant.endpoint, "grant-that-authenticated-the-probe")
    assert outgoing.value.startswith("ed25519-v1.")
    receipt = retire(participant, outgoing.payload(), outgoing.value)
    assert receipt["retired"] and (receipt["authority_gateway_id"], receipt["authority_epoch"]) == (second.install_id, 2)
    assert retire(participant, outgoing.payload(), outgoing.value) == receipt  # idempotent
    with acting(second):
        retirement.acknowledge_notice(second.db, notice=outgoing, response=receipt)
        assert retirement.home_status(second.db)[0]["state"] == "acknowledged"


def test_the_home_a_copy_left_can_no_longer_retire_it(moved):
    net, entry = moved
    home, participant = net["home"], net["third"]
    with acting(home), closing(open_sqlite(home.db)) as conn:
        row = dict(conn.execute(f"SELECT * FROM {retirement.HOME_TABLE} WHERE enrollment_id=?",
                                (entry["enrollment_id"],)).fetchone())
    value = retirement._sign_notice(retirement._signing_seed(SECRET, row), row)  # its own key, still valid
    payload = {key: entry[key] for key in retirement._NOTICE_FIELDS}
    with pytest.raises(retirement.RetirementConflictError, match="differs from the retained copy"):
        retire(participant, payload, value)
    with acting(participant), closing(open_sqlite(participant.db)) as conn:
        assert not retirement.copy_retired_locked(conn, ROOM)


def test_only_the_copys_verified_authority_can_retire_it(moved):
    net, entry = moved
    home, second, participant, fresh = net["home"], net["second"], net["third"], net["fresh"]
    payload = {**{key: entry[key] for key in ("enrollment_id", "room_id", "target_install_id")},
               "authority_gateway_id": second.install_id, "authority_epoch": 2}
    for body, signer in (
            (payload, home),  # signed with another installation's key
            (payload, fresh),  # a key this copy never pinned
            ({**payload, "authority_epoch": 3}, second),  # an epoch this copy never verified
            ({**payload, "authority_gateway_id": fresh.install_id}, fresh),  # no marked transition to it here
            ({**payload, "target_install_id": fresh.install_id}, second)):  # another destination
        with pytest.raises(retirement.RetirementAuthorizationError):
            retire(participant, body, signed(signer, body))
    # A header moved without a marked transition names no authority that can retire the copy.
    with closing(sqlite3.connect(participant.db)) as conn:
        conn.execute("UPDATE hosted_room_replicas SET authority_epoch=3 WHERE room_id=?", (ROOM,))
    forged = {**payload, "authority_epoch": 3}
    with pytest.raises(retirement.RetirementError):
        retire(participant, forged, signed(second, forged))
    with acting(participant), closing(open_sqlite(participant.db)) as conn:
        assert not retirement.copy_retired_locked(conn, ROOM)


def test_only_the_current_host_inherits_a_retirement(moved):
    net, entry = moved
    home, second, participant = net["home"], net["second"], net["third"]
    reported = report(participant, (second.install_id, 2))
    with acting(home):  # the host the copy left is no longer this room's authority at a later epoch
        assert retirement.inherit_home_enrollment(home.db, enrollment=reported, endpoint=participant.endpoint,
                                                  local_gateway_id=home.install_id, proof_grant="grant") is None
    with acting(second), pytest.raises(retirement.RetirementProofUnavailable):
        retirement.inherit_home_enrollment(second.db, enrollment=reported, endpoint=participant.endpoint,
                                           local_gateway_id=second.install_id, proof_grant="")


def test_the_successors_publisher_asks_each_route_once_and_records_the_obligation(moved):
    net, entry = moved
    second, participant = net["second"], net["third"]
    reported, asked = report(participant, (second.install_id, 2)), []
    client = SimpleNamespace(probe=lambda grant: asked.append(grant) or {"retirement_enrollment": reported})
    route = SimpleNamespace(key=(ROOM, "reviewer"), generation="route-1", link=SimpleNamespace(
        grant="continuation-grant", target_url=participant.endpoint,
        catalog=SimpleNamespace(installation_id=participant.install_id)))
    with acting(second):
        pub = publisher.HostedRoomReplicationPublisher(second.db)
        recorded = pub._inherit_retirement(route, client)
        assert (recorded["enrollment_id"], recorded["state"], recorded["authority_gateway_id"]) == (
            entry["enrollment_id"], "enrolled", second.install_id)
        assert pub._inherit_retirement(route, client) is None and asked == ["continuation-grant"]
