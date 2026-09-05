"""Loud model fallback: an unhonored model/provider request must be visible.

Regression cover for the silent-fallback failure mode reported 2026-08-17:

    hermes -m "openrouter/zai/glm-5.3" --provider openrouter -z "ping"

answered "pong" with **glm-5.2 via zai** — a different model from a different
provider — with no error, no warning, and a session row that claimed the served
model had been the requested one. Three things conspired:

1. ``hermes -z`` redirects stdout AND stderr to devnull for the whole run and
   sets ``suppress_status_output``, which short-circuits ``_vprint`` *before*
   its ``force`` check — so even the forced fallback notice was swallowed.
2. ``sessions.model`` holds a single (model, provider) pair, and
   ``update_token_counts``' first-accounted-route reconciliation overwrites it
   with the route that actually billed. The request was destroyed on the first
   API call.
3. Nothing anywhere persisted "what was asked for".

The tests below pin each half of the fix: the state layer keeps the request
beside the delivery, and one-shot says so out loud on the real stderr.
"""

import json

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        yield database
    finally:
        database.close()


# ---------------------------------------------------------------------------
# State layer: the request survives the served route overwriting `model`
# ---------------------------------------------------------------------------

def test_requested_route_survives_first_accounted_route_overwrite(db):
    """The exact reported case: request openrouter, get billed by zai."""
    db.create_session(
        "s_fallback",
        source="cli",
        model="openrouter/zai/glm-5.3",
        requested_model="openrouter/zai/glm-5.3",
        requested_provider="openrouter",
    )

    # First accounted API call comes from the fallback route, not the request.
    db.update_token_counts(
        "s_fallback",
        input_tokens=10,
        output_tokens=5,
        model="glm-5.2",
        billing_provider="zai",
        api_call_count=1,
    )
    db.flush_token_counts()

    row = db.get_session("s_fallback")
    # Served route won the aggregate columns — that part is intended.
    assert row["model"] == "glm-5.2"
    assert row["billing_provider"] == "zai"
    # ...and the request is still recoverable, which is the whole point.
    assert row["requested_model"] == "openrouter/zai/glm-5.3"
    assert row["requested_provider"] == "openrouter"


def test_requested_route_is_never_rewritten_by_an_observation_of_the_served(db):
    """No writer that merely OBSERVES the served route may relabel the request.

    ``update_token_counts``' self-healing insert re-enters the row with the
    model that is actually billing; that is an observation of delivery, and it
    must leave the request alone however far the two diverge.

    Round 5 note — this test used to make the same claim about
    ``record_session_fallback`` by passing it invented values
    (``something-else``/``nowhere``) and asserting they were refused. That
    encoded the wrong contract, and it is the contract that produced the round-5
    lie: this writer's arguments are not an observation of anything, they name
    the route its caller is abandoning right now (see its docstring), so a row
    whose stored request carries no verdict must yield to them. What may never
    be rewritten is the subject of a verdict that already stands — pinned by
    ``test_a_second_fallback_keeps_the_first_abandoned_request`` and the flagged
    half of the table tests.
    """
    db.create_session(
        "s_keep",
        source="cli",
        model="glm-5.3",
        requested_model="glm-5.3",
        requested_provider="zai",
    )
    # A lazy writer (update_token_counts' self-healing insert) re-enters the
    # row with a different model; COALESCE must leave the request alone.
    db.update_token_counts(
        "s_keep", input_tokens=1, model="deepseek-v4-flash",
        billing_provider="deepseek", api_call_count=1,
    )
    db.flush_token_counts()

    row = db.get_session("s_keep")
    assert row["requested_model"] == "glm-5.3"
    assert row["requested_provider"] == "zai"
    assert row["model"] == "deepseek-v4-flash"
    assert row["fallback_activated"] == 0, (
        "an observation of the served route is not a verdict on the request"
    )


def test_fallback_flag_defaults_off_and_is_sticky(db):
    db.create_session(
        "s_flag", source="cli", model="glm-5.3",
        requested_model="glm-5.3", requested_provider="zai",
    )
    assert db.get_session("s_flag")["fallback_activated"] == 0

    db.record_session_fallback("s_flag")
    assert db.get_session("s_flag")["fallback_activated"] == 1
    # Idempotent — a second chain hop must not corrupt the flag.
    db.record_session_fallback("s_flag")
    assert db.get_session("s_flag")["fallback_activated"] == 1


def test_record_session_fallback_backfills_a_row_created_before_the_columns(db):
    """Rows that predate the audit columns still get a usable request."""
    db.create_session("s_backfill", source="cli", model="glm-5.2")
    db.record_session_fallback(
        "s_backfill", requested_model="gpt-5.6-sol", requested_provider="openrouter",
    )
    row = db.get_session("s_backfill")
    assert row["fallback_activated"] == 1
    assert row["requested_model"] == "gpt-5.6-sol"
    assert row["requested_provider"] == "openrouter"


def test_fallback_after_a_switch_names_the_switched_request(db):
    """A ``/model`` switch then a fallback must name the SWITCHED request.

    The switch is the current request; the process-start snapshot stopped being
    current the moment the operator switched. So the route this fallback
    abandons is ``gpt-5.4`` on whatever provider was actually routing it — which
    is what ``_record_fallback_on_session`` passes (see
    ``abandoned_route_for_audit``) — and the row must name that, not
    ``glm-5.3``/``zai``.

    Round 5 note — this test previously fed the writer the process-start
    snapshot (``glm-5.3``/``zai``) and asserted the stored ``gpt-5.4``/NULL
    survived it, i.e. it defended the SQL against a caller passing the wrong
    route. That defence was both incomplete (it could only ever keep a stale
    pair, never record the right one) and the mechanism of the round-5 lie, so
    the caller was fixed instead: the writer is now told the truth and records
    it. Pinned end to end by
    ``test_switch_then_fallback_records_the_switched_route_end_to_end``.
    """
    db.create_session(
        "s_pair", source="cli", model="glm-5.3",
        requested_model="glm-5.3", requested_provider="zai",
    )
    db.update_session_model("s_pair", "gpt-5.4")  # provider-less switch
    switched = db.get_session("s_pair")
    assert (switched["requested_model"], switched["requested_provider"]) == (
        "gpt-5.4", None
    ), "the switch itself still records 'no provider requested'"

    # The fallback later in the same process abandons the LIVE route: the
    # switched model, on the provider that was serving it.
    db.record_session_fallback(
        "s_pair", requested_model="gpt-5.4", requested_provider="deepseek",
    )

    row = db.get_session("s_pair")
    assert row["fallback_activated"] == 1
    assert row["requested_model"] == "gpt-5.4"
    assert row["requested_provider"] == "deepseek"
    assert "zai" not in (row["requested_provider"] or ""), (
        "the abandoned start's provider may not resurface beside a later model"
    )


def test_record_session_fallback_tolerates_a_missing_row(db):
    """Never raise on the recovery path — the row is created lazily."""
    db.record_session_fallback("s_does_not_exist", requested_model="x")
    assert db.get_session("s_does_not_exist") is None


def test_explicit_model_switch_resets_the_request_audit(db):
    """A /model switch is a NEW request, so the stale flag must clear."""
    db.create_session(
        "s_switch", source="cli", model="glm-5.3",
        requested_model="glm-5.3", requested_provider="zai",
    )
    db.record_session_fallback("s_switch")
    db.update_session_model("s_switch", "deepseek-v4-flash", provider="deepseek")

    row = db.get_session("s_switch")
    assert row["requested_model"] == "deepseek-v4-flash"
    assert row["requested_provider"] == "deepseek"
    assert row["fallback_activated"] == 0


def test_provider_less_switch_does_not_keep_the_previous_request_provider(db):
    """The audit pair must describe ONE request, not halves of two.

    ``/model deepseek-v4-flash`` without a provider asks for a model and
    nothing else. COALESCE-ing the provider half of the audit left the PREVIOUS
    request's provider standing beside the NEW model, so the row described a
    route nobody had ever asked for ("requested deepseek-v4-flash via zai") —
    and the `hermes sessions list` warning would have printed exactly that.
    "No provider requested" is NULL, not the last one.
    """
    db.create_session(
        "s_noprov", source="cli", model="glm-5.3",
        requested_model="glm-5.3", requested_provider="zai",
    )
    db.record_session_fallback("s_noprov")

    db.update_session_model("s_noprov", "deepseek-v4-flash")

    row = db.get_session("s_noprov")
    assert row["requested_model"] == "deepseek-v4-flash"
    assert row["requested_provider"] is None
    assert row["fallback_activated"] == 0


def test_switch_audit_and_resume_route_have_separate_provider_semantics(db):
    """The audit column and ``model_config.$.provider`` are different things.

    The stored ``$.provider`` exists so a later resume recombines the model
    with the provider that serves it (#79536) and must survive a provider-less
    switch; the audit column must state what THIS request asked for. Lineage
    markers survive both, since the switch goes through the shared merge.
    """
    db.create_session(
        "s_cfg", source="cli", model="glm-5.3",
        requested_model="glm-5.3", requested_provider="zai",
        model_config={
            "provider": "custom:feather",
            "_branched_from": "parent-session",
            "_delegate_from": "boss-session",
        },
    )
    db.record_session_fallback("s_cfg")

    # Provider-less switch: the audit forgets the provider, the resume route
    # keeps it (the caller made no statement about routing).
    db.update_session_model("s_cfg", "deepseek-v4-flash")
    row = db.get_session("s_cfg")
    config = json.loads(row["model_config"])
    assert config["provider"] == "custom:feather"
    assert config["_branched_from"] == "parent-session"
    assert config["_delegate_from"] == "boss-session"
    assert row["requested_model"] == "deepseek-v4-flash"
    assert row["requested_provider"] is None
    assert row["fallback_activated"] == 0

    # Provider-bearing switch: both halves move to the new request together.
    db.record_session_fallback("s_cfg")
    db.update_session_model("s_cfg", "glm-5.4", provider="zai")
    row = db.get_session("s_cfg")
    config = json.loads(row["model_config"])
    assert config["provider"] == "zai"
    assert config["_branched_from"] == "parent-session"
    assert config["_delegate_from"] == "boss-session"
    assert row["requested_model"] == "glm-5.4"
    assert row["requested_provider"] == "zai"
    assert row["fallback_activated"] == 0


def test_fallback_backfill_completes_the_provider_of_the_same_request(db):
    """Backfilling the pair as a unit is not the same as refusing to backfill.

    The rule is "one route, adopted whole" — so a snapshot naming the model the
    row already records is not a foreign route, it is the SAME route with the
    provider half known. Refusing it (because ``requested_model`` is merely
    non-NULL) drops the provider from the loud warning for every resumed
    session whose recorded model was genuinely re-requested via a real
    provider: `requested glm-5.3 → served grok-4 (xai)` instead of
    `requested glm-5.3 (zai) → ...`.
    """
    db.create_session(
        "s_same", source="cli", model="glm-5.3",
        requested_model="glm-5.3", requested_provider="zai",
    )
    # Provider-less /model switch back to the same model: "no provider
    # requested" is recorded, on purpose.
    db.update_session_model("s_same", "glm-5.3")
    assert db.get_session("s_same")["requested_provider"] is None

    # A later process re-requests that very model, this time knowing the
    # provider; the fallback snapshot may complete the pair it already names.
    db.record_session_fallback(
        "s_same", requested_model="glm-5.3", requested_provider="zai",
    )
    row = db.get_session("s_same")
    assert row["fallback_activated"] == 1
    assert row["requested_model"] == "glm-5.3"
    assert row["requested_provider"] == "zai"


def test_session_upsert_adopts_the_request_pair_only_as_a_unit(db):
    """``create_session``'s upsert is the third writer of the audit pair.

    Every process's first turn re-runs ``create_session`` for an existing
    session id — that is what the ``ON CONFLICT`` upsert is for — carrying THAT
    process's immutable start-of-run snapshot. COALESCE-ing the two halves
    independently lets the snapshot's provider land beside a model some earlier
    ``/model`` switch requested, describing a route nobody ever asked for.
    """
    # Row already records a provider-less request (a `/model` switch, or an
    # ad-hoc --base-url endpoint whose provider key cannot be recovered).
    db.create_session(
        "s_upsert", source="cli", model="glm-5.3",
        requested_model="glm-5.3", requested_provider="zai",
    )
    db.update_session_model("s_upsert", "gpt-5.4")

    # A later process's first turn, whose own request differs.
    db.create_session(
        "s_upsert", source="cli", model="glm-5.4",
        requested_model="glm-5.4", requested_provider="minimax",
    )
    row = db.get_session("s_upsert")
    assert row["requested_model"] == "gpt-5.4"
    assert row["requested_provider"] is None, (
        "the upsert must not pair a foreign provider with the recorded model"
    )

    # A row with NEITHER half still adopts the snapshot's whole pair.
    db.create_session("s_bare", source="cli", model="glm-5.2")
    db.create_session(
        "s_bare", source="cli", model="glm-5.2",
        requested_model="glm-5.3", requested_provider="zai",
    )
    bare = db.get_session("s_bare")
    assert bare["requested_model"] == "glm-5.3"
    assert bare["requested_provider"] == "zai"

    # A row with BOTH halves is never rewritten by a later snapshot.
    db.create_session(
        "s_bare", source="cli", model="glm-5.2",
        requested_model="gpt-5.4", requested_provider="minimax",
    )
    kept = db.get_session("s_bare")
    assert kept["requested_model"] == "glm-5.3"
    assert kept["requested_provider"] == "zai"


# ---------------------------------------------------------------------------
# The whole state machine: 8 row states (4 pairs x 2 flag values) x 3 writers
# ---------------------------------------------------------------------------

#: ``(row state, row pair, snapshot pair, expected pair after the write)`` for
#: ``create_session``'s ON CONFLICT upsert against a row whose
#: ``fallback_activated`` is DOWN (with it up, this writer freezes the pair
#: whole, which the test asserts directly).
#:
#: This writer carries a process-START snapshot: a request that has not been
#: answered yet, so it may only fill what the row leaves unanswered.
#: ``record_session_fallback`` answers to a DIFFERENT gate
#: (``_FALLBACK_ROUTE_TABLE`` below) because its argument is not a snapshot at
#: all — it is the route being abandoned in that very statement. Rounds 1-4
#: shared one table for both writers; that identification is precisely what let
#: round 4 hand a flag to a request its own call site had declined to adopt.
#:
#: The earlier rounds of this fix each enumerated one axis short: two enumerated
#: three of the four pair states and shipped the bug living in the fourth, the
#: third left the flag axis unswept, and the fourth swept single writes only —
#: never asking where the row's pair came from, which is where round 5's lie was
#: (see ``_TWO_PROCESS_*`` below).
_AUDIT_PAIR_TABLE = [
    # Nothing recorded: the snapshot's request is adopted as a whole pair,
    # whatever shape it has.
    ("neither", (None, None), ("glm-5.4", "minimax"), ("glm-5.4", "minimax")),
    ("neither", (None, None), (None, "minimax"), (None, "minimax")),
    ("neither", (None, None), (None, None), (None, None)),
    # Model only: the recorded model can be newer than any snapshot (a /model
    # switch writes it mid-run), so it wins — and its NULL provider means "no
    # provider requested", which only a snapshot naming that SAME model may
    # fill in.
    ("model only", ("gpt-5.4", None), ("glm-5.4", "minimax"), ("gpt-5.4", None)),
    ("model only", ("gpt-5.4", None), ("gpt-5.4", "minimax"), ("gpt-5.4", "minimax")),
    ("model only", ("gpt-5.4", None), (None, "minimax"), ("gpt-5.4", None)),
    # Provider only (`hermes --provider vllm`, no model.default): a request
    # that never named a model, and one no switch can have written. A snapshot
    # that names a model supersedes it as a whole pair — the model may not be
    # stitched onto the stored provider, and refusing the model would strand
    # the warning on a bare provider from an abandoned start.
    ("provider only", (None, "vllm"), ("glm-5.4", "minimax"), ("glm-5.4", "minimax")),
    ("provider only", (None, "vllm"), ("glm-5.4", "vllm"), ("glm-5.4", "vllm")),
    # A snapshot naming no model adds nothing, so the row stands.
    ("provider only", (None, "vllm"), (None, "minimax"), (None, "vllm")),
    ("provider only", (None, "vllm"), (None, None), (None, "vllm")),
    # Both halves: a complete request is never rewritten by a later snapshot.
    ("both", ("gpt-5.4", "vllm"), ("glm-5.4", "minimax"), ("gpt-5.4", "vllm")),
    ("both", ("gpt-5.4", "vllm"), ("gpt-5.4", "minimax"), ("gpt-5.4", "vllm")),
    ("both", ("gpt-5.4", "vllm"), (None, None), ("gpt-5.4", "vllm")),
]


#: Row states are the TRIPLE, so every case above is played twice: once against
#: a row carrying no verdict and once against a row whose ``fallback_activated``
#: is already up. The flag is not a fourth independent column — it is the
#: verdict on the pair beside it ("the request these two name was abandoned") —
#: so what a writer may do to the pair depends on it, and vice versa.
_ROW_FLAGS = [0, 1]


def _setup_row(db, session_id, row, flag):
    """Put the row into one of the eight ``(pair, flag)`` states, and prove it.

    The flag is raised with a snapshot-less ``record_session_fallback``, which
    is a pure flag raise from every pair state (pinned by
    ``test_a_bare_record_session_fallback_raises_only_the_flag``) — so the row
    really is the state the parametrization claims, with no pair write smuggled
    into the setup.
    """
    db.create_session(
        session_id, source="cli", model="glm-5.2",
        requested_model=row[0], requested_provider=row[1],
    )
    if flag:
        db.record_session_fallback(session_id)
    stored = db.get_session(session_id)
    assert (
        stored["requested_model"],
        stored["requested_provider"],
        stored["fallback_activated"],
    ) == (row[0], row[1], flag), "could not set up the row state"


def _assert_one_whole_request(result, *candidates):
    """The stored TRIPLE must be one whole record, never a mix of two.

    The pair rule ("one of the two requests in play, never the model of one
    beside the provider of another") extended to the verdict: a request and the
    verdict reached about it are one record, so ``result`` must equal one of the
    whole records the writer had to choose between — pair and flag together.

    Completing a half is not an exception: a half may only be taken from a
    snapshot that agrees with the row about the half they share, so every legal
    completion equals the snapshot's own pair. And splicing is not a harmless
    approximation in either direction — a flag from record A over pair B cries
    wolf about a request that was honored, while a pair from B over a flag
    belonging to A drops the provider of the request that really was abandoned.
    """
    assert result in candidates, (
        f"{result} is not one of the whole records in play: {candidates}"
    )


@pytest.mark.parametrize("flag", _ROW_FLAGS, ids=["unflagged", "flagged"])
@pytest.mark.parametrize(
    "state,row,snapshot,expected",
    _AUDIT_PAIR_TABLE,
    ids=[
        f"{s}-{'x'.join(str(v) for v in snap)}"
        for s, _row, snap, _exp in _AUDIT_PAIR_TABLE
    ],
)
def test_upsert_audit_pair_table(db, state, row, snapshot, expected, flag):
    """``create_session``'s ON CONFLICT upsert, over the whole state machine.

    The flag axis splits every case in two. ``create_session``'s snapshot is a
    process START — a request that has not been answered yet, let alone
    abandoned — so this writer may never move a raised verdict:

    * flag DOWN: no verdict to contradict, the ordinary pair gate applies, and
      an adopted snapshot arrives with the only verdict a just-made request can
      carry, ``0``. This writer never touches the flag, so that is automatic.
    * flag UP: the stored pair is what the verdict is ABOUT, so it is frozen
      whole. Freezing loses nothing — should this snapshot's own request also be
      abandoned, ``record_session_fallback`` restates pair and flag together at
      the moment the new pair becomes a true statement.
    """
    _setup_row(db, "s_tbl", row, flag)

    # The next process's first turn, carrying its own start-of-run snapshot.
    db.create_session(
        "s_tbl", source="cli", model="glm-5.2",
        requested_model=snapshot[0], requested_provider=snapshot[1],
    )
    stored = db.get_session("s_tbl")
    result = (
        stored["requested_model"],
        stored["requested_provider"],
        stored["fallback_activated"],
    )
    assert result == (
        (row[0], row[1], 1) if flag else (expected[0], expected[1], 0)
    )
    # The verdict is never invented and never discarded by this writer: it may
    # keep the row's whole record, or take a snapshot's pair with the flag down.
    _assert_one_whole_request(result, (row[0], row[1], flag), (*snapshot, 0))


#: ``record_session_fallback``'s own table:
#: ``(row state, row pair, abandoned route, expected pair with the flag DOWN,
#: expected pair with the flag UP)``.
#:
#: The route column is NOT a process-start snapshot — it is what the call site
#: asserts is being abandoned right now — so this writer has its own gate, and
#: the flag axis genuinely changes the answer:
#:
#: * flag DOWN — no verdict stands, so the abandoned route becomes the record's
#:   subject WHOLE. Nothing can be mispaired: both halves come from one route.
#: * flag UP — a verdict already stands and keeps its subject. The pair may only
#:   be named more precisely: a NULL half filled from a route that agrees with
#:   the row on the half the row does record.
#: * a route naming NEITHER half asserts nothing about the pair (the flag-only
#:   raise), so the row's pair stands under either flag.
_FALLBACK_ROUTE_TABLE = [
    # Nothing recorded: the abandoned route is adopted whole either way — a
    # verdict with no subject gains one, which displaces nothing.
    ("neither", (None, None), ("glm-5.4", "minimax"),
     ("glm-5.4", "minimax"), ("glm-5.4", "minimax")),
    ("neither", (None, None), (None, "minimax"),
     (None, "minimax"), (None, "minimax")),
    ("neither", (None, None), (None, None), (None, None), (None, None)),
    # Model only (a /model switch, or a bare-flagged row). Unflagged: the
    # abandoned route wins whole — keeping the switched model beside this
    # route's provider would be the mixed pair, and keeping the switched model
    # while the flag says "abandoned" would name a request nobody gave up on.
    # Flagged: frozen, except the completion in the second line, where the
    # route names the very model the verdict is about.
    ("model only", ("gpt-5.4", None), ("glm-5.4", "minimax"),
     ("glm-5.4", "minimax"), ("gpt-5.4", None)),
    ("model only", ("gpt-5.4", None), ("gpt-5.4", "minimax"),
     ("gpt-5.4", "minimax"), ("gpt-5.4", "minimax")),
    ("model only", ("gpt-5.4", None), (None, "minimax"),
     (None, "minimax"), ("gpt-5.4", None)),
    # Provider only (`hermes --provider vllm`, no model.default).
    ("provider only", (None, "vllm"), ("glm-5.4", "minimax"),
     ("glm-5.4", "minimax"), (None, "vllm")),
    # ...and the completion in the mirror direction: same provider, so the
    # route the verdict is about is simply now known to have named a model.
    ("provider only", (None, "vllm"), ("glm-5.4", "vllm"),
     ("glm-5.4", "vllm"), ("glm-5.4", "vllm")),
    ("provider only", (None, "vllm"), (None, "minimax"),
     (None, "minimax"), (None, "vllm")),
    ("provider only", (None, "vllm"), (None, None),
     (None, "vllm"), (None, "vllm")),
    # Both halves: an unjudged complete request still yields to the route being
    # abandoned (round 5's lie lived exactly here — P1 honored and billed, P2
    # abandoned); a judged one is frozen.
    ("both", ("gpt-5.4", "vllm"), ("glm-5.4", "minimax"),
     ("glm-5.4", "minimax"), ("gpt-5.4", "vllm")),
    ("both", ("gpt-5.4", "vllm"), ("gpt-5.4", "minimax"),
     ("gpt-5.4", "minimax"), ("gpt-5.4", "vllm")),
    ("both", ("gpt-5.4", "vllm"), (None, None),
     ("gpt-5.4", "vllm"), ("gpt-5.4", "vllm")),
]


@pytest.mark.parametrize("flag", _ROW_FLAGS, ids=["unflagged", "flagged"])
@pytest.mark.parametrize(
    "state,row,route,expected_down,expected_up",
    _FALLBACK_ROUTE_TABLE,
    ids=[
        f"{s}-{'x'.join(str(v) for v in rt)}"
        for s, _row, rt, _d, _u in _FALLBACK_ROUTE_TABLE
    ],
)
def test_fallback_route_audit_table(
    db, state, row, route, expected_down, expected_up, flag
):
    """``record_session_fallback`` over the whole state machine.

    The rule this pins is one sentence: *the route being abandoned becomes the
    record's subject unless a verdict already has one*. Both halves of it are
    load-bearing.

    Taking the pair whole while the flag is down is the round-5 fix: coalescing
    instead ("a recorded model wins") let this writer decline the route it was
    judging and stamp its ``= 1`` on a request that had been honored end to end.

    Freezing while the flag is up is what keeps a multi-hop chain naming the
    request it started from, and it is the same rule the upsert follows — this
    writer is no longer exempt from the verdict clause, it simply reaches the
    case where no verdict stands. What it costs is real and documented: with two
    abandoned requests on one session id the row names the first, so the second
    is under-reported. See ``_insert_session_row``'s residual note.
    """
    _setup_row(db, "s_tbl", row, flag)
    db.record_session_fallback(
        "s_tbl", requested_model=route[0], requested_provider=route[1],
    )
    stored = db.get_session("s_tbl")
    result = (
        stored["requested_model"],
        stored["requested_provider"],
        stored["fallback_activated"],
    )
    expected = expected_up if flag else expected_down
    assert result == (expected[0], expected[1], 1)
    assert stored["fallback_activated"] == 1, "the flag is the point of the call"
    # Either request may end up stored, but the verdict this call asserts
    # applies to whichever one does.
    _assert_one_whole_request(result, (row[0], row[1], 1), (*route, 1))
    # The stored pair always names a route that WAS abandoned: the one this call
    # names, or the one the standing verdict was already about.
    assert result[:2] in {route, row}, (
        "a raised flag must sit beside a request that was really abandoned"
    )


@pytest.mark.parametrize("flag", _ROW_FLAGS, ids=["unflagged", "flagged"])
@pytest.mark.parametrize(
    "state,row",
    [(state, row) for state, row, _snap, _exp in _AUDIT_PAIR_TABLE],
    ids=[
        f"{state}-{'x'.join(str(v) for v in row)}"
        for state, row, _snap, _exp in _AUDIT_PAIR_TABLE
    ],
)
@pytest.mark.parametrize(
    "switch", [("glm-5.4", "minimax"), ("glm-5.4", None), ("", "minimax")]
)
def test_update_session_model_audit_pair_table(db, state, row, switch, flag):
    """``update_session_model``: the third writer, from all eight row states.

    A /model switch is a new explicit request, so it writes all THREE columns
    from THIS call — which makes it coherent from every prior state by
    construction, including the provider-only one and including either incoming
    flag. Nothing is coalesced in, so nothing can be mixed in. Clearing the flag
    is legitimate here precisely because the request it judged is being
    discarded in the same statement; the upsert, which discards nothing, must
    freeze a flagged pair instead. The empty-model case pins the ``or None``
    normalization: '' and None both mean "no model requested" and must be stored
    identically, or the NULL gates in the other two writers would read '' as a
    recorded name.
    """
    _setup_row(db, "s_tbl", row, flag)
    db.update_session_model("s_tbl", switch[0], provider=switch[1])

    stored = db.get_session("s_tbl")
    result = (
        stored["requested_model"],
        stored["requested_provider"],
        stored["fallback_activated"],
    )
    assert result == (switch[0] or None, switch[1], 0)
    # This writer has exactly one legal outcome: its own request, unjudged.
    _assert_one_whole_request(result, (switch[0] or None, switch[1], 0))


def test_audit_columns_are_declared_so_existing_dbs_reconcile(tmp_path):
    """The columns are declarative: an older DB gains them on next open."""
    import sqlite3

    from hermes_state_common import SCHEMA_SQL

    declared = SessionDB._parse_schema_columns(SCHEMA_SQL)["sessions"]
    for column in ("requested_model", "requested_provider", "fallback_activated"):
        assert column in declared, column

    # Build a DB, drop the columns out of the picture by recreating the table
    # without them, then reopen: _reconcile_columns must ADD them back.
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.close()
    conn = sqlite3.connect(path)
    conn.execute("ALTER TABLE sessions RENAME TO sessions_old")
    conn.execute(
        "CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT NOT NULL, "
        "model TEXT, started_at REAL NOT NULL)"
    )
    conn.commit()
    conn.close()

    db = SessionDB(path)
    try:
        live = {
            row[1]
            for row in db._conn.execute('PRAGMA table_info("sessions")').fetchall()
        }
        assert {"requested_model", "requested_provider", "fallback_activated"} <= live
    finally:
        db.close()


# ---------------------------------------------------------------------------
# One-shot: the warning reaches the real stderr, stdout stays clean
# ---------------------------------------------------------------------------

class _FakeAgent:
    def __init__(self, requested_model, requested_provider, fallback):
        self.origin_requested_model = requested_model
        self.origin_requested_provider = requested_provider
        self._fallback_activated = fallback


def test_annotate_requested_route_reads_the_immutable_snapshot():
    from hermes_cli import oneshot

    agent = _FakeAgent("openrouter/zai/glm-5.3", "openrouter", True)
    # try_activate_fallback reassigns requested_provider to the fallback; the
    # audit must not read that attribute.
    agent.requested_provider = "zai"
    result = {"model": "glm-5.2", "provider": "zai"}
    oneshot._annotate_requested_route(agent, result)

    assert result["requested_model"] == "openrouter/zai/glm-5.3"
    assert result["requested_provider"] == "openrouter"
    assert result["fallback_activated"] is True


def test_no_warning_when_the_requested_model_answered():
    from hermes_cli import oneshot

    agent = _FakeAgent("glm-5.3", "zai", False)
    result = {"model": "glm-5.3", "provider": "zai"}
    oneshot._annotate_requested_route(agent, result)
    assert oneshot._fallback_warning_line(result) is None


def test_warning_names_both_routes():
    from hermes_cli import oneshot

    agent = _FakeAgent("openrouter/zai/glm-5.3", "openrouter", True)
    result = {"model": "glm-5.2", "provider": "zai"}
    oneshot._annotate_requested_route(agent, result)

    line = oneshot._fallback_warning_line(result)
    assert line is not None
    assert line.endswith("\n")
    assert "openrouter/zai/glm-5.3 via openrouter" in line
    assert "glm-5.2 via zai" in line
    # The whole failure mode was that a wrong-model answer looked normal.
    assert "SERVED" in line


def test_warning_survives_a_missing_request_half():
    from hermes_cli import oneshot

    line = oneshot._fallback_warning_line(
        {"model": "glm-5.2", "provider": "zai", "fallback_activated": True}
    )
    assert line is not None
    assert "an unknown model" in line


def test_oneshot_writes_the_warning_to_the_real_stderr(monkeypatch, capsys):
    """The end-to-end guarantee: -z can no longer answer 200-quiet.

    ``run_oneshot`` swallows every byte the agent writes; only the final
    response reaches stdout. This asserts the fallback notice takes the real
    stderr path instead, and that stdout stays exactly the response.
    """
    from hermes_cli import oneshot

    def _fake_run_agent(prompt, **kwargs):
        result = {
            "final_response": "pong",
            "model": "glm-5.2",
            "provider": "zai",
            "completed": True,
        }
        oneshot._annotate_requested_route(
            _FakeAgent("openrouter/zai/glm-5.3", "openrouter", True), result
        )
        return "pong", result

    monkeypatch.setattr(oneshot, "_run_agent", _fake_run_agent)
    monkeypatch.setattr(oneshot, "declare_stateless_channel", lambda *a, **k: None)

    rc = oneshot.run_oneshot("ping", model="openrouter/zai/glm-5.3", provider="openrouter")
    captured = capsys.readouterr()

    assert rc == 0
    assert captured.out == "pong\n"
    assert "openrouter/zai/glm-5.3" in captured.err
    assert "glm-5.2 via zai" in captured.err


def test_nonexistent_requested_model_still_names_it_in_the_warning(
    monkeypatch, capsys
):
    """Requesting a model that does not exist may not answer quietly either.

    The 2026-08-17 reproduction included ``openai/gpt-5.6-sol`` — a model id
    that does not exist — which was answered by glm-5.2/zai with no signal.
    Whatever the reason a request cannot be honored (unreachable provider,
    bad id, exhausted quota), the fallback notice must name the id exactly as
    typed, so a pipeline pinning a model can see the pin was not honored.
    """
    from hermes_cli import oneshot

    def _fake_run_agent(prompt, **kwargs):
        result = {
            "final_response": "pong",
            "model": "glm-5.2",
            "provider": "zai",
            "completed": True,
        }
        oneshot._annotate_requested_route(
            _FakeAgent("openai/gpt-5.6-sol", "openrouter", True), result
        )
        return "pong", result

    monkeypatch.setattr(oneshot, "_run_agent", _fake_run_agent)
    monkeypatch.setattr(oneshot, "declare_stateless_channel", lambda *a, **k: None)

    rc = oneshot.run_oneshot(
        "ping", model="openai/gpt-5.6-sol", provider="openrouter"
    )
    captured = capsys.readouterr()

    assert rc == 0  # a fallback that answered is not an error...
    assert captured.out == "pong\n"  # ...but stdout stays machine-readable
    # and the nonexistent id is named verbatim on the real stderr.
    assert "openai/gpt-5.6-sol via openrouter" in captured.err
    assert "glm-5.2 via zai" in captured.err


def test_provider_without_credentials_fails_loudly(monkeypatch, capsys):
    """No credentials must never become a quiet answer from an unrelated model.

    When there is no fallback chain to absorb the failure, the run must exit
    non-zero with an explicit error — the same loud contract as the fallback
    warning, on the branch where nothing answered at all.
    """
    from hermes_cli import oneshot

    def _fake_run_agent(prompt, **kwargs):
        raise RuntimeError(
            "openrouter: AuthenticationError: no API key configured"
        )

    monkeypatch.setattr(oneshot, "_run_agent", _fake_run_agent)
    monkeypatch.setattr(oneshot, "declare_stateless_channel", lambda *a, **k: None)

    rc = oneshot.run_oneshot("ping", model="glm-5.3", provider="openrouter")
    captured = capsys.readouterr()

    assert rc == 1
    assert captured.out == ""  # no answer at all — never an unrelated model
    assert "agent failed" in captured.err
    assert "AuthenticationError" in captured.err


def test_oneshot_usage_file_records_request_and_delivery(tmp_path, monkeypatch):
    """Pipelines get the audit in machine-readable form."""
    from hermes_cli import oneshot

    def _fake_run_agent(prompt, **kwargs):
        result = {
            "final_response": "pong",
            "model": "glm-5.2",
            "provider": "zai",
            "completed": True,
        }
        oneshot._annotate_requested_route(
            _FakeAgent("openrouter/zai/glm-5.3", "openrouter", True), result
        )
        return "pong", result

    monkeypatch.setattr(oneshot, "_run_agent", _fake_run_agent)
    monkeypatch.setattr(oneshot, "declare_stateless_channel", lambda *a, **k: None)

    usage = tmp_path / "usage.json"
    oneshot.run_oneshot(
        "ping",
        model="openrouter/zai/glm-5.3",
        provider="openrouter",
        usage_file=str(usage),
    )

    report = json.loads(usage.read_text(encoding="utf-8"))
    assert report["requested_model"] == "openrouter/zai/glm-5.3"
    assert report["requested_provider"] == "openrouter"
    assert report["fallback_activated"] is True
    assert report["model"] == "glm-5.2"


def test_usage_file_marks_an_honored_request_as_not_fallen_back(tmp_path, monkeypatch):
    from hermes_cli import oneshot

    def _fake_run_agent(prompt, **kwargs):
        result = {
            "final_response": "pong",
            "model": "glm-5.3",
            "provider": "zai",
            "completed": True,
        }
        oneshot._annotate_requested_route(_FakeAgent("glm-5.3", "zai", False), result)
        return "pong", result

    monkeypatch.setattr(oneshot, "_run_agent", _fake_run_agent)
    monkeypatch.setattr(oneshot, "declare_stateless_channel", lambda *a, **k: None)

    usage = tmp_path / "usage.json"
    oneshot.run_oneshot("ping", model="glm-5.3", provider="zai", usage_file=str(usage))
    report = json.loads(usage.read_text(encoding="utf-8"))
    assert report["fallback_activated"] is False


# ---------------------------------------------------------------------------
# Agent init: the snapshot is taken, and fallback cannot overwrite it
# ---------------------------------------------------------------------------

def test_fallback_swap_leaves_the_origin_snapshot_intact():
    """try_activate_fallback rewrites requested_provider; not the audit."""
    from agent.chat_completion_helpers import _record_fallback_on_session

    class _Recorder:
        def __init__(self):
            self.calls = []

        def record_session_fallback(self, session_id, **kwargs):
            self.calls.append((session_id, kwargs))

    agent = _FakeAgent("openrouter/zai/glm-5.3", "openrouter", True)
    agent.session_id = "s1"
    agent._session_db = _Recorder()
    _record_fallback_on_session(agent)

    assert agent._session_db.calls == [
        (
            "s1",
            {
                "requested_model": "openrouter/zai/glm-5.3",
                "requested_provider": "openrouter",
            },
        )
    ]


def test_record_fallback_on_session_never_raises():
    """A bookkeeping failure must not abort provider recovery."""
    from agent.chat_completion_helpers import _record_fallback_on_session

    class _Exploding:
        def record_session_fallback(self, *a, **k):
            raise RuntimeError("db is locked")

    agent = _FakeAgent("glm-5.3", "zai", True)
    agent.session_id = "s1"
    agent._session_db = _Exploding()
    _record_fallback_on_session(agent)  # must not raise

    # No session_db / no session_id are also non-events.
    bare = _FakeAgent("glm-5.3", "zai", True)
    bare.session_id = None
    bare._session_db = None
    _record_fallback_on_session(bare)


# ---------------------------------------------------------------------------
# `hermes sessions list` names the divergence
# ---------------------------------------------------------------------------

def test_sessions_list_reports_flagged_rows(capsys):
    from hermes_cli import sessions_cmd

    sessions_cmd._print_fallback_warnings([
        {
            "id": "20260817_191805_ec4afa",
            "model": "glm-5.2",
            "billing_provider": "zai",
            "requested_model": "openrouter/zai/glm-5.3",
            "requested_provider": "openrouter",
            "fallback_activated": 1,
        },
        {"id": "ok", "model": "glm-5.3", "fallback_activated": 0},
    ])
    out = capsys.readouterr().out
    assert "20260817_191805_ec4afa" in out
    assert "openrouter/zai/glm-5.3 (openrouter)" in out
    assert "glm-5.2 (zai)" in out
    assert "ok" not in out.replace("20260817_191805_ec4afa", "")


def test_sessions_list_stays_quiet_when_nothing_fell_back(capsys):
    from hermes_cli import sessions_cmd

    sessions_cmd._print_fallback_warnings(
        [{"id": "s1", "model": "glm-5.3", "fallback_activated": 0}]
    )
    assert capsys.readouterr().out == ""


def test_sessions_list_warns_off_real_listing_rows(db, capsys):
    """Same warning, driven by the real listing query instead of hand-made dicts.

    The tests above feed ``_print_fallback_warnings`` literal dicts, so they
    would keep passing if the listing SELECT stopped carrying the audit
    columns. This walks the whole path: request persisted at creation, served
    route overwriting ``model``, listing row read back, warning printed — and
    then a ``/model`` switch making the warning stop, since it is a new request.
    """
    from hermes_cli import sessions_cmd

    db.create_session(
        "s_listed", source="cli", model="openrouter/zai/glm-5.3",
        requested_model="openrouter/zai/glm-5.3", requested_provider="openrouter",
    )
    db.record_session_fallback("s_listed")
    db.update_token_counts(
        "s_listed", input_tokens=10, output_tokens=5,
        model="glm-5.2", billing_provider="zai", api_call_count=1,
    )
    db.flush_token_counts()

    rows = [s for s in db.list_sessions_rich(limit=10) if s["id"] == "s_listed"]
    assert rows, "the flagged session must be listable"
    sessions_cmd._print_fallback_warnings(rows)
    out = capsys.readouterr().out
    assert "openrouter/zai/glm-5.3 (openrouter)" in out
    assert "glm-5.2 (zai)" in out

    # A provider-less /model switch is a new request: the warning stops, and
    # nothing may reintroduce the abandoned provider as the requested one.
    db.update_session_model("s_listed", "glm-5.4")
    rows = [s for s in db.list_sessions_rich(limit=10) if s["id"] == "s_listed"]
    assert rows[0]["requested_model"] == "glm-5.4"
    assert rows[0]["requested_provider"] is None
    sessions_cmd._print_fallback_warnings(rows)
    assert capsys.readouterr().out == ""


def test_next_process_first_turn_cannot_make_the_warning_lie(db, capsys):
    """The printed warning must never name a route nobody asked for.

    Walks the three writes a real session takes, off a real DB, the real
    listing SELECT and the real printer:

    1. request glm-5.3 via zai, then a provider-less ``/model gpt-5.4`` —
       the audit pair becomes ``gpt-5.4`` / NULL ("no provider requested").
    2. that switched request is abandoned: the flag is raised naming the route
       the swap gave up on (``gpt-5.4`` via the deepseek endpoint that was
       serving it), and the fallback route (grok-4 via xai) bills the turn.
    3. the NEXT process's first turn re-runs ``create_session`` for the same id
       with its own snapshot (``hermes --resume -m glm-5.4 --provider
       minimax`` skips the model restore, so the snapshot need not match the
       row).

    The upsert used to complete the pair's provider half from step 3's
    snapshot, and `hermes sessions list` printed
    ``requested gpt-5.4 (minimax) → served grok-4 (xai)``. Nobody ever
    requested gpt-5.4 via minimax.
    """
    from hermes_cli import sessions_cmd

    db.create_session(
        "s_third_writer", source="cli", model="glm-5.3",
        requested_model="glm-5.3", requested_provider="zai",
    )
    db.update_session_model("s_third_writer", "gpt-5.4")

    db.update_token_counts(
        "s_third_writer", input_tokens=10, output_tokens=5,
        model="grok-4", billing_provider="xai", api_call_count=1,
    )
    db.flush_token_counts()
    db.record_session_fallback(
        "s_third_writer", requested_model="gpt-5.4", requested_provider="deepseek",
    )

    db.create_session(
        "s_third_writer", source="cli", model="glm-5.4",
        requested_model="glm-5.4", requested_provider="minimax",
    )

    rows = [
        s for s in db.list_sessions_rich(limit=10) if s["id"] == "s_third_writer"
    ]
    assert rows, "the flagged session must be listable"
    assert rows[0]["requested_model"] == "gpt-5.4"
    assert rows[0]["requested_provider"] == "deepseek", (
        "the flagged pair names the route the swap abandoned, whole"
    )

    sessions_cmd._print_fallback_warnings(rows)
    out = capsys.readouterr().out
    assert "requested gpt-5.4 (deepseek) → served grok-4 (xai)" in out
    assert "minimax" not in out, (
        "the next process's snapshot may not touch a flagged pair"
    )


def test_provider_only_row_cannot_lend_its_provider_to_a_foreign_model(db, capsys):
    """The mirror of the above: a stored PROVIDER must not adopt a foreign model.

    A provider-only row (``requested_provider`` set, ``requested_model`` NULL)
    is ordinary production state, not a test poke: ``hermes --provider vllm``
    with no ``model.default`` in config leaves ``self.model == ""``, so
    ``agent_init`` snapshots an empty requested model while
    ``requested_provider`` has an ``"auto"`` floor and is effectively never
    empty. The row is written before the first API call (the titler forces
    ``_ensure_db_session``), so it survives even when that model-less request
    400s.

    The next process on that session id then arrives with its own complete
    request. Independent ``COALESCE``s stitched THAT process's model onto the
    stored provider and `hermes sessions list` printed

        s1  requested glm-5.4 (vllm) → served grok-4 (xai)

    Nobody ever requested that: the first process asked for vllm with no model,
    the second asked for glm-5.4 via minimax. Double harm — the stale ``vllm``
    also suppressed the correct ``minimax`` the backfill would have supplied.
    """
    from hermes_cli import sessions_cmd

    # First process: `hermes --provider vllm` with no default model.
    db.create_session(
        "s_provider_only", source="cli", model="",
        requested_model=None, requested_provider="vllm",
    )
    row = db.get_session("s_provider_only")
    assert row["requested_model"] is None
    assert row["requested_provider"] == "vllm"

    # Next process's first turn: a whole request of its own.
    db.create_session(
        "s_provider_only", source="cli", model="glm-5.4",
        requested_model="glm-5.4", requested_provider="minimax",
    )
    # ...which is not honored: another route serves and bills the turn.
    db.update_token_counts(
        "s_provider_only", input_tokens=10, output_tokens=5,
        model="grok-4", billing_provider="xai", api_call_count=1,
    )
    db.flush_token_counts()
    db.record_session_fallback(
        "s_provider_only", requested_model="glm-5.4", requested_provider="minimax",
    )

    rows = [
        s for s in db.list_sessions_rich(limit=10)
        if s["id"] == "s_provider_only"
    ]
    assert rows, "the flagged session must be listable"
    assert rows[0]["requested_model"] == "glm-5.4"
    assert rows[0]["requested_provider"] == "minimax", (
        "the model came from this snapshot, so the provider must too"
    )

    sessions_cmd._print_fallback_warnings(rows)
    out = capsys.readouterr().out
    assert "requested glm-5.4 (minimax) → served grok-4 (xai)" in out
    assert "vllm" not in out


def test_fallback_backfill_alone_cannot_mispair_a_provider_only_row(db):
    """The same mis-pairing arises purely inside ``record_session_fallback``.

    No second ``create_session`` needed: the process that starts provider-only
    and falls back on its very first turn backfills through this writer, whose
    model half was an ungated ``COALESCE`` too.
    """
    db.create_session(
        "s_backfill_only", source="cli", model="",
        requested_model=None, requested_provider="vllm",
    )
    db.record_session_fallback(
        "s_backfill_only", requested_model="glm-5.4", requested_provider="minimax",
    )
    row = db.get_session("s_backfill_only")
    assert row["fallback_activated"] == 1
    assert row["requested_model"] == "glm-5.4"
    assert row["requested_provider"] == "minimax"


# ---------------------------------------------------------------------------
# The flag is part of the record: a raised flag is a verdict ON the stored pair
# ---------------------------------------------------------------------------

def _provider_only_flagged_then_honored_run(db, *, account_the_abandoned_route):
    """Set up the two-process history both flag tests share.

    P1 = ``hermes --provider vllm`` with no ``model.default``: a provider-only
    row is written before the first API call, that model-less request is
    abandoned by ``try_activate_fallback`` (flag up), and the fallback route
    either does or does not manage to bill a turn.

    P2 = ``hermes -c -m glm-5.4 --provider minimax`` on the same session id,
    honored end to end: its own request is what serves and bills.
    """
    db.create_session(
        "s_flagged", source="cli", model="",
        requested_model=None, requested_provider="vllm",
    )
    db.record_session_fallback(
        "s_flagged", requested_model=None, requested_provider="vllm",
    )
    flagged = db.get_session("s_flagged")
    assert (
        flagged["requested_model"],
        flagged["requested_provider"],
        flagged["fallback_activated"],
    ) == (None, "vllm", 1), "P1 must leave a flagged provider-only row"

    if account_the_abandoned_route:
        db.update_token_counts(
            "s_flagged", input_tokens=10, output_tokens=5,
            model="grok-4", billing_provider="xai", api_call_count=1,
        )
        db.flush_token_counts()

    # P2's first turn re-runs create_session with its own start-of-run snapshot.
    db.create_session(
        "s_flagged", source="cli", model="glm-5.4",
        requested_model="glm-5.4", requested_provider="minimax",
    )
    db.update_token_counts(
        "s_flagged", input_tokens=10, output_tokens=5,
        model="glm-5.4", billing_provider="minimax", api_call_count=1,
    )
    db.flush_token_counts()

    rows = [s for s in db.list_sessions_rich(limit=10) if s["id"] == "s_flagged"]
    assert rows, "the flagged session must be listable"
    return rows


def test_flagged_row_does_not_cry_wolf_about_a_request_that_was_honored(db, capsys):
    """A raised flag may not be handed to a request nobody abandoned.

    The upsert learned (round 3) to let a snapshot's whole pair supersede a
    provider-only row's request — correct while the row carries no verdict, but
    that row is exactly the shape ``hermes --provider vllm`` leaves behind when
    its model-less request is the one that got abandoned, and the proof sits in
    the same row: ``fallback_activated = 1``. Superseding the pair without
    reading the flag left P1's verdict standing over P2's request, and
    `hermes sessions list` printed

        s1  requested glm-5.4 (minimax) → served glm-5.4 (minimax)

    announcing that a session "ran a model other than the one requested" about
    a request whose requested and served routes are character-for-character
    identical — the wolf-cry the sticky flag exists to prevent. It also erased
    ``vllm``, the provider of the request that actually WAS abandoned, from the
    row and from the output; before round 3 it was at least still printed.
    """
    from hermes_cli import sessions_cmd

    rows = _provider_only_flagged_then_honored_run(
        db, account_the_abandoned_route=False
    )
    assert (
        rows[0]["requested_model"],
        rows[0]["requested_provider"],
        rows[0]["fallback_activated"],
    ) == (None, "vllm", 1), (
        "the flag is a verdict on P1's request, so P1's request must stay"
    )

    sessions_cmd._print_fallback_warnings(rows)
    out = capsys.readouterr().out
    # The truth: the vllm request that named no model was abandoned, and the
    # session went on to run glm-5.4 via minimax.
    assert "requested vllm → served glm-5.4 (minimax)" in out
    assert "requested glm-5.4 (minimax) → served glm-5.4 (minimax)" not in out


def test_flagged_row_keeps_the_provider_of_the_request_that_was_abandoned(db, capsys):
    """Same history, but the abandoned route billed a turn before P2 arrived.

    The wolf-cry above needs the served columns to coincide with P2's request;
    with P1's fallback route accounted, the served half is grok-4 via xai and
    the printed line stopped looking self-contradictory while saying something
    worse: ``requested glm-5.4 (minimax) → served grok-4 (xai)`` asserts that
    P2's request — honored end to end — was not honored. The flag belongs to
    P1's vllm request, and so must the requested half.
    """
    from hermes_cli import sessions_cmd

    rows = _provider_only_flagged_then_honored_run(
        db, account_the_abandoned_route=True
    )
    assert (
        rows[0]["requested_model"],
        rows[0]["requested_provider"],
        rows[0]["fallback_activated"],
    ) == (None, "vllm", 1)

    sessions_cmd._print_fallback_warnings(rows)
    out = capsys.readouterr().out
    assert "requested vllm → served grok-4 (xai)" in out
    assert "requested glm-5.4 (minimax)" not in out


def test_a_second_fallback_keeps_the_first_abandoned_request(db, capsys):
    """Two abandoned requests, one pair: the row keeps the FIRST, and says so.

    P1 = ``hermes --provider vllm`` with no ``model.default``, abandoned. P2 =
    ``hermes -c -m glm-5.4 --provider minimax`` on the same id, also abandoned.
    Both requests really were given up on, and three columns can name one, so
    the row goes on naming the request the standing verdict is about.

    Round 5 note — this test used to assert the opposite (``glm-5.4``/``minimax``
    replacing ``vllm``), on the reasoning that "the verdict and the request it
    judges always move as one record", i.e. that this writer may re-subject a
    standing verdict because its own call site asserts an abandonment. That
    licence is exactly what let round 4's SQL pin P2's verdict on P1's *honored*
    request in the mirror history, and inside one process it also makes a chain
    hop overwrite the operator's request with an intermediate fallback route
    (pinned by ``test_a_multi_hop_chain_keeps_the_request_it_started_from``). The
    rule is now uniform for both snapshot-carrying writers — a standing verdict
    keeps its subject — so the choice between two truthfully abandoned requests
    is settled by which one was judged first.

    What must NOT happen, and is what this test still guards, is silence: P2's
    fallback is real, so the flag stays up, the session stays listed, and the
    printed line names a request that genuinely was abandoned. The cost is that
    P2's route is not recorded anywhere — a documented boundary of a
    single-pair schema, see ``_insert_session_row``'s residual note.
    """
    from hermes_cli import sessions_cmd

    db.create_session(
        "s_two_falls", source="cli", model="",
        requested_model=None, requested_provider="vllm",
    )
    db.record_session_fallback(
        "s_two_falls", requested_model=None, requested_provider="vllm",
    )
    db.create_session(
        "s_two_falls", source="cli", model="glm-5.4",
        requested_model="glm-5.4", requested_provider="minimax",
    )
    # P2's own request is not honored either: it falls back to grok-4 via xai.
    db.update_token_counts(
        "s_two_falls", input_tokens=10, output_tokens=5,
        model="grok-4", billing_provider="xai", api_call_count=1,
    )
    db.flush_token_counts()
    db.record_session_fallback(
        "s_two_falls", requested_model="glm-5.4", requested_provider="minimax",
    )

    rows = [s for s in db.list_sessions_rich(limit=10) if s["id"] == "s_two_falls"]
    assert (
        rows[0]["requested_model"],
        rows[0]["requested_provider"],
        rows[0]["fallback_activated"],
    ) == (None, "vllm", 1)
    sessions_cmd._print_fallback_warnings(rows)
    out = capsys.readouterr().out
    # Reported, and reported about a request that really was abandoned.
    assert "requested vllm → served grok-4 (xai)" in out


def test_the_guard_does_not_drop_the_provider_from_a_reasserted_request(db):
    """The round-2 completion still reaches the warning, via the right writer.

    A flagged ``model only`` row plus a snapshot naming that very model is the
    case round 2 added the completion arm for: dropping the provider would print
    ``requested glm-5.3 → served ...`` for a session that genuinely re-requested
    glm-5.3 through a known provider. The upsert now declines it — the stored
    verdict is about the provider-less request the row records — but the
    fallback backfill supplies it the moment the re-request is itself abandoned,
    which is the only moment at which the completed pair is a true statement.
    """
    db.create_session(
        "s_complete", source="cli", model="glm-5.3",
        requested_model="glm-5.3", requested_provider=None,
    )
    db.record_session_fallback("s_complete")  # flag only, pair untouched

    # Next process's first turn re-requests glm-5.3, this time naming zai.
    db.create_session(
        "s_complete", source="cli", model="glm-5.3",
        requested_model="glm-5.3", requested_provider="zai",
    )
    frozen = db.get_session("s_complete")
    assert frozen["requested_provider"] is None, (
        "while the flag is up, the verdict's own request may not be re-labelled"
    )

    # ...and that re-request is abandoned too, so the pair is restated whole.
    db.record_session_fallback(
        "s_complete", requested_model="glm-5.3", requested_provider="zai",
    )
    row = db.get_session("s_complete")
    assert row["requested_model"] == "glm-5.3"
    assert row["requested_provider"] == "zai"
    assert row["fallback_activated"] == 1


def test_a_bare_record_session_fallback_raises_only_the_flag(db):
    """The flag-only setup the table tests rely on must really be flag-only.

    ``record_session_fallback(sid)`` with no snapshot is how a caller with no
    request knowledge flags a row (and how ``_record_fallback_on_session``
    behaves when the origin snapshot is empty). It must leave the pair exactly
    as it stands from every row state — otherwise the flagged rows in the table
    below would not be the row states they claim to be.
    """
    for name, pair in (
        ("neither", (None, None)),
        ("model only", ("gpt-5.4", None)),
        ("provider only", (None, "vllm")),
        ("both", ("gpt-5.4", "vllm")),
    ):
        sid = f"s_bare_{name.replace(' ', '_')}"
        db.create_session(
            sid, source="cli", model="glm-5.2",
            requested_model=pair[0], requested_provider=pair[1],
        )
        db.record_session_fallback(sid)
        row = db.get_session(sid)
        assert (row["requested_model"], row["requested_provider"]) == pair, name
        assert row["fallback_activated"] == 1, name


# ---------------------------------------------------------------------------
# The mirror history: P1 honored, P2 abandoned
# ---------------------------------------------------------------------------

def _requested_and_served(out):
    """Split every printed warning line into its two rendered routes."""
    pairs = []
    for line in out.splitlines():
        if "requested " not in line or " → served " not in line:
            continue
        head, served = line.split(" → served ", 1)
        pairs.append((head.split("requested ", 1)[1].strip(), served.strip()))
    return pairs


#: P2's request, in each of the three shapes it can take relative to P1's
#: honored ``glm-5.4 (minimax)``: both halves different, only the provider
#: different, and no model requested at all (``hermes --provider vllm`` with no
#: ``model.default``).
_MIRROR_P2_REQUESTS = [
    ("both halves differ", ("grok-4", "xai"), "grok-4 (xai)"),
    ("only the provider differs", ("glm-5.4", "openrouter"), "glm-5.4 (openrouter)"),
    ("no model requested", (None, "vllm"), "vllm"),
]


def _honored_then_abandoned_run(db, p2_request):
    """P1 honored end to end; P2, on the same session id, is abandoned.

    The exact mirror of ``_provider_only_flagged_then_honored_run``:

    * P1 = ``hermes -m glm-5.4 --provider minimax`` — served by the model it
      asked for, and it bills the turn, so the served columns are P1's route.
    * P2 = ``hermes -c …`` on the same session id. Its first turn re-runs
      ``create_session`` with its own start-of-run snapshot, and then
      ``try_activate_fallback`` abandons *that* request — so
      ``record_session_fallback`` is called naming the route P2 was serving
      when it gave up on it, which for a process that never ran ``/model`` is
      P2's own request.
    """
    db.create_session(
        "s_mirror", source="cli", model="glm-5.4",
        requested_model="glm-5.4", requested_provider="minimax",
    )
    db.update_token_counts(
        "s_mirror", input_tokens=10, output_tokens=5,
        model="glm-5.4", billing_provider="minimax", api_call_count=1,
    )
    db.flush_token_counts()
    honored = db.get_session("s_mirror")
    assert (
        honored["requested_model"],
        honored["requested_provider"],
        honored["fallback_activated"],
    ) == ("glm-5.4", "minimax", 0), "P1 must be an honored, unflagged request"

    db.create_session(
        "s_mirror", source="cli", model=p2_request[0] or "",
        requested_model=p2_request[0], requested_provider=p2_request[1],
    )
    db.record_session_fallback(
        "s_mirror",
        requested_model=p2_request[0],
        requested_provider=p2_request[1],
    )

    rows = [s for s in db.list_sessions_rich(limit=10) if s["id"] == "s_mirror"]
    assert rows, "the flagged session must be listable"
    return rows


@pytest.mark.parametrize(
    "label,p2_request,rendered",
    _MIRROR_P2_REQUESTS,
    ids=[label.replace(" ", "-") for label, _r, _s in _MIRROR_P2_REQUESTS],
)
def test_flagged_row_names_the_request_p2_actually_abandoned(
    db, capsys, label, p2_request, rendered
):
    """The raised flag must keep the request it judges — P2's, not P1's.

    Round 4 froze a flagged pair against ``create_session``'s snapshot, which
    fixed the history where P1 was abandoned and P2 honored. This is its
    mirror, and it was untouched: P1 is honored (so the row carries P1's
    complete pair with the flag DOWN) and P2's request is the one
    ``try_activate_fallback`` abandons. The backfill's "keep a recorded model"
    arms declined P2's pair while the ``= 1`` in the same statement pinned P2's
    verdict on P1's honored request, and `hermes sessions list` printed

        s1  requested glm-5.4 (minimax) → served glm-5.4 (minimax)

    — a warning whose two routes are character-identical, about a request that
    was honored end to end, while the route P2 actually asked for appeared
    nowhere.
    """
    from hermes_cli import sessions_cmd

    rows = _honored_then_abandoned_run(db, p2_request)
    assert (
        rows[0]["requested_model"],
        rows[0]["requested_provider"],
        rows[0]["fallback_activated"],
    ) == (p2_request[0], p2_request[1], 1), (
        "the flag is a verdict on P2's request, so P2's request must be stored"
    )

    sessions_cmd._print_fallback_warnings(rows)
    out = capsys.readouterr().out
    assert f"requested {rendered} → served glm-5.4 (minimax)" in out
    assert "requested glm-5.4 (minimax) → served glm-5.4 (minimax)" not in out
    for requested, served in _requested_and_served(out):
        assert requested != served, (
            "a warning that a session ran a model other than the one requested "
            "may not name the same route twice"
        )


# ---------------------------------------------------------------------------
# The call site: the request recorded is the route the swap ABANDONED
# ---------------------------------------------------------------------------

def _mock_fb_client(base_url="https://fb.example/v1"):
    from unittest.mock import MagicMock

    client = MagicMock()
    client.base_url = base_url
    client.api_key = "fb-key"
    return client


def _agent_with_chain(chain, *, model, provider, db, session_id):
    """A real ``AIAgent`` with a real fallback chain, wired to a real DB.

    Built the way ``tests/run_agent/test_provider_fallback.py`` builds one, so
    ``_try_activate_fallback`` runs its real body — including the audit write —
    with no network.
    """
    from unittest.mock import MagicMock, patch

    from run_agent import AIAgent

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            model=model,
            provider=provider,
            requested_model=model,
            requested_provider=provider,
            fallback_model=chain,
        )
    agent.client = MagicMock()
    agent._session_db = db
    agent.session_id = session_id
    return agent


def _hop(agent, resolved_model):
    """Run one real chain hop with the client resolution mocked out."""
    from unittest.mock import patch

    with patch(
        "agent.auxiliary_client.resolve_provider_client",
        return_value=(_mock_fb_client(), resolved_model),
    ):
        assert agent._try_activate_fallback() is True


def test_the_audit_records_the_route_the_swap_abandoned_not_the_process_start(db):
    """``_record_fallback_on_session`` must name the LIVE pre-swap route.

    ``try_activate_fallback`` captures ``old_model``/``old_provider`` before it
    reassigns them, and calls the audit write after. Passing
    ``origin_requested_*`` instead — the immutable process-start snapshot —
    names the request that failed only while nothing has moved the live route
    since; after a ``/model`` switch it names a request that stopped being
    current, and the flag beside it then judges the wrong one.
    """
    from agent.chat_completion_helpers import abandoned_route_for_audit

    agent = _agent_with_chain(
        [{"provider": "xai", "model": "grok-4"}],
        model="glm-5.4", provider="minimax", db=db, session_id="s_live",
    )
    # A mid-session /model switch moves the live route; the origin snapshot is
    # frozen at process start on purpose and must NOT be what gets recorded.
    agent.model = "gpt-5.4"
    agent.provider = "deepseek"
    assert (agent.origin_requested_model, agent.origin_requested_provider) == (
        "glm-5.4", "minimax"
    )

    db.create_session("s_live", source="cli", model="gpt-5.4",
                      requested_model="gpt-5.4", requested_provider=None)
    _hop(agent, "grok-4")

    assert agent._fallback_abandoned_route == ("gpt-5.4", "deepseek")
    assert abandoned_route_for_audit(agent) == ("gpt-5.4", "deepseek")
    row = db.get_session("s_live")
    assert (
        row["requested_model"], row["requested_provider"], row["fallback_activated"]
    ) == ("gpt-5.4", "deepseek", 1)


def test_switch_then_fallback_records_the_switched_route_end_to_end(db, capsys):
    """A ``/model`` switch then a fallback: the warning names the switch.

    The whole path with no stubs between the swap and the printer: real agent,
    real ``update_session_model``, real ``try_activate_fallback``, real listing
    SELECT, real printer.
    """
    from hermes_cli import sessions_cmd

    db.create_session("s_switch_e2e", source="cli", model="glm-5.3",
                      requested_model="glm-5.3", requested_provider="zai")
    agent = _agent_with_chain(
        [{"provider": "xai", "model": "grok-4"}],
        model="glm-5.3", provider="zai", db=db, session_id="s_switch_e2e",
    )
    # `/model gpt-5.4` — provider-less, so the audit records "no provider
    # requested" and the flag clears.
    db.update_session_model("s_switch_e2e", "gpt-5.4")
    agent.model = "gpt-5.4"
    agent.provider = "deepseek"
    assert db.get_session("s_switch_e2e")["fallback_activated"] == 0, (
        "a new explicit request clears any verdict on the old one"
    )

    _hop(agent, "grok-4")
    db.update_token_counts(
        "s_switch_e2e", input_tokens=10, output_tokens=5,
        model="grok-4", billing_provider="xai", api_call_count=1,
    )
    db.flush_token_counts()

    rows = [s for s in db.list_sessions_rich(limit=10)
            if s["id"] == "s_switch_e2e"]
    sessions_cmd._print_fallback_warnings(rows)
    out = capsys.readouterr().out
    assert "requested gpt-5.4 (deepseek) → served grok-4 (xai)" in out
    assert "glm-5.3" not in out, "the superseded process start is not the request"


def test_a_multi_hop_chain_keeps_the_request_it_started_from(db, capsys):
    """Hop 2 abandons hop 1's fallback; the row must still name the operator's.

    Every successful hop calls ``record_session_fallback``, so "the incoming
    route always wins" would walk the recorded request down the chain and end up
    naming ``grok-4 (xai)`` — a route the operator never asked for — as the
    unhonored request. The verdict raised by hop 1 keeps its subject, so the row
    goes on naming ``glm-5.4 (minimax)`` while the served columns follow the
    route that finally billed.
    """
    from hermes_cli import sessions_cmd

    db.create_session("s_chain", source="cli", model="glm-5.4",
                      requested_model="glm-5.4", requested_provider="minimax")
    agent = _agent_with_chain(
        [{"provider": "xai", "model": "grok-4"},
         {"provider": "openai", "model": "gpt-5.4"}],
        model="glm-5.4", provider="minimax", db=db, session_id="s_chain",
    )

    _hop(agent, "grok-4")
    assert (agent.model, agent.provider) == ("grok-4", "xai")
    first = db.get_session("s_chain")
    assert (first["requested_model"], first["requested_provider"]) == (
        "glm-5.4", "minimax"
    )

    _hop(agent, "gpt-5.4")
    assert (agent.model, agent.provider) == ("gpt-5.4", "openai")
    db.update_token_counts(
        "s_chain", input_tokens=10, output_tokens=5,
        model="gpt-5.4", billing_provider="openai", api_call_count=1,
    )
    db.flush_token_counts()

    row = db.get_session("s_chain")
    assert (
        row["requested_model"], row["requested_provider"], row["fallback_activated"]
    ) == ("glm-5.4", "minimax", 1)

    rows = [s for s in db.list_sessions_rich(limit=10) if s["id"] == "s_chain"]
    sessions_cmd._print_fallback_warnings(rows)
    out = capsys.readouterr().out
    assert "requested glm-5.4 (minimax) → served gpt-5.4 (openai)" in out
    assert "grok-4" not in out, (
        "an intermediate chain route is not a request anybody made"
    )


def test_a_fallback_on_the_very_first_turn_is_re_applied_with_that_route(db):
    """The row is created lazily, so the first-turn swap finds no row to flag.

    ``_ensure_db_session`` re-applies the flag right after the INSERT, and it
    must re-apply the SAME request the swap named — the route it abandoned —
    which it reads through the one shared helper so the two call sites cannot
    drift apart.
    """
    from agent.chat_completion_helpers import abandoned_route_for_audit

    agent = _agent_with_chain(
        [{"provider": "xai", "model": "grok-4"}],
        model="glm-5.4", provider="minimax", db=db, session_id="s_lazy",
    )
    # No row yet: the UPDATE is a silent no-op, as in production.
    _hop(agent, "grok-4")
    assert db.get_session("s_lazy") is None

    # _ensure_db_session inserts the row with the process-start snapshot...
    db.create_session("s_lazy", source="cli", model="grok-4",
                      requested_model="glm-5.4", requested_provider="minimax")
    assert db.get_session("s_lazy")["fallback_activated"] == 0
    # ...and re-applies the flag with the abandoned route.
    _req_model, _req_provider = abandoned_route_for_audit(agent)
    db.record_session_fallback(
        "s_lazy", requested_model=_req_model, requested_provider=_req_provider,
    )

    row = db.get_session("s_lazy")
    assert (
        row["requested_model"], row["requested_provider"], row["fallback_activated"]
    ) == ("glm-5.4", "minimax", 1)


def test_abandoned_route_falls_back_to_the_origin_snapshot(db):
    """With no swap-recorded route, the origin snapshot is the best available.

    A caller that flags a row without ever going through
    ``try_activate_fallback`` (older call paths, and the empty-attribute case)
    still names the only request it knows. Both halves empty means "no route to
    name", which leaves the stored pair untouched.
    """
    from agent.chat_completion_helpers import abandoned_route_for_audit

    agent = _FakeAgent("glm-5.4", "minimax", True)
    assert abandoned_route_for_audit(agent) == ("glm-5.4", "minimax")

    blank = _FakeAgent("", "", True)
    assert abandoned_route_for_audit(blank) == (None, None)


# ---------------------------------------------------------------------------
# Two-process histories: where the row's pair came from is part of the state
# ---------------------------------------------------------------------------

#: Every non-empty request a process can make, as the CLI can produce it:
#: a model and a provider, a model with the provider unresolvable, and
#: `--provider X` with no ``model.default`` (which snapshots an empty model
#: while the provider half has an "auto" floor).
_TWO_PROCESS_MODELS = (None, "glm-5.4", "grok-4")
_TWO_PROCESS_PROVIDERS = (None, "minimax", "xai")
_TWO_PROCESS_REQUESTS = [
    (m, p)
    for m in _TWO_PROCESS_MODELS
    for p in _TWO_PROCESS_PROVIDERS
    if not (m is None and p is None)
]

#: (P1 request) x (P2 request) x (who falls back) x (was the fallback route
#: billed) = 8 x 8 x 3 x 2 = 384 histories.
_TWO_PROCESS_FALLERS = ((1,), (2,), (1, 2))


def _fallback_route(process):
    """The route the chain switched TO — distinct from every request above."""
    return (f"fb{process}-model", f"fb{process}prov")


def _honored_route(request, process):
    """What a request that IS honored ends up billing.

    A request naming no model still bills under some concrete model (the
    provider picks one), which is why the served columns can never be read as
    "the request".
    """
    return (request[0] or f"resolved{process}", request[1] or f"resolvedprov{process}")


def _play_two_process_history(db, sid, r1, r2, fallers, billed):
    """Write one two-process history through the real writers, in real order.

    P<i> = a process whose start-of-run snapshot is r<i>. Its first turn runs
    ``create_session`` (the ON CONFLICT upsert for P2), and then either it is
    honored — its own route bills the turn — or ``try_activate_fallback``
    abandons it, which raises the flag naming the route abandoned (the live
    pre-swap route, which with no ``/model`` switch in play is that process's
    own request) and the fallback route bills only when *billed*.
    """
    for process, request in ((1, r1), (2, r2)):
        db.create_session(
            sid, source="cli", model=request[0] or "",
            requested_model=request[0], requested_provider=request[1],
        )
        if process in fallers:
            db.record_session_fallback(
                sid, requested_model=request[0], requested_provider=request[1],
            )
            served = _fallback_route(process) if billed else None
        else:
            served = _honored_route(request, process)
        if served is not None:
            db.update_token_counts(
                sid, input_tokens=10, output_tokens=5,
                model=served[0], billing_provider=served[1], api_call_count=1,
            )
    db.flush_token_counts()


def test_two_process_history_sweep(db, capsys):
    """384 two-process histories: a raised flag always names an abandoned request.

    This is the axis rounds 1-4 could not see. Their sweeps judged each write
    against ``{row state, snapshot}`` and never asked where the row's pair came
    from, so a triple that was "one whole record in play" locally could still be
    P1's honored request wearing P2's verdict — which is exactly the shape of the
    round-5 lie, and it scored clean on the round-4 metric.

    Two properties, over every history:

    1. the flag is up if and only if some process's request was abandoned; and
    2. when it is up, the stored pair names a request that really WAS abandoned
       (the one the standing verdict is about, possibly completed by a later
       abandonment of the same route) — never one that was honored end to end.

    The printed line is then checked for the wolf-cry: requested and served
    rendering to the same string. Those are enumerated rather than banned,
    because a session id spanning two processes can legitimately have abandoned
    a route that a later process then served — a boundary of one served column
    per session, not a lie. Every such history is asserted to be of exactly that
    shape.
    """
    from hermes_cli import sessions_cmd

    histories = {}
    for r1 in _TWO_PROCESS_REQUESTS:
        for r2 in _TWO_PROCESS_REQUESTS:
            for fallers in _TWO_PROCESS_FALLERS:
                for billed in (False, True):
                    sid = f"h{len(histories):03d}"
                    histories[sid] = (r1, r2, fallers, billed)
                    _play_two_process_history(db, sid, r1, r2, fallers, billed)
    assert len(histories) == 384, "the history space must be swept whole"

    rows = {
        s["id"]: s
        for s in db.list_sessions_rich(limit=1000)
        if s["id"] in histories
    }
    assert len(rows) == 384, "every history must be listable"

    identical = []
    labels = {}
    for sid, (r1, r2, fallers, billed) in histories.items():
        row = rows[sid]
        pair = (row["requested_model"], row["requested_provider"])
        abandoned = {(r1, r2)[i - 1] for i in fallers}

        assert row["fallback_activated"] == 1, (
            f"{sid}: a real fallback must never go unreported ({r1} {r2} "
            f"{fallers} billed={billed})"
        )
        assert pair in abandoned, (
            f"{sid}: the flag names {pair}, which was not abandoned in this "
            f"history (P1={r1} P2={r2} fell back={fallers})"
        )
        # A single write never mixes halves, and neither does a history.
        assert pair[0] is None or pair[0] in _TWO_PROCESS_MODELS
        assert pair[1] is None or pair[1] in _TWO_PROCESS_PROVIDERS

        served = (row["model"] or None, row["billing_provider"] or None)
        labels[sid] = (
            sessions_cmd._format_route(*pair),
            sessions_cmd._format_route(*served),
        )
        if labels[sid][0] == labels[sid][1]:
            identical.append(sid)

    # The printer is fed all 384 rows at once; every line it emits must be one
    # its row's own columns render to — nothing invented in the renderer.
    sessions_cmd._print_fallback_warnings(list(rows.values()))
    out = capsys.readouterr().out
    printed = _requested_and_served(out)
    assert len(printed) == 384, "every flagged row must get exactly one line"
    assert set(printed) <= set(labels.values())

    # Documented boundary, enumerated rather than banned: the histories whose
    # line renders two identical routes, and why each is still literally true.
    assert identical, "the boundary case must actually be reachable"
    for sid in identical:
        r1, r2, fallers, billed = histories[sid]
        row = rows[sid]
        pair = (row["requested_model"], row["requested_provider"])
        if row["billing_provider"] is None:
            # Nothing was ever accounted (both processes fell back and neither
            # fallback route billed), so `model` still holds create_session's
            # seed and the row has no served route at all. Pre-existing property
            # of the served columns — `model` is seeded at creation and only
            # reconciled by the first accounted call — not of the audit pair.
            assert fallers == (1, 2) and not billed, sid
        else:
            # The same route was abandoned by one process and honored by the
            # other. Both halves of the line are true; only a per-process
            # record could tell the two turns apart.
            assert _honored_route(r1, 1) == pair or _honored_route(r2, 2) == pair, (
                f"{sid}: identical routes must come from a request that was "
                f"abandoned once and served once, got P1={r1} P2={r2} "
                f"fell back={fallers}"
            )


def test_a_switch_between_fallbacks_re_captures_the_abandoned_route(db, capsys):
    """A new request must not be judged by an older episode's memo.

    The abandoned route is remembered once per fallback EPISODE so a chain hop
    cannot overwrite the operator's request. It must NOT be remembered once per
    process: after the primary is restored (or ``/model`` switches, both of which
    reset ``_fallback_activated``) the current request is a different one, and a
    later swap abandons THAT. Replaying the first episode's route would flag a
    request the operator has already moved on from — and, because the switch
    cleared the row's verdict, the stale pair would be adopted whole.
    """
    from hermes_cli import sessions_cmd

    db.create_session("s_episodes", source="cli", model="glm-5.4",
                      requested_model="glm-5.4", requested_provider="minimax")
    agent = _agent_with_chain(
        [{"provider": "xai", "model": "grok-4"},
         {"provider": "openai", "model": "gpt-5.4"}],
        model="glm-5.4", provider="minimax", db=db, session_id="s_episodes",
    )

    # Episode 1 abandons the operator's original request.
    _hop(agent, "grok-4")
    assert agent._fallback_abandoned_route == ("glm-5.4", "minimax")

    # The primary is restored and the operator switches: a new request, so the
    # row's verdict clears (real update_session_model).
    agent._fallback_activated = False
    agent._fallback_index = 0
    agent.model = "gpt-5.4"
    agent.provider = "deepseek"
    db.update_session_model("s_episodes", "gpt-5.4", provider="deepseek")
    assert db.get_session("s_episodes")["fallback_activated"] == 0

    # Episode 2 abandons the SWITCHED request.
    _hop(agent, "grok-4")
    assert agent._fallback_abandoned_route == ("gpt-5.4", "deepseek")
    db.update_token_counts(
        "s_episodes", input_tokens=10, output_tokens=5,
        model="grok-4", billing_provider="xai", api_call_count=1,
    )
    db.flush_token_counts()

    row = db.get_session("s_episodes")
    assert (
        row["requested_model"], row["requested_provider"], row["fallback_activated"]
    ) == ("gpt-5.4", "deepseek", 1)
    rows = [s for s in db.list_sessions_rich(limit=10) if s["id"] == "s_episodes"]
    sessions_cmd._print_fallback_warnings(rows)
    assert "requested gpt-5.4 (deepseek) → served grok-4 (xai)" in (
        capsys.readouterr().out
    )
