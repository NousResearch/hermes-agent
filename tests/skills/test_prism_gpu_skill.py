"""Tests for the prism-gpu optional skill.

Stdlib, pytest and unittest.mock only. Nothing here touches the network, a
wallet, or the escrow contract: the two SDK-backed scripts are imported against
a stub ``prismnetwork`` module so their pure logic is exercised without the
dependency being installed in CI.
"""

import importlib.util
import json
import re
import sys
import types
from pathlib import Path
from unittest import mock

import pytest
import yaml

SKILL_DIR = Path(__file__).resolve().parents[2] / "optional-skills" / "mlops" / "prism-gpu"
SKILL_MD = SKILL_DIR / "SKILL.md"
SCRIPTS = SKILL_DIR / "scripts"

SECTIONS = [
    "## When to Use",
    "## Prerequisites",
    "## How to Run",
    "## Quick Reference",
    "## Procedure",
    "## Pitfalls",
    "## Verification",
]


def load_script(name, stub_sdk=False):
    """Import a skill script by path, optionally behind a stub SDK."""
    path = SCRIPTS / name
    spec = importlib.util.spec_from_file_location(f"prism_gpu_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    patched = {"prismnetwork": _stub_sdk()} if stub_sdk else {}
    with mock.patch.dict(sys.modules, patched):
        spec.loader.exec_module(module)
    return module


class _PrismError(Exception):
    """The SDK's error, reproduced: ``broadcast`` defaults to None, which is the
    value the real one raises with when a node grants gateway access."""

    def __init__(self, status, code, body=None, broadcast=None):
        super().__init__(f"prism {status}: {code}")
        self.status = status
        self.code = code
        self.body = body
        self.broadcast = broadcast


def _stub_sdk():
    """Just enough of prismnetwork for the scripts' imports to resolve."""
    stub = types.ModuleType("prismnetwork")
    stub.DEFAULT_ESCROW = "0xfD4228eEEfC49e4b76A0CD40af9fdd546220B2FD"
    stub.DEFAULT_IMAGE = "docker.io/ollama/ollama@sha256:" + "a" * 64
    stub.BudgetError = type("BudgetError", (Exception,), {})
    stub.PrismError = _PrismError
    stub.PrismAgent = mock.MagicMock()
    stub.SpendLedger = mock.MagicMock()
    stub.call_ceiling = mock.MagicMock()
    stub.read_budget = mock.MagicMock()
    stub.record_spend = mock.MagicMock()
    return stub


@pytest.fixture(scope="module")
def frontmatter():
    content = SKILL_MD.read_text(encoding="utf-8")
    assert content.startswith("---"), "frontmatter must start at byte 0"
    end = re.search(r"\n---\s*\n", content[3:])
    assert end, "frontmatter must close with a --- line"
    return yaml.safe_load(content[3:end.start() + 3]), content[end.end() + 3:]


def test_description_meets_the_hardline(frontmatter):
    meta, _ = frontmatter
    description = meta["description"]
    assert len(description) <= 60, f"description is {len(description)} chars"
    assert description.endswith(".")
    assert description.count(".") == 1, "one sentence"
    assert "prism-gpu" not in description.lower()
    assert re.search(r"\bGPU\b", description), "say what it rents"


def test_required_frontmatter_fields(frontmatter):
    meta, _ = frontmatter
    for field in ("name", "version", "author", "license", "platforms"):
        assert field in meta, f"missing {field}"
    assert meta["name"] == "prism-gpu"
    assert meta["author"] == "Mika Winter (winterstacks)"
    assert set(meta["platforms"]) <= {"linux", "macos", "windows"}
    assert meta["metadata"]["hermes"]["tags"]


def test_dependency_declares_an_upper_bound(frontmatter):
    meta, _ = frontmatter
    for spec in meta["dependencies"]:
        assert ",<" in spec, f"{spec} needs a next-major ceiling"


def test_body_uses_the_modern_section_order(frontmatter):
    _, body = frontmatter
    positions = [body.index(section) for section in SECTIONS]
    assert positions == sorted(positions), "sections are out of order"


def test_body_names_no_machine_local_paths(frontmatter):
    _, body = frontmatter
    assert "/Users/" not in body
    assert "/home/" not in body


def test_every_referenced_file_exists(frontmatter):
    _, body = frontmatter
    for target in re.findall(r"\((references/[\w.-]+)\)", body):
        assert (SKILL_DIR / target).is_file(), f"{target} is referenced but missing"
    for script in re.findall(r"scripts/([\w.-]+\.py)", body):
        assert (SCRIPTS / script).is_file(), f"{script} is referenced but missing"


def test_prerequisites_name_the_python_floor(frontmatter):
    """Every published prismnetwork requires 3.10, and pip's refusal below it
    reads as a missing package rather than a version floor."""
    _, body = frontmatter
    prerequisites = body.split("## Prerequisites")[1].split("## How to Run")[0]
    assert "3.10" in prerequisites


def test_offers_is_not_advertised_as_wallet_free(frontmatter):
    """`offers()` signs a wallet session like every other SDK call. The public
    HTTP endpoint is the wallet-free path, and it is the one pick_gpu.py uses."""
    _, body = frontmatter
    quick = body.split("## Quick Reference")[1].split("## Procedure")[0]
    row = next(line for line in quick.splitlines() if line.startswith("| `.offers()`"))
    assert "signs a session" in row
    assert "api.prismnetwork.tech/v1/offers" in quick


# pick_gpu

OFFERS = [
    {"node_id": "0xa", "gpu": {"model": "H100 PCIe", "vram_mib": 81920},
     "rate_per_second": 900, "benchmark_score": 30000, "online": True, "trust_class": "open"},
    {"node_id": "0xb", "gpu": {"model": "RTX A6000", "vram_mib": 49140},
     "rate_per_second": 222, "benchmark_score": 12000, "online": True, "trust_class": "open"},
    {"node_id": "0xc", "gpu": {"model": "RTX 4090", "vram_mib": 24564},
     "rate_per_second": 177, "benchmark_score": 10000, "online": True, "trust_class": "open",
     "staker_only": True},
    {"node_id": "0xd", "gpu": {"model": "L40S", "vram_mib": 46068},
     "rate_per_second": 222, "benchmark_score": 15000, "online": False, "trust_class": "open"},
    {"node_id": "0xe", "gpu": {"model": "RTX 6000Ada", "vram_mib": 49140},
     "rate_per_second": 300, "benchmark_score": 16000, "online": True,
     "trust_class": "confidential"},
]


@pytest.fixture(scope="module")
def pick_gpu():
    return load_script("pick_gpu.py")


def test_cheapest_offer_over_the_vram_floor_wins(pick_gpu):
    eligible, _ = pick_gpu.choose(OFFERS, pick_gpu.floor_mib(40), None, "open", False)
    assert eligible[0]["node_id"] == "0xb"
    assert pick_gpu.why(eligible[0], eligible[1:]).startswith("cheapest of 3")


def test_a_card_of_the_size_asked_for_is_not_rejected(pick_gpu):
    """Nodes publish under nominal: a 24 GB card reports 24564 MiB, so a floor
    converted at 1024 MiB per GiB would reject every card of the size asked for."""
    assert pick_gpu.floor_mib(24) == 24000
    eligible, _ = pick_gpu.choose(OFFERS, pick_gpu.floor_mib(24), None, "open", True)
    assert "0xc" in [offer["node_id"] for offer in eligible], "24564 MiB clears a 24 GB floor"
    _, rejected = pick_gpu.choose(OFFERS, pick_gpu.floor_mib(32), None, "open", True)
    reasons = {offer["node_id"]: "; ".join(text) for offer, text in rejected}
    assert reasons["0xc"] == "24564 MiB VRAM under the 32000 MiB floor"


def test_staker_only_and_offline_offers_are_dropped_with_a_reason(pick_gpu):
    _, rejected = pick_gpu.choose(OFFERS, pick_gpu.floor_mib(16), None, "open", False)
    reasons = {offer["node_id"]: "; ".join(text) for offer, text in rejected}
    assert "reserved for stakers" in reasons["0xc"]
    assert "offline" in reasons["0xd"]


def test_staker_only_is_kept_when_asked_for(pick_gpu):
    eligible, _ = pick_gpu.choose(OFFERS, pick_gpu.floor_mib(16), None, "open", True)
    assert eligible[0]["node_id"] == "0xc", "the staker offer is the cheapest"


def test_a_price_ceiling_excludes_dearer_offers(pick_gpu):
    ceiling = int(0.70 * 1_000_000 / 3600)          # USDG per hour to base units per second
    eligible, rejected = pick_gpu.choose(OFFERS, pick_gpu.floor_mib(16), ceiling, "open", False)
    assert eligible == []
    assert any("over the ceiling" in reason for _, text in rejected for reason in text)


def test_trust_class_floor_is_ordered_not_matched(pick_gpu):
    eligible, _ = pick_gpu.choose(OFFERS, pick_gpu.floor_mib(16), None, "confidential", False)
    assert [offer["node_id"] for offer in eligible] == ["0xe"]


def test_deposit_is_the_whole_window(pick_gpu):
    winner = OFFERS[1]
    payload = pick_gpu.as_json(winner, [], [], 600, pick_gpu.floor_mib(16))
    assert '"deposit_base_units": 133200' in payload


def test_a_24_gb_request_matches_a_24_gb_card_end_to_end(pick_gpu, tmp_path, capsys):
    feed = tmp_path / "offers.json"
    feed.write_text(json.dumps([OFFERS[2] | {"staker_only": False}]), encoding="utf-8")
    assert pick_gpu.main(["--offers", str(feed), "--min-vram-gb", "24"]) == pick_gpu.MATCHED
    assert "24564 MiB" in capsys.readouterr().out


def test_no_match_exits_one(pick_gpu, tmp_path):
    offers = tmp_path / "offers.json"
    offers.write_text("[]", encoding="utf-8")
    assert pick_gpu.main(["--offers", str(offers)]) == pick_gpu.NO_MATCH


def test_unreadable_capacity_exits_three(pick_gpu, tmp_path):
    missing = tmp_path / "absent.json"
    assert pick_gpu.main(["--offers", str(missing)]) == pick_gpu.UNREACHABLE


# verify_receipt

@pytest.fixture(scope="module")
def verify_receipt():
    return load_script("verify_receipt.py")


def test_pinned_receipts_verify_offline(verify_receipt):
    assert verify_receipt.self_test() == verify_receipt.OK


def test_an_edited_receipt_is_rejected(verify_receipt):
    tampered = dict(verify_receipt.PINNED["clean"], refunded_base_units=0)
    assert verify_receipt.report(tampered) == verify_receipt.FAILED


def test_an_interrupted_run_is_flagged_rather_than_passed(verify_receipt):
    assert verify_receipt.report(verify_receipt.PINNED["interrupted"]) == verify_receipt.NOT_CLEAN


def test_canonical_form_is_declaration_order(verify_receipt):
    payload = verify_receipt.canonical(verify_receipt.PINNED["clean"]).decode()
    assert payload.index('"receipt_id"') < payload.index('"lease_id"') < payload.index('"gpu_model"')
    assert '"escrow_address"' not in payload, "fields outside the canonical set are dropped"
    assert '"failure_class":null' in payload, "failure_class survives as an explicit null"


def test_a_lease_with_no_receipt_reports_it(verify_receipt, capsys):
    with mock.patch.object(verify_receipt, "get", return_value={"receipts": []}):
        assert verify_receipt.main(["--lease", "9999"]) == verify_receipt.USAGE


def test_a_lease_id_resolves_to_its_receipt(verify_receipt, capsys):
    clean = verify_receipt.PINNED["clean"]
    index = {"receipts": [{"lease_id": "34", "receipt_id": clean["receipt_id"],
                           "escrow_address": verify_receipt.DEFAULT_ESCROW.lower()}]}
    with mock.patch.object(verify_receipt, "get", side_effect=[index, clean]):
        assert verify_receipt.main(["--lease", "34"]) == verify_receipt.OK
    assert f"receipt      {clean['receipt_id']}" in capsys.readouterr().out


def test_a_lease_id_shared_across_escrows_resolves_to_the_named_one(verify_receipt):
    """Ids restart at 1 with each escrow deployment, so three published receipts
    carry lease_id 34. Matching on the number alone returns a stranger's run."""
    live = verify_receipt.escrow_in_force(verify_receipt.DEFAULT_ESCROW)
    chosen = verify_receipt.resolve(verify_receipt.COLLIDING_INDEX, 34, live)
    assert chosen == verify_receipt.PINNED["clean"]["receipt_id"]
    assert chosen != verify_receipt.COLLIDING_INDEX["receipts"][1]["receipt_id"]


def test_a_lease_id_on_another_escrow_is_refused_not_substituted(verify_receipt):
    with pytest.raises(LookupError) as raised:
        verify_receipt.resolve(verify_receipt.COLLIDING_INDEX, 34, "0x" + "11" * 20)
    assert "none settled by the escrow" in str(raised.value)


def test_two_receipts_on_one_escrow_refuse_to_resolve(verify_receipt):
    index = {"receipts": [
        {"lease_id": "5", "receipt_id": "a", "escrow_address": verify_receipt.DEFAULT_ESCROW},
        {"lease_id": "5", "receipt_id": "b", "escrow_address": verify_receipt.DEFAULT_ESCROW},
    ]}
    with pytest.raises(LookupError) as raised:
        verify_receipt.resolve(index, 5, verify_receipt.escrow_in_force(verify_receipt.DEFAULT_ESCROW))
    assert "does not identify a run" in str(raised.value)


# lease_run_release

@pytest.fixture(scope="module")
def lease_run_release():
    return load_script("lease_run_release.py", stub_sdk=True)


def test_a_failure_before_broadcast_costs_nothing(lease_run_release):
    error = mock.Mock(code="wallet_unfunded", body=None, broadcast=False)
    code, message = lease_run_release.explain(error)
    assert code == lease_run_release.UNSPENT_FAILURE
    assert "before anything was signed" in message


def test_a_failure_after_broadcast_names_the_transaction(lease_run_release):
    error = mock.Mock(code="access_timeout", body={"lease_id": 41}, broadcast="0xdead")
    code, message = lease_run_release.explain(error)
    assert code == lease_run_release.MONEY_AT_RISK
    assert "0xdead" in message
    assert "scripts/release_lease.py --lease 41" in message


def test_a_failure_with_no_lease_id_points_at_the_funding_hash(lease_run_release):
    """``lease()`` can fail after funding and before an id comes back, so telling
    the caller to release 'the lease named above' names nothing."""
    error = mock.Mock(code="chain_error", body={"hint": "receipt unreadable"}, broadcast="0xf00")
    code, message = lease_run_release.explain(error)
    assert code == lease_run_release.MONEY_AT_RISK
    assert "scripts/release_lease.py --list" in message
    assert "0xf00" in message
    assert "named above" not in message


def test_recovery_is_shell_commands_the_skill_actually_ships(lease_run_release):
    """The heading says to run these through the `terminal` tool, so every line
    has to be a command and every script named has to exist."""
    with_id = lease_run_release.recovery(77, "0xabc")
    assert "scripts/release_lease.py --lease 77" in with_id
    assert "scripts/verify_receipt.py --lease 77" in with_id

    without_id = lease_run_release.recovery(None, "0xabc")
    assert "scripts/release_lease.py --list" in without_id
    assert "0xabc" in without_id, "with no id, the funding hash is the only handle"

    for message in (with_id, without_id):
        assert message.strip(), "an empty message strands a running meter"
        assert "agent." not in message, "a shell has no agent binding"
        for line in message.splitlines():
            body = line.strip()
            if body.startswith("python3"):
                assert (SKILL_DIR / body.split()[1]).is_file(), f"{body} names no shipped script"


def test_an_undetermined_failure_is_treated_as_spent(lease_run_release):
    error = mock.Mock(code="confirmation_timeout", body=None, broadcast=None)
    code, _ = lease_run_release.explain(error)
    assert code == lease_run_release.MONEY_AT_RISK


def test_a_released_lease_is_never_reported_as_open(lease_run_release):
    """``broadcast`` is None on the SDK's own ``ssh_access_unavailable``. The
    release already happened, so the answer is the bill, not a rescue."""
    error = lease_run_release.PrismError(400, "ssh_access_unavailable",
                                         {"mode": "gateway", "lease_id": 99})
    code, message = lease_run_release.explain(error, released=True)
    assert code == lease_run_release.FUNDED_NOT_RUN
    assert "The meter is stopped." in message
    assert "release_lease" not in message


def test_the_deposit_is_read_from_the_funding_receipt(lease_run_release):
    lease = mock.Mock(deposit_micros=133200)
    assert lease_run_release.deposited(lease, 1_000_000) == 133200


def test_an_unreadable_deposit_falls_back_to_the_ceiling(lease_run_release):
    for held in (None, 0, -1, True, 2_000_000):
        assert lease_run_release.deposited(mock.Mock(deposit_micros=held), 1_000_000) == 1_000_000


def test_no_wallet_is_refused_before_the_network(lease_run_release, monkeypatch):
    monkeypatch.delenv("PRISM_AGENT_KEY", raising=False)
    with pytest.raises(lease_run_release.NoWallet):
        lease_run_release.agent_from_env()


def test_an_empty_command_is_rejected(lease_run_release):
    with pytest.raises(SystemExit):
        lease_run_release.parse(["   "])


def test_a_node_without_its_rate_is_rejected(lease_run_release):
    with pytest.raises(SystemExit):
        lease_run_release.parse(["nvidia-smi", "--node", "0xabc"])


def test_a_priced_node_is_carried_through_untouched(lease_run_release):
    args = lease_run_release.parse(["nvidia-smi", "--node", "0xabc", "--rate-per-second", "222"])
    assert lease_run_release.price(args) == ("0xabc", 222, "node 0xabc at the rate given")


def test_pricing_picks_the_cheapest_rentable_offer(lease_run_release):
    args = lease_run_release.parse(["nvidia-smi", "--min-vram-gb", "16"])
    with mock.patch.object(lease_run_release.pick_gpu, "load", return_value=OFFERS):
        node, rate, _ = lease_run_release.price(args)
    assert (node, rate) == ("0xb", 222), "the staker-only 177 offer is not rentable"


def test_no_matching_capacity_raises_before_a_wallet_is_touched(lease_run_release):
    args = lease_run_release.parse(["nvidia-smi", "--min-vram-gb", "80"])
    with mock.patch.object(lease_run_release.pick_gpu, "load", return_value=OFFERS[2:]):
        with pytest.raises(lease_run_release.NoCapacity):
            lease_run_release.price(args)


def test_the_deposit_reserved_is_the_deposit_funded(lease_run_release):
    """The ledger reservation, the printed deposit and the escrow's ``max_deposit``
    are one figure, so budget_check.py and this script cannot disagree."""
    lease = mock.Mock(lease_id=91, funding_hash="0xf00", deposit_micros=199_800)
    lease.access = {"ssh_host": "10.0.0.1", "ssh_port": 2222}
    agent = mock.MagicMock()
    agent.lease.return_value = lease
    agent.run.return_value = {"code": 0, "stdout": "ok", "stderr": ""}
    args = lease_run_release.parse(["nvidia-smi", "--duration", "900"])
    reservations = []

    def reserving(book, tool, micros, run):
        reservations.append(micros)
        return run()["value"]

    with mock.patch.object(lease_run_release, "record_spend", side_effect=reserving):
        assert lease_run_release.run_lease(agent, mock.MagicMock(), args, "0xNODE", 199_800) == \
            lease_run_release.RELEASED

    assert reservations == [199_800], "the day is booked for the deposit, not the ceiling"
    kwargs = agent.lease.call_args.kwargs
    assert kwargs["max_deposit"] == 199_800
    assert kwargs["preferred_node_id"] == "0xNODE"


def test_a_dead_stdout_still_releases_the_lease(lease_run_release):
    """A closed pipe must not cost a whole window.

    Piping this script into something that exits first (``| head -1``) makes the
    next print raise BrokenPipeError. The prints that report the funded lease sit
    inside the try for exactly this reason, so the release still runs.
    """
    lease = mock.Mock(lease_id=7, funding_hash="0xdead", deposit_micros=133_200)
    lease.access = {"ssh_host": "10.0.0.1", "ssh_port": 2222}
    agent = mock.MagicMock()
    agent.lease.return_value = lease
    agent.run.return_value = {"code": 0, "stdout": "ok", "stderr": ""}
    args = lease_run_release.parse(["nvidia-smi"])

    def broken_pipe(*_a, **_k):
        raise BrokenPipeError(32, "Broken pipe")

    with mock.patch.object(lease_run_release, "record_spend", lambda book, tool, micros, run: run()["value"]), \
         mock.patch("builtins.print", side_effect=broken_pipe):
        with pytest.raises(BrokenPipeError):
            lease_run_release.run_lease(agent, mock.MagicMock(), args, "0xNODE", 133_200)

    assert agent.end_lease.called, "the lease was funded and the meter was left running"


def test_both_streams_survive_the_machine(lease_run_release, capsys):
    """The box is destroyed in the ``finally`` below, so a crash reason dropped
    here costs another deposit to see again."""
    lease = mock.Mock(lease_id=92, funding_hash="0xf01", deposit_micros=133_200)
    lease.access = {"ssh_host": "10.0.0.1", "ssh_port": 2222}
    agent = mock.MagicMock()
    agent.lease.return_value = lease
    agent.run.return_value = {"code": 1, "stdout": "partial results",
                              "stderr": "CUDA out of memory"}
    args = lease_run_release.parse(["train", "--duration", "600"])

    with mock.patch.object(lease_run_release, "record_spend",
                           side_effect=lambda book, tool, micros, run: run()["value"]):
        lease_run_release.run_lease(agent, mock.MagicMock(), args, "0xNODE", 133_200)

    printed = capsys.readouterr().out
    assert "partial results" in printed
    assert "CUDA out of memory" in printed


def funded_lease():
    lease = mock.Mock(lease_id=99, funding_hash="0xfeed", deposit_micros=133_200)
    lease.access = {"ssh_host": "10.0.0.1", "ssh_port": 2222}
    return lease


def drive(lease_run_release, agent):
    args = lease_run_release.parse(["nvidia-smi", "--duration", "600"])
    with mock.patch.object(lease_run_release, "record_spend",
                           side_effect=lambda book, tool, micros, run: run()["value"]):
        return lease_run_release.run_lease(agent, mock.MagicMock(), args, "0xNODE", 133_200)


def test_a_run_failure_on_a_released_lease_asks_for_no_rescue(lease_run_release, capsys):
    """The finally block already released it. Exit 3 would send the agent after a
    lease that is closed, and a caller branching on 3 would take the rescue path."""
    agent = mock.MagicMock()
    agent.lease.return_value = funded_lease()
    agent.run.side_effect = lease_run_release.PrismError(
        400, "ssh_access_unavailable", {"mode": "gateway", "lease_id": 99})

    assert drive(lease_run_release, agent) == lease_run_release.FUNDED_NOT_RUN

    printed = capsys.readouterr()
    assert "released     lease 99" in printed.out
    assert "release_lease" not in printed.err
    assert "meter is running" not in printed.err


def test_a_release_that_fails_is_the_case_that_needs_the_rescue(lease_run_release, capsys):
    agent = mock.MagicMock()
    agent.lease.return_value = funded_lease()
    agent.run.return_value = {"code": 0, "stdout": "ok", "stderr": ""}
    agent.end_lease.side_effect = lease_run_release.PrismError(502, "release_refused",
                                                              {"lease_id": 99})

    assert drive(lease_run_release, agent) == lease_run_release.MONEY_AT_RISK
    assert "scripts/release_lease.py --lease 99" in capsys.readouterr().err


def test_the_vram_floor_reaching_the_paid_call_is_the_one_that_matches(lease_run_release):
    """The same conversion the offer filter uses goes to the escrow, so a floor
    that selects a card cannot then be refused by the control plane as no_match."""
    agent = mock.MagicMock()
    agent.lease.return_value = funded_lease()
    agent.run.return_value = {"code": 0, "stdout": "", "stderr": ""}
    args = lease_run_release.parse(["nvidia-smi", "--min-vram-gb", "24"])
    with mock.patch.object(lease_run_release, "record_spend",
                           side_effect=lambda book, tool, micros, run: run()["value"]):
        lease_run_release.run_lease(agent, mock.MagicMock(), args, "0xNODE", 133_200)
    assert agent.lease.call_args.kwargs["min_vram_mib"] == 24_000


# release_lease

@pytest.fixture(scope="module")
def release_lease():
    return load_script("release_lease.py", stub_sdk=True)


def test_open_leases_are_marked_as_still_billing(release_lease):
    assert release_lease.billing({"lease_id": 7, "state": "active"})
    assert not release_lease.billing({"lease_id": 7, "state": "finalized"})
    assert not release_lease.billing({"lease_id": 7, "state": "refunded"})


def test_the_list_names_what_to_run_next(release_lease, capsys):
    records = [{"lease_id": 7, "state": "active", "node_id": "0xa"},
               {"lease_id": 6, "state": "finalized"}]
    assert release_lease.show(records, False) == release_lease.RELEASED
    printed = capsys.readouterr().out
    assert "2 lease(s), 1 still billing" in printed
    assert "7 BILLING" in printed
    assert "--lease 7" in printed


def test_a_field_the_control_plane_adds_is_counted_not_dropped(release_lease):
    line = release_lease.summarise({"lease_id": 7, "state": "active", "surprise": "value"})
    assert "+1 more fields" in line


def test_a_release_needs_a_target(release_lease):
    with pytest.raises(SystemExit):
        release_lease.parse([])
    with pytest.raises(SystemExit):
        release_lease.parse(["--lease", "7", "--list"])


def test_no_wallet_cannot_release(release_lease, monkeypatch):
    monkeypatch.delenv("PRISM_AGENT_KEY", raising=False)
    with pytest.raises(release_lease.NoWallet):
        release_lease.agent_from_env()


# budget_check

@pytest.fixture(scope="module")
def budget_check():
    return load_script("budget_check.py", stub_sdk=True)


def test_a_deposit_inside_both_limits_has_no_problems(budget_check):
    assert budget_check.verdict(133200, 1_000_000, 5_000_000) == []


def test_a_deposit_over_the_per_lease_cap_is_blocked(budget_check):
    problems = budget_check.verdict(7_992_000, 1_000_000, 5_000_000)
    assert len(problems) == 2, "over the per-lease cap and over the day"
    assert "per-lease cap" in problems[0]
    assert "left of today" in problems[1]


def test_an_unlimited_day_only_checks_the_per_lease_cap(budget_check):
    problems = budget_check.verdict(2_000_000, 1_000_000, None)
    assert len(problems) == 1
    assert "per-lease cap" in problems[0]


def test_duration_and_rate_travel_together(budget_check):
    with pytest.raises(SystemExit):
        budget_check.parse(["--duration", "600"])


# the guards that only main() reaches

def test_an_unpriced_offer_is_rejected_rather_than_ranked_free(pick_gpu):
    """``.get("rate_per_second", 0)`` reads a missing rate as free, which sorts it
    to the front of a ranking that is cheapest-first."""
    unpriced = dict(OFFERS[0])
    unpriced.pop("rate_per_second", None)
    eligible, rejected = pick_gpu.choose(
        [unpriced] + list(OFFERS[1:2]), pick_gpu.floor_mib(16), None, "open", False)
    assert [offer["node_id"] for offer in eligible] == [OFFERS[1]["node_id"]]
    assert any("no published rate" in reason for _, text in rejected for reason in text)


def test_a_malformed_offer_list_does_not_crash_the_ranking(pick_gpu):
    eligible, _ = pick_gpu.choose(
        [1, "two", None, dict(OFFERS[1])], pick_gpu.floor_mib(16), None, "open", False)
    assert [offer["node_id"] for offer in eligible] == [OFFERS[1]["node_id"]]


def test_a_deposit_over_the_ceiling_never_reaches_a_wallet(lease_run_release, capsys):
    budget = mock.Mock(ledger_path="/tmp/none.json", daily_micros=None, max_per_call_micros=100)
    with mock.patch.object(lease_run_release, "read_budget", return_value=budget), \
         mock.patch.object(lease_run_release, "SpendLedger") as ledger, \
         mock.patch.object(lease_run_release, "call_ceiling", return_value=100), \
         mock.patch.object(lease_run_release, "agent_from_env") as wallet:
        ledger.return_value.remaining.return_value = None
        code = lease_run_release.main(
            ["nvidia-smi", "--node", "0xNODE", "--rate-per-second", "222", "--duration", "900"])
    assert code == lease_run_release.USAGE
    assert not wallet.called, "no wallet is built for a lease that cannot be afforded"
    assert "per-lease cap" in capsys.readouterr().err


def test_the_planned_deposit_is_the_rate_times_the_window(lease_run_release, capsys):
    """main() is where the deposit is priced. run_lease only receives the figure."""
    budget = mock.Mock(ledger_path="/tmp/none.json", daily_micros=None, max_per_call_micros=10**9)
    with mock.patch.object(lease_run_release, "read_budget", return_value=budget), \
         mock.patch.object(lease_run_release, "SpendLedger") as ledger, \
         mock.patch.object(lease_run_release, "call_ceiling", return_value=10**9):
        ledger.return_value.remaining.return_value = None
        code = lease_run_release.main(
            ["nvidia-smi", "--node", "0xNODE", "--rate-per-second", "222",
             "--duration", "900", "--dry-run"])
    assert code == lease_run_release.RELEASED
    assert "0.199800 USDG planned for 900s" in capsys.readouterr().out


def test_a_timeout_that_outlives_its_window_is_refused(lease_run_release):
    with pytest.raises(SystemExit):
        lease_run_release.parse(["nvidia-smi", "--duration", "300", "--timeout", "300"])


def test_a_command_too_large_for_the_argument_list_is_refused(lease_run_release):
    with pytest.raises(SystemExit):
        lease_run_release.parse(["x" * (lease_run_release.COMMAND_LIMIT + 1)])


def test_a_bad_ceiling_is_a_usage_error_not_a_broken_ledger(budget_check):
    """Exit 3 sends the caller to inspect the ledger. A negative flag is exit 2."""
    with pytest.raises(SystemExit) as exit_code:
        budget_check.parse(["--max-usdg", "-1"])
    assert exit_code.value.code == 2


def test_an_unknown_lease_says_ids_restart_with_each_escrow(release_lease, capsys):
    agent = mock.MagicMock()
    agent.release.side_effect = release_lease.PrismError(404, "lease_not_found")
    with mock.patch.object(release_lease, "agent_from_env", return_value=agent):
        assert release_lease.main(["--lease", "34"]) == release_lease.REFUSED
    assert "Ids restart at 1" in capsys.readouterr().err


def test_a_lease_list_that_is_not_a_list_is_not_an_all_clear(release_lease, capsys):
    """"no leases" on an unreadable answer is a false all-clear on a wallet that
    may still be paying for one."""
    assert release_lease.show({"leases": []}, False) == release_lease.REFUSED
    assert "not a list" in capsys.readouterr().err


def test_an_unreadable_lease_record_counts_as_billing(release_lease):
    assert release_lease.billing("not-a-dict") is True


def test_a_document_that_is_not_a_receipt_is_refused(verify_receipt, capsys):
    with mock.patch.object(verify_receipt, "load", return_value={"receipt_hash": "0x1"}):
        assert verify_receipt.main(["8f3e0c1d"]) == verify_receipt.USAGE
    assert "not a settlement receipt" in capsys.readouterr().err


def test_a_local_receipt_file_is_read_from_disk(verify_receipt, tmp_path, monkeypatch):
    """The usage line advertises a path. Requiring a slash in it sent
    ``verify_receipt.py receipt.json`` to the network and reported a 404."""
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(verify_receipt.PINNED["clean"]), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    with mock.patch.object(verify_receipt, "get", side_effect=AssertionError("went to the network")):
        assert verify_receipt.load("receipt.json") == verify_receipt.PINNED["clean"]


def test_a_blank_escrow_does_not_match_every_receipt(verify_receipt, monkeypatch):
    monkeypatch.setenv("PRISM_ESCROW", "   ")
    assert verify_receipt.escrow_in_force() == verify_receipt.DEFAULT_ESCROW.lower()


def test_the_self_test_still_checks_under_optimised_python(verify_receipt):
    """``python3 -O`` strips assert. SKILL.md points at --self-test to back the
    billing claims, so it has to check something under any interpreter flag."""
    import subprocess
    script = SCRIPTS / "verify_receipt.py"
    source = script.read_text(encoding="utf-8")
    broken = source.replace('"runtime_seconds": 28,', '"runtime_seconds": 29,', 1)
    assert broken != source, "the pinned receipt this test edits has moved"

    clean = subprocess.run([sys.executable, "-O", str(script), "--self-test"],
                           capture_output=True, text=True)
    assert clean.returncode == 0, clean.stderr

    edited = SCRIPTS.parent / "_self_test_under_o.py"
    edited.write_text(broken, encoding="utf-8")
    try:
        result = subprocess.run([sys.executable, "-O", str(edited), "--self-test"],
                                capture_output=True, text=True)
    finally:
        edited.unlink()
    assert result.returncode != 0, "an edited pinned receipt passed with asserts stripped"
    assert "SELF-TEST FAILED" in result.stderr
    assert verify_receipt.FAILED == 1


def test_a_release_that_fails_below_the_sdk_still_reports_the_bill(lease_run_release, capsys):
    """A reset connection raises OSError, not PrismError. Letting it out of the
    finally skips the only message that says the meter is still running."""
    lease = mock.Mock(lease_id=101, funding_hash="0xbeef", deposit_micros=133_200)
    lease.access = {"ssh_host": "10.0.0.1", "ssh_port": 2222}
    agent = mock.MagicMock()
    agent.lease.return_value = lease
    agent.run.return_value = {"code": 0, "stdout": "ok", "stderr": ""}
    agent.end_lease.side_effect = ConnectionResetError(54, "Connection reset by peer")
    args = lease_run_release.parse(["nvidia-smi"])

    with mock.patch.object(lease_run_release, "record_spend",
                           lambda book, tool, micros, run: run()["value"]):
        code = lease_run_release.run_lease(agent, mock.MagicMock(), args, "0xNODE", 133_200)

    assert code == lease_run_release.MONEY_AT_RISK
    printed = capsys.readouterr()
    assert "RELEASE FAILED ConnectionResetError" in printed.err
    assert "scripts/release_lease.py --lease 101" in printed.err


def test_capacity_is_filtered_here_not_by_the_endpoint(pick_gpu):
    """Asking the endpoint for one trust class returns a pre-filtered list, and
    then a constraint that matched nothing reports "0 were published" while six
    machines are online."""
    seen = {}

    def record(url, timeout=15):
        seen["url"] = url
        return list(OFFERS)

    with mock.patch.object(pick_gpu, "fetch", side_effect=record):
        pick_gpu.load(None)
    assert "min_trust" not in seen["url"]


def test_an_empty_feed_is_not_reported_as_a_missed_constraint(pick_gpu, tmp_path, capsys):
    feed = tmp_path / "offers.json"
    feed.write_text("[]", encoding="utf-8")
    assert pick_gpu.main(["--offers", str(feed)]) == pick_gpu.NO_MATCH
    assert "no capacity is published right now" in capsys.readouterr().err


def test_a_tie_counts_only_offers_that_are_actually_tied(pick_gpu):
    """"tied on every criterion" counted the whole matching field, so two cards
    of different sizes were reported as identical."""
    same = dict(OFFERS[0])
    smaller = dict(OFFERS[0], node_id="0xsmall", gpu=dict(OFFERS[0]["gpu"], vram_mib=24564))
    assert "tied with 1 other offer(s) of 2" in pick_gpu.why(same, [dict(same, node_id="0xb")])
    assert "largest card" in pick_gpu.why(same, [smaller])


def test_a_staker_only_winner_is_flagged_before_the_next_command(pick_gpu, capsys):
    winner = dict(OFFERS[0], staker_only=True)
    printed = pick_gpu.report(winner, [], [], 600)
    assert "only rents to bonded stakers" in printed
    assert printed.index("caution") < printed.index("next"), "the warning has to precede the command"


def test_an_image_without_a_digest_is_refused_before_pricing(lease_run_release):
    """--dry-run is the pre-flight for the rule SKILL.md states, so it has to
    make the check the control plane makes."""
    with pytest.raises(SystemExit):
        lease_run_release.parse(["nvidia-smi", "--image", "ollama/ollama:latest", "--dry-run"])


def test_a_path_that_does_not_exist_is_not_blamed_on_the_network(verify_receipt, capsys):
    with pytest.raises(FileNotFoundError):
        verify_receipt.load("/tmp/prism-gpu-no-such-receipt.json")
    assert verify_receipt.main(["/tmp/prism-gpu-no-such-receipt.json"]) == verify_receipt.USAGE
    assert "no such file" in capsys.readouterr().err


def test_an_unreachable_feed_is_its_own_exit_code(verify_receipt, capsys):
    """Exit 2 means the arguments were wrong. An outage is not a wrong argument."""
    with mock.patch.object(verify_receipt, "get",
                           side_effect=OSError("[Errno 61] Connection refused")):
        assert verify_receipt.main(["--lease", "34"]) == verify_receipt.UNREACHABLE
    assert "the feed could not be reached" in capsys.readouterr().err


def test_the_self_test_refuses_arguments_it_would_ignore(verify_receipt):
    with pytest.raises(SystemExit):
        verify_receipt.main(["--self-test", "--lease", "34"])


def test_a_lease_id_that_is_not_a_number_is_caught_before_the_feed(verify_receipt):
    with pytest.raises(SystemExit):
        verify_receipt.main(["--lease", "abc"])


def test_an_escrow_that_is_not_an_address_is_caught_before_the_feed(verify_receipt):
    with pytest.raises(SystemExit):
        verify_receipt.main(["--lease", "34", "--escrow", "potato"])
    assert verify_receipt.is_address("0x" + "ab" * 20)
    assert not verify_receipt.is_address("0x" + "ab" * 19)


def test_a_ceiling_above_the_environment_says_it_was_clamped(budget_check, capsys):
    budget = mock.Mock(ledger_path="/tmp/none.json", daily_micros=None, max_per_call_micros=10**6)
    with mock.patch.object(budget_check, "read_budget", return_value=budget), \
         mock.patch.object(budget_check, "SpendLedger") as ledger, \
         mock.patch.object(budget_check, "call_ceiling", return_value=10**6):
        ledger.return_value.status.return_value = {
            "daily_budget": "5.000000 USDG", "spent_last_24h": "0.000000 USDG",
            "remaining_today": "5.000000 USDG", "max_per_call": "1.000000 USDG",
            "ledger": "/tmp/none.json", "charges_last_24h": [],
        }
        ledger.return_value.remaining.return_value = None
        budget_check.main(["--max-usdg", "50"])
    assert "clamped to 1.000000 USDG" in capsys.readouterr().out
