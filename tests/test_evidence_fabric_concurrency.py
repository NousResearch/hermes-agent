from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from hermes_state import SessionDB
from research.evidence_fabric import EvidenceFabricService, EvidenceScope


def test_identical_evidence_race_converges_without_integrity_error(tmp_path):
    path = tmp_path / "state.db"
    setup = SessionDB(path)
    try:
        service = EvidenceFabricService(setup, EvidenceScope("scope", None, None, "setup"))
        run = service.create_research_run("race")
    finally:
        setup.close()

    barrier = Barrier(2)

    def writer(agent):
        db = SessionDB(path)
        try:
            service = EvidenceFabricService(db, EvidenceScope("scope", None, None, agent))
            barrier.wait()
            return service.add_evidence(
                run.id,
                source_type="WEB_PAGE",
                retrieval_method="DIRECT_HTTP",
                content="identical",
                source_uri="https://example.test/race",
            )
        finally:
            db.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(writer, ("agent-a", "agent-b")))

    assert len({result.evidence.id for result in results}) == 1
    assert sorted(result.created for result in results) == [False, True]
    check = SessionDB(path)
    try:
        service = EvidenceFabricService(check, EvidenceScope("scope", None, None, "check"))
        assert len(service.list_evidence(run.id)) == 1
    finally:
        check.close()


def test_distinct_concurrent_evidence_is_not_lost(tmp_path):
    path = tmp_path / "state.db"
    setup = SessionDB(path)
    try:
        service = EvidenceFabricService(setup, EvidenceScope("scope", None, None, "setup"))
        run = service.create_research_run("distinct")
    finally:
        setup.close()

    barrier = Barrier(2)

    def writer(value):
        db = SessionDB(path)
        try:
            service = EvidenceFabricService(db, EvidenceScope("scope", None, None, f"agent-{value}"))
            barrier.wait()
            return service.add_evidence(
                run.id,
                source_type="FILE",
                retrieval_method="FILE_READ",
                content=value,
                raw_reference=f"artifact:{value}",
            )
        finally:
            db.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(writer, ("one", "two")))
    check = SessionDB(path)
    try:
        service = EvidenceFabricService(check, EvidenceScope("scope", None, None, "check"))
        assert {record.raw_reference for record in service.list_evidence(run.id)} == {"artifact:one", "artifact:two"}
    finally:
        check.close()
