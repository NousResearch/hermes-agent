import XCTest
import HermesCore
@testable import HermesFeatures

@MainActor final class ScheduledJobsTests: XCTestCase {
    func job(_ profile: String, id: String = "same-id") -> JSON {
        .object(["id": .string(id), "profile": .string(profile), "name": .string("Check"), "prompt": .string("Audit"),
                 "enabled": .bool(true), "schedule": .object(["kind": .string("cron"), "expr": .string("0 9 * * *")])])
    }
    func testExactUpdateEnvelopeAndCompositeProfileIdentity() async {
        let wire = FakeTransport(); let rows = [job("one"), job("two")]
        let store = ScheduledJobsStore(transport: wire)
        store.configure(connected: true, profile: "")
        wire.httpHandler = { path, method, body, query in
            if method == "GET" { return .array(rows) }
            XCTAssertEqual(path, "/api/cron/jobs/same-id")
            XCTAssertEqual(method, "PUT")
            XCTAssertEqual(query["profile"], "two")
            XCTAssertEqual(body, .object(["updates": .object(["name": .string("New")])]))
            return rows[1]
        }
        await store.refresh()
        let saved = await store.update(rows[1], changed: .object(["name": .string("New")]), context: store.contextID)
        XCTAssertTrue(saved)
    }
    func testRunsUseConcreteProfileAndStaleSelectionCannotReplaceNewHistory() async {
        let wire = FakeTransport(); let rows = [job("one"), job("two")]
        var waiting: CheckedContinuation<JSON, Error>?
        wire.httpHandler = { path, _, _, query in
            if !path.hasSuffix("/runs") { return .array(rows) }
            XCTAssertNotEqual(query["profile"], "all")
            if query["profile"] == "one" { return try await withCheckedThrowingContinuation { waiting = $0 } }
            return .object(["runs": .array([.object(["id": .string("two-run")])])])
        }
        let store = ScheduledJobsStore(transport: wire); store.configure(connected: true, profile: "")
        await store.refresh()
        let old = Task { await store.select(rows[0]) }
        while waiting == nil { await Task.yield() }
        await store.select(rows[1])
        waiting?.resume(returning: .object(["runs": .array([.object(["id": .string("old-run")])])]))
        await old.value
        XCTAssertEqual(store.runs.first?["id"].string, "two-run")
    }
    func testMismatchedTriggerReceiptIsUnknownAndNotRetried() async {
        let wire = FakeTransport(); let row = job("one"); var writes = 0
        wire.httpHandler = { _, method, _, _ in
            if method == "GET" { return .array([row]) }
            writes += 1; return .object(["id": .string("wrong")])
        }
        let store = ScheduledJobsStore(transport: wire); store.configure(connected: true, profile: "one")
        await store.refresh(); let context = store.contextID
        let first = await store.perform(.runNow, job: row, context: context)
        let retry = await store.perform(.runNow, job: row, context: context)
        XCTAssertFalse(first); XCTAssertFalse(retry); XCTAssertEqual(writes, 1)
        XCTAssertNotNil(store.unknownOutcomeMessage)
    }
    func testMalformedListClearsLoadingAndBlocksControls() async {
        let wire = FakeTransport(); wire.httpHandler = { _, _, _, _ in .object(["jobs": .array([])]) }
        let store = ScheduledJobsStore(transport: wire); store.configure(connected: true, profile: "one")
        await store.refresh()
        XCTAssertFalse(store.loading); XCTAssertTrue(store.needsRefresh); XCTAssertFalse(store.canMutate)
        XCTAssertNotNil(store.errorMessage)
    }
    func testStaleEditorCannotReportSavedAfterDisconnect() async {
        let wire = FakeTransport(); let row = job("one")
        wire.httpHandler = { _, _, _, _ in .array([row]) }
        let store = ScheduledJobsStore(transport: wire); store.configure(connected: true, profile: "one")
        await store.refresh(); let context = store.contextID
        store.configure(connected: false, profile: "one")
        let saved = await store.update(row, changed: .object(["name": .string("New")]), context: context)
        XCTAssertFalse(saved)
        XCTAssertFalse(wire.httpCalls.contains { $0.1 != "GET" })
    }
    func testCreateRejectsAllAndMissingReceiptDoesNotReportSaved() async {
        let wire = FakeTransport()
        wire.httpHandler = { _, method, _, _ in method == "GET" ? .array([]) : .object(["ok": .bool(true)]) }
        let store = ScheduledJobsStore(transport: wire); store.configure(connected: true, profile: "")
        await store.refresh(); let context = store.contextID
        let body: JSON = .object(["prompt": .string("Audit"), "schedule": .string("0 9 * * *")])
        let invalid = await store.create(body, concreteProfile: "all", context: context)
        XCTAssertFalse(invalid); XCTAssertFalse(wire.httpCalls.contains { $0.1 == "POST" })
        let unknown = await store.create(body, concreteProfile: "one", context: context)
        XCTAssertFalse(unknown); XCTAssertTrue(store.needsRefresh)
    }
    func testScheduleAndUnchangedOptionalValuesRoundTrip() {
        let row = job("one")
        let draft = ScheduledJobDraft(job: row)
        XCTAssertEqual(draft.schedule, "0 9 * * *")
        XCTAssertEqual(draft.changes(from: row), .object([:]))
    }
}
