import XCTest
import HermesCore
@testable import HermesFeatures

@MainActor final class WorkspaceLibraryTests: XCTestCase {
    func testProfileIsSentAndMalformedCatalogDoesNotBecomeFresh() async {
        let wire = FakeTransport()
        let library = WorkspaceLibrary(transport: wire)
        library.configure(connected: true, profile: "research")
        wire.httpHandler = { _, _, _, query in
            XCTAssertEqual(query["profile"], "research")
            return .object(["unexpected": .bool(true)])
        }
        await library.refresh(.skills)
        XCTAssertFalse(library.fresh.contains(.skills))
        XCTAssertFalse(library.loading.contains(.skills))
        XCTAssertNotNil(library.errors[.skills])
    }
    func testDelayedOldProfileCannotPopulateNewProfile() async {
        let wire = FakeTransport()
        var waiting: CheckedContinuation<JSON, Error>?
        wire.httpHandler = { _, _, _, _ in try await withCheckedThrowingContinuation { waiting = $0 } }
        let library = WorkspaceLibrary(transport: wire)
        library.configure(connected: true, profile: "old")
        let first = Task { await library.refresh(.skills) }
        while waiting == nil { await Task.yield() }
        library.configure(connected: true, profile: "new")
        waiting?.resume(returning: .array([.object(["name": .string("old-private-skill")])]))
        await first.value
        XCTAssertNil(library.values[.skills])
        XCTAssertTrue(library.fresh.isEmpty)
    }
    func testToggleRequiresCurrentRowAndMatchingReceiptWithoutReplay() async {
        let wire = FakeTransport()
        let library = WorkspaceLibrary(transport: wire)
        library.configure(connected: true, profile: "research")
        var writes = 0
        wire.httpHandler = { path, method, body, query in
            if method == "GET" { return .array([.object(["name": .string("audit"), "enabled": .bool(false)])]) }
            writes += 1
            XCTAssertEqual(path, "/api/skills/toggle")
            XCTAssertEqual(query["profile"], "research")
            XCTAssertEqual(body?["name"].string, "audit")
            return .object(["ok": .bool(true), "name": .string("different"), "enabled": .bool(true)])
        }
        await library.refresh(.skills)
        let result = await library.setEnabled(.skills, name: "audit", enabled: true, expectedProfile: "research")
        XCTAssertFalse(result)
        let retry = await library.setEnabled(.skills, name: "audit", enabled: true, expectedProfile: "research")
        XCTAssertFalse(retry)
        XCTAssertEqual(writes, 1)
    }
    func testDisconnectBlocksDetailAndMutation() async {
        let wire = FakeTransport()
        let library = WorkspaceLibrary(transport: wire)
        library.configure(connected: false, profile: "research")
        let detail = await library.detail(.skills, id: "audit")
        XCTAssertNil(detail)
        XCTAssertTrue(wire.httpCalls.isEmpty)
    }
}
