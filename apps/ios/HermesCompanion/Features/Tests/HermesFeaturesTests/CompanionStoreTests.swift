import XCTest
import HermesCore
@testable import HermesFeatures

@MainActor
final class FakeTransport: CompanionTransport {
    var onEvent: ((JSON) -> Void)?
    var onDisconnect: ((Error) -> Void)?
    var calls: [(String, JSON)] = []
    var httpCalls: [(String, String, [String: String])] = []
    var httpHandler: ((String, String, JSON?, [String: String]) async throws -> JSON)?
    var mutationBodies: [JSON] = []
    var mutationReceipt: JSON = .object(["task": .object(["id": .string("task-a")])])
    var failSubmit = false
    var delayed: CheckedContinuation<JSON, Error>?
    var delayRoster = false
    var rosterWait: CheckedContinuation<JSON, Error>?
    var delayConnect = false
    var connectWait: CheckedContinuation<Void, Error>?
    func connect(endpoint: Endpoint, username: String, password: String) async throws {
        if delayConnect { try await withCheckedThrowingContinuation { connectWait = $0 } }
    }
    func reconnect() async throws {}
    func disconnect() {}
    func signOut() {}
    func request(_ method: String, params: JSON) async throws -> JSON {
        calls.append((method, params))
        switch method {
        case "profiles.list":
            if delayRoster { return try await withCheckedThrowingContinuation { rosterWait = $0 } }
            return .object(["profiles": .array([.object(["name": .string("default")])])])
        case "session.list": return .object(["sessions": .array([])])
        case "session.resume":
            if params["session_id"].string == "slow" {
                return try await withCheckedThrowingContinuation { delayed = $0 }
            }
            return snapshot(params["session_id"].string ?? "a")
        case "prompt.submit":
            if failSubmit { throw URLError(.networkConnectionLost) }
            return .object(["status": .string("streaming")])
        case "approval.respond": return .object(["resolved": .number(1)])
        case "session.interrupt": return .object(["status": .string("interrupted")])
        case "approval.pending": return .object(["approvals": .array([])])
        default: return .object([:])
        }
    }
    func snapshot(_ id: String) -> JSON {
        .object(["session_id": .string("live-" + id), "session_key": .string(id), "resumed": .string(id),
                 "messages": .array([]), "running": .bool(false)])
    }
    func http(_ path: String, method: String, body: JSON?, query: [String: String]) async throws -> JSON {
        httpCalls.append((path, method, query))
        if let httpHandler { return try await httpHandler(path, method, body, query) }
        if method != "GET" { mutationBodies.append(body ?? .null); return mutationReceipt }
        return .object(["boards": .array([]), "columns": .array([])])
    }
}
@MainActor
final class CompanionStoreTests: XCTestCase {
    func testPreviewInvalidatesPendingLoginAndExitClearsFixtures() async {
        let wire = FakeTransport(); wire.delayConnect = true
        let store = CompanionStore(transport: wire)
        let login = Task { await store.connect(url: "https://mac.example.ts.net", username: "user", password: "password") }
        while wire.connectWait == nil { await Task.yield() }
        store.beginPreview()
        store.profiles = [.object(["name": .string("Sample")])]
        store.scheduledJobs.installPreview([.object(["id": .string("sample")])])
        wire.connectWait?.resume()
        await login.value
        XCTAssertTrue(store.isPreview)
        XCTAssertFalse(store.isConnected)
        XCTAssertFalse(store.connecting)
        XCTAssertTrue(wire.calls.isEmpty)
        store.signOut()
        XCTAssertFalse(store.isPreview)
        XCTAssertTrue(store.profiles.isEmpty)
        XCTAssertTrue(store.scheduledJobs.jobs.isEmpty)
        XCTAssertTrue(store.host.isEmpty)
    }
    func connected(_ wire: FakeTransport) async -> CompanionStore {
        let store = CompanionStore(transport: wire)
        await store.connect(url: "https://mac.example.ts.net", username: "user", password: "password")
        return store
    }
    func testUncertainSubmitIsNotAutomaticallyRepeated() async {
        let wire = FakeTransport()
        let store = await connected(wire)
        await store.openSession("a")
        wire.failSubmit = true
        await store.send("do work")
        await store.send("do work")
        XCTAssertEqual(wire.calls.filter { $0.0 == "prompt.submit" }.count, 1)
        XCTAssertEqual(store.session?.delivery, .uncertain)
        XCTAssertNotNil(store.errorMessage)
    }
    func testNewSessionPreservesChosenProviderAndProfileChangeClearsOverride() async {
        let wire = FakeTransport()
        let store = await connected(wire)
        store.modelOverride = "shared-model-name"
        await store.createChat(title: "Audit", cwd: "/workspace", provider: "openrouter")
        let request = wire.calls.first { $0.0 == "session.create" }?.1
        XCTAssertEqual(request?["provider"].string, "openrouter")
        XCTAssertEqual(request?["model"].string, "shared-model-name")
        XCTAssertEqual(request?["cwd"].string, "/workspace")
        await store.chooseProfile("other")
        XCTAssertTrue(store.modelOverride.isEmpty)
    }
    func testOldSessionLoadCannotOverwriteNewSelection() async {
        let wire = FakeTransport()
        let store = await connected(wire)
        let first = Task { await store.openSession("slow") }
        while wire.delayed == nil { await Task.yield() }
        await store.openSession("new")
        wire.delayed?.resume(returning: wire.snapshot("slow"))
        await first.value
        XCTAssertEqual(store.session?.storedID, "new")
    }
    func testReconnectResumesDurableConversationID() async {
        let wire = FakeTransport()
        let store = await connected(wire)
        await store.openSession("saved-conversation")
        await store.reconnect()
        let ids = wire.calls.filter { $0.0 == "session.resume" }.compactMap { $0.1["session_id"].string }
        XCTAssertEqual(ids, ["saved-conversation", "saved-conversation"])
    }
    func testReconnectCannotRestoreOldChatIntoNewProfileSelection() async {
        let wire = FakeTransport()
        let store = await connected(wire)
        await store.openSession("old")
        wire.delayRoster = true
        let reconnect = Task { await store.reconnect() }
        while wire.rosterWait == nil { await Task.yield() }
        await store.chooseProfile("other")
        wire.rosterWait?.resume(returning: .object(["profiles": .array([])]))
        await reconnect.value
        XCTAssertNil(store.session)
        XCTAssertEqual(wire.calls.filter { $0.0 == "session.resume" }.count, 1)
    }
    func testForeignOrExpiredApprovalCannotSendCommand() async {
        let wire = FakeTransport()
        let store = await connected(wire)
        await store.openSession("a")
        let request = PendingRequest(id: "absent", kind: .approval, payload: .object([:]))
        await store.respond(request, answer: "once")
        XCTAssertFalse(wire.calls.contains { $0.0 == "approval.respond" })
    }
    func testNumericApprovalReceiptConfirmsResolution() async {
        let wire = FakeTransport()
        let store = await connected(wire)
        await store.openSession("a")
        wire.onEvent?(.object(["type": .string("approval.request"), "session_id": .string("live-a"),
            "payload": .object(["request_id": .string("request-a"), "command": .string("check"),
                                 "choices": .array([.string("once"), .string("deny")])])]))
        guard let request = store.session?.requests.first else { return XCTFail("Missing request") }
        await store.respond(request, answer: "once")
        XCTAssertNil(store.errorMessage)
        XCTAssertTrue(store.session?.requests.isEmpty == true)
        XCTAssertFalse(store.session?.needsRefresh == true)
    }
    func testNoBoardSelectedMeansNoTaskMutation() async {
        let wire = FakeTransport()
        let store = await connected(wire)
        await store.createTask(title: "Audit", body: "check", assignee: "")
        XCTAssertFalse(wire.httpCalls.contains { $0.1 != "GET" })
        XCTAssertNotNil(store.errorMessage)
    }
    func testDisconnectClearsControlsAndPreventsSend() async {
        let wire = FakeTransport()
        let store = await connected(wire)
        await store.openSession("a")
        wire.onDisconnect?(URLError(.networkConnectionLost))
        await store.send("run now")
        XCTAssertFalse(wire.calls.contains { $0.0 == "prompt.submit" })
        XCTAssertFalse(store.isConnected)
    }
    func testStopAcknowledgementDoesNotClaimMacHasStopped() async {
        let wire = FakeTransport()
        let store = await connected(wire)
        await store.openSession("a")
        await store.send("work")
        await store.stop()
        XCTAssertTrue(store.session?.running == true)
        XCTAssertFalse(store.session?.canSend == true)
        XCTAssertEqual(store.session?.activity, "Stop requested · waiting for Mac")
        wire.onEvent?(.object(["type": .string("message.complete"), "session_id": .string("live-a"), "payload": .object([:])]))
        XCTAssertFalse(store.session?.running == true)
    }
    func testTaskEditOnlyWritesChangedFields() async {
        let wire = FakeTransport()
        let store = await connected(wire)
        store.selectedBoard = "work"
        let task: JSON = .object(["id": .string("task-a"), "title": .string("Old"), "body": .string("Keep"), "assignee": .string("agent"), "status": .string("ready")])
        store.columns = [.object(["tasks": .array([task])])]
        await store.refreshBoard()
        store.columns = [.object(["tasks": .array([task])])]
        await store.updateTask(original: task, title: "New", body: "Keep", assignee: "agent", status: "ready")
        XCTAssertEqual(wire.mutationBodies, [.object(["title": .string("New")])])
    }
    func testMissingTaskReceiptRequiresRefresh() async {
        let wire = FakeTransport()
        let store = await connected(wire)
        store.selectedBoard = "work"
        await store.refreshBoard()
        wire.mutationReceipt = .object(["task": .null])
        await store.createTask(title: "Check", body: "", assignee: "")
        XCTAssertNotNil(store.errorMessage)
        XCTAssertTrue(store.boardNeedsRefresh)
    }
}
