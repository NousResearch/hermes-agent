import XCTest
import HermesCore
@testable import HermesFeatures

final class SessionStateTests: XCTestCase {
    func event(_ type: String, _ text: String, sid: String = "live-a", seq: Int = 1) -> JSON {
        .object(["type": .string(type), "session_id": .string(sid), "seq": .number(Double(seq)),
                 "payload": .object(["text": .string(text)])])
    }
    func testForeignSessionCannotChangeTranscriptOrSequence() {
        var state = SessionState(sessionID: "live-a", storedID: "saved-a")
        state.apply(event("message.delta", "private", sid: "live-b", seq: 12))
        XCTAssertTrue(state.messages.isEmpty)
        XCTAssertNil(state.lastSequence)
    }
    func testStreamingDuplicateAndFinalTextReconcile() {
        var state = SessionState(sessionID: "live-a", storedID: "saved-a")
        state.apply(event("message.start", "", seq: 1))
        state.apply(event("message.delta", "Hel", seq: 2))
        state.apply(event("message.delta", "Hel", seq: 2))
        state.apply(event("message.delta", "lo", seq: 3))
        XCTAssertEqual(state.messages.last?.text, "Hello")
        state.apply(event("message.complete", "Hello!", seq: 4))
        XCTAssertEqual(state.messages.map(\.text), ["Hello!"])
        XCTAssertFalse(state.running)
    }
    func testSequenceGapInvalidatesControlAndNeedsRefresh() {
        var state = SessionState(sessionID: "live-a", storedID: "saved-a")
        state.apply(event("message.start", "", seq: 1))
        state.apply(event("message.delta", "missing context", seq: 4))
        XCTAssertTrue(state.needsRefresh)
        XCTAssertFalse(state.canSend)
    }
    func testAlreadyStreamedInterimSealsWithoutDuplicatingText() {
        var state = SessionState(sessionID: "live-a", storedID: "saved-a")
        state.apply(event("message.delta", "Checking now", seq: 1))
        state.apply(.object(["type": .string("message.interim"), "session_id": .string("live-a"), "seq": .number(2),
            "payload": .object(["text": .string("Checking now"), "already_streamed": .bool(true)])]))
        state.apply(event("message.delta", "Result", seq: 3))
        XCTAssertEqual(state.messages.map(\.text), ["Checking now", "Result"])
    }
    func testResumeRestoresPendingClarification() throws {
        let state = try SessionState(snapshot: .object(["session_id": .string("live-a"), "session_key": .string("saved-a"),
            "messages": .array([]), "pending_clarify": .object(["request_id": .string("q-a"), "question": .string("Which folder?")])]))
        XCTAssertEqual(state.requests.first?.id, "q-a")
        XCTAssertEqual(state.requests.first?.kind, .clarify)
    }
    func testFinalSettlesSealedInterimButPreservesDistinctReply() {
        for final in ["partial", "partial answer continued", "different result"] {
            var state = SessionState(sessionID: "live-a", storedID: "saved-a")
            state.apply(event("message.start", "", seq: 1))
            state.apply(event("message.interim", "partial", seq: 2))
            state.apply(event("message.start", "", seq: 3))
            state.apply(event("message.complete", final, seq: 4))
            XCTAssertEqual(state.messages.map(\.text), final.hasPrefix("partial") ? [final] : ["partial", final])
        }
    }
    func testUnknownDeliveryBlocksAnotherSendUntilReload() {
        var state = SessionState(sessionID: "live-a", storedID: "saved-a")
        XCTAssertTrue(state.canSend)
        state.beginSend("do work")
        state.markUncertain()
        XCTAssertFalse(state.canSend)
        XCTAssertEqual(state.delivery, .uncertain)
        XCTAssertEqual(state.messages.filter { $0.role == "user" }.count, 1)
        state.disconnect()
        XCTAssertEqual(state.delivery, .uncertain)
    }
    func testApprovalResponseUsesExactRequestAndSessionAndOnlyOnceOrDeny() throws {
        let request = PendingRequest(id: "req-a", kind: .approval, payload: .object([
            "request_id": .string("req-a"), "command": .string("python check.py"),
            "choices": .array([.string("once"), .string("deny")])]))
        let command = try Commands.respond(request, sessionID: "live-a", answer: "once")
        XCTAssertEqual(command.method, "approval.respond")
        XCTAssertEqual(command.params["request_id"].string, "req-a")
        XCTAssertEqual(command.params["session_id"].string, "live-a")
        XCTAssertEqual(command.params["choice"].string, "once")
        XCTAssertEqual(command.params["all"].bool, false)
        XCTAssertThrowsError(try Commands.respond(request, sessionID: "live-a", answer: "always"))
    }
    func testDisconnectClearsPendingApprovals() {
        var state = SessionState(sessionID: "live-a", storedID: "saved-a")
        state.apply(.object(["type": .string("approval.request"), "session_id": .string("live-a"),
                            "payload": .object(["request_id": .string("req-a"), "command": .string("work")])]))
        XCTAssertEqual(state.requests.count, 1)
        state.disconnect()
        XCTAssertTrue(state.requests.isEmpty)
        XCTAssertFalse(state.canSend)
    }
    func testTaskPathRejectsTraversalAndCreateDefaultsToTriage() throws {
        XCTAssertThrowsError(try Commands.taskPath("../config"))
        XCTAssertEqual(try Commands.taskPath("t_123"), "/api/plugins/kanban/tasks/t_123")
        let body = try Commands.createTask(title: "Audit", body: "Check", assignee: "")
        XCTAssertEqual(body["triage"].bool, true)
        XCTAssertEqual(body["workspace_kind"].string, "scratch")
        XCTAssertNotNil(body["idempotency_key"].string)
        XCTAssertThrowsError(try Commands.createTask(title: "  ", body: "", assignee: ""))
    }
    func testUnknownBackendContractKeepsHistoryButDisablesControls() throws {
        let result: JSON = .object(["session_id": .string("live-a"), "messages": .array([
            .object(["role": .string("assistant"), "text": .string("saved reply")])]),
            "info": .object(["desktop_contract": .number(999)])])
        let state = try SessionState(snapshot: result)
        XCTAssertEqual(state.messages.first?.text, "saved reply")
        XCTAssertFalse(state.canSend)
        XCTAssertTrue(state.needsRefresh)
    }
    func testHistorySkipsHiddenAndPreservesSourceRole() throws {
        let result: JSON = .object(["session_id": .string("live-a"), "stored_session_id": .string("saved-a"),
            "messages": .array([
                .object(["role": .string("user"), "text": .string("hello")]),
                .object(["role": .string("assistant"), "text": .string("hidden"), "display_kind": .string("hidden")]),
                .object(["role": .string("assistant"), "text": .string("answer")])])])
        let state = try SessionState(snapshot: result)
        XCTAssertEqual(state.messages.map(\.text), ["hello", "answer"])
        XCTAssertEqual(state.storedID, "saved-a")
    }
}
