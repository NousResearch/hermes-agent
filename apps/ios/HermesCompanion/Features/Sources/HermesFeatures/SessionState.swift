import Foundation
import HermesCore

public enum FeatureError: LocalizedError {
    case invalid(String)
    public var errorDescription: String? { if case .invalid(let message) = self { return message }; return nil }
}
public struct ChatMessage: Identifiable, Equatable {
    public let id: UUID
    public var role: String
    public var text: String
    public init(role: String, text: String) { self.id = UUID(); self.role = role; self.text = text }
}
public enum Delivery: Equatable { case idle, sending, accepted, uncertain }
public struct PendingRequest: Identifiable, Equatable {
    public enum Kind: String { case approval, clarify }
    public let id: String
    public let kind: Kind
    public let payload: JSON
    public init(id: String, kind: Kind, payload: JSON) { self.id = id; self.kind = kind; self.payload = payload }
    public var title: String { payload["command"].string ?? payload["question"].string ?? "Action required" }
}
public struct SessionState {
    public let sessionID: String
    public let storedID: String
    public var messages: [ChatMessage] = []
    public var requests: [PendingRequest] = []
    public var lastSequence: Int?
    public var running = false
    public var needsRefresh = false
    public private(set) var contractSupported = true
    public var delivery: Delivery = .idle
    public var activity = "Ready"
    private var streamIndex: Int?
    private var sealedInterimIndex: Int?
    public var canSend: Bool { !needsRefresh && !running && delivery != .sending && delivery != .uncertain }
    public init(sessionID: String, storedID: String) { self.sessionID = sessionID; self.storedID = storedID }
    public init(snapshot: JSON) throws {
        guard let sid = snapshot["session_id"].string, !sid.isEmpty,
              let rows = snapshot["messages"].array else {
            throw FeatureError.invalid("Hermes returned an incomplete conversation. Refresh before sending.")
        }
        let durableID = ["stored_session_id", "session_key", "resumed"].compactMap { snapshot[$0].string }.first { !$0.isEmpty }
        self.init(sessionID: sid, storedID: durableID ?? sid)
        messages = rows.compactMap { row in
            guard row["display_kind"].string != "hidden", let role = row["role"].string,
                  let text = row["text"].string ?? row["content"].string else { return nil }
            return ChatMessage(role: role, text: text)
        }
        running = snapshot["running"].bool ?? snapshot["info"]["running"].bool ?? false
        activity = running ? "Working on Mac" : "Ready"
        addPending(snapshot["pending_clarify"], kind: .clarify)
        addPending(snapshot["pending_approval"], kind: .approval)
        inspectContract(snapshot["info"])
    }
    public mutating func beginSend(_ text: String) {
        guard canSend else { return }
        messages.append(ChatMessage(role: "user", text: text))
        sealedInterimIndex = nil
        delivery = .sending; running = true; activity = "Sending"
    }
    public mutating func markAccepted() { if delivery == .sending { delivery = .accepted }; activity = "Working on Mac" }
    public mutating func markUncertain() { delivery = .uncertain; needsRefresh = true; activity = "Delivery unknown · refresh" }
    public mutating func disconnect() {
        if delivery == .sending { delivery = .uncertain }
        requests.removeAll(); needsRefresh = true; activity = "Disconnected · Mac may still be working"
    }
    public mutating func addPending(_ payload: JSON, kind: PendingRequest.Kind) {
        guard let id = payload["request_id"].string, !id.isEmpty else { return }
        requests.removeAll { $0.id == id }
        requests.append(PendingRequest(id: id, kind: kind, payload: payload))
    }
    public mutating func apply(_ event: JSON) {
        guard event["session_id"].string == sessionID else { return }
        if let seq = event["seq"].int {
            if let last = lastSequence {
                guard seq > last else { return }
                if seq > last + 1 { needsRefresh = true; requests.removeAll(); activity = "Stream gap · refresh"; return }
            }
            lastSequence = seq
        }
        let type = event["type"].string ?? ""
        let payload = event["payload"]
        let text = payload["text"].string ?? ""
        switch type {
        case "session.info":
            if let active = payload["running"].bool { running = active }
            inspectContract(payload)
        case "message.start":
            running = true; streamIndex = nil; activity = "Working on Mac"
        case "message.delta":
            appendStream(text); running = true
        case "message.interim":
            if payload["already_streamed"].bool == true, let index = streamIndex, messages.indices.contains(index) {
                if !text.isEmpty { messages[index].text = text }
                sealedInterimIndex = index
            } else if !text.isEmpty {
                messages.append(ChatMessage(role: "assistant", text: text)); sealedInterimIndex = messages.count - 1
            }
            streamIndex = nil
        case "message.complete":
            if let index = streamIndex, messages.indices.contains(index) {
                if !text.isEmpty { messages[index].text = text }
            } else if let index = sealedInterimIndex, messages.indices.contains(index),
                      !text.isEmpty, text.hasPrefix(messages[index].text) {
                messages[index].text = text
            } else if !text.isEmpty {
                messages.append(ChatMessage(role: "assistant", text: text))
            }
            streamIndex = nil; sealedInterimIndex = nil; running = false; delivery = .idle; activity = "Ready"
        case "tool.start", "tool.progress", "tool.complete":
            activity = (payload["name"].string ?? payload["tool"].string ?? "Tool") + (type == "tool.complete" ? " · complete" : " · working")
        case "approval.request": addPending(payload, kind: .approval); activity = "Approval required"
        case "clarify.request": addPending(payload, kind: .clarify); activity = "Your answer is needed"
        case "clarify.expire", "approval.expire", "approval.resolved":
            requests.removeAll { $0.id == payload["request_id"].string }
        case "secret.request", "sudo.request": activity = "Secure input required on Mac"
        case "error":
            needsRefresh = true; activity = payload["message"].string ?? "Hermes reported an error · refresh"
        default: break
        }
    }
    private mutating func inspectContract(_ info: JSON) {
        let value = info["desktop_contract"]
        // Some existing resume paths omit this field. Source-surface verification
        // covers that legacy shape; an explicitly unknown contract is never accepted.
        if value != .null, value.int != 6 {
            contractSupported = false; needsRefresh = true; requests.removeAll()
            activity = "Backend protocol not verified · update companion"
        }
    }
    private mutating func appendStream(_ text: String) {
        guard !text.isEmpty else { return }
        if let index = streamIndex, messages.indices.contains(index) { messages[index].text += text }
        else { messages.append(ChatMessage(role: "assistant", text: text)); streamIndex = messages.count - 1 }
    }
}
