import Foundation
import HermesCore
public struct RPCCommand {
    public let method: String
    public let params: JSON
}
public enum Commands {
    public static func respond(_ request: PendingRequest, sessionID: String, answer: String) throws -> RPCCommand {
        guard !sessionID.isEmpty, !request.id.isEmpty else { throw FeatureError.invalid("Request identity is missing. Refresh first.") }
        var values: [String: JSON] = ["session_id": .string(sessionID), "request_id": .string(request.id)]
        if request.kind == .approval {
            let choices = request.payload["choices"].array?.compactMap(\.string) ?? ["once", "deny"]
            guard ["once", "deny"].contains(answer), choices.contains(answer) else {
                throw FeatureError.invalid("This approval choice is not available.")
            }
            values["choice"] = .string(answer); values["all"] = .bool(false)
            return RPCCommand(method: "approval.respond", params: .object(values))
        }
        guard request.payload["questions"].array == nil else {
            throw FeatureError.invalid("Answer this multi-question request on the Mac.")
        }
        guard !answer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw FeatureError.invalid("Enter an answer before continuing.")
        }
        values["answer"] = .string(answer)
        return RPCCommand(method: "clarify.respond", params: .object(values))
    }
    public static func taskPath(_ id: String) throws -> String {
        guard !id.isEmpty, id.unicodeScalars.allSatisfy({ CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "_-" )).contains($0) }) else {
            throw FeatureError.invalid("Invalid task identity.")
        }
        return "/api/plugins/kanban/tasks/" + id
    }
    public static func createTask(title: String, body: String, assignee: String) throws -> JSON {
        let title = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !title.isEmpty else { throw FeatureError.invalid("A task needs a title.") }
        var values: [String: JSON] = ["title": .string(title), "body": .string(body), "triage": .bool(true),
                                      "workspace_kind": .string("scratch"), "idempotency_key": .string(UUID().uuidString)]
        if !assignee.isEmpty { values["assignee"] = .string(assignee) }
        return .object(values)
    }
}
